"""Knowledge-graph retriever: entity resolution, candidate generation, and graph expansion.

Retrieval pipeline:
  user terms → exact/alias/fuzzy/BM25 candidates → cross-encoder seed reranking
             → Cypher graph expansion → GraphEvidence objects with provenance

The retriever is domain-agnostic at the interface level — it resolves entities
and traverses relationships.  Healthcare-specific node labels (Disease, Symptom,
Treatment) are configured via the domain adapter.
"""

import os
import time
from typing import Any, Optional

from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi

from synapse.core.config import SynapseSettings, load_settings
from synapse.core.exceptions import GraphConnectionError, RetrievalError
from synapse.core.logging import get_logger
from synapse.core.models import (
    GraphEvidence,
    RetrievalCandidate,
    SourceType,
)
from synapse.retrieval.reranker import rerank_pairs

logger = get_logger("retrieval.graph")


class GraphRetriever:
    """Retrieves entities and relationships from a Neo4j knowledge graph.

    Candidate generation uses a four-stage cascade:
      1. Exact canonical name match
      2. Alias match
      3. Fuzzy match (token_sort_ratio ≥ threshold)
      4. BM25 keyword fallback

    After candidates are gathered they are reranked with a cross-encoder and
    the top seeds are expanded through the graph.
    """

    def __init__(self, settings: Optional[SynapseSettings] = None):
        self._cfg = settings or load_settings()
        self._reranker_model = self._cfg.reranker_model

        # In-memory indexes (built from the graph on connect)
        self._entity_canon: list[str] = []
        self._alias_to_canon: dict[str, str] = {}
        self._surface_strings: list[str] = []
        self._surface_to_canon: list[str] = []
        self._primary_names: list[str] = []  # e.g. disease names
        self._bm25: Optional[BM25Okapi] = None

        self._driver = None
        self._database = self._cfg.neo4j_database
        self.connected = False
        self._connect()

    # ── Connection management ──────────────────────────────────────────

    def _connect(self):
        """Establish a Neo4j driver connection and build in-memory indexes."""
        try:
            self._driver = GraphDatabase.driver(
                self._cfg.neo4j_uri,
                auth=(self._cfg.neo4j_username, self._cfg.neo4j_password),
                keep_alive=True,
                max_connection_lifetime=300,
            )
            with self._driver.session(database=self._database) as s:
                s.run("RETURN 1")
            self.connected = True
            logger.info("Connected to Neo4j knowledge graph")
            self._build_indexes()
        except Exception as exc:
            logger.warning("Failed to connect to Neo4j: %s", exc)
            self.connected = False
            self._driver = None

    def _reconnect(self):
        """Re-establish the Neo4j driver with retries."""
        for attempt in range(3):
            try:
                if self._driver:
                    try:
                        self._driver.close()
                    except Exception:
                        pass
                self._driver = GraphDatabase.driver(
                    self._cfg.neo4j_uri,
                    auth=(self._cfg.neo4j_username, self._cfg.neo4j_password),
                    keep_alive=True,
                    max_connection_lifetime=300,
                )
                with self._driver.session(database=self._database) as s:
                    s.run("RETURN 1")
                logger.info("Reconnected to Neo4j")
                self.connected = True
                return
            except Exception as exc:
                logger.warning("Reconnect attempt %d/3 failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(3)
        self.connected = False

    def _execute(self, cypher: str, **params) -> list[dict]:
        """Run a Cypher query, reconnecting once on failure."""
        for attempt in range(2):
            try:
                with self._driver.session(database=self._database) as s:
                    return [r.data() for r in s.run(cypher, **params)]
            except Exception as exc:
                if attempt == 0:
                    logger.warning("Query error, reconnecting: %s", exc)
                    self._reconnect()
                    if not self.connected:
                        return []
                else:
                    logger.error("Query failed after reconnect: %s", exc)
                    return []
        return []

    # ── Index construction ─────────────────────────────────────────────

    def _build_indexes(self):
        """Load entity names and aliases into in-memory structures for fast lookup."""
        rows = self._execute(
            "MATCH (s:Symptom) RETURN s.name AS name, s.aliases AS aliases"
        )
        for r in rows:
            canon = r["name"]
            self._entity_canon.append(canon)
            self._alias_to_canon[canon] = canon
            self._surface_strings.append(canon)
            self._surface_to_canon.append(canon)
            for alias in r.get("aliases") or []:
                self._alias_to_canon[alias] = canon
                self._surface_strings.append(alias)
                self._surface_to_canon.append(canon)

        if self._surface_strings:
            self._bm25 = BM25Okapi([s.split() for s in self._surface_strings])

        self._primary_names = [
            r["name"]
            for r in self._execute("MATCH (d:Disease) RETURN d.name AS name")
        ]
        logger.info(
            "Indexed %d entity terms, %d primary records",
            len(self._entity_canon),
            len(self._primary_names),
        )

    # ── Candidate generation ───────────────────────────────────────────

    def _generate_candidates(self, term: str) -> dict[str, str]:
        """Find candidate entity matches via exact → alias → fuzzy → BM25."""
        term = term.strip().lower()
        found: dict[str, str] = {}
        if not term:
            return found

        # Stage 1 & 2: exact canonical / alias match
        if term in self._alias_to_canon:
            method = "exact" if term in self._entity_canon else "alias"
            found[self._alias_to_canon[term]] = method

        # Stage 3: fuzzy match
        cutoff = self._cfg.fuzzy_match_cutoff
        for surface, score, idx in process.extract(
            term,
            self._surface_strings,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=cutoff,
            limit=10,
        ):
            canon = self._surface_to_canon[idx]
            found.setdefault(canon, "fuzzy")

        # Stage 4: BM25 keyword fallback
        if self._bm25 is not None:
            scores = self._bm25.get_scores(term.split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in ranked[:5]:
                if scores[i] <= 0:
                    break
                found.setdefault(self._surface_to_canon[i], "bm25")

        return found

    def _select_seeds(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Rerank candidate entities and return the top seed nodes."""
        if not candidates:
            return []
        pool = candidates[: self._cfg.max_graph_candidates]
        return rerank_pairs(query, pool, self._reranker_model, self._cfg.kg_seed_count)

    # ── Public retrieval interface ─────────────────────────────────────

    def retrieve_by_attributes(
        self, attribute_terms: list[str], two_hop: bool = False,
    ) -> dict[str, Any]:
        """Map attribute terms (e.g. symptoms) to primary entities via the KG.

        Returns a dict with ``entities`` (list of dicts), ``seeds``, and provenance
        metadata.
        """
        if not self.connected:
            return self._empty_result("Knowledge graph not connected")
        if not self._entity_canon:
            return self._empty_result(
                "No indexed entities (run graph ingestion first)"
            )

        logger.info("Attribute retrieval: %s", attribute_terms)
        query_text = ", ".join(attribute_terms)

        all_candidates: dict[str, str] = {}
        for term in attribute_terms:
            for canon, method in self._generate_candidates(term).items():
                all_candidates.setdefault(canon, method)

        if not all_candidates:
            return self._empty_result("No matching entities in knowledge graph")

        seeds = self._select_seeds(query_text, list(all_candidates))
        seed_names = [name for name, _ in seeds]
        logger.info("Seeds after reranking: %s", seed_names)

        entities = self._expand_from_attributes(seed_names, two_hop)
        return {
            "entities": entities,
            "source": "knowledge_graph",
            "query_type": "attribute_mapping",
            "seeds": [{"entity": n, "score": round(s, 3)} for n, s in seeds],
        }

    def retrieve_entity_details(
        self, entity_name: str, two_hop: bool = False,
    ) -> dict[str, Any]:
        """Return the full record for a single named primary entity."""
        if not self.connected:
            return self._empty_result("Knowledge graph not connected")

        logger.info("Entity detail retrieval: %s", entity_name)
        target = self._resolve_primary_name(entity_name)
        if not target:
            return self._empty_result(
                f"Entity '{entity_name}' not found in knowledge graph"
            )

        entities = self._expand_entity(target, two_hop)
        return {
            "entities": entities,
            "source": "knowledge_graph",
            "query_type": "entity_detail",
        }

    # ── Graph expansion ────────────────────────────────────────────────

    def _expand_from_attributes(
        self, seed_names: list[str], two_hop: bool,
    ) -> list[dict]:
        """Find primary entities sharing the seed attribute nodes."""
        rows = self._execute(
            """
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.name IN $seeds
            WITH d, count(DISTINCT s) AS matched
            ORDER BY matched DESC
            LIMIT $limit
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
            RETURN d.name AS name, d.disease_code AS code,
                   d.contagious AS contagious, d.chronic AS chronic,
                   d.raw_treatments AS raw_treatments,
                   collect(DISTINCT all_s.name) AS symptoms,
                   collect(DISTINCT t.name) AS treatments,
                   matched
            """,
            seeds=seed_names,
            limit=self._cfg.max_expansion_nodes,
        )
        entities = [self._format_entity(r, matched=r.get("matched")) for r in rows]

        if two_hop and entities:
            entities = self._attach_related(entities)
        logger.info("Expanded to %d entities", len(entities))
        return entities

    def _expand_entity(self, name: str, two_hop: bool) -> list[dict]:
        """Fetch one primary entity with its full attribute set."""
        rows = self._execute(
            """
            MATCH (d:Disease {name: $name})
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
            RETURN d.name AS name, d.disease_code AS code,
                   d.contagious AS contagious, d.chronic AS chronic,
                   d.raw_treatments AS raw_treatments,
                   collect(DISTINCT s.name) AS symptoms,
                   collect(DISTINCT t.name) AS treatments
            """,
            name=name,
        )
        entities = [self._format_entity(r) for r in rows]
        if two_hop and entities:
            entities = self._attach_related(entities)
        return entities

    def _attach_related(self, entities: list[dict]) -> list[dict]:
        """Add 2-hop related entities that share a treatment edge."""
        max_hops = self._cfg.max_hops
        if max_hops < 2:
            return entities

        for entity in entities:
            if not entity.get("treatments"):
                continue
            related = self._execute(
                """
                MATCH (d:Disease {name: $name})-[:TREATED_BY]->(t:Treatment)
                      <-[:TREATED_BY]-(other:Disease)
                WHERE other.name <> $name
                RETURN DISTINCT other.name AS name,
                       collect(DISTINCT t.name) AS shared_treatments
                LIMIT $limit
                """,
                name=entity["name"],
                limit=self._cfg.max_expansion_nodes,
            )
            entity["related_entities"] = related
        return entities

    # ── Name resolution ────────────────────────────────────────────────

    def _resolve_primary_name(self, query: str) -> Optional[str]:
        """Resolve a query to a canonical primary entity name."""
        q = query.strip().lower()
        if q in self._primary_names:
            return q
        for n in self._primary_names:
            if q in n or n in q:
                return n
        match = process.extractOne(
            q,
            self._primary_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self._cfg.fuzzy_match_cutoff,
        )
        return match[0] if match else None

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_entity(row: dict, matched: Optional[int] = None) -> dict:
        """Shape a raw Cypher row into a clean entity dict."""
        out = {
            "name": row.get("name", "Unknown"),
            "symptoms": [s for s in (row.get("symptoms") or []) if s],
            "treatments": [t for t in (row.get("treatments") or []) if t],
            "raw_treatments": row.get("raw_treatments") or "",
            "contagious": row.get("contagious"),
            "chronic": row.get("chronic"),
        }
        if matched is not None:
            out["matched_attributes"] = matched
        return out

    def _empty_result(self, error_msg: str) -> dict:
        """Return an empty result with an error message."""
        return {
            "entities": [],
            "source": "knowledge_graph",
            "error": error_msg,
            "query_type": "fallback",
        }

    def as_graph_evidence(self, entities: list[dict]) -> list[GraphEvidence]:
        """Convert raw entity dicts to typed GraphEvidence objects."""
        evidence = []
        for entity in entities:
            evidence.append(
                GraphEvidence(
                    entity_name=entity.get("name", ""),
                    entity_type="Disease",
                    properties={
                        k: v
                        for k, v in entity.items()
                        if k not in ("name", "related_entities")
                    },
                    relationships=[
                        {"type": "RELATED_TO", **rel}
                        for rel in entity.get("related_entities", [])
                    ],
                    traversal_depth=1,
                    provenance=f"Neo4j entity:{entity.get('name', '')}",
                )
            )
        return evidence

    def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
