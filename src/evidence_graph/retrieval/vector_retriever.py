"""Dense vector retriever: semantic search, exact lookup, and composition matching.

Pipeline:
  query → embedding (BGE) → ANN search (ChromaDB) → cross-encoder reranking
        → top-k VectorEvidence objects with provenance

The retriever also supports direct record lookup by name or by composition,
bypassing the embedding path when an exact match exists.
"""

import json
from pathlib import Path
from typing import Any, Optional

import chromadb

from evidence_graph.core.config import EvidenceGraphSettings, load_settings
from evidence_graph.core.exceptions import VectorDBError
from evidence_graph.core.logging import get_logger
from evidence_graph.core.models import RetrievalCandidate, SourceType, VectorEvidence
from evidence_graph.retrieval.reranker import load_reranker

logger = get_logger("retrieval.vector")


class VectorRetriever:
    """Retrieves documents from ChromaDB via exact lookup or semantic search.

    Three retrieval paths:
      1. **Exact name** — direct lookup in the in-memory name index.
      2. **Composition** — matches a composition string against the index.
      3. **Semantic** — embeds the query, searches the ANN index, reranks.
    """

    def __init__(self, settings: Optional[EvidenceGraphSettings] = None):
        self._cfg = settings or load_settings()

        # Lazy-loaded heavy models
        self._embedder = None

        # In-memory record store and indexes
        self._records_by_id: dict[int, dict] = {}
        self._name_index: dict[str, int] = {}
        self._composition_index: dict[str, list[int]] = {}
        self._load_records()

        # ChromaDB connection
        self._collection = None
        self.connected = False
        self._connect_chroma()

    # ── Initialization ─────────────────────────────────────────────────

    def _load_records(self):
        """Load the canonical record store and build name/composition indexes."""
        path = Path(self._cfg.corpus_json_path)
        if not path.exists():
            logger.warning(
                "Corpus records not found at %s (run vector ingestion first)", path
            )
            return

        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for rec in records:
            rid = rec["medicine_id"]
            self._records_by_id[rid] = rec
            self._name_index[rec["product_name"].lower().strip()] = rid
            composition = (rec.get("salt_composition") or "").lower().strip()
            if composition:
                self._composition_index.setdefault(composition, []).append(rid)

        logger.info("Loaded %d corpus records", len(self._records_by_id))

    def _connect_chroma(self):
        """Connect to the persistent ChromaDB collection."""
        try:
            client = chromadb.PersistentClient(path=self._cfg.chroma_persist_dir)
            self._collection = client.get_collection(name=self._cfg.chroma_collection)
            self.connected = True
            logger.info(
                "Connected to ChromaDB (%d chunks)", self._collection.count()
            )
        except Exception as exc:
            logger.warning("Failed to connect to ChromaDB: %s", exc)
            self.connected = False
            self._collection = None

    def _get_embedder(self):
        """Lazily load the sentence-transformer embedding model."""
        if self._embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading embedding model on %s", device)
            self._embedder = SentenceTransformer(
                self._cfg.embedding_model, device=device
            )
        return self._embedder

    # ── Public retrieval interface ─────────────────────────────────────

    def retrieve_by_query(
        self, query: str, context: Optional[str] = None
    ) -> dict[str, Any]:
        """Retrieve documents by composition lookup or semantic search.

        Tries composition matching first; falls back to dense retrieval.
        """
        if not self._records_by_id:
            return self._empty_result("No corpus records loaded")

        composition_ids = self._match_composition(query)
        if composition_ids:
            return self._build_response(
                composition_ids, query, match_type="composition_lookup"
            )

        enhanced = f"{query} {context}".strip() if context else query
        return self._semantic_retrieve(enhanced)

    def retrieve_by_name(self, name: str) -> dict[str, Any]:
        """Exact name lookup, falling back to semantic search."""
        if not self._records_by_id:
            return self._empty_result("No corpus records loaded")

        ids = self._match_name(name)
        if ids:
            return self._build_response(ids, name, match_type="exact_name")

        logger.info("No exact name match for '%s', using semantic search", name)
        return self._semantic_retrieve(name)

    def retrieve_for_entity(self, entity_name: str) -> dict[str, Any]:
        """Find documents related to a named entity via semantic search."""
        query = (
            f"Documents related to {entity_name}. "
            f"Treatment options for {entity_name}."
        )
        return self.retrieve_by_query(query, context=entity_name)

    # ── Name / composition matching ────────────────────────────────────

    def _match_name(self, name: str) -> list[int]:
        """Match a document name exactly, then by substring."""
        key = name.lower().strip()
        if not key:
            return []
        if key in self._name_index:
            return [self._name_index[key]]
        hits = [rid for n, rid in self._name_index.items() if key in n]
        return hits[: self._cfg.top_k_retrieval]

    def _match_composition(self, query: str) -> list[int]:
        """Match the query to a composition entry."""
        q = query.lower()
        if q.strip() in self._composition_index:
            return self._composition_index[q.strip()][: self._cfg.top_k_retrieval]
        for comp, ids in self._composition_index.items():
            if len(comp) >= 5 and comp in q:
                return ids[: self._cfg.top_k_retrieval]
        return []

    # ── Semantic retrieval ─────────────────────────────────────────────

    def _semantic_retrieve(self, query: str) -> dict[str, Any]:
        """Embed, search ANN index, cross-encoder rerank, deduplicate."""
        if not self.connected:
            return self._empty_result("Vector index not connected")

        logger.info("Semantic search: '%s'", query[:100])
        try:
            embedder = self._get_embedder()
            q_emb = embedder.encode(
                self._cfg.embedding_query_prefix + query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

            result = self._collection.query(
                query_embeddings=[q_emb],
                n_results=self._cfg.candidate_pool_size,
            )

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            if not docs:
                return self._build_response([], query, match_type="semantic")

            # Cross-encoder rerank
            reranker = load_reranker(self._cfg.reranker_model)
            scores = reranker.predict([(query, d) for d in docs])

            ranked = sorted(zip(scores, metas), key=lambda x: x[0], reverse=True)

            # Deduplicate by product name
            ordered_ids: list[int] = []
            score_by_id: dict[int, float] = {}
            seen_names: set = set()
            for score, meta in ranked:
                rid = meta.get("medicine_id")
                pname = meta.get("product_name") or str(rid)
                if rid is None or pname in seen_names:
                    continue
                seen_names.add(pname)
                score_by_id[rid] = float(score)
                ordered_ids.append(rid)
                if len(ordered_ids) >= self._cfg.top_k_retrieval:
                    break

            return self._build_response(
                ordered_ids, query, match_type="semantic", scores=score_by_id
            )
        except Exception as exc:
            logger.error("Semantic retrieval error: %s", exc)
            return self._empty_result(str(exc))

    # ── Response construction ──────────────────────────────────────────

    def _build_response(
        self,
        record_ids: list[int],
        query: str,
        match_type: str,
        scores: Optional[dict[int, float]] = None,
    ) -> dict[str, Any]:
        """Assemble full records for the given IDs into a response."""
        documents: list[dict] = []
        for rid in record_ids:
            rec = self._records_by_id.get(rid)
            if not rec:
                continue
            documents.append(
                {
                    "document_id": rid,
                    "name": rec["product_name"],
                    "description": rec.get("medicine_desc", ""),
                    "salt_composition": rec.get("salt_composition", ""),
                    "sub_category": rec.get("sub_category", ""),
                    "manufacturer": rec.get("manufacturer", ""),
                    "price": rec.get("price"),
                    "side_effects": rec.get("side_effects", []),
                    "drug_interactions": rec.get("drug_interactions", []),
                    "similarity_score": (scores or {}).get(rid),
                    "match_type": match_type,
                }
            )

        logger.info("Retrieved %d documents via %s", len(documents), match_type)
        return {
            "documents": documents,
            "source": "vector_index",
            "query": query,
            "query_type": match_type,
            "top_k": self._cfg.top_k_retrieval,
        }

    def as_vector_evidence(self, documents: list[dict]) -> list[VectorEvidence]:
        """Convert raw document dicts to typed VectorEvidence objects."""
        return [
            VectorEvidence(
                document_id=str(doc.get("document_id", "")),
                text=doc.get("description", "")[:500],
                similarity_score=doc.get("similarity_score") or 0.0,
                metadata={
                    k: v
                    for k, v in doc.items()
                    if k not in ("description", "similarity_score", "document_id")
                },
                provenance=f"ChromaDB doc:{doc.get('name', '')}",
            )
            for doc in documents
        ]

    def _empty_result(self, error_msg: str) -> dict:
        return {
            "documents": [],
            "source": "vector_index",
            "error": error_msg,
            "query": "fallback",
        }
