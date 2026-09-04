"""Hybrid retriever: orchestrates graph, vector, and lexical backends.

Accepts a RetrievalRequest (or a QueryPlan) and dispatches to the
appropriate retriever(s), then delegates to evidence fusion.

This is the single entry point that the agent layer calls — individual
retrievers are not invoked directly by the agent.
"""

from typing import Any, Optional

from evidence_graph.core.config import EvidenceGraphSettings, load_settings
from evidence_graph.core.logging import get_logger
from evidence_graph.core.models import (
    EvidenceBundle,
    QueryPlan,
    SourceType,
)
from evidence_graph.retrieval.graph_retriever import GraphRetriever
from evidence_graph.retrieval.vector_retriever import VectorRetriever
from evidence_graph.retrieval.lexical_retriever import LexicalRetriever

logger = get_logger("retrieval.hybrid")


class HybridRetriever:
    """Unified retrieval interface that routes queries to the appropriate backends.

    Given a QueryPlan, it activates the requested sources and collects
    raw results from each.  Fusion and reranking happen downstream in
    ``evidence_fusion.py``.
    """

    def __init__(
        self,
        graph: Optional[GraphRetriever] = None,
        vector: Optional[VectorRetriever] = None,
        lexical: Optional[LexicalRetriever] = None,
        settings: Optional[EvidenceGraphSettings] = None,
    ):
        self._cfg = settings or load_settings()
        self.graph = graph or GraphRetriever(self._cfg)
        self.vector = vector or VectorRetriever(self._cfg)
        self.lexical = lexical or LexicalRetriever(self._cfg)

    # ── Public interface ───────────────────────────────────────────────

    def retrieve(self, plan: QueryPlan) -> dict[str, Any]:
        """Execute retrieval according to the query plan.

        Returns a raw results dict with keys ``graph_results``,
        ``vector_results``, ``lexical_results`` (each may be empty if the
        plan did not request that source or the backend is offline).
        """
        query = plan.original_query
        results: dict[str, Any] = {
            "graph_results": {},
            "vector_results": {},
            "lexical_results": {},
            "plan": plan.model_dump(),
        }

        for source in plan.sources:
            if source == SourceType.KNOWLEDGE_GRAPH:
                results["graph_results"] = self._retrieve_graph(query, plan)
            elif source == SourceType.VECTOR_INDEX:
                results["vector_results"] = self._retrieve_vector(query)
            elif source == SourceType.LEXICAL_INDEX:
                results["lexical_results"] = self._retrieve_lexical(query)

        logger.info(
            "Hybrid retrieval complete — graph:%d  vector:%d  lexical:%d",
            len(results["graph_results"].get("entities", [])),
            len(results["vector_results"].get("documents", [])),
            len(results["lexical_results"].get("documents", [])),
        )
        return results

    def retrieve_all(self, query: str) -> dict[str, Any]:
        """Retrieve from every available backend (used for ablation studies)."""
        plan = QueryPlan(
            original_query=query,
            query_type="multi_hop",
            sources=[
                SourceType.KNOWLEDGE_GRAPH,
                SourceType.VECTOR_INDEX,
                SourceType.LEXICAL_INDEX,
            ],
        )
        return self.retrieve(plan)

    # ── Backend dispatch ───────────────────────────────────────────────

    def _retrieve_graph(self, query: str, plan: QueryPlan) -> dict[str, Any]:
        """Route to the graph retriever based on query semantics."""
        if not self.graph.connected:
            logger.warning("Graph retriever unavailable — skipping")
            return {"entities": [], "error": "knowledge graph offline"}

        try:
            # For multi-hop or relational queries, treat the query as attribute terms
            terms = [t.strip() for t in query.split(",") if t.strip()]
            if not terms:
                terms = [query]

            if plan.query_type.value in ("relational_query", "multi_hop"):
                return self.graph.retrieve_by_attributes(
                    terms, two_hop=True
                )
            elif plan.query_type.value == "direct_lookup":
                return self.graph.retrieve_entity_details(query)
            else:
                return self.graph.retrieve_by_attributes(terms)
        except Exception as exc:
            logger.error("Graph retrieval failed: %s", exc)
            return {"entities": [], "error": str(exc)}

    def _retrieve_vector(self, query: str) -> dict[str, Any]:
        """Route to the vector retriever."""
        if not self.vector.connected:
            logger.warning("Vector retriever unavailable — skipping")
            return {"documents": [], "error": "vector index offline"}

        try:
            return self.vector.retrieve_by_query(query)
        except Exception as exc:
            logger.error("Vector retrieval failed: %s", exc)
            return {"documents": [], "error": str(exc)}

    def _retrieve_lexical(self, query: str) -> dict[str, Any]:
        """Route to the BM25 lexical retriever."""
        if not self.lexical.ready:
            logger.warning("Lexical retriever unavailable — skipping")
            return {"documents": [], "error": "lexical index offline"}

        try:
            return self.lexical.retrieve(query)
        except Exception as exc:
            logger.error("Lexical retrieval failed: %s", exc)
            return {"documents": [], "error": str(exc)}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self):
        """Release resources held by the retrieval backends."""
        self.graph.close()
