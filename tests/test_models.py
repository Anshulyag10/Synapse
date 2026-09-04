"""Tests for typed data models."""

import pytest

from evidence_graph.core.models import (
    QueryCategory,
    QueryPlan,
    SourceType,
    EvidenceBundle,
    RetrievalCandidate,
    GraphEvidence,
    VectorEvidence,
    AnswerTrace,
    Citation,
    ClaimStatus,
    GroundingResult,
)


class TestQueryPlan:
    def test_create_basic_plan(self):
        plan = QueryPlan(
            original_query="test",
            query_type=QueryCategory.SEMANTIC_SEARCH,
            sources=[SourceType.VECTOR_INDEX],
        )
        assert plan.query_type == QueryCategory.SEMANTIC_SEARCH
        assert plan.requires_reranking is True

    def test_serialise_round_trip(self):
        plan = QueryPlan(
            original_query="fever cough",
            query_type=QueryCategory.MULTI_HOP,
            sources=[SourceType.KNOWLEDGE_GRAPH, SourceType.VECTOR_INDEX],
        )
        d = plan.model_dump()
        restored = QueryPlan(**d)
        assert restored.original_query == "fever cough"


class TestEvidenceBundle:
    def test_empty_bundle(self):
        bundle = EvidenceBundle()
        assert len(bundle.graph_evidence) == 0
        assert len(bundle.fused_candidates) == 0

    def test_bundle_with_graph(self):
        ev = GraphEvidence(entity_name="flu", properties={"symptoms": ["fever"]})
        bundle = EvidenceBundle(graph_evidence=[ev])
        assert bundle.graph_evidence[0].entity_name == "flu"


class TestAnswerTrace:
    def test_trace_defaults(self):
        trace = AnswerTrace(query="test query")
        assert trace.answer == ""
        assert trace.latency_ms == 0.0
        assert trace.timestamp > 0

    def test_trace_with_plan(self):
        plan = QueryPlan(
            original_query="test",
            query_type=QueryCategory.DIRECT_LOOKUP,
            sources=[SourceType.KNOWLEDGE_GRAPH],
        )
        trace = AnswerTrace(query="test", query_plan=plan, answer="result")
        assert trace.query_plan.query_type == QueryCategory.DIRECT_LOOKUP


class TestCitation:
    def test_citation_defaults(self):
        cit = Citation(claim_text="aspirin treats pain")
        assert cit.status == ClaimStatus.UNSUPPORTED
        assert cit.confidence == 0.0

    def test_grounding_result(self):
        result = GroundingResult(total_claims=5, grounded_ratio=0.8)
        assert result.total_claims == 5
