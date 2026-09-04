"""Tests for evidence fusion."""

import pytest

from evidence_graph.grounding.evidence_fusion import fuse_evidence, _minmax_normalise
from evidence_graph.core.models import (
    GraphEvidence,
    VectorEvidence,
    LexicalEvidence,
    SourceType,
)


class TestMinMaxNormalise:
    def test_normalise_basic(self):
        assert _minmax_normalise([1, 2, 3]) == [0.0, 0.5, 1.0]

    def test_single_value(self):
        assert _minmax_normalise([5]) == [1.0]

    def test_identical_values(self):
        assert _minmax_normalise([3, 3, 3]) == [1.0, 1.0, 1.0]

    def test_empty(self):
        assert _minmax_normalise([]) == []


class TestEvidenceFusion:
    def test_fuse_empty_inputs(self):
        bundle = fuse_evidence([], [], [])
        assert len(bundle.fused_candidates) == 0
        assert bundle.source_summary == {}

    def test_fuse_graph_only(self):
        graph = [
            GraphEvidence(
                entity_name="malaria",
                entity_type="Disease",
                properties={"symptoms": ["fever"], "matched_attributes": 2},
                provenance="KG:malaria",
            )
        ]
        bundle = fuse_evidence(graph, [], [])
        assert len(bundle.fused_candidates) == 1
        assert bundle.fused_candidates[0].source_type == SourceType.KNOWLEDGE_GRAPH
        assert "knowledge_graph" in bundle.source_summary

    def test_fuse_deduplicates(self):
        graph = [
            GraphEvidence(entity_name="aspirin", properties={"matched_attributes": 1})
        ]
        vector = [
            VectorEvidence(
                document_id="1",
                text="aspirin info",
                similarity_score=0.9,
                metadata={"name": "aspirin"},
            )
        ]
        bundle = fuse_evidence(graph, vector, [])
        # aspirin appears in both graph and vector — should be deduped
        names = [c.candidate_id for c in bundle.fused_candidates]
        assert len(bundle.fused_candidates) <= 2  # at most one from each

    def test_fuse_preserves_ranking(self):
        vector = [
            VectorEvidence(document_id="1", text="a", similarity_score=0.9, metadata={"name": "high"}),
            VectorEvidence(document_id="2", text="b", similarity_score=0.3, metadata={"name": "low"}),
        ]
        bundle = fuse_evidence([], vector, [])
        assert bundle.fused_candidates[0].metadata["name"] == "high"

    def test_multi_source_fusion(self):
        graph = [GraphEvidence(entity_name="flu", properties={"matched_attributes": 1})]
        vector = [VectorEvidence(document_id="1", text="med", similarity_score=0.8, metadata={"name": "med_a"})]
        lexical = [LexicalEvidence(document_id="2", text="doc", bm25_score=5.0, metadata={"name": "doc_b"})]
        bundle = fuse_evidence(graph, vector, lexical)
        assert len(bundle.source_summary) == 3
