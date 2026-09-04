"""Tests for the grounding verifier."""

import pytest

from evidence_graph.grounding.verifier import (
    extract_claims,
    check_claim_support,
    verify_grounding,
)
from evidence_graph.core.models import (
    ClaimStatus,
    EvidenceBundle,
    VectorEvidence,
)


class TestClaimExtraction:
    def test_extracts_factual_sentences(self):
        text = (
            "Aspirin is used to treat pain and inflammation. "
            "It is also used as a blood thinner. "
            "This is for educational purposes only — not medical advice."
        )
        claims = extract_claims(text)
        assert len(claims) == 2  # disclaimer should be filtered out

    def test_filters_short_sentences(self):
        claims = extract_claims("Yes. No. Maybe a longer sentence here.")
        assert len(claims) == 1  # only the longer one

    def test_empty_text(self):
        assert extract_claims("") == []


class TestClaimSupport:
    def test_supported_claim(self):
        evidence = ["Aspirin is commonly used to treat headache and pain"]
        status, conf, supporting = check_claim_support(
            "Aspirin is used to treat headache", evidence
        )
        assert status == ClaimStatus.SUPPORTED
        assert conf > 0.5

    def test_unsupported_claim(self):
        evidence = ["Ibuprofen is used for inflammation"]
        status, conf, supporting = check_claim_support(
            "Aspirin cures cancer and diabetes", evidence
        )
        assert status == ClaimStatus.UNSUPPORTED

    def test_empty_evidence(self):
        status, conf, supporting = check_claim_support("any claim", [])
        assert status == ClaimStatus.UNSUPPORTED
        assert conf == 0.0


class TestVerifyGrounding:
    def test_fully_grounded_answer(self):
        bundle = EvidenceBundle(
            vector_evidence=[
                VectorEvidence(
                    document_id="1",
                    text="Aspirin is used to treat headache and pain. It may cause stomach irritation.",
                    similarity_score=0.9,
                )
            ]
        )
        result = verify_grounding(
            "Aspirin treats headache and may cause stomach irritation.",
            bundle,
        )
        assert result.total_claims > 0
        assert result.grounded_ratio > 0.0

    def test_empty_bundle(self):
        result = verify_grounding(
            "Some claim about something.", EvidenceBundle()
        )
        assert result.grounded_ratio == 0.0
