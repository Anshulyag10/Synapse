"""Grounding verifier: checks whether generated claims are supported by evidence.

Before returning a final answer, the verifier:
  1. Extracts factual claims from the generated answer.
  2. Checks each claim against the retrieved evidence (keyword overlap).
  3. Classifies claims as supported / partially_supported / unsupported.
  4. Returns a GroundingResult with an overall confidence score.

This does NOT claim perfect hallucination prevention — it is a best-effort
check documented as such.
"""

import re
from typing import Optional

from synapse.core.config import SynapseSettings, load_settings
from synapse.core.logging import get_logger
from synapse.core.models import (
    Citation,
    ClaimStatus,
    EvidenceBundle,
    GroundingResult,
)

logger = get_logger("grounding.verifier")


def extract_claims(answer: str) -> list[str]:
    """Extract factual claims from an answer text.

    Uses sentence splitting as a simple heuristic.  Each non-trivial
    sentence is treated as a claim.  Disclaimers and meta-sentences are
    filtered out.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = []
    skip_patterns = [
        r"educational.*only",
        r"not.*(?:medical|professional).*advice",
        r"consult.*(?:doctor|physician|healthcare)",
        r"our dataset does not",
        r"i could not find",
        r"please note",
    ]
    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        if any(re.search(pat, s, re.IGNORECASE) for pat in skip_patterns):
            continue
        claims.append(s)
    return claims


def check_claim_support(
    claim: str,
    evidence_texts: list[str],
    threshold: float = 0.6,
) -> tuple[ClaimStatus, float, list[str]]:
    """Check whether a claim is supported by any evidence text.

    Uses keyword overlap as a lightweight proxy.  Returns
    ``(status, confidence, supporting_evidence_ids)``.
    """
    claim_words = set(
        w.lower()
        for w in re.findall(r"\b\w{4,}\b", claim)  # words ≥ 4 chars
    )
    if not claim_words:
        return ClaimStatus.UNSUPPORTED, 0.0, []

    best_overlap = 0.0
    supporting: list[str] = []

    for i, ev_text in enumerate(evidence_texts):
        ev_words = set(
            w.lower() for w in re.findall(r"\b\w{4,}\b", ev_text)
        )
        if not ev_words:
            continue
        overlap = len(claim_words & ev_words) / len(claim_words)
        if overlap > best_overlap:
            best_overlap = overlap
        if overlap >= threshold:
            supporting.append(f"evidence_{i}")

    if best_overlap >= threshold:
        status = ClaimStatus.SUPPORTED
    elif best_overlap >= threshold * 0.5:
        status = ClaimStatus.PARTIALLY_SUPPORTED
    else:
        status = ClaimStatus.UNSUPPORTED

    return status, round(best_overlap, 3), supporting


def verify_grounding(
    answer: str,
    bundle: EvidenceBundle,
    settings: Optional[SynapseSettings] = None,
) -> GroundingResult:
    """Verify whether an answer is grounded in the retrieved evidence.

    Parameters
    ----------
    answer
        The generated answer text.
    bundle
        The evidence bundle used to generate the answer.
    settings
        Configuration (uses ``grounding_similarity_threshold``).

    Returns
    -------
    GroundingResult
        Claim-level grounding analysis.
    """
    cfg = settings or load_settings()
    threshold = cfg.grounding_similarity_threshold

    # Collect all evidence texts
    evidence_texts: list[str] = []
    for ev in bundle.graph_evidence:
        parts = [ev.entity_name]
        for k, v in ev.properties.items():
            if isinstance(v, list):
                parts.append(", ".join(str(x) for x in v))
            elif v:
                parts.append(str(v))
        evidence_texts.append(" ".join(parts))
    for ev in bundle.vector_evidence:
        evidence_texts.append(ev.text)
    for ev in bundle.lexical_evidence:
        evidence_texts.append(ev.text)
    for cand in bundle.fused_candidates:
        evidence_texts.append(cand.text)

    # Extract and verify claims
    claims = extract_claims(answer)
    supported_citations: list[Citation] = []
    unsupported_citations: list[Citation] = []

    for claim_text in claims:
        status, confidence, supporting = check_claim_support(
            claim_text, evidence_texts, threshold
        )
        citation = Citation(
            claim_text=claim_text,
            supporting_evidence=supporting,
            status=status,
            confidence=confidence,
        )
        if status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED):
            supported_citations.append(citation)
        else:
            unsupported_citations.append(citation)

    total = len(claims)
    grounded = len(supported_citations)
    ratio = grounded / total if total > 0 else 0.0
    overall = round(ratio, 3)

    result = GroundingResult(
        supported_claims=supported_citations,
        unsupported_claims=unsupported_citations,
        overall_confidence=overall,
        total_claims=total,
        grounded_ratio=ratio,
    )

    logger.info(
        "Grounding: %d/%d claims supported (%.1f%%)",
        grounded,
        total,
        ratio * 100,
    )
    return result
