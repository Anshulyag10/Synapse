"""Citation generator: maps answer claims to their supporting evidence.

After the grounding verifier has classified claims, this module creates
a human-readable citation mapping:

  Claim A → Evidence 3
  Claim B → Evidence 1 + Evidence 4

Each citation includes the claim text, supporting evidence references,
status, and confidence score.
"""

from evidence_graph.core.logging import get_logger
from evidence_graph.core.models import Citation, ClaimStatus, GroundingResult

logger = get_logger("grounding.citation")


def generate_citations(grounding: GroundingResult) -> list[Citation]:
    """Collect all citations (supported and unsupported) from a grounding result."""
    all_citations = list(grounding.supported_claims) + list(
        grounding.unsupported_claims
    )
    logger.info(
        "Generated %d citations (%d supported, %d unsupported)",
        len(all_citations),
        len(grounding.supported_claims),
        len(grounding.unsupported_claims),
    )
    return all_citations


def format_citations(citations: list[Citation]) -> str:
    """Format citations as a human-readable string for display."""
    if not citations:
        return "No citations available."

    lines = ["### Sources & Evidence"]
    supported = [c for c in citations if c.status != ClaimStatus.UNSUPPORTED]
    unsupported = [c for c in citations if c.status == ClaimStatus.UNSUPPORTED]

    if supported:
        lines.append("\n**Supported claims:**")
        for i, cit in enumerate(supported, 1):
            refs = ", ".join(cit.supporting_evidence) if cit.supporting_evidence else "general evidence"
            status_label = "✓" if cit.status == ClaimStatus.SUPPORTED else "~"
            lines.append(
                f"  {status_label} [{i}] {cit.claim_text[:120]} → {refs}"
            )

    if unsupported:
        lines.append("\n**Unverified claims:**")
        for cit in unsupported:
            lines.append(f"  ⚠ {cit.claim_text[:120]}")

    return "\n".join(lines)


def citation_coverage(citations: list[Citation]) -> float:
    """Compute the fraction of claims that are supported or partially supported."""
    if not citations:
        return 0.0
    supported = sum(
        1 for c in citations if c.status != ClaimStatus.UNSUPPORTED
    )
    return round(supported / len(citations), 3)
