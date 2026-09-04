"""Evidence fusion: merges multi-source retrieval results into a unified EvidenceBundle.

Pipeline:
  graph evidence + vector evidence + lexical evidence
    → candidate normalisation (common schema)
    → deduplication across sources
    → score normalisation (min-max per source type)
    → unified ranking
    → EvidenceBundle

Every fused candidate carries its source type, normalised score, rank,
text, metadata, and provenance string.
"""

from typing import Optional

from evidence_graph.core.config import EvidenceGraphSettings, load_settings
from evidence_graph.core.logging import get_logger
from evidence_graph.core.models import (
    EvidenceBundle,
    GraphEvidence,
    LexicalEvidence,
    RetrievalCandidate,
    SourceType,
    VectorEvidence,
)

logger = get_logger("grounding.fusion")


def _minmax_normalise(scores: list[float]) -> list[float]:
    """Normalise a list of scores to [0, 1] via min-max scaling."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_evidence(
    graph: list[GraphEvidence],
    vector: list[VectorEvidence],
    lexical: list[LexicalEvidence],
    settings: Optional[EvidenceGraphSettings] = None,
) -> EvidenceBundle:
    """Merge multi-source evidence into a single ranked EvidenceBundle.

    Parameters
    ----------
    graph, vector, lexical
        Evidence lists from each retriever (may be empty).
    settings
        Configuration for fusion weights.

    Returns
    -------
    EvidenceBundle
        The unified evidence with fused candidates ranked by weighted score.
    """
    cfg = settings or load_settings()
    weights = cfg.fusion_weights
    candidates: list[RetrievalCandidate] = []

    # ── Normalise graph evidence ───────────────────────────────────────
    if graph:
        raw_scores = [
            float(e.properties.get("matched_attributes", 1)) for e in graph
        ]
        norm_scores = _minmax_normalise(raw_scores)
        for ev, ns in zip(graph, norm_scores):
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"graph:{ev.entity_name}",
                    text=_graph_to_text(ev),
                    score=ns * weights.get("graph", 0.35),
                    source_type=SourceType.KNOWLEDGE_GRAPH,
                    metadata=ev.properties,
                    provenance=ev.provenance or f"KG entity:{ev.entity_name}",
                )
            )

    # ── Normalise vector evidence ──────────────────────────────────────
    if vector:
        raw_scores = [e.similarity_score for e in vector]
        norm_scores = _minmax_normalise(raw_scores)
        for ev, ns in zip(vector, norm_scores):
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"vector:{ev.document_id}",
                    text=ev.text,
                    score=ns * weights.get("vector", 0.40),
                    source_type=SourceType.VECTOR_INDEX,
                    metadata=ev.metadata,
                    provenance=ev.provenance or f"ChromaDB:{ev.document_id}",
                )
            )

    # ── Normalise lexical evidence ─────────────────────────────────────
    if lexical:
        raw_scores = [e.bm25_score for e in lexical]
        norm_scores = _minmax_normalise(raw_scores)
        for ev, ns in zip(lexical, norm_scores):
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"lexical:{ev.document_id}",
                    text=ev.text,
                    score=ns * weights.get("lexical", 0.25),
                    source_type=SourceType.LEXICAL_INDEX,
                    metadata=ev.metadata,
                    provenance=ev.provenance or f"BM25:{ev.document_id}",
                )
            )

    # ── Deduplicate by candidate name ──────────────────────────────────
    seen: dict[str, int] = {}
    deduped: list[RetrievalCandidate] = []
    for cand in candidates:
        key = _dedup_key(cand)
        if key in seen:
            # Keep the higher-scored version
            idx = seen[key]
            if cand.score > deduped[idx].score:
                deduped[idx] = cand
        else:
            seen[key] = len(deduped)
            deduped.append(cand)

    # ── Rank by fused score ────────────────────────────────────────────
    deduped.sort(key=lambda c: c.score, reverse=True)
    for rank, cand in enumerate(deduped, 1):
        cand.rank = rank

    # ── Build summary ──────────────────────────────────────────────────
    source_summary = {}
    for cand in deduped:
        st = cand.source_type.value
        source_summary[st] = source_summary.get(st, 0) + 1

    bundle = EvidenceBundle(
        graph_evidence=list(graph),
        vector_evidence=list(vector),
        lexical_evidence=list(lexical),
        fused_candidates=deduped,
        source_summary=source_summary,
    )

    logger.info(
        "Fused %d candidates (graph:%d  vector:%d  lexical:%d  → deduped:%d)",
        len(candidates),
        len(graph),
        len(vector),
        len(lexical),
        len(deduped),
    )
    return bundle


def _dedup_key(cand: RetrievalCandidate) -> str:
    """Generate a dedup key from the candidate's name or ID."""
    name = cand.metadata.get("name", "") or cand.metadata.get("entity_name", "")
    return (name or cand.candidate_id).lower().strip()


def _graph_to_text(ev: GraphEvidence) -> str:
    """Convert graph evidence to a text representation for scoring."""
    parts = [ev.entity_name]
    for key, val in ev.properties.items():
        if isinstance(val, list):
            parts.append(f"{key}: {', '.join(str(v) for v in val)}")
        elif val:
            parts.append(f"{key}: {val}")
    return "; ".join(parts)
