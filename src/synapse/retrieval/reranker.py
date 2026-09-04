"""Shared cross-encoder reranker loaded once and reused by all retrieval backends.

The reranker is a BAAI/bge-reranker-v2-m3 CrossEncoder from sentence-transformers.
It scores (query, passage) pairs and is used after initial candidate retrieval to
improve ranking precision before the final top-k selection.
"""

import functools
from typing import Optional

from synapse.core.logging import get_logger

logger = get_logger("retrieval.reranker")

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@functools.lru_cache(maxsize=2)
def load_reranker(model_name: str = _DEFAULT_MODEL):
    """Load a CrossEncoder reranker model (cached per model name for the process).

    Falls back to CPU when CUDA is unavailable.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading cross-encoder %s on %s (one-time)", model_name, device)
    return CrossEncoder(model_name, device=device)


def rerank_pairs(
    query: str,
    candidates: list[str],
    model_name: str = _DEFAULT_MODEL,
    top_n: Optional[int] = None,
) -> list[tuple[str, float]]:
    """Score (query, candidate) pairs and return them sorted descending.

    Parameters
    ----------
    query
        The user query.
    candidates
        Passage texts to rerank.
    model_name
        CrossEncoder model identifier.
    top_n
        If given, return only the top *n* results.

    Returns
    -------
    list[tuple[str, float]]
        ``(candidate_text, score)`` pairs, highest score first.
    """
    if not candidates:
        return []

    reranker = load_reranker(model_name)
    scores = reranker.predict([(query, c) for c in candidates])
    ranked = sorted(
        zip(candidates, (float(s) for s in scores)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked
