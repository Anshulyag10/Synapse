"""Lexical retriever: BM25 keyword search over the document corpus.

Complements dense vector retrieval by catching exact terminology that
embedding models may dilute.  Particularly useful for queries containing
specific drug names, compositions, or technical terms.

Pipeline:
  query → tokenize → BM25 score → top-k LexicalEvidence objects
"""

import json
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from synapse.core.config import SynapseSettings, load_settings
from synapse.core.logging import get_logger
from synapse.core.models import LexicalEvidence

logger = get_logger("retrieval.lexical")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class LexicalRetriever:
    """BM25 retrieval over the canonical document corpus.

    The index is built from the same ``corpus_records.json`` used by the
    vector retriever so that both indexes cover the same document set.
    """

    def __init__(self, settings: Optional[SynapseSettings] = None):
        self._cfg = settings or load_settings()
        self._records: list[dict] = []
        self._corpus_texts: list[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self.ready = False
        self._build_index()

    def _build_index(self):
        """Load records and build the BM25 index."""
        path = Path(self._cfg.corpus_json_path)
        if not path.exists():
            logger.warning(
                "Corpus records not found at %s — lexical retrieval unavailable", path
            )
            return

        with open(path, "r", encoding="utf-8") as f:
            self._records = json.load(f)

        # Build searchable text from each record: name + description + side effects
        for rec in self._records:
            parts = [rec.get("product_name", "")]
            if rec.get("salt_composition"):
                parts.append(rec["salt_composition"])
            if rec.get("medicine_desc"):
                parts.append(rec["medicine_desc"])
            if rec.get("side_effects"):
                parts.append(
                    " ".join(rec["side_effects"])
                    if isinstance(rec["side_effects"], list)
                    else str(rec["side_effects"])
                )
            self._corpus_texts.append(" ".join(parts))

        tokenized = [_tokenize(t) for t in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        self.ready = True
        logger.info("BM25 index built over %d documents", len(self._records))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> dict[str, Any]:
        """Run BM25 retrieval and return scored documents.

        Parameters
        ----------
        query
            Natural-language query.
        top_k
            Override the configured top-k.

        Returns
        -------
        dict
            ``documents`` list plus metadata.
        """
        if not self.ready or self._bm25 is None:
            return self._empty_result("BM25 index not built")

        k = top_k or self._cfg.bm25_top_k
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        documents: list[dict] = []
        seen_names: set = set()
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            rec = self._records[idx]
            pname = rec.get("product_name", "").lower()
            if pname in seen_names:
                continue
            seen_names.add(pname)
            documents.append(
                {
                    "document_id": rec.get("medicine_id", idx),
                    "name": rec.get("product_name", ""),
                    "description": rec.get("medicine_desc", ""),
                    "salt_composition": rec.get("salt_composition", ""),
                    "sub_category": rec.get("sub_category", ""),
                    "manufacturer": rec.get("manufacturer", ""),
                    "price": rec.get("price"),
                    "side_effects": rec.get("side_effects", []),
                    "drug_interactions": rec.get("drug_interactions", []),
                    "bm25_score": float(scores[idx]),
                    "match_type": "lexical",
                }
            )
            if len(documents) >= k:
                break

        logger.info("BM25 retrieved %d documents", len(documents))
        return {
            "documents": documents,
            "source": "lexical_index",
            "query": query,
            "query_type": "lexical",
            "top_k": k,
        }

    def as_lexical_evidence(self, documents: list[dict]) -> list[LexicalEvidence]:
        """Convert raw document dicts to typed LexicalEvidence objects."""
        return [
            LexicalEvidence(
                document_id=str(doc.get("document_id", "")),
                text=doc.get("description", "")[:500],
                bm25_score=doc.get("bm25_score", 0.0),
                metadata={
                    k: v
                    for k, v in doc.items()
                    if k not in ("description", "bm25_score", "document_id")
                },
                provenance=f"BM25 doc:{doc.get('name', '')}",
            )
            for doc in documents
        ]

    def _empty_result(self, error_msg: str) -> dict:
        return {
            "documents": [],
            "source": "lexical_index",
            "error": error_msg,
            "query": "fallback",
        }
