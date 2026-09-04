"""Vector indexer: clean CSV → corpus_records.json → chunks → embeddings → ChromaDB.

Pipeline:
  1. Clean each CSV row into a structured record.
  2. Write corpus_records.json (the canonical record store).
  3. Build description and side-effect chunks for each record.
  4. Embed with BAAI/bge-base-en-v1.5.
  5. Store in ChromaDB with metadata.
"""

import csv
import json
import logging
import pickle
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb

from synapse.core.config import SynapseSettings, load_settings
from synapse.core.logging import get_logger

logger = get_logger("ingestion.vector")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class CorpusRecord:
    """Cleaned document record for the corpus store."""
    medicine_id: int
    product_name: str
    sub_category: str
    salt_composition: str
    manufacturer: str
    price: Optional[float]
    medicine_desc: str
    side_effects: list[str] = field(default_factory=list)
    drug_interactions: list[dict] = field(default_factory=list)


def _clean_price(raw: str) -> Optional[float]:
    if not raw:
        return None
    cleaned = re.sub(r"[^\d.]", "", raw)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_side_effects(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_interactions(raw: str) -> list[dict]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    drugs = data.get("drug", []) or []
    effects = data.get("effect", []) or []
    interactions = []
    for i, drug in enumerate(drugs):
        drug = (drug or "").strip()
        if not drug:
            continue
        effect = (effects[i] if i < len(effects) else "UNKNOWN") or "UNKNOWN"
        interactions.append({"drug": drug, "effect": effect.strip()})
    return interactions


def load_and_clean_csv(csv_path: str) -> list[CorpusRecord]:
    """Read and clean the source CSV into CorpusRecord objects."""
    records: list[CorpusRecord] = []
    next_id = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("product_name") or "").strip()
            if not name:
                continue
            records.append(
                CorpusRecord(
                    medicine_id=next_id,
                    product_name=name,
                    sub_category=(row.get("sub_category") or "").strip(),
                    salt_composition=(row.get("salt_composition") or "").strip(),
                    manufacturer=(row.get("product_manufactured") or "").strip(),
                    price=_clean_price(row.get("product_price", "")),
                    medicine_desc=(row.get("medicine_desc") or "").strip(),
                    side_effects=_parse_side_effects(row.get("side_effects", "")),
                    drug_interactions=_parse_interactions(row.get("drug_interactions", "")),
                )
            )
            next_id += 1
    logger.info("Loaded and cleaned %d records from CSV", len(records))
    return records


def save_corpus_json(records: list[CorpusRecord], path: str):
    """Write the canonical record store."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False)
    logger.info("Wrote %d records to %s", len(records), path)


def _build_description_chunk(rec: CorpusRecord) -> dict:
    parts = [f"medicine name: {rec.product_name}"]
    if rec.sub_category:
        parts.append(f"category: {rec.sub_category}")
    if rec.salt_composition:
        parts.append(f"composition: {rec.salt_composition}")
    if rec.medicine_desc:
        parts.append(f"description:\n{rec.medicine_desc}")
    return {
        "chunk_id": f"{rec.medicine_id}_desc",
        "text": "\n".join(parts),
        "metadata": _chunk_metadata(rec, "description"),
    }


def _build_side_effects_chunk(rec: CorpusRecord) -> Optional[dict]:
    if not rec.side_effects:
        return None
    text = f"medicine: {rec.product_name}\nside effects:\n" + "\n".join(rec.side_effects)
    return {
        "chunk_id": f"{rec.medicine_id}_side_effects",
        "text": text,
        "metadata": _chunk_metadata(rec, "side_effects"),
    }


def _chunk_metadata(rec: CorpusRecord, chunk_type: str) -> dict:
    return {
        "medicine_id": rec.medicine_id,
        "chunk_type": chunk_type,
        "product_name": rec.product_name.lower(),
        "salt_composition": rec.salt_composition.lower(),
        "sub_category": rec.sub_category.lower(),
    }


def build_chunks(records: list[CorpusRecord]) -> tuple[list, list, list]:
    """Build description and side-effect chunks for all records."""
    ids, documents, metadatas = [], [], []
    for rec in records:
        desc = _build_description_chunk(rec)
        ids.append(desc["chunk_id"])
        documents.append(desc["text"])
        metadatas.append(desc["metadata"])

        se = _build_side_effects_chunk(rec)
        if se is not None:
            ids.append(se["chunk_id"])
            documents.append(se["text"])
            metadatas.append(se["metadata"])

    n_se = sum(1 for cid in ids if cid.endswith("_side_effects"))
    logger.info(
        "Built %d chunks (%d description + %d side_effects)",
        len(documents),
        len(records),
        n_se,
    )
    return ids, documents, metadatas


class DocumentEmbedder:
    """Embeds document chunks using a sentence-transformer model."""

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model: %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Model loaded (dim: %d)", self.model.get_sentence_embedding_dimension()
        )

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


def generate_embeddings(
    records: list[CorpusRecord], model_name: str, cache_path: str
) -> tuple:
    """Build chunks, embed them, cache the result."""
    ids, documents, metadatas = build_chunks(records)
    embedder = DocumentEmbedder(model_name)
    logger.info("Generating embeddings for %d chunks", len(documents))
    embeddings = embedder.embed_batch(documents)

    # Cache
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {"ids": ids, "documents": documents, "embeddings": embeddings, "metadatas": metadatas},
            f,
        )
    logger.info("Embeddings cached to %s", cache_path)
    return ids, documents, embeddings, metadatas


def add_to_chromadb(
    ids, documents, embeddings, metadatas,
    persist_dir: str, collection_name: str,
    start_idx: int = 0, chunk_size: int = 1000,
):
    """Write embedded chunks into ChromaDB."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    if start_idx == 0:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Synapse document corpus", "hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' created", collection_name)
    else:
        collection = client.get_collection(collection_name)
        logger.info("Resuming from index %d", start_idx)

    emb_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

    for i in range(start_idx, len(documents), chunk_size):
        end = min(i + chunk_size, len(documents))
        try:
            collection.add(
                ids=ids[i:end],
                embeddings=emb_list[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            logger.info("Added %d/%d chunks", end, len(documents))
        except Exception as exc:
            logger.error("Error at index %d: %s", i, exc)
            logger.info("Resume by running with --resume %d", i)
            raise

    logger.info("Vector indexing complete (%d chunks)", collection.count())


def main():
    """Entry point: clean CSV → embed → index."""
    cfg = load_settings()
    csv_path = Path(cfg.data_dir) / "medicine_data.csv"
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return

    resume_idx = 0
    if len(sys.argv) > 2 and sys.argv[1] == "--resume":
        resume_idx = int(sys.argv[2])

    cache_path = str(Path(cfg.chroma_persist_dir).parent / "embeddings_cache.pkl")
    cache = None
    if Path(cache_path).exists() and "--regenerate" not in sys.argv:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logger.info("Using cached embeddings")

    if cache:
        ids, documents, embeddings, metadatas = (
            cache["ids"], cache["documents"], cache["embeddings"], cache["metadatas"]
        )
    else:
        records = load_and_clean_csv(str(csv_path))
        save_corpus_json(records, cfg.corpus_json_path)
        ids, documents, embeddings, metadatas = generate_embeddings(
            records, cfg.embedding_model, cache_path
        )

    add_to_chromadb(
        ids, documents, embeddings, metadatas,
        cfg.chroma_persist_dir, cfg.chroma_collection,
        start_idx=resume_idx,
    )


if __name__ == "__main__":
    main()
