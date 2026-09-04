"""Centralised configuration for EvidenceGraph.

All tuneable parameters live here as a single Pydantic *BaseSettings* object.
Values come from environment variables (with dotenv support) or from explicit
keyword arguments.  Nothing in the rest of the codebase should hard-code an
API key, model name, threshold, or database URL.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/../..


class EvidenceGraphSettings(BaseSettings):
    """Single source of truth for every configurable knob."""

    # ── Neo4j knowledge graph ──────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j Bolt URI")
    neo4j_username: str = Field("neo4j", description="Neo4j user")
    neo4j_password: str = Field("password", description="Neo4j password")
    neo4j_database: str = Field("neo4j", description="Neo4j database name")

    # ── ChromaDB vector store ──────────────────────────────────────────
    chroma_persist_dir: str = Field(
        str(_PROJECT_ROOT / "datastore" / "chroma_db"),
        description="Directory for ChromaDB persistent storage",
    )
    chroma_collection: str = Field("evidence_corpus", description="Chroma collection name")

    # ── Canonical record store ─────────────────────────────────────────
    corpus_json_path: str = Field(
        str(_PROJECT_ROOT / "datastore" / "corpus_records.json"),
        description="JSON file holding the full document records",
    )

    # ── Embedding model ────────────────────────────────────────────────
    embedding_model: str = Field(
        "BAAI/bge-base-en-v1.5",
        description="SentenceTransformer model for dense embedding",
    )
    embedding_query_prefix: str = Field(
        "Represent this sentence for searching relevant passages: ",
        description="Query-side instruction prefix for the embedding model",
    )

    # ── Cross-encoder reranker ─────────────────────────────────────────
    reranker_model: str = Field(
        "BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model used by all retrievers",
    )

    # ── Agent LLM ──────────────────────────────────────────────────────
    groq_api_key: Optional[str] = Field(None, description="Groq API key")
    agent_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Model used for the orchestration agent",
    )
    judge_model: str = Field(
        "llama-3.1-8b-instant",
        description="Model used for LLM-as-judge evaluation",
    )
    agent_temperature: float = Field(0.2, ge=0.0, le=2.0)
    agent_max_tokens: int = Field(1200, ge=100)
    max_tool_rounds: int = Field(6, ge=1, le=20)

    # ── Normalization (local LLM) ──────────────────────────────────────
    ollama_model: str = Field(
        "qwen2.5:7b-instruct",
        description="Local Ollama model for KG ingestion normalization and benchmarks",
    )

    # ── Retrieval tuning ───────────────────────────────────────────────
    top_k_retrieval: int = Field(5, ge=1, description="Final top-k after reranking")
    candidate_pool_size: int = Field(60, ge=1, description="ANN candidates before reranking")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Min similarity to keep")
    fuzzy_match_cutoff: int = Field(80, ge=0, le=100, description="RapidFuzz score floor")
    bm25_top_k: int = Field(20, ge=1, description="BM25 lexical retrieval top-k")

    # ── Knowledge graph traversal ──────────────────────────────────────
    max_hops: int = Field(2, ge=1, le=5, description="Max relationship hops in graph expansion")
    max_graph_candidates: int = Field(25, ge=1)
    max_expansion_nodes: int = Field(5, ge=1)
    kg_seed_count: int = Field(5, ge=1, description="Seed nodes after reranking")

    # ── Entity resolution ──────────────────────────────────────────────
    symptom_merge_threshold: int = Field(90, ge=0, le=100)
    treatment_merge_threshold: int = Field(90, ge=0, le=100)

    # ── Evidence fusion ────────────────────────────────────────────────
    fusion_weights: dict = Field(
        default={"graph": 0.35, "vector": 0.40, "lexical": 0.25},
        description="Relative weight of each retrieval source in fusion scoring",
    )

    # ── Grounding ──────────────────────────────────────────────────────
    grounding_similarity_threshold: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Min similarity for a claim to be considered grounded",
    )

    # ── API ─────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)

    # ── Data paths ──────────────────────────────────────────────────────
    data_dir: str = Field(
        str(_PROJECT_ROOT / "data" / "healthcare"),
        description="Directory containing domain CSV data",
    )

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Root log level")
    log_dir: str = Field(
        str(_PROJECT_ROOT / "logs"),
        description="Directory for log files",
    )

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def load_settings(**overrides) -> EvidenceGraphSettings:
    """Build a settings instance, allowing test/CLI overrides."""
    return EvidenceGraphSettings(**overrides)
