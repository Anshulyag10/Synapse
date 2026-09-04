"""Shared test fixtures for EvidenceGraph."""

import pytest

from evidence_graph.core.config import EvidenceGraphSettings


@pytest.fixture
def settings():
    """Settings with all external services disabled (for unit tests)."""
    return EvidenceGraphSettings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="test",
        groq_api_key=None,
        chroma_persist_dir="/tmp/test_chroma",
        corpus_json_path="/tmp/test_corpus.json",
        log_level="WARNING",
    )
