"""Tests for the configuration system."""

import pytest

from evidence_graph.core.config import EvidenceGraphSettings, load_settings


class TestConfiguration:
    def test_defaults_are_valid(self):
        cfg = EvidenceGraphSettings()
        assert cfg.neo4j_uri
        assert cfg.embedding_model == "BAAI/bge-base-en-v1.5"
        assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_override_via_kwargs(self):
        cfg = EvidenceGraphSettings(agent_model="test-model")
        assert cfg.agent_model == "test-model"

    def test_load_settings_returns_instance(self):
        cfg = load_settings()
        assert isinstance(cfg, EvidenceGraphSettings)

    def test_fusion_weights_sum(self):
        cfg = EvidenceGraphSettings()
        total = sum(cfg.fusion_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            EvidenceGraphSettings(agent_temperature=3.0)
