"""Tests for the query router."""

import pytest

from evidence_graph.agents.router import classify_query_rules, QueryRouter
from evidence_graph.core.models import QueryCategory


class TestRuleBasedRouting:
    """Verify the rule-based fallback classifies queries correctly."""

    def test_symptom_query_routes_to_multi_hop(self):
        """Symptom descriptions should trigger multi-hop (symptoms→disease→treatment)."""
        assert classify_query_rules("I have fever and cough") == QueryCategory.MULTI_HOP

    def test_direct_lookup_query(self):
        """'What is X' should route to direct_lookup."""
        assert classify_query_rules("What is malaria?") == QueryCategory.DIRECT_LOOKUP

    def test_interaction_query_routes_to_relational(self):
        """Drug interaction questions should route to relational."""
        assert classify_query_rules("does aspirin interact with ibuprofen?") == QueryCategory.RELATIONAL_QUERY

    def test_generic_query_defaults_to_semantic(self):
        """A query with no specific signals should default to semantic search."""
        assert classify_query_rules("medicines for headache") == QueryCategory.SEMANTIC_SEARCH

    def test_empty_query_defaults_to_semantic(self):
        assert classify_query_rules("") == QueryCategory.SEMANTIC_SEARCH


class TestQueryRouter:
    """Verify the stateful router logs decisions."""

    def test_decisions_are_recorded(self):
        router = QueryRouter(use_llm=False)
        router.classify("I have a headache")
        router.classify("what is diabetes?")
        assert len(router.decisions) == 2

    def test_reset_clears_decisions(self):
        router = QueryRouter(use_llm=False)
        router.classify("test query")
        router.reset()
        assert len(router.decisions) == 0
