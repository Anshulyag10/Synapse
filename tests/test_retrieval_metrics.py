"""Tests for retrieval metrics."""

import pytest

from evidence_graph.evaluation.retrieval_metrics import (
    rank_of,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    same_item,
    compute_metric_block,
)


class TestSameItem:
    def test_exact_match(self):
        assert same_item("aspirin", "aspirin") is True

    def test_substring_match(self):
        assert same_item("aspirin", "aspirin tablets 500mg") is True

    def test_no_match(self):
        assert same_item("aspirin", "ibuprofen") is False

    def test_empty_strings(self):
        assert same_item("", "aspirin") is False
        assert same_item("aspirin", "") is False


class TestRankOf:
    def test_first_position(self):
        assert rank_of(["aspirin"], ["aspirin", "ibuprofen", "tylenol"]) == 1

    def test_third_position(self):
        assert rank_of(["tylenol"], ["aspirin", "ibuprofen", "tylenol"]) == 3

    def test_not_found(self):
        assert rank_of(["missing"], ["aspirin", "ibuprofen"]) is None

    def test_multiple_golds(self):
        assert rank_of(["a", "b"], ["x", "b", "a"]) == 2


class TestRecallAtK:
    def test_within_k(self):
        assert recall_at_k(1, 5) == 1
        assert recall_at_k(3, 5) == 1

    def test_outside_k(self):
        assert recall_at_k(6, 5) == 0

    def test_none_rank(self):
        assert recall_at_k(None, 5) == 0


class TestReciprocalRank:
    def test_rank_1(self):
        assert reciprocal_rank(1) == 1.0

    def test_rank_2(self):
        assert reciprocal_rank(2) == 0.5

    def test_not_found(self):
        assert reciprocal_rank(None) == 0.0


class TestNdcgAtK:
    def test_rank_1(self):
        assert ndcg_at_k(1) == 1.0

    def test_outside_k(self):
        assert ndcg_at_k(6, 5) == 0.0

    def test_none_rank(self):
        assert ndcg_at_k(None) == 0.0


class TestMetricBlock:
    def test_empty_records(self):
        block = compute_metric_block([])
        assert block["n"] == 0

    def test_basic_aggregation(self):
        records = [
            {"recall@1": 1, "recall@3": 1, "recall@5": 1, "reciprocal_rank": 1.0,
             "ndcg@5": 1.0, "latency_ms": {"retrieval": 10.0}},
            {"recall@1": 0, "recall@3": 1, "recall@5": 1, "reciprocal_rank": 0.5,
             "ndcg@5": 0.63, "latency_ms": {"retrieval": 20.0}},
        ]
        block = compute_metric_block(records)
        assert block["n"] == 2
        assert block["recall@1"] == 0.5
