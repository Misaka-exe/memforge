"""Unit tests for temporal stress benchmark (V2.1 Phase 3)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from benchmarks.stress.temporal.benchmark import (
    TemporalCategory,
    TemporalCase,
    TemporalMemory,
    evaluate_temporal_system,
    generate_temporal_cases,
    hard_temporal_filter,
    no_temporal_filter,
    soft_temporal_filter,
)


class TestGenerateCases:
    def test_six_categories(self):
        cases = generate_temporal_cases(n_per_category=5)
        categories = {c.category for c in cases}
        assert len(categories) == 6
        assert TemporalCategory.CURRENT in categories
        assert TemporalCategory.FUTURE_CONTAMINATION in categories

    def test_count(self):
        cases = generate_temporal_cases(n_per_category=10)
        assert len(cases) == 60

    def test_gold_memories_exist(self):
        cases = generate_temporal_cases(n_per_category=3)
        for case in cases:
            assert len(case.gold_memory_ids) > 0
            gold_ids = set(case.gold_memory_ids)
            mem_ids = {m.memory_id for m in case.memories}
            assert gold_ids.issubset(mem_ids)


class TestFilters:
    def _make_memories(self):
        base = datetime(2024, 1, 1)
        return [
            TemporalMemory("past", "past fact", base.isoformat(),
                          (base + timedelta(days=100)).isoformat()),
            TemporalMemory("current", "current fact",
                          (base + timedelta(days=100)).isoformat(), None),
            TemporalMemory("future", "future fact",
                          (base + timedelta(days=400)).isoformat(), None, is_future=True),
        ]

    def test_hard_filter_removes_future(self):
        mems = self._make_memories()
        query = datetime(2024, 6, 1)  # day 152
        result = hard_temporal_filter(mems, query)
        ids = {m.memory_id for m in result}
        assert "future" not in ids
        assert "current" in ids
        assert "past" not in ids  # expired

    def test_no_temporal_keeps_all(self):
        mems = self._make_memories()
        result = no_temporal_filter(mems, datetime(2024, 6, 1))
        assert len(result) == 3

    def test_soft_decay_keeps_all_but_orders(self):
        mems = self._make_memories()
        result = soft_temporal_filter(mems, datetime(2024, 6, 1))
        assert len(result) == 3
        # current should be first (score 1.0)
        assert result[0].memory_id == "current"


class TestEvaluate:
    def test_basic_metrics(self):
        cases = generate_temporal_cases(n_per_category=5)
        metrics = evaluate_temporal_system(cases, no_temporal_filter)
        assert metrics["n_cases"] == 30
        assert 0 <= metrics["recall@1"] <= 1
        assert 0 <= metrics["future_leakage_rate"] <= 1
        assert "by_category" in metrics

    def test_hard_filter_zero_flr(self):
        cases = generate_temporal_cases(n_per_category=10)
        metrics = evaluate_temporal_system(cases, hard_temporal_filter)
        # Hard filter should have zero future leakage
        assert metrics["future_leakage_rate"] == 0.0

    def test_no_temporal_zero_elr(self):
        cases = generate_temporal_cases(n_per_category=10)
        metrics = evaluate_temporal_system(cases, no_temporal_filter)
        # No temporal filter should have zero evidence loss
        assert metrics["evidence_loss_rate"] == 0.0
