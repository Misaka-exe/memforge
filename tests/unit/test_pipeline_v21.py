"""Unit tests for V2.1 unified pipeline (Phase 5+6)."""
from __future__ import annotations

import pytest

from memforge.core.types import Memory, MemoryStatus
from memforge.evaluation.metrics_v21 import V21Metrics
from memforge.evaluation.pipeline_v21 import V21Config, V21Pipeline, V21Result
from memforge.retrieval.score import RetrievalFeatures


def make_memory(content="test memory", mid=None):
    import uuid
    return Memory(
        id=mid or uuid.uuid4(),
        content=content,
        status=MemoryStatus.ACTIVE,
    )


def make_features(top1=0.8):
    return RetrievalFeatures(
        question_id="q1", n_retrieved=10,
        top1_score=top1, top5_mean=0.6, score_margin=0.1,
        mean_score=0.5, std_score=0.1,
    )


class TestV21Config:
    def test_defaults(self):
        c = V21Config()
        assert c.enable_gate is False
        assert c.enable_temporal is False
        assert c.enable_reconsolidation is False
        assert c.mode == "static"

    def test_to_dict(self):
        c = V21Config(enable_gate=True)
        d = c.to_dict()
        assert d["enable_gate"] is True


class TestV21Pipeline:
    def test_static_no_gate(self):
        config = V21Config(mode="static")
        pipe = V21Pipeline(config)
        mem = make_memory("Alice lives in Beijing.")
        result = pipe.run_question(
            question_id="q1",
            question="Where does Alice live?",
            gold_answer="Beijing",
            memories=[mem],
        )
        assert result.answered is True
        assert result.gate_decision == "DISABLED"
        assert result.verification_result in ("PASS", "FAIL")

    def test_gate_insufficient_blocks(self):
        config = V21Config(enable_gate=True, gate_threshold=0.9)
        pipe = V21Pipeline(config)
        mem = make_memory()
        result = pipe.run_question(
            question_id="q1",
            question="test",
            gold_answer="test",
            memories=[mem],
            retrieval_features=make_features(top1=0.2),
        )
        assert result.gate_decision == "INSUFFICIENT"
        assert result.answered is False

    def test_gate_sufficient_allows(self):
        config = V21Config(enable_gate=True, gate_threshold=0.3)
        pipe = V21Pipeline(config)
        mem = make_memory("test answer")
        result = pipe.run_question(
            question_id="q1",
            question="test",
            gold_answer="test",
            memories=[mem],
            retrieval_features=make_features(top1=0.8),
        )
        assert result.gate_decision == "SUFFICIENT"
        assert result.answered is True

    def test_adaptive_reconsolidation_on_fail(self):
        config = V21Config(
            mode="adaptive",
            enable_reconsolidation=True,
        )
        # Use a verify_fn that always returns FAIL
        pipe = V21Pipeline(
            config,
            verify_fn=lambda a, m: "FAIL",
        )
        mem = make_memory("Alice lives in Beijing.")
        result = pipe.run_question(
            question_id="q1",
            question="Where does Alice live?",
            gold_answer="Shanghai",
            memories=[mem],
        )
        assert result.verification_result == "FAIL"
        assert result.reconsolidation_decision in ("UPDATE", "KEEP", "ABSTAIN")

    def test_static_no_reconsolidation(self):
        config = V21Config(mode="static", enable_reconsolidation=True)
        pipe = V21Pipeline(
            config,
            verify_fn=lambda a, m: "FAIL",
        )
        mem = make_memory()
        result = pipe.run_question("q1", "q", "gold", [mem])
        # Static mode: no reconsolidation even if enabled
        assert result.reconsolidation_decision == ""

    def test_compute_metrics(self):
        config = V21Config()
        pipe = V21Pipeline(config)
        for i in range(5):
            mem = make_memory(f"answer {i}")
            pipe.run_question(f"q{i}", f"q{i}", f"answer {i}", [mem])
        metrics = pipe.compute_metrics("test_system")
        assert metrics.n_questions == 5
        assert metrics.system_name == "test_system"
        assert 0 <= metrics.overall_accuracy <= 1
        assert metrics.api_calls > 0

    def test_save_results(self, tmp_path):
        config = V21Config()
        pipe = V21Pipeline(config)
        mem = make_memory("test")
        pipe.run_question("q1", "q", "gold", [mem])
        out = tmp_path / "results.jsonl"
        pipe.save_results(str(out))
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1


class TestV21Metrics:
    def test_to_dict(self):
        m = V21Metrics(system_name="test", n_questions=10)
        d = m.to_dict()
        assert d["system"] == "test"
        assert d["n_questions"] == 10
        assert "retrieval" in d
        assert "qa" in d
        assert "gate" in d
        assert "evidence" in d
        assert "memory" in d
        assert "efficiency" in d
