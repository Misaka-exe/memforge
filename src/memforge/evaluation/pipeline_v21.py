"""V2.1 Unified Adaptive Pipeline.

Integrates: Retrieval → Calibrated Gate → Grounded QA → Verification → Reconsolidation

Two modes:
- Mode A (Static): memory frozen, used for V2 comparison
- Mode B (Adaptive): question → retrieve → answer → verify → reconsolidate → next question

Config: V21Config(enable_gate, enable_temporal, enable_reconsolidation)
All three can be toggled independently for ablation.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from memforge.core.types import Memory, MemoryStatus
from memforge.evaluation.metrics_v21 import V21Metrics
from memforge.memory.reconsolidation import MemoryReconsolidator, ReconsolidationDecision
from memforge.retrieval.gate_v2 import BaseCalibratedGate, Sufficiency, ThresholdGate
from memforge.retrieval.score import RetrievalFeatures, compute_retrieval_features


@dataclass
class V21Config:
    """Configuration for V2.1 experiment. All toggles independent for ablation."""
    enable_gate: bool = False
    enable_temporal: bool = False
    enable_reconsolidation: bool = False
    mode: str = "static"  # "static" or "adaptive"
    top_k: int = 10
    gate_threshold: float = 0.5
    reconsolidation_min_confidence: float = 0.6

    def to_dict(self) -> dict:
        return {
            "enable_gate": self.enable_gate,
            "enable_temporal": self.enable_temporal,
            "enable_reconsolidation": self.enable_reconsolidation,
            "mode": self.mode,
            "top_k": self.top_k,
            "gate_threshold": self.gate_threshold,
        }


@dataclass
class V21Result:
    """Result for a single question in V2.1 pipeline."""
    question_id: str
    category: str = ""
    is_abstention: bool = False
    gate_decision: str = ""
    gate_score: float = 0.0
    answered: bool = True
    answer: str = ""
    gold_answer: str = ""
    correct: bool = False
    verification_result: str = ""  # PASS/WARN/FAIL/None
    reconsolidation_decision: str = ""  # KEEP/UPDATE/ABSTAIN/None
    reconsolidation_new_memory_id: str = ""
    citations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "is_abstention": self.is_abstention,
            "gate_decision": self.gate_decision,
            "gate_score": round(self.gate_score, 4),
            "answered": self.answered,
            "answer": self.answer,
            "gold_answer": self.gold_answer,
            "correct": self.correct,
            "verification_result": self.verification_result,
            "reconsolidation_decision": self.reconsolidation_decision,
            "reconsolidation_new_memory_id": self.reconsolidation_new_memory_id,
            "citations": self.citations,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class V21Pipeline:
    """V2.1 unified evaluation pipeline.

    This is the infrastructure layer. It supports both mock and real LLM
    via the answer_fn callback. Real API calls are NOT made here; the caller
    provides an answer_fn.
    """

    def __init__(
        self,
        config: V21Config,
        gate: Optional[BaseCalibratedGate] = None,
        reconsolidator: Optional[MemoryReconsolidator] = None,
        answer_fn=None,  # answer_fn(question, memories) -> (answer, citations, tokens)
        verify_fn=None,  # verify_fn(answer, memories) -> "PASS"|"WARN"|"FAIL"
    ):
        self.config = config
        self.gate = gate or ThresholdGate(threshold=config.gate_threshold)
        self.reconsolidator = reconsolidator or MemoryReconsolidator(
            min_confidence=config.reconsolidation_min_confidence
        )
        self.answer_fn = answer_fn or self._default_answer
        self.verify_fn = verify_fn or self._default_verify
        self.memory_store: dict[str, Memory] = {}
        self.results: list[V21Result] = []

    def _default_answer(self, question: str, memories: list[Memory]) -> tuple[str, list[str], dict]:
        """Mock answer function: returns first memory content or abstain."""
        if not memories:
            return "INSUFFICIENT_INFORMATION", [], {"prompt": 10, "completion": 5}
        citations = [str(m.id) for m in memories[:3]]
        return memories[0].content, citations, {"prompt": 50, "completion": 20}

    def _default_verify(self, answer: str, memories: list[Memory]) -> str:
        """Mock verify: PASS if answer matches any memory content."""
        if answer == "INSUFFICIENT_INFORMATION":
            return "None"
        for m in memories:
            if answer.strip() in m.content or m.content in answer:
                return "PASS"
        return "FAIL"

    def run_question(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
        memories: list[Memory],
        category: str = "",
        is_abstention: bool = False,
        retrieval_features: Optional[RetrievalFeatures] = None,
    ) -> V21Result:
        """Run a single question through the pipeline."""
        start = time.time()
        result = V21Result(
            question_id=question_id,
            category=category,
            is_abstention=is_abstention,
            gold_answer=gold_answer,
        )

        # Step 1: Gate (if enabled)
        if self.config.enable_gate and retrieval_features:
            decision = self.gate.assess(retrieval_features)
            result.gate_decision = decision.sufficiency.value
            result.gate_score = decision.score
            if decision.sufficiency == Sufficiency.INSUFFICIENT:
                result.answered = False
                result.answer = "INSUFFICIENT_INFORMATION"
                result.verification_result = "None"
                result.latency_ms = (time.time() - start) * 1000
                self.results.append(result)
                return result
        else:
            result.gate_decision = "DISABLED"

        # Step 2: Answer
        answer, citations, tokens = self.answer_fn(question, memories)
        result.answer = answer
        result.citations = citations
        result.prompt_tokens = tokens.get("prompt", 0)
        result.completion_tokens = tokens.get("completion", 0)
        result.answered = answer != "INSUFFICIENT_INFORMATION"

        # Step 3: Correctness (simple match for mock)
        result.correct = self._check_correct(answer, gold_answer, is_abstention)

        # Step 4: Verification
        if result.answered:
            result.verification_result = self.verify_fn(answer, memories)
        else:
            result.verification_result = "None"

        # Step 5: Reconsolidation (if enabled + adaptive mode + verification FAIL)
        if (self.config.enable_reconsolidation and
                self.config.mode == "adaptive" and
                result.verification_result == "FAIL" and
                memories):
            recon_result = self.reconsolidator.reconsolidate(
                memory=memories[0],
                new_evidence=answer,
                verification_fail_reason="verification FAIL",
                confidence=0.8,
            )
            result.reconsolidation_decision = recon_result.decision.value
            if recon_result.decision == ReconsolidationDecision.UPDATE and recon_result.new_memory_id:
                result.reconsolidation_new_memory_id = str(recon_result.new_memory_id)
                # Update memory store
                new_mem = Memory(
                    id=recon_result.new_memory_id,
                    content=f"{memories[0].content} [Updated] {answer}",
                    status=MemoryStatus.ACTIVE,
                    metadata={"lineage_id": recon_result.lineage_id, "version": recon_result.version},
                )
                self.memory_store[str(new_mem.id)] = new_mem

        result.latency_ms = (time.time() - start) * 1000
        self.results.append(result)
        return result

    def _check_correct(self, answer: str, gold: str, is_abstention: bool) -> bool:
        """Simple correctness check for mock pipeline."""
        if is_abstention:
            return answer == "INSUFFICIENT_INFORMATION"
        if not answer or answer == "INSUFFICIENT_INFORMATION":
            return False
        return gold.lower() in answer.lower() or answer.lower() in gold.lower()

    def compute_metrics(self, system_name: str = "") -> V21Metrics:
        """Compute aggregate metrics from results."""
        n = len(self.results)
        if n == 0:
            return V21Metrics(system_name=system_name)

        answered = [r for r in self.results if r.answered]
        answerable = [r for r in self.results if not r.is_abstention]
        abstention = [r for r in self.results if r.is_abstention]

        correct = sum(1 for r in self.results if r.correct)
        answerable_correct = sum(1 for r in answerable if r.correct)
        abstention_correct = sum(1 for r in abstention if r.correct)

        gate_sufficient = sum(1 for r in self.results if r.gate_decision == "SUFFICIENT")
        gate_uncertain = sum(1 for r in self.results if r.gate_decision == "UNCERTAIN")
        gate_insufficient = sum(1 for r in self.results if r.gate_decision == "INSUFFICIENT")

        verif_pass = sum(1 for r in self.results if r.verification_result == "PASS")
        verif_warn = sum(1 for r in self.results if r.verification_result == "WARN")
        verif_fail = sum(1 for r in self.results if r.verification_result == "FAIL")

        recon_updates = sum(1 for r in self.results if r.reconsolidation_decision == "UPDATE")

        avg_citations = sum(len(r.citations) for r in answered) / max(len(answered), 1)
        avg_latency = sum(r.latency_ms for r in self.results) / n
        total_prompt = sum(r.prompt_tokens for r in self.results)
        total_completion = sum(r.completion_tokens for r in self.results)

        # By category
        by_category: dict[str, dict] = {}
        for r in self.results:
            cat = r.category or "unknown"
            if cat not in by_category:
                by_category[cat] = {"n": 0, "correct": 0}
            by_category[cat]["n"] += 1
            if r.correct:
                by_category[cat]["correct"] += 1
        for cat in by_category:
            by_category[cat]["accuracy"] = round(
                by_category[cat]["correct"] / by_category[cat]["n"], 4
            )

        return V21Metrics(
            system_name=system_name,
            n_questions=n,
            mode=self.config.mode,
            enable_gate=self.config.enable_gate,
            enable_temporal=self.config.enable_temporal,
            enable_reconsolidation=self.config.enable_reconsolidation,
            overall_accuracy=correct / n,
            answerable_accuracy=answerable_correct / max(len(answerable), 1),
            abstention_accuracy=abstention_correct / max(len(abstention), 1),
            abstention_rate=len([r for r in self.results if not r.answered]) / n,
            gate_coverage=len(answered) / n,
            gate_sufficient_rate=gate_sufficient / n,
            gate_uncertain_rate=gate_uncertain / n,
            gate_insufficient_rate=gate_insufficient / n,
            avg_citations=avg_citations,
            verification_pass_rate=verif_pass / n,
            verification_warn_rate=verif_warn / n,
            verification_fail_rate=verif_fail / n,
            reconsolidation_triggered=recon_updates,
            memory_update_count=recon_updates,
            avg_latency_ms=avg_latency,
            total_tokens=total_prompt + total_completion,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            api_calls=len(answered),
            by_category=by_category,
        )

    def save_results(self, path: str | Path) -> None:
        """Save raw results to JSONL."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
