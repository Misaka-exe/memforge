"""Unit tests for evidence-grounded reconsolidation (V2.1 Phase 4)."""
from __future__ import annotations

import uuid

import pytest

from memforge.core.types import Memory, MemoryStatus
from memforge.memory.reconsolidation import (
    MemoryReconsolidator,
    ReconsolidationDecision,
    compute_evidence_loss_rate,
    compute_evidence_preservation_rate,
)


def make_memory(content="Alice lives in Beijing.", importance=0.5, confidence=0.5):
    return Memory(
        content=content,
        importance=importance,
        confidence=confidence,
        status=MemoryStatus.ACTIVE,
    )


class TestReconsolidator:
    def test_update_creates_new_version(self):
        recons = MemoryReconsolidator()
        old = make_memory("Alice lives in Beijing and works as Engineer.")
        result = recons.reconsolidate(
            memory=old,
            new_evidence="Alice moved to Shanghai.",
            verification_fail_reason="answer said Beijing but evidence says Shanghai",
            confidence=0.9,
        )
        assert result.decision == ReconsolidationDecision.UPDATE
        assert result.new_memory_id is not None
        assert result.new_memory_id != old.id
        assert result.lineage_id is not None
        assert result.version == 2
        assert "location" in result.updated_fields

    def test_low_confidence_abstains(self):
        recons = MemoryReconsolidator(min_confidence=0.6)
        old = make_memory()
        result = recons.reconsolidate(
            memory=old,
            new_evidence="Alice moved to Shanghai.",
            confidence=0.3,
        )
        assert result.decision == ReconsolidationDecision.ABSTAIN
        assert result.new_memory_id is None

    def test_empty_evidence_keeps(self):
        recons = MemoryReconsolidator()
        old = make_memory()
        result = recons.reconsolidate(
            memory=old,
            new_evidence="   ",
            confidence=0.9,
        )
        assert result.decision == ReconsolidationDecision.KEEP

    def test_no_updatable_fields_keeps(self):
        recons = MemoryReconsolidator()
        old = make_memory("The weather is nice today.")
        result = recons.reconsolidate(
            memory=old,
            new_evidence="The stock market went up.",
            confidence=0.9,
        )
        # No overlapping slots → KEEP
        assert result.decision == ReconsolidationDecision.KEEP

    def test_lineage_preserved_across_versions(self):
        recons = MemoryReconsolidator()
        v1 = make_memory("Alice lives in Beijing.")
        r1 = recons.reconsolidate(v1, "Alice moved to Shanghai.", confidence=0.9)
        assert r1.decision == ReconsolidationDecision.UPDATE

        # Simulate v2 memory
        v2 = Memory(
            content="Alice lives in Shanghai.",
            status=MemoryStatus.ACTIVE,
            metadata={
                "lineage_id": r1.lineage_id,
                "version": 2,
                "derived_from": str(v1.id),
            },
        )
        r2 = recons.reconsolidate(v2, "Alice moved to Shenzhen.", confidence=0.9)
        assert r2.decision == ReconsolidationDecision.UPDATE
        assert r2.lineage_id == r1.lineage_id  # same lineage
        assert r2.version == 3

    def test_deprecate_old(self):
        recons = MemoryReconsolidator()
        old = make_memory()
        deprecated = recons.deprecate_old(old)
        assert deprecated.status == MemoryStatus.DEPRECATED

    def test_never_overwrites_original(self):
        recons = MemoryReconsolidator()
        old = make_memory("Alice lives in Beijing.")
        old_content = old.content
        old_id = old.id
        result = recons.reconsolidate(old, "Alice moved to Shanghai.", confidence=0.9)
        # Original memory object is NOT modified
        assert old.content == old_content
        assert old.id == old_id
        assert old.status == MemoryStatus.ACTIVE  # caller must deprecate separately


class TestEvidencePreservation:
    def test_epr_all_preserved(self):
        m1 = make_memory()
        gold = {str(m1.id)}
        epr = compute_evidence_preservation_rate(gold, [m1])
        assert epr == 1.0

    def test_epr_none_preserved(self):
        m1 = make_memory()
        m1.status = MemoryStatus.DEPRECATED
        gold = {str(m1.id)}
        epr = compute_evidence_preservation_rate(gold, [m1])
        assert epr == 0.0

    def test_elr_complement(self):
        m1 = make_memory()
        gold = {str(m1.id)}
        epr = compute_evidence_preservation_rate(gold, [m1])
        elr = compute_evidence_loss_rate(gold, [m1])
        assert abs(epr + elr - 1.0) < 0.01

    def test_epr_empty_gold(self):
        epr = compute_evidence_preservation_rate(set(), [])
        assert epr == 1.0
