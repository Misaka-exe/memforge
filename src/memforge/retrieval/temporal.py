"""Temporal memory: v1 hard filter + v2 soft scoring.

v1 (kept for backward compatibility):
    memory is active at T iff status==ACTIVE and
    valid_from <= T < valid_until (half-open right bound).

v2 (soft decay, new in this file):
    A TemporalScorer maps a (memory, as_of) pair to a score in [floor, 1.0].
    Memories whose validity interval does not contain as_of are NOT deleted;
    they are down-weighted so they can still surface when nothing else matches.
    This fixes the v1 finding that hard temporal filtering removed valid gold
    evidence (future / slightly-out-of-window memories).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

from memforge.core.types import Memory, MemoryStatus


def memory_active_at(memory: Memory, as_of: datetime) -> bool:
    if memory.status != MemoryStatus.ACTIVE:
        return False
    if memory.valid_from is not None and memory.valid_from > as_of:
        return False
    if memory.valid_until is not None and as_of >= memory.valid_until:
        return False
    return True


def filter_temporal(memories: Iterable[Memory], as_of: datetime) -> list[Memory]:
    return [m for m in memories if memory_active_at(m, as_of)]


# ---------------------------------------------------------------------------
# v2 soft temporal scoring
# ---------------------------------------------------------------------------

def _days_outside_window(memory: Memory, as_of: datetime) -> float:
    """Days between as_of and the nearest edge of the validity interval.

    Returns 0.0 when as_of is inside [valid_from, valid_until) (or when the
    memory has no temporal constraints). Positive number of days outside.
    """
    if memory.valid_from is None and memory.valid_until is None:
        return 0.0
    if memory.valid_from is not None and as_of < memory.valid_from:
        delta = memory.valid_from - as_of
        return max(delta.total_seconds() / 86400.0, 0.0)
    if memory.valid_until is not None and as_of >= memory.valid_until:
        delta = as_of - memory.valid_until
        return max(delta.total_seconds() / 86400.0, 0.0)
    return 0.0


class TemporalScorer(ABC):
    """Map (memory, as_of) -> temporal score in [floor, 1.0]."""

    floor: float = 0.0

    @abstractmethod
    def score(self, memory: Memory, as_of: datetime) -> float: ...

    def score_batch(self, memories: list[Memory], as_of: datetime) -> list[float]:
        return [self.score(m, as_of) for m in memories]


class NoDecayScorer(TemporalScorer):
    """Always 1.0. Temporal signal contributes nothing (v1 default)."""

    floor = 1.0

    def score(self, memory: Memory, as_of: datetime) -> float:
        return 1.0


class WindowScorer(TemporalScorer):
    """Hard window expressed as a SOFT score: 1.0 inside, ``out_of_window`` outside.

    Future / expired memories are retained in the candidate set but heavily
    down-weighted, instead of being dropped outright.
    """

    def __init__(self, out_of_window: float = 0.1) -> None:
        self.out_of_window = out_of_window
        self.floor = out_of_window

    def score(self, memory: Memory, as_of: datetime) -> float:
        if memory_active_at(memory, as_of):
            return 1.0
        return self.out_of_window


class ExponentialDecayScorer(TemporalScorer):
    """Inside window -> 1.0. Outside -> floor + (1-floor)*2^(-days/half_life)."""

    def __init__(self, floor: float = 0.1, half_life_days: float = 30.0) -> None:
        self.floor = floor
        self.half_life_days = half_life_days

    def score(self, memory: Memory, as_of: datetime) -> float:
        if memory_active_at(memory, as_of):
            return 1.0
        days = _days_outside_window(memory, as_of)
        if days <= 0.0:
            return 1.0
        decay = math.exp(-math.log(2) * days / self.half_life_days)
        return self.floor + (1.0 - self.floor) * decay


class LinearDecayScorer(TemporalScorer):
    """Inside window -> 1.0. Outside -> max(floor, 1 - days/max_days)."""

    def __init__(self, floor: float = 0.1, max_days: float = 365.0) -> None:
        self.floor = floor
        self.max_days = max_days

    def score(self, memory: Memory, as_of: datetime) -> float:
        if memory_active_at(memory, as_of):
            return 1.0
        days = _days_outside_window(memory, as_of)
        if days <= 0.0:
            return 1.0
        linear = max(0.0, 1.0 - days / self.max_days)
        return max(self.floor, linear)