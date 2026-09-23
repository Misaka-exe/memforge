"""Temporal memory filtering using the half-open interval [valid_from, valid_until).

A memory is valid at time T when:
    valid_from <= T  AND  (valid_until IS NULL OR T < valid_until)

The half-open right bound means that at the exact instant a new memory takes
effect, the old one is no longer considered valid — no boundary ambiguity.
"""
from __future__ import annotations

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