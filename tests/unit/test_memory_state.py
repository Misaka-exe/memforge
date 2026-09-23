"""Stage 0: pure state machine tests (22 tests, zero external deps)."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from memforge.core.lifecycle import IllegalTransitionError, LifecycleManager
from memforge.core.types import (
    Memory,
    MemoryEventType,
    MemoryStatus,
    utcnow,
)


def make_memory(**kwargs) -> Memory:
    defaults = {"content": "User likes coffee.", "user_id": "u1"}
    defaults.update(kwargs)
    return Memory(**defaults)


def make_manager(t0: datetime | None = None) -> LifecycleManager:
    return LifecycleManager(now=t0 or datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------

def test_initial_status_is_candidate():
    m = make_memory()
    assert m.status == MemoryStatus.CANDIDATE


def test_candidate_to_active_allowed():
    m = make_memory()
    make_manager().transition(m, MemoryStatus.ACTIVE)
    assert m.status == MemoryStatus.ACTIVE


def test_active_to_dormant_allowed():
    m = make_memory(status=MemoryStatus.ACTIVE)
    make_manager().transition(m, MemoryStatus.DORMANT)
    assert m.status == MemoryStatus.DORMANT


def test_active_to_consolidated_allowed():
    m = make_memory(status=MemoryStatus.ACTIVE)
    make_manager().transition(m, MemoryStatus.CONSOLIDATED)
    assert m.status == MemoryStatus.CONSOLIDATED


def test_consolidated_to_dormant_allowed():
    m = make_memory(status=MemoryStatus.CONSOLIDATED)
    make_manager().transition(m, MemoryStatus.DORMANT)
    assert m.status == MemoryStatus.DORMANT


def test_active_to_deprecated_allowed():
    m = make_memory(status=MemoryStatus.ACTIVE)
    make_manager().transition(m, MemoryStatus.DEPRECATED)
    assert m.status == MemoryStatus.DEPRECATED


def test_dormant_to_active_allowed():
    m = make_memory(status=MemoryStatus.DORMANT)
    make_manager().transition(m, MemoryStatus.ACTIVE)
    assert m.status == MemoryStatus.ACTIVE


def test_dormant_to_deprecated_allowed():
    m = make_memory(status=MemoryStatus.DORMANT)
    make_manager().transition(m, MemoryStatus.DEPRECATED)
    assert m.status == MemoryStatus.DEPRECATED


def test_deprecated_to_forgotten_allowed():
    m = make_memory(status=MemoryStatus.DEPRECATED)
    make_manager().transition(m, MemoryStatus.FORGOTTEN)
    assert m.status == MemoryStatus.FORGOTTEN


# ---------------------------------------------------------------------------
# Illegal transitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (MemoryStatus.FORGOTTEN, MemoryStatus.ACTIVE),
        (MemoryStatus.FORGOTTEN, MemoryStatus.DORMANT),
        (MemoryStatus.CANDIDATE, MemoryStatus.DEPRECATED),
        (MemoryStatus.CANDIDATE, MemoryStatus.CONSOLIDATED),
        (MemoryStatus.CANDIDATE, MemoryStatus.FORGOTTEN),
        (MemoryStatus.ACTIVE, MemoryStatus.FORGOTTEN),
        (MemoryStatus.DORMANT, MemoryStatus.FORGOTTEN),
        (MemoryStatus.CONSOLIDATED, MemoryStatus.ACTIVE),
        (MemoryStatus.CONSOLIDATED, MemoryStatus.DEPRECATED),
        (MemoryStatus.DEPRECATED, MemoryStatus.ACTIVE),
    ],
)
def test_illegal_transitions_raise(from_status, to_status):
    m = make_memory(status=from_status)
    with pytest.raises(IllegalTransitionError):
        make_manager().transition(m, to_status)


def test_same_status_transition_raises():
    m = make_memory(status=MemoryStatus.ACTIVE)
    with pytest.raises(IllegalTransitionError):
        make_manager().transition(m, MemoryStatus.ACTIVE)


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------

def test_transition_records_event_with_from_to():
    m = make_memory()
    mgr = make_manager()
    event = mgr.transition(m, MemoryStatus.ACTIVE, reason="activated by user")
    assert event.memory_id == m.id
    assert event.event == MemoryEventType.ACTIVATE
    assert event.from_status == MemoryStatus.CANDIDATE
    assert event.to_status == MemoryStatus.ACTIVE
    assert event.reason == "activated by user"


def test_transition_event_timestamp_matches_manager_clock():
    t0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    m = make_memory()
    event = make_manager(t0).transition(m, MemoryStatus.ACTIVE)
    assert event.timestamp == t0


def test_deprecate_event_type_is_deprecate():
    m = make_memory(status=MemoryStatus.ACTIVE)
    event = make_manager().transition(m, MemoryStatus.DEPRECATED)
    assert event.event == MemoryEventType.DEPRECATE


def test_event_records_source_memory_id():
    m = make_memory(status=MemoryStatus.ACTIVE)
    src = uuid.uuid4()
    event = make_manager().transition(m, MemoryStatus.DEPRECATED, source_memory_id=str(src))
    assert str(event.source_memory_id) == str(src)


def test_full_lifecycle_event_chain():
    m = make_memory()
    mgr = make_manager()
    events = [
        mgr.transition(m, MemoryStatus.ACTIVE),
        mgr.transition(m, MemoryStatus.DORMANT),
        mgr.transition(m, MemoryStatus.DEPRECATED),
        mgr.transition(m, MemoryStatus.FORGOTTEN),
    ]
    assert [e.event for e in events] == [
        MemoryEventType.ACTIVATE,
        MemoryEventType.SLEEP,
        MemoryEventType.DEPRECATE,
        MemoryEventType.FORGET,
    ]
    assert m.status == MemoryStatus.FORGOTTEN


# ---------------------------------------------------------------------------
# valid_until stamping on first DEPRECATE
# ---------------------------------------------------------------------------

def test_first_deprecate_stamps_valid_until():
    m = make_memory(status=MemoryStatus.ACTIVE)
    mgr = make_manager()
    mgr.transition(m, MemoryStatus.DEPRECATED)
    assert m.valid_until == mgr.now


def test_deprecated_memory_keeps_valid_until_on_forget():
    m = make_memory(status=MemoryStatus.ACTIVE)
    mgr = make_manager()
    mgr.transition(m, MemoryStatus.DEPRECATED)
    stamped = m.valid_until
    mgr.transition(m, MemoryStatus.FORGOTTEN)
    assert m.valid_until == stamped


def test_valid_until_manual_not_overwritten():
    manual = datetime(2026, 5, 1, tzinfo=timezone.utc)
    m = make_memory(status=MemoryStatus.ACTIVE, valid_until=manual)
    mgr = make_manager()
    mgr.transition(m, MemoryStatus.DEPRECATED)
    assert m.valid_until == manual


# ---------------------------------------------------------------------------
# Memory fields & access counting
# ---------------------------------------------------------------------------

def test_touch_increments_access_count():
    m = make_memory()
    m.touch()
    m.touch()
    assert m.access_count == 2


def test_touch_sets_last_accessed_at():
    m = make_memory()
    m.touch()
    assert m.last_accessed_at is not None


def test_transition_updates_updated_at():
    mgr = make_manager()
    before = mgr.now
    m = make_memory()
    m.updated_at = before - timedelta(days=1)
    mgr.transition(m, MemoryStatus.ACTIVE)
    assert m.updated_at == mgr.now
    assert m.updated_at >= before


def test_memory_defaults_are_sane():
    m = make_memory()
    assert m.importance == 0.5
    assert m.confidence == 0.5
    assert m.utility == 0.0
    assert m.embedding == []
    assert m.supersedes is None