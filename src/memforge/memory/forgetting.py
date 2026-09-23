"""Forgetting: discover stale memories, never auto-transition.

Stage 3 only *reports* candidates. Automated DORMANT -> FORGOTTEN must wait
for benchmark evidence that it does not hurt recall.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from memforge.core.types import Memory, MemoryStatus
from memforge.storage.repository import MemoryRepository


class ForgettingManager:
    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    async def find_stale(
        self,
        user_id: str | None = None,
        now: datetime | None = None,
        dormant_max_age_days: int = 180,
        active_max_unused_days: int = 365,
    ) -> list[Memory]:
        """Return memories that are candidates for forgetting.

        - DORMANT memories older than dormant_max_age_days (updated_at)
        - ACTIVE memories not accessed within active_max_unused_days
        """
        now = now or datetime.now()
        all_memories = await self.repository.list_by_user(user_id or "default", limit=10000)
        stale: list[Memory] = []
        for m in all_memories:
            ref = m.updated_at or m.created_at
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=now.tzinfo)
            age_days = (now - ref).total_seconds() / 86400.0
            if m.status == MemoryStatus.DORMANT and age_days >= dormant_max_age_days:
                stale.append(m)
            elif m.status == MemoryStatus.ACTIVE and m.last_accessed_at is not None:
                last = m.last_accessed_at
                if last.tzinfo is None:
                    last = last.replace(tzinfo=now.tzinfo)
                if (now - last).total_seconds() / 86400.0 >= active_max_unused_days:
                    stale.append(m)
        return stale