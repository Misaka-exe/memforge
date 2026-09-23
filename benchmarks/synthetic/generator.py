"""Synthetic scenario generator with known ground truth.

- Fixed seed for reproducibility.
- Deterministic hash embedding (no API, no network).
- Injects controlled conflicts, temporal updates and optional noise.
"""
from __future__ import annotations

import hashlib
import math
import random
import uuid

from memforge.embeddings.hash import hash_embedding
from benchmarks.synthetic.ground_truth import (
    BenchmarkMemory,
    BenchmarkQuery,
    BenchmarkScenario,
)

EMBED_DIM = 64

LOCATION_HISTORY = [
    ("2023-01-01", "Beijing"),
    ("2024-06-01", "Shanghai"),
    ("2025-09-01", "Shenzhen"),
]

PHONE_HISTORY = [
    ("2024-03-01", "prefers lightweight phones"),
    ("2025-11-01", "now prefers large-screen phones"),
]

HOBBIES = ["coffee", "cycling", "photography", "reading", "gaming", "cooking"]
JOBS = ["software engineer", "product manager", "designer", "data scientist"]


def _h(token: str, salt: str) -> int:
    return int(hashlib.md5(f"{salt}:{token}".encode()).hexdigest()[:8], 16)


def hash_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in text.lower().split():
        vec[_h(tok, "t") % dim] += 1.0
    for tok in text.lower():
        vec[_h(tok, "c") % dim] += 0.3
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class ScenarioGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def _mid(self) -> str:
        return f"mem_{uuid.uuid4().hex[:8]}"

    def generate(self, num_users: int = 8, noise_rate: float = 0.0) -> list[BenchmarkScenario]:
        scenarios: list[BenchmarkScenario] = []
        for u in range(num_users):
            scenarios.append(self._one_user(u, noise_rate))
        return scenarios

    def _one_user(self, idx: int, noise_rate: float) -> BenchmarkScenario:
        uid = f"u{idx}"
        memories: list[BenchmarkMemory] = []
        queries: list[BenchmarkQuery] = []

        # ---- stable facts ----
        job = self.rng.choice(JOBS)
        hobby = self.rng.choice(HOBBIES)
        fact_job = self._add(memories, uid, f"User works as a {job}.", mtype="fact", importance=0.8)
        fact_hobby = self._add(memories, uid, f"User enjoys {hobby}.", mtype="preference", importance=0.7)

        # ---- temporal location history ----
        loc_memories: list[BenchmarkMemory] = []
        for i, (since, city) in enumerate(LOCATION_HISTORY):
            is_last = i == len(LOCATION_HISTORY) - 1
            m = self._add(
                memories,
                uid,
                f"User lives in {city}.",
                mtype="fact",
                importance=0.9,
                valid_from=since,
                status="active" if is_last else "deprecated",
            )
            loc_memories.append(m)
        current_city = LOCATION_HISTORY[-1][1]
        queries.append(
            BenchmarkQuery(
                query="Where does the user currently live?",
                gold_memory_ids=[loc_memories[-1].id],
                category="retrieval",
                user_id=uid,
            )
        )
        queries.append(
            BenchmarkQuery(
                query="Where did the user live in 2023?",
                gold_memory_ids=[loc_memories[0].id],
                category="temporal",
                user_id=uid,
            )
        )

        # ---- preference change (update / conflict) ----
        old_phone = self._add(memories, uid, f"User {PHONE_HISTORY[0][1]}.", mtype="preference", importance=0.85, status="deprecated")
        new_phone = self._add(memories, uid, f"User {PHONE_HISTORY[1][1]}.", mtype="preference", importance=0.85, status="active")
        queries.append(
            BenchmarkQuery(
                query="What kind of phone does the user prefer now?",
                gold_memory_ids=[new_phone.id],
                category="update",
                user_id=uid,
            )
        )

        # ---- plain retrieval over preferences ----
        queries.append(
            BenchmarkQuery(
                query=f"What does the user like doing in free time?",
                gold_memory_ids=[fact_hobby.id],
                category="retrieval",
                user_id=uid,
            )
        )

        # ---- irrelevant / distractor memories ----
        for i in range(3):
            self._add(memories, uid, f"Random topic {idx}-{i} mentioned once.", mtype="fact", importance=0.2, is_noise=True)

        # ---- injected noise (wrong memories) ----
        if noise_rate > 0:
            n_wrong = int(noise_rate * len(memories))
            for i in range(n_wrong):
                self._add(memories, uid, f"User lives in {self.rng.choice(['Tokyo', 'London', 'Paris'])}.", mtype="fact", importance=0.9, is_noise=True)

        return BenchmarkScenario(scenario_id=f"scn_{idx}", user_id=uid, memories=memories, queries=queries)

    def _add(
        self,
        memories: list[BenchmarkMemory],
        uid: str,
        content: str,
        mtype: str = "fact",
        importance: float = 0.5,
        valid_from: str | None = None,
        status: str = "active",
        is_noise: bool = False,
    ) -> BenchmarkMemory:
        m = BenchmarkMemory(
            id=self._mid(),
            content=content,
            user_id=uid,
            memory_type=mtype,
            importance=importance,
            valid_from=valid_from,
            status=status,
            is_noise=is_noise,
        )
        memories.append(m)
        return m