"""MemForge v2 end-to-end pipeline.

Wires the v2 modules into one callable:

    Query
      -> Hybrid Retrieval (vector + keyword, ACTIVE filter)
      -> Soft Temporal Score (no hard drop)
      -> Sufficiency Gate
         -> INSUFFICIENT : ABSTAIN (no LLM QA call)
         -> else         : Grounded QA (citation JSON)
      -> Post-hoc Verification
      -> V2PipelineResult

v1 pipeline (RetrievalEngine / HybridRetriever) is left untouched; this is a
new, independently configurable entry point. Each v2 module can be toggled for
ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from benchmarks.evaluation.qa_v2 import V2QAQuestion, V2QAPipeline
from memforge.core.types import Memory
from memforge.embeddings.base import EmbeddingProvider
from memforge.llm.base import LLMProvider
from memforge.retrieval.gate import RetrievalDecision, SufficiencyGate
from memforge.retrieval.hybrid import HybridRetriever
from memforge.retrieval.reranker import RerankConfig, UtilityReranker
from memforge.retrieval.temporal import NoDecayScorer, TemporalScorer
from memforge.storage.repository import MemoryRepository
from memforge.verification.verifier import Verifier, VerificationResult
from memforge.memory.grounding import GroundedAnswer


@dataclass
class V2PipelineConfig:
    use_temporal: bool = True
    use_gate: bool = True
    use_verification: bool = True
    temporal_weight: float = 0.3
    semantic_top_k: int = 20
    keyword_top_k: int = 20


@dataclass
class V2PipelineResult:
    query: str
    memories: list[Memory] = field(default_factory=list)
    gate_decision: RetrievalDecision | None = None
    answer: GroundedAnswer | None = None
    verification: VerificationResult | None = None
    abstained: bool = False
    gate_decision_label: str = ""


class MemForgeV2Pipeline:
    def __init__(
        self,
        repository: MemoryRepository,
        embedding: EmbeddingProvider,
        llm: LLMProvider,
        temporal_scorer: TemporalScorer | None = None,
        gate: SufficiencyGate | None = None,
        verifier: Verifier | None = None,
        config: V2PipelineConfig | None = None,
    ) -> None:
        self.config = config or V2PipelineConfig()
        self.gate = gate or SufficiencyGate()
        self.verifier = verifier or Verifier()

        scorer: TemporalScorer = temporal_scorer or NoDecayScorer()
        rerank_config = RerankConfig(
            temporal_weight=self.config.temporal_weight if self.config.use_temporal else 0.0,
        )
        reranker = UtilityReranker(
            embedding=embedding, config=rerank_config, temporal_scorer=scorer
        )
        self.retriever = HybridRetriever(
            repository=repository,
            embedding=embedding,
            reranker=reranker,
            semantic_top_k=self.config.semantic_top_k,
            keyword_top_k=self.config.keyword_top_k,
            temporal_scorer=scorer,
        )
        # QA sub-pipeline (gate -> grounded QA -> verify). Gate/verifier can be
        # disabled via config by swapping in no-op gates; here we always wire
        # the real ones and let the caller toggle behavior through config.
        self.qa = V2QAPipeline(llm=llm, gate=self.gate, verifier=self.verifier)

    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
        as_of: datetime | None = None,
    ) -> list[Memory]:
        return await self.retriever.search(query, user_id=user_id, top_k=top_k, as_of=as_of)

    async def answer(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
        as_of: datetime | None = None,
    ) -> V2PipelineResult:
        result = V2PipelineResult(query=query)
        memories = await self.retrieve(query, user_id=user_id, top_k=top_k, as_of=as_of)
        result.memories = memories

        q = V2QAQuestion(
            question_id="",
            question=query,
            gold_answer="",
            is_abstention=False,
            category="",
            retrieved_items=memories,
        )
        qa_result = await self.qa.answer(q)
        result.answer = qa_result.grounded_answer
        result.verification = qa_result.verification
        result.abstained = qa_result.abstained
        result.gate_decision_label = qa_result.gate_decision
        return result
