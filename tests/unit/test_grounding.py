"""Phase 5: Citation / evidence grounding tests."""
from __future__ import annotations

from memforge.memory.grounding import (
    Citation,
    GroundedAnswer,
    citation_validity,
    looks_like_abstention,
    parse_grounded_answer,
)
from benchmarks.evaluation.qa_v2 import GROUNDED_PROMPT_TEMPLATE, build_grounded_prompt
from memforge.core.types import Memory, MemoryStatus, MemoryType


def test_grounded_answer_parsing():
    item_ids = ["mem-1", "mem-2", "mem-3"]
    raw = '{"answer": "User lives in Shanghai and likes tea.", "citations": [1, 3], "abstain": false}'
    g = parse_grounded_answer(raw, item_ids)
    assert g.abstain is False
    assert "Shanghai" in g.answer
    assert [c.memory_id for c in g.citations] == ["mem-1", "mem-3"]


def test_grounded_answer_parses_string_citations():
    item_ids = ["mem-1", "mem-2"]
    raw = '{"answer": "x", "citations": ["mem-2"], "abstain": false}'
    g = parse_grounded_answer(raw, item_ids)
    assert [c.memory_id for c in g.citations] == ["mem-2"]


def test_citation_coverage_calculation():
    g = GroundedAnswer(
        answer="x",
        citations=[
            Citation(memory_id="m1", supported=True),
            Citation(memory_id="m2", supported=True),
            Citation(memory_id="m3", supported=False),
        ],
    )
    assert abs(g.citation_coverage() - (2 / 3)) < 1e-9


def test_citation_validity_check():
    g = GroundedAnswer(
        answer="x",
        citations=[
            Citation(memory_id="m1", supported=True),
            Citation(memory_id="ghost", supported=True),
        ],
    )
    invalid = citation_validity(g, valid_memory_ids={"m1", "m2", "m3"})
    assert invalid == ["ghost"]


def test_abstain_flag_from_json():
    item_ids = ["m1"]
    raw = '{"answer": "", "citations": [], "abstain": true}'
    g = parse_grounded_answer(raw, item_ids)
    assert g.abstain is True
    assert g.answer == ""


def test_abstain_flag_from_marker_text():
    g = parse_grounded_answer("INSUFFICIENT_INFORMATION", ["m1"])
    assert g.abstain is True


def test_malformed_response_falls_back():
    g = parse_grounded_answer("I think the answer is maybe Shanghai", ["m1"])
    # Not JSON -> plain answer, no fabricated citations.
    assert g.abstain is False
    assert g.citations == []
    assert "Shanghai" in g.answer


def test_looks_like_abstention():
    assert looks_like_abstention("INSUFFICIENT_INFORMATION")
    assert looks_like_abstention("I DO NOT KNOW")
    assert not looks_like_abstention("User lives in Shanghai")


def test_v2_prompt_requires_citations():
    # The grounded prompt must explicitly demand citations and abstention.
    assert "citation" in GROUNDED_PROMPT_TEMPLATE.lower()
    assert "abstain" in GROUNDED_PROMPT_TEMPLATE.lower()
    assert "ONLY" in GROUNDED_PROMPT_TEMPLATE

    mems = [
        Memory(content="User lives in Shanghai.", status=MemoryStatus.ACTIVE,
               memory_type=MemoryType.FACT),
        Memory(content="User likes tea.", status=MemoryStatus.ACTIVE,
               memory_type=MemoryType.FACT),
    ]
    prompt = build_grounded_prompt("Where does the user live?", mems)
    assert "[1]" in prompt and "[2]" in prompt
    assert "Shanghai" in prompt
