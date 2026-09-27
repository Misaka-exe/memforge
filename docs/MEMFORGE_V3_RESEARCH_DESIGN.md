# MemForge V3 Research Design

## Memory Genome: Genotype-Phenotype Separation with Evidence-Grounded Mutation and Selection

**Status**: Design Document (pre-implementation)
**Date**: 2026-09-27
**Prerequisite**: V2.1-final (tag v2.1-final) frozen

---

## 1. Research Question

**Can separating memory genotype from phenotype, together with evidence-grounded mutation and selection, improve memory reliability compared with direct memory updating?**

V2.1 demonstrated that evidence-grounded reconsolidation improves QA (+1.8pp overall, +6.6pp abstention) but has two limitations:
1. Reconsolidation uses keyword-based slot detection, not a principled encoding
2. Updates are applied directly to memory content, with no separation between the stored representation (genotype) and the expression used for retrieval/QA (phenotype)

V3 introduces a **Memory Genome** abstraction: memory is stored as a structured genotype (slots + metadata + evidence), and expressed as a phenotype (natural language text) only at retrieval time. Mutations produce candidate genomes that undergo evidence validation and fitness selection before activation.

---

## 2. Core Hypothesis

**H1**: Genotype-phenotype separation reduces false update rate by ensuring mutations only modify genotype fields with explicit evidence support.

**H2**: Evidence-grounded selection reduces destructive memory transitions by requiring candidate genomes to pass validation before activation.

**H3**: Phenotype translation produces more consistent retrieval/QA behavior than raw genotype text, because translation can be task-aware.

**H4**: Genome-based mutation preserves more historical evidence than direct overwrite, because old genomes are archived rather than destroyed.

---

## 3. Architecture

```
                    Evidence
                       │
                       ▼
              ┌────────────────┐
              │  Mutation      │
              │  Engine        │
              │  (slot-level)  │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │  Candidate     │
              │  Genome        │
              └────────┬───────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐          ┌─────────────┐
   │  Evidence   │          │   Fitness   │
   │  Validation │          │  Evaluation │
   └──────┬──────┘          └──────┬──────┘
          └────────────┬───────────┘
                       ▼
              ┌────────────────┐
              │  Selection     │
              │  (threshold)   │
              └────────┬───────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐          ┌─────────────┐
   │  Reject /   │          │   Activate  │
   │  Archive    │          │   New       │
   │  (old       │          │   Genome    │
   │   lineage)  │          └──────┬──────┘
   └─────────────┘                 │
                                   ▼
                          ┌────────────────┐
                          │  Translation   │
                          │  (genotype →   │
                          │   phenotype)   │
                          └────────┬───────┘
                                   ▼
                          ┌────────────────┐
                          │  Phenotype     │
                          │  (NL text for  │
                          │   retrieval)   │
                          └────────┬───────┘
                                   ▼
                          Retrieval / QA
```

---

## 4. Data Model

### 4.1 Memory Genome

```python
@dataclass
class MemoryGenome:
    genome_id: str              # unique
    lineage_id: str             # shared across versions
    version: int                # incremented on mutation
    derived_from: str | None    # parent genome_id
    status: GenomeStatus        # ACTIVE / CANDIDATE / ARCHIVED / REJECTED

    # Genotype: structured slots (the "DNA")
    slots: dict[str, SlotValue]
    # SlotValue = {value, confidence, evidence_ids, last_updated}

    # Metadata
    created_at: float
    activated_at: float | None
    evidence_ids: list[str]     # all evidence supporting this genome
    fitness: float              # computed by selection
```

### 4.2 Phenotype

```python
@dataclass
class MemoryPhenotype:
    genome_id: str
    text: str                   # translated natural language
    translation_strategy: str   # e.g., "concise", "detailed", "qa_focused"
    token_count: int
```

Phenotype is **derived**, never stored as source of truth. Multiple phenotypes can exist for one genome (different translation strategies).

### 4.3 Mutation

```python
@dataclass
class Mutation:
    mutation_id: str
    parent_genome_id: str
    candidate_genome: MemoryGenome
    mutated_slots: list[str]
    evidence: list[str]
    confidence: float
    timestamp: float
```

---

## 5. Key Mechanisms

### 5.1 Mutation Engine

- Input: current genome + new evidence + verification result
- Produces: candidate genome with modified slots
- Only slots with explicit evidence support are mutated
- Unmutated slots inherit parent's values + evidence
- Candidate genome status = CANDIDATE (not yet active)

### 5.2 Evidence Validation

- Each mutated slot must have at least one supporting evidence ID
- Confidence below threshold → mutation rejected
- Contradictory evidence → candidate flagged for manual review (or ABSTAIN)

### 5.3 Fitness Evaluation

Fitness score combines:
- Evidence strength (number + confidence of supporting evidence)
- Usage frequency (how often this genome is retrieved)
- Retrieval success (does this genome's phenotype lead to correct answers?)
- Temporal consistency (no future-dated claims for current questions)

### 5.4 Selection

- Candidate genome activated only if fitness > threshold AND evidence validation passes
- Activated genome becomes ACTIVE; parent becomes ARCHIVED (not deleted)
- Rejected candidates are retained with status=REJECTED for auditability

### 5.5 Translation

Genotype → Phenotype strategies:
- **concise**: "User lives in Shanghai."
- **detailed**: "User currently lives in Shanghai (confidence 0.91, evidence E123, updated 2024-03-15). Previously lived in Beijing."
- **qa_focused**: optimized for question answering (includes temporal context)

Translation is deterministic and reproducible.

---

## 6. Experimental Design

### 6.1 Core Comparison: Direct Update vs Genome-Based Mutation

**Baseline (V2.1-Recon)**: direct slot-level update on verification FAIL.

**V3 Genome**: mutation → candidate → validation → selection → activation → translation.

Both use the same retrieval backbone, LLM, prompt, and evaluator.

### 6.2 Synthetic Causal Benchmark (extended)

Extend V2.1's 30-case benchmark to test:
- Multi-step mutations (v1 → v2 → v3)
- Concurrent mutations (two candidates from same parent)
- Selection rejection (low-confidence mutation rejected)
- Phenotype consistency (same genome → same phenotype across translations)
- Historical recall via archived genomes

### 6.3 Adaptive Memory Benchmark (NEW)

LongMemEval-S is per-question independent, so reconsolidation doesn't propagate. V3 needs a dedicated adaptive benchmark:

- Sequential questions where memory updates from Q_i affect Q_{i+1}
- Knowledge update chains (A → B → C)
- Contradictory evidence over time
- Measure: does genome-based updating preserve more evidence than direct overwrite?

### 6.4 Metrics

| Metric | Definition |
|--------|-----------|
| False Update Rate | mutations that modify slots without evidence |
| Evidence Preservation | gold evidence still retrievable after updates |
| Lineage Completeness | all versions traceable via lineage_id |
| Unchanged-Slot Preservation | non-targeted slots retain values |
| Current Recall | current version answers current questions |
| Historical Recall | archived versions answer historical questions |
| Mutation Acceptance Rate | candidates activated / candidates generated |
| Selection Precision | activated candidates that are correct / total activated |
| Phenotype Consistency | same genome → same phenotype (determinism check) |
| Downstream QA | overall/answerable/abstention accuracy |
| Genome Drift | how much phenotype changes across versions |

### 6.5 Ablation

| Variant | Mutation | Selection | Translation |
|---------|----------|-----------|-------------|
| V2.1-Recon (baseline) | direct update | none | raw text |
| V3-Mutation-only | genome mutation | auto-accept | raw text |
| V3-Selection | genome mutation | evidence+fitness | raw text |
| V3-Full | genome mutation | evidence+fitness | phenotype translation |

---

## 7. Phase Plan

### Phase 0: Freeze V2.1 (done)
- Tag v2.1-final
- Freeze all V2/V2.1 results, evaluators, data

### Phase 1: Genome Data Model + Mutation Engine
- Implement MemoryGenome, SlotValue, Mutation
- Mutation engine (slot-level, evidence-gated)
- Unit tests: mutation correctness, unchanged slot preservation

### Phase 2: Validation + Selection + Translation
- Evidence validation
- Fitness evaluation
- Selection threshold
- Phenotype translation (3 strategies)
- Unit tests: selection precision, phenotype determinism

### Phase 3: Synthetic Causal Benchmark (extended)
- Multi-step mutations
- Concurrent candidates
- Selection rejection
- Historical recall

### Phase 4: Adaptive Memory Benchmark
- Sequential question design
- Knowledge update chains
- Compare direct update vs genome-based

### Phase 5: Real LongMemEval-S Evaluation
- V3-Full vs V2.1-Recon baseline
- 500 questions, same LLM/retrieval/evaluator
- Measure all metrics in 6.4

### Phase 6: Analysis + Report
- Ablation analysis
- Failure analysis
- V3 research report

---

## 8. Relationship to V2.1

| V2.1 Component | V3 Successor |
|---------------|-------------|
| Memory.content (string) | MemoryPhenotype (derived from genome) |
| Memory.metadata (slots) | MemoryGenome.slots (structured, evidence-tracked) |
| Reconsolidation (direct update) | Mutation Engine (candidate genome) |
| MemoryEvidence | Genome.evidence_ids + SlotValue.evidence_ids |
| MemoryStatus (ACTIVE/DEPRECATED) | GenomeStatus (ACTIVE/CANDIDATE/ARCHIVED/REJECTED) |
| Verification FAIL trigger | Mutation trigger (same, but produces candidate) |
| No selection | Evidence Validation + Fitness Selection |
| No translation | Phenotype Translation |

V3 is **additive**: all V2.1 code remains frozen. V3 introduces new modules under `src/memforge/genome/`.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Genome overhead slows retrieval | Phenotype caching; translation only on mutation |
| Selection threshold tuning | Calibrate on synthetic benchmark, freeze before real eval |
| Phenotype translation loses information | Detailed strategy includes all slots + evidence |
| Adaptive benchmark too artificial | Combine synthetic causal + real LongMemEval |
| Mutation acceptance too low | Start with permissive threshold, analyze rejection reasons |

---

## 10. Success Criteria

V3 is successful if:
1. False Update Rate < V2.1 direct update (measured on synthetic causal)
2. Evidence Preservation Rate > 0.95 after 10 sequential mutations
3. Mutation Acceptance Rate between 0.3-0.7 (not too permissive, not too strict)
4. Downstream QA on LongMemEval-S >= V2.1-Recon (0.422 overall)
5. Phenotype Consistency = 1.0 (deterministic translation)

If V3 fails these criteria, the negative result is itself valuable: it would show that genotype-phenotype separation does not improve memory reliability in this setting.

---

*End of V3 Research Design. Implementation begins after V2.1-final freeze and user approval.*
