# ML for Scientific Extraction Learning Plan

## Purpose

This is a working learning log for building ML and LLM systems knowledge through
the repository's real scientific extraction problems.

Short-term goal:

- understand and debug table evidence and value extraction in the current
  pipeline
- learn enough Python and artifact tracing to answer why a value belongs to a
  specific formulation row

Long-term goal:

- build toward AI for science systems: automatic literature reading, database
  construction, modeling, hypothesis generation, and experiment design

## Current Learning Contract

- Primary learning style: whiteboard concepts plus repository-grounded
  engineering practice.
- Weekly time budget: 6-8 hours.
- Daily rhythm: 45-60 minute small steps.
- Paper case study: `QLYKLPKT` first; `UFXX9WXE` as the DOE-specific backup.
- First practice artifact:
  `src/analysis/trace_formulation_value_evidence_v1.py`.

## Repository Mental Model

The learning path follows the value-evidence chain:

```text
Stage1 CSV/table assets
  -> Stage2 normalized_table_payloads_v1.json
  -> Stage2 evidence_blocks_v1.json
  -> S2-4a prompt
  -> S2-5 semantic objects
  -> S2-7 weak_labels TSV
  -> Stage3 relation records
  -> Stage5 final_formulation_table_v1.tsv
  -> Layer3 compare / audit / evidence binding sidecars
```

Ownership map:

- Stage1 owns source table extraction assets.
- Stage2 S2-2 owns execution-grade table authority and LLM-facing evidence.
- Stage2 LLM output owns semantic authorization and markers.
- Stage2 S2-7/function units expand rows only inside authorized scope.
- Stage3 owns formulation relation and shared-parameter structure.
- Stage5 owns final formulation closure and value carry-through, without
  reinventing formulation semantics.
- Compare and audit surfaces diagnose output behavior; current DEV15 outputs
  are diagnostic-only, not benchmark-valid final evidence.

## First Four Weeks

### Week 1 - Code Literacy And Artifact Map

Goals:

- distinguish variables, functions, fields, paths, CLI arguments, and artifact
  names
- read TSV/JSON/JSONL files with small scripts
- resolve current source artifacts through explicit paths or `ACTIVE_RUN.json`

Practice:

- run the trace script for `QLYKLPKT`
- inspect which active pointer paths exist or are missing in this checkout
- filter Stage2 and Stage5 TSV rows by `key == QLYKLPKT`

### Week 2 - Table Evidence Tracing

Goals:

- understand the difference between source CSVs, normalized table payloads, and
  prompt summaries
- connect retrieval/ranking concepts to evidence selection

Practice:

- extend the trace script to report Stage2 normalized table payload and evidence
  block status
- inspect whether the prompt/audit surfaces mention the target paper and tables

### Week 3 - Value Belongs To Row

Goals:

- distinguish extraction from assignment proof
- understand provenance, row identity, scope, and uncertainty

Practice:

- trace fields such as `encapsulation_efficiency_percent`, ratio,
  concentration, drug name, solvent, and surfactant
- compare Stage2 rows, Stage3 relation records, Stage5 final rows, and Layer3
  value compare rows for the same paper

### Week 4 - Evaluation Science

Goals:

- explain diagnostic-only vs benchmark-valid
- understand GT authority, lineage alignment, ablation, and error taxonomy

Practice:

- classify one paper-level or field-level mismatch by failure layer:
  table not recovered, evidence not selected, LLM scope missing, row expansion
  failed, relation binding failed, Stage5 closure issue, value evidence missing,
  or GT ambiguity

## Concept Roadmap

1. Evaluation science: benchmark, diagnostic-only, GT authority, ablation,
   error taxonomy.
2. Retrieval and ranking: evidence recall, precision, prompt packing, table
   visibility.
3. Modern LLM systems: RAG, structured output, replay, validation, tool-use,
   memory.
4. Representation learning: embeddings, latent space, similarity, reranking.
5. Probabilistic thinking: uncertainty, confidence, calibration, risk.
6. Classical ML essentials: supervised learning, train/test split,
   cross-validation, feature engineering.
7. Agent and RL basics: policy, reward, credit assignment, orchestration.

## Checkpoint

Current checkpoint:

- date: 2026-05-12
- stage: Week 1 / value evidence tracing
- current case: `QLYKLPKT`
- current finding:
  - Stage2 has 8 rows and Stage5 has 7 final rows in the local 425/426/427
    diagnostic chain.
  - `f1` is filtered by Stage5 as `candidate_non_formulation`, but it carries
    shared context values such as `surfactant_name`, `organic_solvent`, and
    `drug_name`.
  - Stage3 puts `f1` and the 7 final rows in the same `relation_graph_id`, but
    in separate `method_group_id` values. The `f1` shared fields are shared
    only inside the `f1` method group, not across the single-variable final-row
    method groups.
  - Current Stage5 resolved-field carry-through is exact candidate-id based:
    fields resolved for `f1` are not automatically applied to final rows whose
    representative ids are `QLYKLPKT__single_variable__...`.
- next task:
  - inspect whether a lawful graph-level or parent/context relation exists or
    needs to be designed before any shared-field propagation could be valid.
