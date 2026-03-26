# Minimal Final-Output Layer Design

This document defines the minimal final-output layer contract for the repository.
It remains the durable design contract for the layer even after the first controlled implementation.

Implementation status on `2026-03-25`:
- phase 1 runtime scripts exist under `src/stage5_benchmark/`
- phase 1 final-output artifacts can be produced from the compatibility-projected
  legacy wide-row surface emitted after semantic Stage2
- the downstream benchmark comparison step now exists under `src/stage5_benchmark/`
- the canonical full path is manual Stage 0 to Stage 5 reproduction defined in `project/ACTIVE_PIPELINE_FLOW.md`

## Why this layer is needed

The repository now has an explicit benchmark policy: official GT comparison may be reported only from the complete pipeline final-output layer. The current active DEV-15 path stops earlier. It extracts candidate formulation-instance rows and evaluates those candidate rows directly against the fixed skeleton benchmark workbook. That path is useful for diagnosis, but it does not yet produce a benchmark-valid final formulation table.

The missing layer is therefore not optional housekeeping. It is the smallest downstream layer needed to turn high-recall candidate-instance output into a final one-row-per-formulation artifact that can legitimately support benchmark claims and later modeling exports.

## Why current Stage2 -> Stage4 candidate comparison is not sufficient

The active semantic Stage2 emitter produces semantic objects, and the
deterministic compatibility adapter projects those objects into the legacy
candidate formulation-instance rows still consumed by unchanged Stage3, Stage4,
and Stage5. `src/stage4_eval/eval_weak_labels_v7pilot3.py` evaluates those
compatibility-projected rows as candidate-instance behavior. That comparison is
intentionally useful for under-segmentation, over-segmentation, and paper-level
debugging, but it does not own final formulation closure.

Candidate-instance semantics and final formulation-table semantics are different. A paper may legitimately produce more candidate rows than final formulation rows. Provenance labels such as baseline, optimized, or sweep-derived are still valuable, but they do not by themselves define whether two rows should remain separate in a benchmark-valid final table.

## Architectural position

In the documented eight-layer architecture, this layer sits between:

- Layer 6: instance-level evaluation and diagnostics
- Layer 8: final formulation table and modeling

More precisely, it is the concrete implementation of Layer 7: light guardrail and final formulation filtering.

It consumes candidate-instance outputs from the active path, applies narrow downstream closure rules, and emits a final formulation table suitable for benchmark comparison and later modeling/export work.

## Minimal responsibilities owned by this layer

The minimal final-output layer owns only the following responsibilities:

1. Final non-formulation row filtering.
2. Limited core-parameter-based row collapse when the duplicate or redundancy case is clear and defensible.
3. Final one-row-per-formulation semantics for the benchmark-facing output.
4. Decision-trace export explaining which candidate rows were retained, suppressed, or collapsed.
5. Benchmark-valid final formulation table export.

This layer should remain small. Its purpose is final row closure, not broad scientific reinterpretation.

## Responsibilities explicitly excluded from this layer

The minimal final-output layer does not own:

1. Evidence packing.
2. LLM candidate formulation extraction.
3. Initial parent/variant proposal.
4. Broad rule-heavy formulation reconstruction.
5. Large-scale scientific context extraction beyond what is required for final row closure.
6. Historical benchmark bootstrap logic.
7. Release-specific modeling filters such as PLGA-only subsetting for downstream analysis.

Those responsibilities belong upstream, downstream, or in historical reference code.

The final-output layer may consume Stage 3 relation artifacts as deterministic
provenance, but relation materialization itself belongs to Stage 3, not Stage 5.

## Input artifacts

Minimum required input:

- a run-scoped candidate-instance TSV produced by the deterministic
  compatibility adapter, currently `weak_labels__v7pilot_r3_fixparse.tsv`

Optional but useful supporting inputs:

- the split manifest used for the run
- Stage 3 relation records from `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- candidate-instance evaluation diagnostics from the active Stage4 evaluator
- cleaned text/table assets only if a later narrow guardrail needs traceable evidence lookup

The final-output layer must not require a historical benchmark bootstrap path as input.

## Output artifacts

The minimal layer is expected to produce at least:

1. A decision-trace artifact recording row-level keep/drop/collapse outcomes.
2. A grouped or normalized intermediate artifact showing which candidate rows map to the same final formulation identity.
3. A benchmark-valid final formulation table with one row per final formulation.
4. A benchmark-facing comparison artifact produced downstream from that final table.

The final formulation table is the contract boundary for future benchmark-valid reporting.

## Required benchmark-valid semantics

Before a run may be called `full_pipeline_benchmark_run`, the following must be true:

1. Candidate-instance extraction has completed.
2. Deterministic Stage 3 relation materialization has completed for the declared scope.
3. The minimal final-output layer has executed on those candidate rows.
4. The benchmark comparison is performed on the final formulation table, not directly on candidate rows.
5. The run specification explicitly records that the run is benchmark-valid and names the final-output artifact used for comparison.

If any of those conditions is missing, the run remains diagnostic-only.

## Relationship to current candidate-instance runs

Current candidate-instance runs remain valid engineering assets. They are still the right place to inspect boundary loss, over-segmentation, and evidence-packing failures. The final-output layer is downstream of those runs, not a replacement for them.

The current DEV-15 path should therefore be read as:

- candidate extraction and candidate diagnostics are implemented
- deterministic relation materialization is implemented as an explicit Stage 3 layer
- final-output closure is implemented in phase 1
- final-table comparison is implemented as a Stage 5 step and must be run manually as part of the complete path

## Relationship to historical Stage5 benchmark ideas

The surviving `src/stage5_benchmark/` family contains useful reference concepts, especially around core signatures, two-table schemas, derivation, and projection. Those scripts are not the current active final-output path. They should be treated as reference assets or adapter candidates, not as proof that the minimal final-output layer already exists.

Historical skeleton bootstrap scripts under `archive/code/dev15_skeleton_bootstrap/` are also reference-only. They explain how the benchmark artifact was prepared, not how the current pipeline produces benchmark-valid final rows.

## full_pipeline_benchmark_run contract

A `full_pipeline_benchmark_run` must include, in order:

1. corpus/split selection and cleaned-input confirmation
2. semantic Stage2 object generation
3. deterministic compatibility projection into the legacy wide-row surface
4. deterministic Stage 3 relation materialization
5. optional candidate-instance diagnostic evaluation
6. minimal final-output layer execution
7. benchmark comparison against GT using the final formulation table
8. run-scoped reproducibility documentation naming the final-output artifact

Without steps 4 and 5, the run is not benchmark-valid.

## Phase 1 minimal viable final-output layer

Phase 1 should do only the following:

1. consume candidate-instance rows
2. drop explicit non-formulation rows
3. compute a narrow core-parameter signature for collapse decisions
4. collapse only clearly redundant rows with a retained decision trace
5. emit a final one-row-per-formulation table
6. hand that table to a benchmark comparison step

This phase is intentionally smaller than the historical rule-heavy reconstruction families.

## Later optional extensions

Possible later extensions, which are not part of the minimum contract:

1. richer DoE-specific coordinate alignment
2. stronger Stage 3 relation-aware parent/variant inheritance resolution
3. measurement-level separation beyond the benchmark table
4. modeling-target-specific export variants
5. broader schema harmonization with older Stage5 experiments

These should remain deferred until the minimal layer exists and is validated.
