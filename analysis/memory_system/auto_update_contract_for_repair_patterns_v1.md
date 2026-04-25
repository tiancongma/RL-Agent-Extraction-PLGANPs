# Auto Update Contract For Repair Patterns v1

## Facts
- The governed supporting memory layer lives under `data/mem/v1/`.
- `AGENTS.md` requires a reproducibility-grade run specification for every governed run root used as accepted evidence.
- Auto-append is allowed for candidate records only.
- Promotion above `candidate_historical_pattern` is forbidden automatically and requires evidence checks.

## Automatic Append Eligibility

Append a new candidate row only when all of the following are present in one governed run root:

1. `RUN_CONTEXT.md`
2. a run-local `Feature Unit Activation` surface
   - accepted evidence:
     - `analysis/feature_activation_report_v1.tsv`
     - or an explicitly versioned successor with equivalent columns
3. explicit `Boundary Governance` fields inside `RUN_CONTEXT.md`
4. explicit input and output provenance
   - accepted evidence:
     - `RUN_CONTEXT.md` `source_files`
     - or exact source paths section
     - plus named output artifacts under the run root

If any of those four are missing, do not append a machine-memory candidate row.

## Automatic Append Row Shape

For an auto-appended candidate row, populate at minimum:

- `pattern_id`
- `title`
- `failure_type`
- `earliest_break_boundary`
- `paper_class`
- `trigger_signature`
- `repair_unit_type`
- `code_entrypoints`
- `supporting_docs`
- `required_inputs`
- `activation_evidence`
- `validation_artifacts`
- `benchmark_effect`
- `no_regression_scope`
- `adoption_status`
- `supersedes_pattern_id`
- `notes`
- `covered_change_ids`
- `evidence_run_path`
- `evidence_run_context_path`
- `evidence_feature_activation_path`
- `evidence_boundary_governance_path`
- `activation_evidence_strength`
- `reproducibility_status`
- `pattern_scope_type`
- `linked_governance_changes`
- `acceptance_audit_status`

Hard fill rule:
- If a field cannot be populated from governed evidence, leave it blank or mark it `unknown_from_governed_evidence`.
- Do not infer missing values from folder names, recency, or remembered history.

## Automatic Classification Rule

When a new governed run passes append eligibility:

- append as `candidate_historical_pattern` only
- do not auto-write `validated_replay_pattern`
- do not auto-write `active_mainline_pattern`
- do not auto-write `rejected_or_obsolete_pattern` unless a human explicitly marks that classification

## Promotion Blocks

Promotion above `candidate_historical_pattern` is blocked automatically when any of the following is true:

- `RUN_CONTEXT.md` is missing
- the governed run surface itself is missing
- explicit `Boundary Governance` is missing
- run-local feature activation evidence is missing
- activation depends only on code presence

Required classification when promotion is blocked:

- missing run surface:
  - classify as `historical_unverified` or `supporting_run_without_pattern`
- missing `RUN_CONTEXT.md`:
  - classify as `supporting_run_without_pattern`
- missing governed activation:
  - keep at or below `candidate_historical_pattern`

## Promotion Rules

### Promotion to `validated_replay_pattern`
Requires explicit human review plus all of:

1. a specific run path
2. the run surface exists on disk
3. `RUN_CONTEXT.md` exists
4. run-local `Feature Unit Activation` evidence exists
5. explicit `Boundary Governance` exists
6. the repaired run targets the same failure class as the candidate row
7. the repaired run has before or after comparison evidence against a named failing run or baseline

Minimum accepted evidence set:
- failing run path
- repaired run path
- repaired run `RUN_CONTEXT.md`
- repaired run feature activation report
- repaired run boundary evidence artifact
- repaired run comparison artifact or validation summary

### Promotion to `active_mainline_pattern`
Requires explicit human review plus all of:

1. maintained-surface wiring
   - evidence must come from:
     - `project/ACTIVE_PIPELINE_RUNBOOK.md`
     - `docs/maintained_script_surface.tsv`
     - or a maintained script path referenced there
2. governed run activation visibility
   - activation must be explicit in run artifacts, not only implied by code existence
3. contract alignment
   - the pattern must not violate current Stage2, Stage3, or Stage5 authority contracts
4. `activation_evidence_strength = explicit_governed_activation`

If current code is wired but current run artifacts do not prove activation, keep the row below `active_mainline_pattern`.

## Missing Surface Rule

Missing run surface forces conservative classification.

Examples:
- missing validating run directory:
  - `reproducibility_status = missing_run_surface`
  - keep the row at `candidate_historical_pattern` or lower
- run root exists but lacks `RUN_CONTEXT.md`:
  - classify the run as `supporting_run_without_pattern`
  - do not promote from that run

Example from this audit:
- `data/results/20260417_385b6e1/10_qlyk_capability_restoration_v1`
  contains artifacts but no `RUN_CONTEXT.md`
  and therefore cannot enter the machine index as accepted replay evidence
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
  is missing on disk
  and therefore cannot support replay validation

## Post-Update Validation

Every repair-index update must run all of the following:

1. schema check
   - verify that `docs/repair_index/success_pattern_index_v1.tsv` contains the required evidence fields
2. coverage check
   - compare the current repair index against post-baseline changed maintained scripts and materially relevant runs
3. memory update or rebuild path
   - use `python3 src/utils/build_mem_v1.py` when authoritative markdown or status classifications changed materially
   - use `python3 src/utils/update_mem_v1.py ...` only for additive candidate-only updates
4. memory validation
   - run `python3 src/utils/check_mem_v1.py`

## Material Change Rule

Rebuild the memory-compatible machine index whenever either of the following changes materially:

- governed source markdown relevant to pattern governance
  - examples:
    - `project/ACTIVE_PIPELINE_FLOW.md`
    - `project/ACTIVE_PIPELINE_RUNBOOK.md`
    - `project/2_ARCHITECTURE.md`
    - `project/4_DECISIONS_LOG.md`
    - `docs/maintained_script_surface.tsv`
    - `docs/src_script_registry.tsv`
    - `docs/repair_index/success_pattern_index_v1.md`
    - `analysis/baseline_regressions/post_baseline_repair_audit_20260417.md`
- governed `RUN_CONTEXT.md` artifacts used by the pattern index

Required follow-up when material change happens:

1. refresh `docs/repair_index/success_pattern_index_v1.tsv`
2. refresh `docs/repair_index/success_pattern_index_v1.md`
3. refresh any linked acceptance audit surfaces
4. run `python3 src/utils/build_mem_v1.py`
5. run `python3 src/utils/check_mem_v1.py`

## Incremental Update Rule

Small append-only updates may use incremental memory maintenance only when:

- the change is additive
- no existing row classification changes
- no authoritative markdown changed materially
- the affected run already satisfies append eligibility

In that narrow case:

1. append the new candidate row to `docs/repair_index/success_pattern_index_v1.tsv`
2. append the matching human-readable entry to `docs/repair_index/success_pattern_index_v1.md`
3. run `python3 src/utils/update_mem_v1.py ...` for the new row only
4. run `python3 src/utils/check_mem_v1.py`

Do not use the incremental path when a row changes status, supersedes another row, or depends on newly changed governance markdown.

## Strategy
- Use governed runs to capture candidate evidence early.
- Keep automatic growth limited to candidate evidence capture.
- Require explicit replay evidence and governed activation proof before promotion.
