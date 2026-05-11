# DEV15 Full Diagnostic Baseline Plan — Clean Text/Marker/Selector to Final Table

> **For Hermes cron:** Execute this plan autonomously from repo root with `PYTHONPATH=.`. The user has authorized a fresh DEV15 live Stage2 LLM diagnostic run for this plan. Do not update `data/results/ACTIVE_RUN.json`; do not claim benchmark-valid performance; produce a governed diagnostic baseline with explicit lineage and Layer3 GT field/value compare counts.

## 1. Final objective

Create a new system-managed, reproducible **DEV15 diagnostic baseline** that incorporates the recent clean-text, Marker table/cell authority, selector, Stage2 projection/materialization, Stage3 relation, and Stage5 closure repairs, then runs the complete maintained pipeline from source/clean-text boundary through final table and Layer3 GT field comparison.

Required terminal outcome:

```text
new diagnostic run directory under data/results/<new_run_id>/
completed Stage2 live LLM raw responses and completed Stage2 artifact
completed Stage3 relation artifacts
completed Stage5 final_formulation_table_v1.tsv
completed Layer3 value compare against frozen source-completed GT
status-count summary similar to:
  present_and_match: N
  missing_in_system: N
  present_but_mismatch: N
  extra_in_system: N
  blocked_alignment: N
  not_reported_in_gt: N
  error_rows: N if emitted by compare/audit surface
```

This is a **diagnostic baseline**, not a benchmark-valid promotion. `ACTIVE_RUN.json` must remain unchanged unless the user later explicitly approves promotion.

## 2. Hard governance constraints

1. Use only maintained execution entrypoints from `project/ACTIVE_PIPELINE_RUNBOOK.md` and `docs/maintained_script_surface.tsv`.
2. Resolve all `data/results/` inputs from explicit paths or `data/results/ACTIVE_RUN.json`; never infer by latest timestamp, glob-first, or directory name similarity.
3. Stage2 semantic authority remains LLM discovery plus deterministic post-LLM completion. Deterministic Stage2 table reconstruction may only expand within LLM-authorized semantic scope.
4. Marker/Stage1 extraction artifacts must be generated/frozen once and reused deterministically like the prior clean-text path; do not repeatedly rerun Marker unless raw PDF or parser version changes.
5. Selector may rank/summarize prompt evidence but must not erase execution-grade full table/cell authority.
6. If a blocker appears, query repair index and memory first, then repair the generic capability only; no paper-key-specific runtime branch or hard-coded value map.
7. Credentials come from repo `.env` through maintained scripts. Do not print keys; report only `SET:[REDACTED]` or `MISSING`.
8. No Weixin/WeChat/gateway messaging attempts.

## 3. Context already resolved before writing this plan

### 3.1 Mandatory governance files read

- `project/0_PROJECT_CHARTER.md` — empty file in current checkout.
- `project/1_REQUIREMENTS.md`
- `project/2_ARCHITECTURE.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `project/ACTIVE_DATA_SOURCE_CONTRACT.md`
- `docs/maintained_script_surface.tsv`

### 3.2 Active baseline authority

Resolved from `data/results/ACTIVE_RUN.json`:

```text
active_run_id: 20260423_9c4a03f
active_compare_run_dir: data/results/20260423_9c4a03f/174_layer3_compare_source_completed_gt_authority_v1_diagnostic
active_compare_cells_tsv: data/results/20260423_9c4a03f/174_layer3_compare_source_completed_gt_authority_v1_diagnostic/layer3_value_compare_cells_v1.tsv
active_compare_summary_tsv: data/results/20260423_9c4a03f/174_layer3_compare_source_completed_gt_authority_v1_diagnostic/layer3_value_compare_summary_v1.tsv
layer3_gt_path: data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv
benchmark_valid: no
compare_mode: diagnostic
```

Baseline status counts computed from `layer3_value_compare_cells_v1.tsv`:

```text
not_reported_in_gt: 3677
present_and_match: 1913
missing_in_system: 813
blocked_alignment: 588
present_but_mismatch: 203
extra_in_system: 30
```

### 3.3 Recent repairs and audits to integrate

The plan must account for these recent files/ledgers:

- `docs/plans/2026-05-05-stage1-stage2-cleantext-selector-visibility-repair-plan.md`
- `docs/plans/2026-05-05-stage1-stage2-cleantext-selector-visibility-progress.tsv`
  - clean text visibility full=6 / partial=9 / absent=0
  - table authority exact partial=8 / absent=7
  - selector recall preserved authority with violations=0
- `docs/plans/2026-05-05-characterization-measurement-full-line-repair-and-live-llm-plan.md`
- `docs/plans/2026-05-05-characterization-measurement-full-line-progress-ledger.md`
  - selector/prompt numeric visibility repairs improved replay surfaces but old frozen LLM responses could not reflect newly visible evidence
  - fresh live LLM remained blocked until selector/clean-text gates were repaired; this user message now authorizes the fresh DEV15 diagnostic baseline
- `docs/plans/2026-05-06-end-to-end-boundary-repair-plan.md`
- `docs/plans/2026-05-06-end-to-end-boundary-repair-progress.tsv`
  - documents the stage-by-stage input/output/function/principle split
  - T08/T09/T10 diagnostic chain improved row universe and compare but remained no-live-LLM / diagnostic-only
- `docs/plans/2026-05-07-stage1-marker-current-cleantext-fusion-plan.md`
- `docs/plans/2026-05-07-stage1-marker-current-cleantext-fusion-progress.tsv`
  - Stage1 section/noise sidecar and optional Stage2 structure sidecar metadata attach without defining candidate universe
- `docs/plans/2026-05-07-marker-table-authority-b-phase-repair-plan.md`
- `docs/plans/2026-05-07-marker-table-authority-b-phase-progress.tsv`
  - Marker table/cell metadata, caption binding, continuation metadata, symbol provenance, noise tagging, S2-2 sidecar propagation
  - B09 shows residual caption coverage and Marker rerun reuse problem; these must not block a DEV15 baseline unless the run chooses to promote Marker PDF sidecar as required input. For this diagnostic baseline, prefer frozen/reusable Stage1 surfaces and record Marker sidecar activation status explicitly.

### 3.4 Repair index and memory bootstrap

Before any blocker fix, read/query:

```bash
PYTHONPATH=. python3 src/utils/mem_bootstrap_v1.py --query "DEV15 baseline clean text Marker selector Stage1 Stage2 Stage5 diagnostic baseline complete run GT compare recent fixes"
```

Also read:

```text
docs/repair_index/success_pattern_index_v1.tsv
```

Relevant pattern families include Stage2 table-row/header projection, Stage5 shared preparation carrythrough, Layer3 compare-surface visibility, and alignment bridge patterns. Treat rows without explicit governed activation as historical/partial only.

## 4. Execution design

### Phase A — preflight and frozen input contract

1. Create a new diagnostic run root, e.g.:

```text
data/results/20260507_dev15_full_cleantext_marker_selector_live_diagnostic/
```

If that exact path exists, create a monotonic child suffix under the same date theme. Record the chosen path in `RUN_CONTEXT.md` and progress ledger.

2. Create/resolve explicit DEV15 scope manifest from the governed baseline scope, preferring the existing active baseline scope manifest if recorded in ACTIVE_RUN. Do not select by mtime.

3. Verify `.env` credentials without printing values:

```bash
set -a; . ./.env; set +a
python3 - <<'PY'
import os
for k in ['GEMINI_API_KEY','GOOGLE_API_KEY','NVIDIA_API_KEY']:
    print(k, 'SET:[REDACTED]' if os.getenv(k) else 'MISSING')
PY
```

4. Verify current worktree and test status. Dirty worktree is expected, but runtime script list must be recorded.

### Phase B — Stage1 / Marker / sidecar freeze

Goal: ensure Stage1 clean-text and table/cell sidecar inputs are reusable fixed artifacts before Stage2.

1. Prefer existing canonical clean text and manifest:

```text
data/cleaned/index/manifest_current.tsv
data/cleaned/index/key2txt.tsv
```

2. If Marker sidecars are needed for this diagnostic, generate or freeze them once under the new run root or a stable Stage1 marker artifact root, then consume only frozen artifacts. Do not rerun Marker repeatedly for downstream replay.

3. Required metadata for any frozen Marker artifact:

```text
paper_key
source_pdf_path
source_pdf_sha256
marker/parser version or unknown explicitly recorded
artifact paths and sha256
block/table/cell counts
caption_binding coverage
benchmark_valid=no
```

4. If residual Marker caption coverage blocks sidecar promotion, do not paper-special-case. Either repair the generic binding rule or run baseline with Marker activation status explicitly recorded as partial/diagnostic sidecar only.

### Phase C — Stage2 live LLM DEV15 diagnostic

Use maintained Stage2 composite entrypoint:

```bash
PYTHONPATH=. python3 src/stage2_sampling_labels/run_stage2_composite_v1.py \
  --run-dir <RUN_ROOT>/stage2_live_llm_dev15 \
  --manifest-tsv <DEV15_SCOPE_MANIFEST> \
  --source-mode live_llm \
  --llm-backend gemini \
  --model gemini-2.5-flash \
  --request-timeout-seconds 180 \
  --request-retries 0 \
  --retry-sleep-sec 3.0 \
  --stage1-table-cell-sidecar-root <FROZEN_STAGE1_TABLE_CELL_SIDECAR_ROOT_IF_AVAILABLE>
```

Notes:

- The maintained scripts load `.env` from repo root.
- If a sidecar root is not ready, omit the sidecar flag and record why; do not invent paths.
- Preserve all raw responses, request metadata, evidence blocks, normalized table payloads, candidate blocks, prompt preview, and completed Stage2 TSV/JSONL.
- If live call fails for a subset, preserve partial raw responses and metadata, classify controlled failure, and attempt safe retry only according to the maintained retry policy or a bounded rerun of failed papers.

### Phase D — Stage3 relation materialization

Use maintained Stage3 entrypoint:

```bash
PYTHONPATH=. python3 src/stage3_relation/build_formulation_relation_artifacts_v1.py \
  --weak-labels-tsv <STAGE2_RUN>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv \
  --weak-labels-jsonl <STAGE2_RUN>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl \
  --scope-manifest-tsv <DEV15_SCOPE_MANIFEST> \
  --out-dir <RUN_ROOT>/stage3_relation
```

Record:

```text
candidate_count
relation_row_count
resolved_relation_field_row_count
```

### Phase E — Stage5 final table

Use maintained Stage5 final-output entrypoint:

```bash
PYTHONPATH=. python3 src/stage5_benchmark/build_minimal_final_output_v1.py \
  --input-tsv <STAGE2_RUN>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv \
  --relation-records-tsv <STAGE3_RUN>/formulation_relation_records_v1.tsv \
  --resolved-relation-fields-tsv <STAGE3_RUN>/resolved_relation_fields_v1.tsv \
  --out-dir <RUN_ROOT>/stage5_final
```

Do not add S5-3 live residual value extraction unless a maintained, source/evidence-gated S5-3 path is explicitly ready and the plan records its lawful input boundary. The baseline may be a complete Stage5 final-output diagnostic without S5-3a if S5-3a is not yet governed-ready.

### Phase F — Layer3 GT field/value compare

Use maintained Layer3 compare surface for the requested Run165-style status counts:

```bash
PYTHONPATH=. python3 src/stage5_benchmark/compare_layer3_values_to_gt_v1.py \
  --final-table-tsv <RUN_ROOT>/stage5_final/final_formulation_table_v1.tsv \
  --layer3-gt-tsv data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv \
  --scope-manifest-tsv <DEV15_SCOPE_MANIFEST> \
  --alignment-scaffold-tsv data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv \
  --decision-trace-tsv <RUN_ROOT>/stage5_final/formulation_decision_trace_v1.tsv \
  --out-dir <RUN_ROOT>/layer3_compare
```

Then compute status counts from:

```text
<RUN_ROOT>/layer3_compare/layer3_value_compare_cells_v1.tsv
```

using `compare_status`.

### Phase G — baseline report and acceptance gates

Write:

```text
<RUN_ROOT>/RUN_CONTEXT.md
<RUN_ROOT>/diagnostic_baseline_summary_v1.md
<RUN_ROOT>/diagnostic_status_counts_v1.tsv
<RUN_ROOT>/execution_ledger_v1.tsv
```

Acceptance gates:

1. All maintained stages complete or controlled failures are recorded with exact paper keys and request metadata.
2. Final table exists and has row count recorded.
3. Layer3 compare exists and status counts are reported.
4. Lineage names exact Stage2, Stage3, Stage5, compare, GT, scope, and sidecar paths.
5. No ACTIVE_RUN promotion and no benchmark-valid language.
6. Any blocker repair follows repair-index + memory + generic-capability workflow.

## 5. Blocker handling algorithm

When a stage fails:

1. Stop interpretation and label the failing boundary.
2. Query repair index and memory before code changes.
3. Inspect the precise input/output contract for that boundary.
4. If repair is needed, write a minimal generic patch plus tests.
5. Run bounded replay first, then resume the full diagnostic chain from the lawful boundary.
6. Record all failures, repairs, and replays in the run ledger.

## 6. Cron execution instruction

The cron job should load this plan, execute it autonomously, and continue until it produces either:

```text
completed diagnostic baseline with Layer3 compare status counts
```

or:

```text
blocked report with exact boundary, evidence, repair-index/memory query results, and generic repair attempt status
```

It must not recursively create cron jobs.

## 7. Execution result — 2026-05-07 cron diagnostic baseline

- run_root: `data/results/20260507_c1ad6ca`
- benchmark_valid: no
- compare_mode: diagnostic
- ACTIVE_RUN_json_updated: no
- scope_manifest_tsv: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- layer3_gt_path: `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`
- alignment_scaffold_tsv: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv`
- Stage1/Marker sidecar activation: `partial/omitted` — no verified frozen Stage1 table-cell sidecar root was resolved; `--stage1-table-cell-sidecar-root` was omitted rather than inventing a path.
- Stage2 run dir: `data/results/20260507_c1ad6ca/01_stage2_live_llm_dev15`
- Stage2 completed TSV: `data/results/20260507_c1ad6ca/01_stage2_live_llm_dev15/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage2 status: `success_count=12`, `failure_count=3`, `contract_validation_status=success`, `compatibility_projection_status=success`
- Stage2 live-call failures: `5GIF3D8W`, `BB3JUVW7`, `WFDTQ4VX` (`DeadlineExceeded`, `504 Deadline Exceeded`, request metadata preserved, no raw payload persisted)
- Stage3 relation records: `data/results/20260507_c1ad6ca/02_stage3_relation/formulation_relation_records_v1.tsv`
- Stage3 resolved fields: `data/results/20260507_c1ad6ca/02_stage3_relation/resolved_relation_fields_v1.tsv`
- Stage5 final table: `data/results/20260507_c1ad6ca/03_stage5_final/final_formulation_table_v1.tsv`
- Stage5 decision trace: `data/results/20260507_c1ad6ca/03_stage5_final/final_output_decision_trace_v1.tsv`
- Layer3 compare cells: `data/results/20260507_c1ad6ca/04_layer3_compare/layer3_value_compare_cells_v1.tsv`
- Layer3 compare summary: `data/results/20260507_c1ad6ca/04_layer3_compare/layer3_value_compare_summary_v1.tsv`
- Diagnostic summary: `data/results/20260507_c1ad6ca/diagnostic_baseline_summary_v1.md`
- Status-count TSV: `data/results/20260507_c1ad6ca/diagnostic_status_counts_v1.tsv`

Layer3 `compare_status` counts from `layer3_value_compare_cells_v1.tsv`:

```text
blocked_alignment: 2772
not_reported_in_gt: 2345
present_and_match: 1229
missing_in_system: 636
present_but_mismatch: 223
extra_in_system: 19
```

This is diagnostic-only, not benchmark-valid final performance evidence.


## Superseding continuation summary — actual completed lineage (appended 2026-05-07T12:28:40Z)

The earlier `Completed diagnostic baseline outputs` section in this RUN_CONTEXT references a non-existent `01_stage2_live_llm_dev15` lineage from an interrupted/stale attempt. The actual completed lineage for this cron continuation is the following:

- benchmark_valid: no
- compare_mode: diagnostic
- ACTIVE_RUN_json_updated: no
- Stage2 run dir: `data/results/20260507_c1ad6ca/01_stage2_live_llm`
- Stage2 completed TSV: `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage2 completed JSONL: `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
- Stage2 request summary: `data/results/20260507_c1ad6ca/01_stage2_live_llm/analysis/request_summary.tsv`
- Stage2 raw responses: `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/raw_responses/` (14 persisted; `UFXX9WXE` failed live request)
- Stage2 request metadata: `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/request_metadata/` (15 metadata sidecars)
- Stage2 status: `success_count=14`, `failure_count=1`, failed paper `UFXX9WXE`, `DeadlineExceeded: 504 Deadline Exceeded`; compatibility projection and contract validation succeeded.
- Retry attempt: `data/results/20260507_c1ad6ca/02_stage2_live_llm_retry_complete` was attempted with one retry but stalled during live calls and was killed; it is diagnostic-incomplete and was not used downstream.
- Stage3 run dir: `data/results/20260507_c1ad6ca/03_stage3_relation_from_01_stage2`
- Stage3 relation records: `data/results/20260507_c1ad6ca/03_stage3_relation_from_01_stage2/formulation_relation_records_v1.tsv`
- Stage3 resolved relation fields: `data/results/20260507_c1ad6ca/03_stage3_relation_from_01_stage2/resolved_relation_fields_v1.tsv`
- Stage3 counts: `paper_count=14`, `candidate_count=258`, `relation_row_count=2600`, `resolved_relation_field_row_count=1284`
- Stage5 run dir: `data/results/20260507_c1ad6ca/04_stage5_final_from_01_stage2_03_stage3`
- Final table: `data/results/20260507_c1ad6ca/04_stage5_final_from_01_stage2_03_stage3/final_formulation_table_v1.tsv`
- Stage5 decision trace: `data/results/20260507_c1ad6ca/04_stage5_final_from_01_stage2_03_stage3/final_output_decision_trace_v1.tsv`
- Stage5 counts: `input_rows=258`, `final_rows=158`, `filtered_rows=99`, `collapsed_rows=1`, `downstream_variant_rows=2`
- Layer3 compare dir: `data/results/20260507_c1ad6ca/05_layer3_compare_source_completed_gt_from_04_stage5`
- Layer3 compare cells: `data/results/20260507_c1ad6ca/05_layer3_compare_source_completed_gt_from_04_stage5/layer3_value_compare_cells_v1.tsv`
- Layer3 compare summary: `data/results/20260507_c1ad6ca/05_layer3_compare_source_completed_gt_from_04_stage5/layer3_value_compare_summary_v1.tsv`
- Layer3 status counts TSV: `data/results/20260507_c1ad6ca/analysis/layer3_status_counts_05_compare_v1.tsv`
- Layer3 compare_status counts:
  - blocked_alignment: 1554
  - extra_in_system: 38
  - missing_in_system: 694
  - not_reported_in_gt: 3159
  - present_and_match: 1393
  - present_but_mismatch: 386

Diagnostic-only statement: this run is a system-managed DEV15 diagnostic baseline from the completed Stage2 boundary through Stage5 final table and Layer3 value compare. Because Stage2 had one live LLM request failure (`UFXX9WXE`) and this is a diagnostic run, these outputs are **not benchmark-valid performance evidence**.
