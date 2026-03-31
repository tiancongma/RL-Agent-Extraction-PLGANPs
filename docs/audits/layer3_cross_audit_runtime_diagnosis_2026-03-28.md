# Layer 3 Cross-Audit Runtime Diagnosis

Date: `2026-03-28`

Script audited:

- `src/stage5_benchmark/run_layer3_cross_audit_v1.py`

Workbook audited:

- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`

## Scope

This note diagnoses the interrupted command:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py run `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx
```

This was a code and bounded-runtime audit only.

The workbook was not modified.

No corrected workbook was generated.

## Exact Current Invocation Pattern Before This Fix

The `run` subcommand executed this path:

1. resolve workbook and run context
2. run deterministic `rule_audit(...)`
3. export task packs with `export_model_tasks(...)`
4. execute live Gemini calls with `execute_model_audit(..., backend=\"gemini\")`
5. execute live NVIDIA calls with `execute_model_audit(..., backend=\"nvidia\")`
6. merge rule and model outputs with `merge_reports(...)`

Observed implementation details before the fix:

- backends attempted: both Gemini and NVIDIA
- ordering: sequential, not parallel
- request unit: one workbook cell per request
- batching: none at the API level
- execution style: synchronous
- checkpointing: backend TSV written only after the full backend loop completed
- progress logging: effectively none inside the per-cell model loop
- candidate cap: none
- per-backend call cap: none
- explicit request timeout:
  - Gemini: none in code
  - NVIDIA: fixed `180` seconds
- retries:
  - Gemini: `retries=1` default, so up to `2` attempts
  - NVIDIA: `retries=1` default, so up to `2` attempts

## What `run` Was Likely Doing When It Appeared To Hang

The rule pass is fast and not the likely stall.

The likely long-running stage was live model execution, especially after Gemini
finished and the script entered the NVIDIA loop.

Evidence from run-local artifacts before this fix:

- `layer3_cross_audit_rule_flags_v4.tsv` existed with `32` rows
- `layer3_cross_audit_gemini_results_v4.tsv` existed with `32` rows
- `layer3_cross_audit_nvidia_results_v4.tsv` did not yet exist

That pattern is most consistent with:

- rule audit completed
- Gemini completed
- NVIDIA was still running when the command was interrupted

## Why Only Google / gRPC ALTS Warnings Were Visible

The visible lines such as:

- `alts_credentials.cc:93] ALTS creds ignored. Not running on GCP ...`

are noisy stderr emitted by the Google / gRPC stack used by the Gemini client.

They are not the root cause of the long runtime.

Bounded smoke execution after this fix showed the same warnings while Gemini
still completed successfully on all tested requests.

Conclusion:

- ALTS lines are noisy logging
- they are not a fatal authentication error in this local environment
- the real issue was missing observability during slow sequential model calls

## Most Likely Failure Mode

The interrupted command most likely looked hung because the script combined all
of these behaviors:

- sequential live Gemini then NVIDIA execution
- one request per candidate cell
- no max-candidate or max-call cap
- no per-batch progress logging
- no incremental partial-result writes during the loop
- long per-request waits, especially on hosted model backends

This is primarily:

- missing observability
- unbounded per-cell model execution
- long wall-clock time

It is not primarily:

- workbook reading
- merge time
- ALTS warning spam

## Error Handling / Suppression Findings

There was no infinite retry loop in the audited code.

However, the previous implementation had weak runtime observability:

- backend setup failures were caught and reduced to `blocked_reason`
- per-cell request exceptions broke the backend loop after the first failure
- the user saw almost no explicit progress or per-backend state while waiting

So the issue was not a silent infinite loop, but rather a long opaque live-call
loop with limited runtime reporting.

## Bounded Runtime Controls Added

The script now supports:

- `--rules-only`
- `--skip-gemini`
- `--skip-nvidia`
- `--max-candidates`
- `--max-gemini-calls`
- `--max-nvidia-calls`
- `--batch-size`
- `--request-timeout-seconds`
- `--max-retries`
- `--write-partial-every-batch`

Runtime behavior improvements added:

- explicit backend progress logging
- batch index and total batch count logging
- per-backend request, success, failure, and parse-empty counts
- fail-fast backend blocking on missing credentials or setup failure
- partial TSV writes after each batch when requested
- bounded live execution for smoke tests

## Bounded Smoke Tests Executed

The following bounded tests were actually executed:

Rule audit:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py rule-audit `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx
```

Gemini-only smoke test:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py execute-models `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --skip-nvidia `
  --max-candidates 3 `
  --max-gemini-calls 3 `
  --batch-size 2 `
  --request-timeout-seconds 30 `
  --max-retries 0 `
  --write-partial-every-batch `
  --gemini-tsv data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_cross_audit_gemini_results_v4_smoke.tsv
```

NVIDIA-only smoke test:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py execute-models `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --skip-gemini `
  --max-candidates 3 `
  --max-nvidia-calls 3 `
  --batch-size 2 `
  --request-timeout-seconds 30 `
  --max-retries 0 `
  --write-partial-every-batch `
  --nvidia-tsv data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_cross_audit_nvidia_results_v4_smoke.tsv
```

Smoke-test outcomes:

- refined rule candidate rows: `32`
- Gemini smoke:
  - candidate cells: `3`
  - requests: `3`
  - successes: `3`
  - failures: `0`
  - returned rows: `3`
- NVIDIA smoke:
  - candidate cells: `3`
  - requests: `3`
  - successes: `3`
  - failures: `0`
  - returned rows: `3`

## Recommended Smoke-Test Commands Going Forward

Rules only:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py run `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --rules-only
```

Rules plus bounded Gemini:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py run `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --skip-nvidia `
  --max-candidates 10 `
  --max-gemini-calls 10 `
  --batch-size 5 `
  --request-timeout-seconds 30 `
  --max-retries 0 `
  --write-partial-every-batch
```

Rules plus bounded NVIDIA:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py run `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --skip-gemini `
  --max-candidates 10 `
  --max-nvidia-calls 10 `
  --batch-size 5 `
  --request-timeout-seconds 30 `
  --max-retries 0 `
  --write-partial-every-batch
```

## Files Inspected

- `src/stage5_benchmark/run_layer3_cross_audit_v1.py`
- `docs/methods/layer3_cross_audit_v1.md`
- `docs/src_script_registry.tsv`
- `docs/maintained_script_surface.tsv`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_cross_audit_rule_flags_v4.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_cross_audit_gemini_results_v4.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_cross_audit_nvidia_results_v4_smoke.tsv`

## Bottom Line

The interrupted command was not blocked by the ALTS warnings themselves.

It was spending time in live per-cell model execution, with both backends
called sequentially and almost no progress logging or partial checkpointing.

The runtime controls added in this change make the workflow observable,
bounded, and safe to smoke-test without rerunning another opaque full audit.
