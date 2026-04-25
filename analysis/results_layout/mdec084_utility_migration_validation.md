# MDEC084 Utility Migration Validation

This note records the compatibility-first utility migration checks performed for
the first MDEC084 code pass.

## Utilities Changed

- `src/utils/run_id.py`
- `src/utils/run_latest.py`
- `src/utils/run_preflight.py`
- `src/utils/active_data_source.py`
- `src/utils/audit_run_lineage_layout_v1.py`
- `src/utils/audit_results_top_level_semantics_v1.py`

`src/utils/paths.py` was inspected but did not require a code change for this
pass.

## Legacy Compatibility Checks Performed

- compiled the targeted utility files successfully with:
  - `python -m py_compile ...`
- verified legacy basename classification still returns `legacy_run_root` for:
  - `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`
- verified `read_active_run_pointer()` still resolves the current
  `data/results/ACTIVE_RUN.json` authority and classifies its target as:
  - `legacy_run_root`
- verified `resolve_run_context()` still resolves:
  - the current authority pointer
  - an explicit legacy `--run-dir`
- verified `src/utils/audit_run_lineage_layout_v1.py` and
  `src/utils/audit_results_top_level_semantics_v1.py` both execute
  successfully against the current repository tree

## V2 Path Compatibility Checks Performed

- verified future-name classifier acceptance for:
  - `20260331_03e5d25` -> `v2_bucket`
  - `01_stage2` -> `v2_child`
- verified synthetic path classification under `data/results/` for:
  - `data/results/20260331_03e5d25` -> `v2_bucket_root`
  - `data/results/20260331_03e5d25/01_stage2` -> `v2_child_execution`
- verified future-name generation helpers without creating any real run:
  - `python -m src.utils.run_id --format v2-bucket`
  - `python -m src.utils.run_id --format v2-child --ordinal 2 --cue relation`

## Default Generation Status

- default legacy generation remains unchanged
- `build_run_id(...)` still emits legacy
  `run_YYYYMMDD_HHMM_<short_hash>_<suffix>` identities
- `python -m src.utils.run_id` still defaults to legacy mode
- future v2 name generation is explicit opt-in only through the new
  `--format v2-bucket` and `--format v2-child` modes
- no writer was switched to emit the v2 layout by default

## What Is Still Not Enabled

- no repository utility creates a real MDEC084 bucket-and-child lineage by
  default
- no child-ordinal allocator exists yet for real execution-time v2 writes
- no active script was switched to write future v2 bucket/child directories
- no historical run directory was renamed or migrated
- `ACTIVE_RUN.json` was not switched

## Remaining Work Before Real V2 Run Creation Is Safe

- update actual run-writing entrypoints to use explicit bucket and child
  allocation instead of legacy `run_id`-only writes
- define the governed writer contract for:
  - bucket creation
  - child ordinal assignment
  - bucket-level lineage note/index creation
- validate all maintained execution entrypoints that currently require
  `--run-id` or write directly under `data/results/<run_id>/...`
- add end-to-end guarded tests for a synthetic v2 bucket root plus child
  execution folder before enabling any default rollout

## Unresolved Ambiguity

- `--run-id` remains a legacy compatibility selector only in this pass.
  A future short-name selector for v2 bucket roots or child executions was not
  introduced here, because child folder names such as `01_stage2` are not
  globally unique without their parent bucket path.
