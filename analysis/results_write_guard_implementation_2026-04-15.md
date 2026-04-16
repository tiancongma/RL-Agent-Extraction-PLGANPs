# Results Write Guard Implementation (2026-04-15)

## What Shared Guard Was Added

I added two shared helpers in [run_id.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\utils\run_id.py:368):

- `resolve_governed_results_artifact_path(...)`
  - validates that a write target stays under one of the only two governed `data/results` scopes:
    - legacy run roots: `data/results/run_.../...`
    - v2 child execution roots: `data/results/YYYYMMDD_<hash>/NN_<cue>/...`
  - rejects:
    - uncontrolled top-level roots such as `data/results/dev15_review/...`
    - writes directly under a v2 bucket without a child execution folder
    - any path outside `data/results`
  - can also require:
    - the governed root already exists
    - `RUN_CONTEXT.md` already exists at that governed root

- `build_governed_results_artifact_path(...)`
  - builds a validated artifact path under an existing governed run root or v2 child execution path
  - keeps rich meaning in functional subdirectories such as `analysis/...` rather than in uncontrolled top-level names

This is additive. It does not rename historical directories and does not change how old results are read.

## Which Write Patterns Are Now Blocked

The new shared guard blocks uncontrolled top-level results writes such as:

- `data/results/dev15_review/...`
- `data/results/20260415_targeted_core_a_repair_v1/...`
- any other `data/results/<top-level-name>/...` where `<top-level-name>` is not:
  - a legacy `run_*` root
  - or a v2 bucket name followed by a governed `NN_<cue>` child

It also blocks bucket-only writes such as:

- `data/results/20260415_23c14f0/...`

unless the path continues into a governed child execution root:

- `data/results/20260415_23c14f0/08_stage4_review/...`

For guarded nondefault writers, I also required the governed root to already exist, so they can no longer silently mint new top-level legacy run roots.

## How build_dev15_review_workbook_v1.py Was Repaired

The proof-case repair is in [build_dev15_review_workbook_v1.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\stage4_eval\build_dev15_review_workbook_v1.py:303).

Changes:

- Source resolution now uses governed authority instead of filesystem discovery.
  - Added `--run-dir` and `--run-id`
  - Added `--weak-labels-tsv`
  - The script resolves the source run with `resolve_run_context(...)`
  - The weak-label input now resolves from:
    - explicit `--weak-labels-tsv`, or
    - `ACTIVE_RUN.json` key `stage2_compatibility_tsv`
  - The GT counts input now resolves from:
    - explicit `--gt-counts-tsv`, or
    - `ACTIVE_RUN.json` key `layer1_gt_path`

- Recency-based source discovery was removed.
  - The old `DATA_RESULTS_DIR.rglob("weak_labels__v7pilot_r3_fixparse.tsv")`
    plus latest-mtime selection path was deleted.

- The output no longer writes to fixed top-level `data/results/dev15_review`.
  - Default output is now built under the resolved governed source run:
    - `analysis/dev15_review_workbook_v1/dev15_instance_review_v1.xlsx`
  - Explicit `--out-xlsx` is allowed only when it also stays under a governed results scope.

- Output provenance is explicit.
  - The script now writes a metadata sidecar via
    `write_artifact_metadata_json(...)`
  - It prints:
    - `resolved_source_run_dir`
    - `resolved_source_run_id`
    - exact input file paths
    - generated workbook path
    - generated metadata path

- Registry discoverability was repaired.
  - Added the Stage 4 workbook surface to
    [maintained_script_surface.tsv](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\docs\maintained_script_surface.tsv)
    with `must_use_active_data_source_contract=yes`

## Legacy Scripts That Remain Intentionally Nondefault

These were not migrated into a new runtime model. They now have guardrails only.

- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\stage2_sampling_labels\auto_extract_weak_labels_v7pilot_r3_fixparse.py:3236)
  - remains deprecated legacy Stage 2 extraction
  - no longer creates a timestamped top-level `data/results/run_...` root by default
  - now requires explicit `--out-dir` under an existing governed run scope

- [run_alignment_eval_v1.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\run_alignment_eval_v1.py:325)
  - remains `diagnostic_nondefault`
  - still accepts `--run-id` / `--out-dir`
  - but now validates that the output path stays under an existing governed run scope before creating directories

I did not do a wider migration of the remaining medium-risk helpers in this change.

## Exact Commands And Checks Used To Verify The Guard

Read-only syntax check after edits:

```powershell
@'
from pathlib import Path
paths = [
    Path(r'src/utils/run_id.py'),
    Path(r'src/stage4_eval/build_dev15_review_workbook_v1.py'),
    Path(r'src/stage5_benchmark/run_alignment_eval_v1.py'),
    Path(r'src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py'),
]
for path in paths:
    source = path.read_text(encoding='utf-8')
    compile(source, str(path), 'exec')
    print(f'syntax_ok={path}')
'@ | python -
```

Direct shared-guard proof:

```powershell
@'
from pathlib import Path
from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import resolve_governed_results_artifact_path, build_governed_results_artifact_path

valid = build_governed_results_artifact_path(
    run_dir=Path(r'data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1'),
    artifact_subdir='analysis/dev15_review_workbook_v1',
    filename='dev15_instance_review_v1.xlsx',
    results_root=DATA_RESULTS_DIR,
)
print(f'valid_target={valid}')
print(resolve_governed_results_artifact_path(valid, results_root=DATA_RESULTS_DIR, require_existing_governed_root=True, require_run_context=True)["governed_run_kind"])
try:
    resolve_governed_results_artifact_path(Path(r'data/results/dev15_review/dev15_instance_review_v1.xlsx'), results_root=DATA_RESULTS_DIR)
except Exception as exc:
    print(f'blocked_top_level={type(exc).__name__}:{exc}')
'@ | python -
```

Static proof that the workbook no longer uses the old discovery/write pattern:

```powershell
Select-String -Path src\stage4_eval\build_dev15_review_workbook_v1.py -Pattern 'rglob\(','dev15_review'
```

Shared-guard wiring check:

```powershell
Select-String -Path src\stage4_eval\build_dev15_review_workbook_v1.py,src\stage5_benchmark\run_alignment_eval_v1.py,src\stage2_sampling_labels\auto_extract_weak_labels_v7pilot_r3_fixparse.py -Pattern 'resolve_governed_results_artifact_path','build_governed_results_artifact_path','resolve_artifact_path','resolve_run_context'
```

Note:

- A first `py_compile` attempt failed because Windows denied `__pycache__` writes in this environment, so I switched to the read-only `compile(...)` check above.
