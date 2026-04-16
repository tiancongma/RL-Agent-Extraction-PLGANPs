# RUN_CONTEXT Enforcement (2026-04-15)

## What enforcement was added

`RUN_CONTEXT.md` is now a hard write-path contract for governed writes under `data/results/`.

- Added [run_context_v1.py](/d:/tiancong/GitHub/RL-Agent-Extraction-PLGANPs/src/utils/run_context_v1.py) with:
  - `validate_run_context(path)`
  - `require_run_context_for_write(run_root)`
  - `create_minimal_run_context(run_root, metadata)`
- Extended [run_id.py](/d:/tiancong/GitHub/RL-Agent-Extraction-PLGANPs/src/utils/run_id.py) so `resolve_governed_results_artifact_path(...)` no longer treats `RUN_CONTEXT.md` as a presence-only convention.
- Patched [build_dev15_review_workbook_v1.py](/d:/tiancong/GitHub/RL-Agent-Extraction-PLGANPs/src/stage4_eval/build_dev15_review_workbook_v1.py) so it explicitly validates the governed output run root before writing.

The enforced minimum contract is:

- `run_class`
- `stage_coverage`
- `boundary_class`
- `lawful_resume_boundary`
- `upstream_authority_source`
- `created_by_script`

Missing or empty required fields now fail validation immediately.

## What cases now fail

These cases now fail loudly:

- Writing to an uncontrolled top-level results root such as `data/results/dev15_review/...`
- Writing under a governed run root that exists but has no `RUN_CONTEXT.md`
- Writing under a governed run root whose `RUN_CONTEXT.md` cannot satisfy the six required fields
- Attempting to create a new governed run root through the shared guard without explicit `RUN_CONTEXT` metadata

`resolve_governed_results_artifact_path(...)` now has an explicit creation path for future governed callers:

- `allow_create_governed_root=True`
- `new_run_context_metadata={...required fields...}`

Without that explicit metadata, new governed root creation fails.

## How existing governed runs pass

Existing governed runs under `data/results/` were not rewritten. Instead, the validator normalizes the required semantics from current repo formats.

Examples:

- The active governed run [RUN_CONTEXT.md](/d:/tiancong/GitHub/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md) passes because the validator can read:
  - run type from `## 2. Run Type`
  - stage coverage from the benchmark-contract chain text
  - upstream authority source from `Active baseline run resolved through`
  - owner script from `Script Paths Used`
- New-format stage-local runs such as [RUN_CONTEXT.md](/d:/tiancong/GitHub/RL-Agent-Extraction-PLGANPs/data/results/20260415_8a2502a/05_s2_5_semantic_parsing/RUN_CONTEXT.md) already expose several of these fields directly in bullet form and pass through the same validator path.

The frozen snapshot reference at `data/frozen/dev15_full_pipeline_freeze_v1/RUN_CONTEXT.md` was used as a format reference only. It is not part of the governed `data/results` write contract enforced here.

## How new runs must supply metadata

New governed run roots are not auto-filled.

Callers that want to create a new governed root through the shared guard must provide all six required fields explicitly to `create_minimal_run_context(...)` or to the guarded creation path in `resolve_governed_results_artifact_path(...)`.

No field guessing was added.

## How build_dev15_review_workbook_v1.py was repaired

The Stage 4 workbook proof-case already used the governed results guard. This change makes the pre-write contract explicit too:

- it resolves the output path under a governed run root
- it requires the governed root to already exist
- it requires a valid `RUN_CONTEXT.md`
- it calls `require_run_context_for_write(...)` before creating the workbook output directory

That means the script now fails before any workbook artifact write if the selected governed run root is missing a valid run context contract.

## Example failure message

Missing `RUN_CONTEXT.md` under an otherwise governed path now fails like this:

```text
FileNotFoundError: RUN_CONTEXT.md not found: D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20990101_abcdef0\01_contract_test\RUN_CONTEXT.md
```

## Exact validation commands used

Blocked uncontrolled top-level root:

```powershell
@'
from pathlib import Path
from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import resolve_governed_results_artifact_path

try:
    resolve_governed_results_artifact_path(
        Path(r'data/results/dev15_review/dev15_instance_review_v1.xlsx'),
        results_root=DATA_RESULTS_DIR,
        require_existing_governed_root=True,
        require_run_context=True,
    )
except Exception as exc:
    print(f'blocked_top_level={type(exc).__name__}:{exc}')
'@ | python -
```

Blocked governed child without `RUN_CONTEXT.md`:

```powershell
@'
from pathlib import Path
from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import resolve_governed_results_artifact_path

run_root = Path(r'data/results/20990101_abcdef0/01_contract_test')
run_root.mkdir(parents=True, exist_ok=True)
target = run_root / 'artifact.tsv'
try:
    resolve_governed_results_artifact_path(
        target,
        results_root=DATA_RESULTS_DIR,
        require_existing_governed_root=True,
        require_run_context=True,
    )
except Exception as exc:
    print(f'blocked_missing_run_context={type(exc).__name__}:{exc}')
'@ | python -
```

Passed governed run with valid `RUN_CONTEXT.md`:

```powershell
@'
from pathlib import Path
from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import build_governed_results_artifact_path, resolve_governed_results_artifact_path

valid = build_governed_results_artifact_path(
    run_dir=Path(r'data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1'),
    artifact_subdir='analysis/dev15_review_workbook_v1',
    filename='dev15_instance_review_v1.xlsx',
    results_root=DATA_RESULTS_DIR,
)
result = resolve_governed_results_artifact_path(
    valid,
    results_root=DATA_RESULTS_DIR,
    require_existing_governed_root=True,
    require_run_context=True,
)
print(f'passed_existing_valid_run={result["governed_run_kind"]}:{result["governed_run_dir"]}')
'@ | python -
```

Direct validator check:

```powershell
python src/utils/run_context_v1.py validate data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1
```

Read-only syntax check:

```powershell
@'
from pathlib import Path
paths = [
    Path(r'src/utils/run_context_v1.py'),
    Path(r'src/utils/run_id.py'),
    Path(r'src/stage4_eval/build_dev15_review_workbook_v1.py'),
]
for path in paths:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print(f'syntax_ok={path}')
'@ | python -
```

## Note

A temporary validation directory was created at `data/results/20990101_abcdef0/01_contract_test` for the missing-`RUN_CONTEXT` failure proof. Cleanup hit a local Windows access-denied lock after validation, so that empty test directory may still be present.
