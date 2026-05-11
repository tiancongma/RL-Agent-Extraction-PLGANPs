# Baseline Object Model Design (2026-04-15)

## Summary

This change introduces the minimum governed baseline object model needed to make baselines first-class machine-readable objects in this repository. The source of truth remains the repository files under `data/baselines/`, not MCP. The new model adds one central TSV registry plus one per-baseline JSON manifest, keeps compatibility with `data/results/ACTIVE_RUN.json` and existing frozen roots, and registers the current `2026-04-15` operational replay baseline as a governed object rooted at `data/frozen/dev15_full_pipeline_freeze_v1/`. The design intentionally does not redesign pipeline stages, benchmark semantics, or existing run/freeze authority.

## Design Goals

- Make each governed baseline addressable by a stable `baseline_id`
- Separate baseline identity from run-folder naming and narrative docs
- Preserve existing authority surfaces:
  - `data/results/ACTIVE_RUN.json`
  - `data/frozen/...`
  - `RUN_CONTEXT.md`
  - `FREEZE_MANIFEST.md`
- Fail loudly when required baseline fields are missing or inconsistent
- Keep MCP additive:
  - baseline files are the source of truth
  - repo-mcp reads and validates them later

## Implemented Object Model

### Registry Layer

Implemented path:

- `data/baselines/BASELINE_REGISTRY.tsv`

Role:

- one machine-readable index of governed baseline objects
- one row per baseline
- stable lookup surface for date-, id-, or manifest-based resolution

Required fields:

- `baseline_id`
- `baseline_date`
- `baseline_type`
- `authority_root`
- `primary_lineage_root`
- `stage_coverage`
- `lawful_resume_boundary`
- `benchmark_validity`
- `active_status`
- `manifest_path`
- `notes`

Contract notes:

- `authority_root` is the baseline object root, not necessarily the active run
- `primary_lineage_root` records the principal provenance lineage root behind the baseline
- `stage_coverage` is a compact registry summary; the manifest carries the structured list
- `lawful_resume_boundary` records the downstream authority boundary exposed by the baseline
- `manifest_path` must point to a governed per-baseline manifest file

### Manifest Layer

Implemented path:

- `data/baselines/baseline_20260415_operational_replay_v1/BASELINE_MANIFEST.json`

Role:

- structured machine-readable details for one baseline object
- exact artifact chain, lineage chain, coverage, limitations, and linked audits

Required fields:

- `baseline_id`
- `baseline_type`
- `date`
- `purpose`
- `authority_root`
- `source_artifacts`
- `artifact_chain`
- `lineage_chain`
- `stage_coverage`
- `boundary_classification`
- `benchmark_validity`
- `limitations`
- `linked_audits`

Contract notes:

- `source_artifacts` pins the governing evidence surfaces behind the baseline
- `artifact_chain` is the ordered operational chain a consumer can inspect directly
- `lineage_chain` records how the baseline relates to frozen and run-local provenance roots
- `boundary_classification` records the lawful resume boundary plus boundary classes relevant to the baseline object
- `limitations` is mandatory because baseline membership and benchmark legality are not implied by the presence of frozen artifacts

## Baseline Type And Validity Vocabularies

Implemented baseline type vocabulary:

- `operational_replay_baseline`
- `full_pipeline_baseline`
- `benchmark_baseline`

Implemented benchmark validity vocabulary:

- `benchmark_valid`
- `not_benchmark_valid`
- `diagnostic_only`

## Implemented 2026-04-15 Baseline

Registered baseline:

- `baseline_20260415_operational_replay_v1`

Judgment:

- type:
  - `operational_replay_baseline`
- authority root:
  - `data/frozen/dev15_full_pipeline_freeze_v1`
- primary lineage root:
  - `data/results/20260412_8517d36/18_full_pipeline_benchmark_dev15_v1`
- lawful resume boundary:
  - `S2-7`
- benchmark validity:
  - `not_benchmark_valid`

Why this is not a full-pipeline baseline:

- the checked operational replay baseline starts at frozen `S2-4b` raw responses
- baseline membership audit evidence shows the checked lineage does not prove `S2-2` traversal inside this baseline
- identity freeze failed, so the lineage is not benchmark-valid overall

Explicit limitations recorded in the manifest:

- lack of `S2-2` coverage inside the checked replay baseline
- failed identity freeze
- no GT compare in the frozen lineage
- observed Stage5 exclusion behavior remains stage-local evidence only

## Helper Tooling

Implemented utility:

- `src/utils/baseline_registry_v1.py`

Supported commands:

- `list`
- `show --query <baseline_id_or_date>`
- `validate`

Current resolution behavior:

- exact `baseline_id` match
- exact `manifest_path` match
- exact `baseline_date` match
- compact date query such as `20260415`

Failure behavior:

- missing registry file: hard error
- empty registry: hard error
- missing required registry fields: hard error
- missing required manifest fields: hard error
- invalid vocabulary values: hard error
- ambiguous date match: hard error
- registry/manifest mismatch: hard error

This supports the first requested use case directly. Example:

```powershell
python src/utils/baseline_registry_v1.py show --query 20260415
```

That command resolves one stable baseline object and returns:

- baseline identity
- baseline type
- authority root
- artifact chain
- lineage chain
- stage coverage
- benchmark validity
- linked audits

## Governance Checks Used

The repaired repo-local MCP server was used as a governance check before implementation.

Validated through repo-mcp:

- active run resolution still points to:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`
- lawful downstream resume boundary:
  - `S2-7`
- proposed new paths were allowed for:
  - `data/baselines/BASELINE_REGISTRY.tsv`
  - `data/baselines/baseline_20260415_operational_replay_v1/BASELINE_MANIFEST.json`
  - `src/utils/baseline_registry_v1.py`
  - `analysis/baseline_regressions/baseline_object_model_design_2026-04-15.md`

The new baseline object model therefore stays additive relative to the existing repository authority contract.

## Compatibility Notes

### With `ACTIVE_RUN.json`

- `ACTIVE_RUN.json` remains the authority pointer for current `data/results` workflows
- it is not replaced by the baseline registry
- baselines and active runs are related but not identical concepts
- a baseline object may point to a frozen root, a results root, or another governed authority root without changing `ACTIVE_RUN.json`

### With Frozen Roots

- `data/frozen/...` roots remain the actual stored baseline-like artifacts
- the new registry/manifest layer indexes and explains them
- frozen roots are no longer expected to stand alone as the only baseline identity surface

### With Narrative Docs

- `docs/archive_project/ARCHIVED_BASELINES.md` remains a historical narrative reference
- it is no longer the only place baseline identity can live

## How To Register A New Baseline

1. Create a stable `baseline_id` using the governed pattern:
   - `baseline_<YYYYMMDD>_<baseline_type_cue>_vN`
2. Create a baseline directory under:
   - `data/baselines/<baseline_id>/`
3. Write `BASELINE_MANIFEST.json` with all required fields.
4. Add one row to `data/baselines/BASELINE_REGISTRY.tsv`.
5. Run:

```powershell
python src/utils/baseline_registry_v1.py validate
```

6. If a new `src/utils/` helper was added for baseline work, record it in:
   - `docs/src_script_registry.tsv`
7. Link the baseline to any governing audits that justify its coverage and limitations.

Registration rules:

- do not infer baseline identity from recency alone
- do not omit limitations when coverage is partial
- do not mark a baseline `benchmark_valid` unless the manifest can point to the legality evidence that supports that state
- keep the manifest rooted in actual authority files, not only descriptive prose

## How Baseline Lookup Should Work In MCP After This Change

1. MCP should read `data/baselines/BASELINE_REGISTRY.tsv` as the baseline index.
2. MCP should resolve a user query such as `20260415 baseline` by:
   - exact `baseline_id` match first
   - exact `baseline_date` match second
   - fail loudly if the date match is ambiguous
3. MCP should then load the manifest from the registry row's `manifest_path`.
4. MCP should return the manifest-backed baseline object, not a heuristic guess from `data/frozen/` names or `data/results/` recency.
5. MCP should treat:
   - `authority_root`
   - `artifact_chain`
   - `lineage_chain`
   - `stage_coverage`
   - `lawful_resume_boundary`
   - `benchmark_validity`
   as first-class fields from the baseline object.
