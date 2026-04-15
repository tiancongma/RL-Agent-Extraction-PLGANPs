# WFDTQ4VX Prevention Plan

## Goal

Prevent the failure pattern:

- discussed yesterday
- believed fixed
- today's replay still fails
- repo users cannot tell whether the fix lived in code, config, lineage,
  compare-only reconciliation, or a one-off local run

This plan is intentionally governance-first and additive.

It does not redesign Stage2, Stage3, or Stage5 semantics.

## Minimal prevention system

### 1. Add a paper-fix contract registry

Add one governed registry file under `docs/audits/`:

- `docs/audits/paper_fix_contract_registry_v1.tsv`

Each row should record one blocker-paper fix claim.

Minimum columns:

- `paper_key`
- `issue_id`
- `issue_summary`
- `owning_stage_boundary`
- `maintained_script_path`
- `required_input_boundary`
- `expected_observable_artifact`
- `expected_observable_predicate`
- `first_diagnosed_run`
- `local_patch_run`
- `merged_maintained_run`
- `replay_verified_run`
- `benchmark_verified_run`
- `current_status`
- `notes`

Why this is minimal:

- one TSV gives a single governed place to answer:
  - what was fixed
  - where it lives
  - which boundary it assumes
  - whether replay verification actually happened

### 2. Add one paper-specific contract file per blocker paper

For blocker papers only, add one short markdown contract under:

- `docs/audits/paper_fix_contracts/<paper_key>_fix_contract_v1.md`

For WFDTQ4VX, the contract would bind the fix claim to:

- stage boundary:
  - `Stage2 primary`
  - `Stage5 secondary narrow collapse support`
- maintained script path:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
- required input boundary:
  - either `current live S2-4b raw-response boundary`
  - or a named replay-frozen raw-response lineage if one is explicitly blessed
- expected observable artifact:
  - Stage2 completed TSV contains WFDTQ4VX batch-level DOE rows or equivalent
    formulation universe
  - Stage5 final table no longer collapses to only `2` rows
- explicit non-claims:
  - Stage5 checkpoint collapse alone is not sufficient
  - compare-only reconciliation does not count as row-generation repair

Why this matters:

- a paper can no longer be called "fixed" without an explicit contract saying
  which boundary the statement applies to

### 3. Add a paper-specific replay regression check

Add one run artifact convention under `data/results/<run>/analysis/`:

- `paper_regression_check_<paper_key>_v1.tsv`
- optional companion markdown:
  - `paper_regression_check_<paper_key>_v1.md`

For each registered blocker paper, the replay regression check must record:

- `paper_key`
- `issue_id`
- `tested_run_dir`
- `tested_start_boundary`
- `maintained_chain_used`
- `expected_observable_artifact`
- `expected_predicate`
- `observed_value`
- `pass_or_fail`
- `reason`

For WFDTQ4VX, the replay check should answer at minimum:

- did the declared replay boundary produce more than the legacy 2-row final
  table outcome?
- did the completed Stage2 artifact contain the expected batch-level DOE row
  universe?
- if not, failure is upstream of Stage5

This check should be run only on:

- a declared boundary
- a declared maintained chain
- a declared paper contract

### 4. Introduce status labels with strict promotion rules

Add these status values to the paper-fix registry:

- `diagnosed`
- `locally_patched`
- `merged_into_maintained_path`
- `replay_verified`
- `benchmark_verified`

Promotion rules:

- `diagnosed`
  - problem localized with explicit run evidence
- `locally_patched`
  - a run-local or paper-local patch improved the case
  - not enough to claim mainline resolution
- `merged_into_maintained_path`
  - maintained script path changed or maintained live path clearly reflects the
    fix
  - still not enough to claim replay safety
- `replay_verified`
  - the fix survives a replay from the declared frozen boundary using only
    maintained scripts
- `benchmark_verified`
  - the full maintained pipeline plus benchmark legality checks confirm the
    fix

Hard rule:

- no paper may be described as "fixed" unless the registry says at least:
  - `replay_verified`
  - and the exact verified input boundary is named

## How this prevents the WFDTQ4VX confusion

Applied to the current case, the registry would have made the distinction
visible immediately:

- older Stage4 reconciliation rule:
  - status could be `merged_into_maintained_path`
  - boundary would be `compare-only`
- Stage5 narrow checkpoint collapse rule:
  - status could be `merged_into_maintained_path`
  - boundary would be `Stage5 closure`
- `2026-04-14` live-current Stage2 recovery:
  - status could be `locally_patched` or `merged_into_maintained_path`
  - boundary would be `live S2-4b`
- operational replay from old frozen raw responses:
  - would remain `not replay_verified`

That would have prevented the phrase "WFDTQ4VX is fixed" from being used
without the missing qualifier:

- `fixed at live current Stage2 boundary only`

## Recommended WFDTQ4VX contract content

### Paper

- `paper_key = WFDTQ4VX`

### Issue summary

- replay from frozen raw responses still collapses to `2` final rows, while a
  newer live-current Stage2 lineage reached `33`

### Owning stage boundary

- `Stage2`

### Maintained scripts

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- `src/stage3_relation/run_formulation_relation_artifacts_v1.py`
- `src/stage5_benchmark/build_minimal_final_output_v1.py`

### Required input boundary

- explicitly one of:
  - `S2-4b live current raw-response boundary`
  - or a named replay-frozen raw-response directory

### Expected observable artifact

- Stage2 completed TSV for WFDTQ4VX contains batch-level DOE row emission or an
  equivalent explicit formulation-row universe

### Expected downstream observable

- Stage5 final table for WFDTQ4VX is no longer the legacy `2`-row outcome

### Replay regression check

- must start from the same frozen raw-response boundary that the contract names
- if the fix only works from a fresh live boundary, the paper may not be marked
  `replay_verified`

## Best minimal implementation sequence

1. create `paper_fix_contract_registry_v1.tsv`
2. add one paper contract for WFDTQ4VX
3. require every blocker-paper fix claim to reference:
   - one registry row
   - one maintained script path
   - one input boundary
   - one expected observable artifact
4. require a replay regression artifact before promoting status to
   `replay_verified`

## Final recommendation

The best minimal prevention mechanism is:

- **a paper-fix contract registry plus a required paper-specific replay
  regression check**

That is small, additive, and compatible with the repo's current governance
style.

It prevents the exact current ambiguity:

- a live-path recovery can be recorded honestly
- but it cannot be mistaken for a replay-verified operational baseline fix
