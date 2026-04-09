# Feature Unit Governance

## What a feature unit is

A feature unit is a reusable functional intervention that is expected to solve a specific problem class at one or more explicit points in the pipeline.

In this repository, a feature unit is not just "some code exists." A feature unit is governed by three separate surfaces:

1. registry identity
   - what the feature is
   - which problem class it addresses
   - where it is supposed to intervene
2. intervention matrix
   - which pipeline surfaces should show the feature
   - whether that intervention is required, optional, or not applicable
3. run-level activation report
   - whether a specific run's artifacts prove that the feature actually intervened
4. run metadata section
   - whether the run's `RUN_CONTEXT.md` records the activation report path, required features, missing required features, and the activation gate
   - for governed runs, this section is required rather than optional

## Why logs alone are insufficient

Decision logs and method notes are necessary, but they are not enough to prevent lineage reuse failures.

UFXX9WXE is the motivating case:

- the repo documented the numbered DOE under-enumeration failure
- later child validation and support runs showed the DOE recovery benefit
- the benchmark-valid parent run still reused an older Stage2 child artifact and therefore did not receive the intended intervention

Without a run-level activation surface, the repo could say a fix existed while a benchmark-valid run still lacked evidence that the fix actually fired.

## Code existence vs run activation

These are different states:

- code exists in the repo
  - example: a deterministic enumerator or governance layer is present in `src/`
- feature is registered
  - example: the registry records the feature's problem class and intervention points
- feature is active in a run
  - example: the run-local weak-label or compare artifacts prove that the feature actually intervened

This governance layer also covers observability-only feature units. For example,
`stage2_input_evidence_packing` is intended to make live Stage2 prompt assembly
auditable through a run-local prompt preview artifact. That unit does not change
Stage2 semantics; it records whether the governed input order was actually
materialized in the run artifacts.

The same governance layer can also describe execution feature units. In that
case, the run-local prompt preview is still required for activation evidence, but
the feature changes the Stage2 live input assembly path when the governed
packing mode is enabled. The key distinction remains the same: code presence is
not activation, and activation is only credited when run artifacts prove that
the controlled order was actually used.

`stage2_input_evidence_packing` is a formal member of this system. It is not a
one-off prompt option: its governed activation is evidenced by the prompt
preview artifact and the generated Feature Unit Activation section inside
`RUN_CONTEXT.md`.

The same governance layer also tracks post-raw-output parse-repair units for
diagnostic evaluation:

- `stage2_json_sanitation_path1`
  - syntax-first sanitation before strict parse
  - no semantic reconstruction
- `stage2_legacy_fixparse_fallback_path2`
  - historical wide-row fixparse fallback basis
  - diagnostic-only fallback, not mainline authority

These parse-repair units are audited through dedicated parse-repair comparison
artifacts rather than by reusing the mainline Stage2 prompt-preview contract.

The activation report is intentionally evidence-based. A feature must not be marked `active` only because code for it exists in the repository.

For numbered DOE row activation specifically, governed recovery rows are accepted when they preserve the explicit numbered table-row anchor pattern and the run-level report can prove the same downstream evidence structure.

## Files

- registry:
  - `project/feature_units/feature_unit_registry.json`
- intervention matrix:
  - `project/feature_units/feature_intervention_matrix.tsv`
- report builder:
  - `src/utils/build_feature_activation_report_v1.py`
- run-context updater:
  - `src/utils/update_run_context_with_feature_activation_v1.py`

## How to register a new feature unit

1. Add a new entry to `project/feature_units/feature_unit_registry.json`.
2. Add the feature to `project/feature_units/feature_intervention_matrix.tsv`.
3. Define at least one deterministic activation signal that can be checked from run artifacts.
4. If needed, extend `src/utils/build_feature_activation_report_v1.py` with a new evidence check.
5. Generate a run-level activation report for at least one validating run.

## How to generate an activation report

Example:

```powershell
python src/utils/build_feature_activation_report_v1.py --run-dir data/results/<run_id>
```

Optional explicit paths:

```powershell
python src/utils/build_feature_activation_report_v1.py --registry project/feature_units/feature_unit_registry.json --matrix project/feature_units/feature_intervention_matrix.tsv --run-dir data/results/<run_id>
```

The default output path is:

- `analysis/feature_activation_report_v1.tsv` inside the target run directory

## How to refresh RUN_CONTEXT.md

Example:

```powershell
python src/utils/update_run_context_with_feature_activation_v1.py --run-dir data/results/<run_id>
```

This does two things:

1. refreshes `analysis/feature_activation_report_v1.tsv`
2. injects or replaces the `## Feature Unit Activation` section inside the run's `RUN_CONTEXT.md`

The governed Stage2, Stage 3, and Stage 5 run wrappers call this updater after
their own `RUN_CONTEXT.md` write so the observability record is produced as part
of normal mainline execution rather than as a detached follow-up step.

The governed Stage 5 compare entrypoint and the supported full-pipeline
comparison wrappers also refresh this section so feature activation lineage
cannot disappear from benchmark-facing or diagnostic comparison runs.

## Activation gate

Run-level feature activation uses a small gate vocabulary:

- `pass`
  - all required feature units for the run are active
- `warn`
  - no required feature is missing, but one or more required features are `unclear`
- `fail`
  - one or more required feature units are missing

`RUN_CONTEXT.md` must record this gate alongside the feature activation report path and compact activation summary.

The gate is informational. It helps diagnose missing or unclear feature evidence in run artifacts, but it does not block mainline execution or change benchmark semantics.

## Why this helps

This layer makes one previously invisible failure mode visible:

- a feature can exist in code
- a child validation run can prove that it works
- the benchmark-valid parent run can still miss the intervention because it reused an older artifact

The activation report and the `RUN_CONTEXT.md` feature section are meant to catch that difference directly from the run artifacts.
