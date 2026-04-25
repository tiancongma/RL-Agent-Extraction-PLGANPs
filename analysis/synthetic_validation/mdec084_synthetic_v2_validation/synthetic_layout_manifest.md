# Synthetic V2 Layout Manifest

This directory is an analysis-only synthetic validation surface for the future
MDEC084 writer contract.

It is intentionally non-authoritative.

- it does not live under `data/results/`
- it does not change `data/results/ACTIVE_RUN.json`
- it does not represent a real production run
- it exists only to validate bucket/child layout expectations safely

## Synthetic Layout

- bucket:
  - `20260331_ab12cd3`
- synthetic children:
  - `01_stage2`
  - `02_relation`
  - `03_stage5`

## Expected Contract Behavior

- bucket name follows `YYYYMMDD_<short_hash>`
- child names follow `NN_<cue>`
- every child contains `RUN_CONTEXT.md`
- child paths are meaningful only with their parent bucket path
- no active authority is inferred from this synthetic surface
