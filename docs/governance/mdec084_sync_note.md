# MDEC084 Sync Note

This note records the narrow synchronization work performed after the addition
of MDEC084.

## Surfaces Updated

1. Human-readable decision surface:
   - already present in `project/4_DECISIONS_LOG.md`
2. Structured governed decision memory:
   - added `MDEC084` to `data/mem/v1/dec.tsv`
   - added the corresponding index row to `data/mem/v1/idx.tsv`
3. Authoritative naming/governance specification:
   - updated `project/FILE_NAMING_AND_VERSIONING.md`

## What Changed

- Future-facing naming/governance language now states:
  - top-level run bucket: `YYYYMMDD_<short_hash>`
  - child execution folder: `NN_<cue>` or `NNN_<cue>`
  - rich execution meaning belongs in `RUN_CONTEXT.md`
  - `data/results/ACTIVE_RUN.json` remains the active authority surface
  - future lineage must not use nested repeated full `run_id` directories
  - historical `run_*` directories remain frozen legacy surfaces
- Closely coupled guidance was updated in:
  - `project/ACTIVE_PIPELINE_RUNBOOK.md`
  - `docs/run_spec_template.md`
- The structured memory row captures the MDEC084 title, decision statement,
  rationale, forward implication, tags, and authoritative source path.

## What Was Intentionally Not Changed

- no historical run directories were renamed, moved, or deleted
- no active run authority was switched
- no scientific Stage2-Stage5 semantics were changed
- no utility code was modified
- no historical run migration was started
- no broader cleanup of `project/` contents was performed in this sync step

## What Still Remains Before Code-Level v2 Run Generation

- update writer and resolver utilities named by MDEC084:
  - `src/utils/run_id.py`
  - `src/utils/run_latest.py`
  - `src/utils/run_preflight.py`
  - `src/utils/active_data_source.py`
- update any remaining audit or lineage utilities that still assume the old
  global `run_*` regex as the only valid governed results-path identity
- decide and document the exact future writer interface for bucket creation and
  ordinal child allocation before enabling the v2 naming scheme in code
