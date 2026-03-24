# Active Data Source Contract

This document defines the repository-wide authority rule for resolving the
current data source in `data/results/` workflows.

## Purpose

Prevent silent use of the wrong run artifacts.

This repository must not infer the active source from:

- directory recency
- lexical sort order
- modification time
- parent fallback
- glob-first matching
- unstated human memory

## Terminology

- `active run`
  - the currently declared results lineage root for repository-level
    data/results-based workflows
- `authoritative data source`
  - the exact run directory and exact source artifact paths declared by the
    current contract
- `terminal surface`
  - the reviewer-facing or benchmark-facing artifact set currently treated as
    the output-layer source of truth for a workflow
- `lineage child`
  - a child execution nested under one parent run root at
    `data/results/<parent_run_id>/lineage/children/...`

## Authority Precedence

For benchmark, alignment, comparison, workbook-generation, or audit workflows,
resolve inputs only by this precedence:

1. explicit CLI path such as `--run-dir`
2. repository authority pointer:
   - `data/results/ACTIVE_RUN.json`
3. otherwise hard error

Compatibility note:

- Exact explicit file paths such as `--final-table-tsv` remain allowed and are
  more specific than run-level resolution.
- Exact explicit `--run-id` may remain as a compatibility alias for selecting a
  known run root, but it must not trigger heuristic discovery.

## Forbidden Behaviors

These behaviors are forbidden for current `data/results/` authority resolution:

- latest-by-sort
- latest-by-mtime
- parent fallback
- glob-first match
- silent defaulting

`runs/latest.txt` may remain for legacy compatibility helpers, but it is not
the primary authority for current `data/results/run_*` and lineage-child
workflows.

## Repository Authority Pointer

Primary machine-readable authority file:

- `data/results/ACTIVE_RUN.json`

Required fields:

- `active_run_id`
- `active_run_dir`
- `authoritative_terminal_files`
- `lineage_policy`
- `updated_at`
- `note`

`authoritative_terminal_files` must pin the exact source artifacts used by the
current terminal surface, for example:

- Stage5 final table
- decision trace
- scope manifest
- audit-ready export
- workbook seed rows
- active workbook path
- GT workbook path

## Metadata Requirements

Workbook, alignment, comparison, and audit artifacts must record:

- `source_run_dir`
- `source_run_id`
- `source_files`
- `generated_by`
- `generated_at`

Preferred sidecar pattern:

- `<artifact_name>.metadata.json`

At runtime, scripts must print:

- the resolved source run directory
- the resolved source run id
- the exact input files being consumed

## Lineage Rule

The parent lineage run remains the human-facing entrypoint, but child terminal
artifacts must be pinned explicitly in `ACTIVE_RUN.json`.

The contract must not assume:

- the newest child is authoritative
- the parent root contains the latest terminal artifacts
- one child can be inferred from another by naming or timestamp

## Migration Note

Historical `runs/latest.txt` remains legacy-only:

- acceptable for backward-compatible helper workflows
- not sufficient as the sole authority for current `data/results` benchmark,
  alignment, comparison, workbook, or audit workflows

## Failure Policy

If neither an explicit CLI source nor `data/results/ACTIVE_RUN.json` can
resolve the required source artifacts, the script must fail loudly with a clear
message.
