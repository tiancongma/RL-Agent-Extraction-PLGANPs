# Results Top-Level Governance v1

## Purpose

`data/results/` is the repository entry-point layer for run-scoped outputs and
their immediate governance surfaces.

It exists so a human or tool can open one directory and quickly distinguish:

- current canonical parent runs,
- governed historical exceptions,
- archive roots,
- review-only surfaces,
- top-level index or registry files.

Top level is an entry-point layer, not a dumping ground for every output.

## Rule Distinction

### Top-level containment rule

Top-level containment is governed by the broad family container rule:

- one top-level parent per `(YYYYMMDD, short_commit)` family
- all top-level `run_*` directories matching:
  - `run_<same_date>_<any_time>_<same_short_commit>_*`
  must be contained under one parent directory
- the earliest timestamp in that family is retained as the top-level parent

### Semantic lineage rule

Semantic lineage is narrower than top-level containment.

Different child runs inside the same `(date, short_commit)` family may still
represent:

- different purposes,
- different stages,
- repair or recovery work,
- benchmark versus diagnostic work,
- separate sub-lineages that only share the same engineering day and commit.

The broader top-level containment rule is a directory-governance rule, not a
scientific equivalence rule.

## Allowed At Top Level

The following categories are allowed at the immediate child level of
`data/results/`:

- `canonical_parent_run`
- `frozen_historical_exception`
- `non_run_review_surface`
- `non_run_artifact`
- `archive_root`
- `index_or_registry_file`

## Not Allowed At Top Level

The following should not remain top-level once policy-guided normalization is
performed:

- `lineage_child_run`
- unnamed or unexplained scratch directories
- loose artifacts that clearly belong under a parent run
- duplicate sibling runs that are retries, refreshes, completion steps, or
  stage-only child executions of the same lineage

## Category Definitions

### `canonical_parent_run`

A top-level `run_*` directory that is the authoritative entry point for one
independent experiment or benchmark lineage.

Expected properties:

- its own `RUN_CONTEXT.md`
- authoritative outputs for that lineage
- optional nested child executions under `lineage/children/`

### `lineage_child_run`

A run directory that belongs to an already-declared parent lineage and should
not remain top-level.

Typical examples:

- retries
- partial reruns
- deterministic refreshes
- stage-only child executions
- merge or completion helpers

Decision rule:

- if a top-level `run_*` directory shares the same `(date, short_commit)`
  family as another run and is not the earliest-timestamp parent, it becomes a
  normalization candidate for containment under that parent
- narrower semantic sub-lineages may still be expressed inside the parent
  directory through nested lineage artifacts

### `frozen_historical_exception`

A top-level historical run directory retained temporarily because authoritative
documentation still references the old path.

Decision rule:

- retain top-level only when repository evidence explicitly marks the run as a
  frozen exception or documents a compatibility reason.

### `non_run_review_surface`

A top-level directory that is not a canonical run lineage but exists as a
human-facing review or audit surface.

Examples include:

- review workbook drops
- deterministic reconciliation review packs
- audit-only comparison surfaces

These should remain rare and should eventually move under a more explicit
governed namespace if they continue to grow.

### `non_run_artifact`

A top-level file or directory containing results-like content that is not a
run parent and not an index/registry surface.

This category is allowed to exist during audit, but it is a likely
normalization candidate.

### `archive_root`

A governed top-level container that holds archived or historical material and
is intentionally separate from current canonical parent runs.

### `index_or_registry_file`

A top-level file whose purpose is governance, discovery, or audit of the
results root itself.

Examples:

- run indexes
- historical indexes
- registry templates
- audit summary files for top-level hygiene

## Decision Rule: Stay Top-Level Or Become A Normalization Candidate

An immediate child of `data/results/` stays top-level only if at least one of
the following is true:

- it is the canonical parent for an independent run lineage,
- it is an explicit archive root,
- it is a frozen historical exception backed by repository evidence,
- it is an index or registry file for governing the results root,
- it is a clearly justified non-run review surface that serves as an entry
  point and does not yet have a better governed namespace.

Otherwise it is a normalization candidate.

Normalization-candidate examples:

- a child retry run left at top level,
- a stage-only rerun that belongs under an existing lineage,
- a run with a different purpose but the same `(date, short_commit)` family as
  an earlier parent run,
- a loose artifact file that belongs under a parent run,
- a free-form directory with unclear ownership.

## Policy Ordering

Policy comes first, migration later.

This document defines only top-level classification and governance of
`data/results/`.

It does not itself move, rename, or delete anything.

A migration or normalization pass should happen only after:

1. the top-level categories are applied consistently,
2. a registry or audit artifact records the current state,
3. each proposed move has an explicit retention or normalization reason.
