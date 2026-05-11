# MDEC084 Writer Contract v1

This document defines the governed writer contract for future MDEC084-style run
creation.

This contract is future-facing only.

- it does not activate the v2 layout by default
- it does not rename or migrate historical `data/results/run_*` surfaces
- it does not change production authority resolution

## 1. Bucket Creation Rule

Future top-level bucket format:

- `YYYYMMDD_<short_hash>`

Bucket identity is determined only by:

- local execution date in `YYYYMMDD`
- the governed short git hash token

A new bucket is created when either of the following changes:

- the date token changes
- the short hash token changes

An existing same-day same-hash bucket is reused when it already exists under
the chosen governed root.

Rule:

- same-day same-hash future executions share one bucket
- separate executions within that bucket are represented as child execution
  folders, not new sibling buckets

Canonical helper behavior:

- bucket path = `<root>/<YYYYMMDD_<short_hash>>/`
- the helper must not infer a different bucket from recency, sort order, or
  partial-name matching

## 2. Child Ordinal Assignment Rule

Future child execution folder format:

- `NN_<cue>`
- `NNN_<cue>`

Child ordinal assignment rule:

- scan only existing valid child folders in the chosen bucket
- parse existing ordinals from child folder names
- assign the next ordinal as `max(existing) + 1`
- if no valid child folders exist, start at `01`

Collision rule:

- the allocator must never auto-reuse an existing child folder name
- collision avoidance is achieved by monotonic ordinal assignment within the
  bucket

Gap rule:

- gaps are allowed
- gaps are not backfilled automatically

Rerun rule:

- a new rerun or new execution step gets a new child folder
- automatic reuse of an existing child folder is not allowed by the allocator
- direct reuse of an already chosen child path, if ever needed for manual
  recovery, must be explicit and must not be inferred by the helper

Cue rule:

- the caller supplies a short cue token describing the execution role
- the helper normalizes the cue to lowercase safe-token form
- normalization uses:
  - lowercase
  - spaces and punctuation collapsed to underscores
  - repeated underscores collapsed
  - leading or trailing underscores removed
- rich semantics must remain in `RUN_CONTEXT.md`, not in the cue text

## 3. Required Child Contents

Every future child execution folder must contain:

- `RUN_CONTEXT.md`

Recommended bucket-level helper file when multiple children exist:

- `LINEAGE.md`

Minimum required child-level reproducibility metadata follows
`docs/run_spec_template.md`, including:

- exact bucket path
- exact child execution folder
- run type
- benchmark status
- starting inputs
- script paths used
- step order
- intermediate artifacts
- final outputs

Optional child-local helper notes are allowed only when they remain
non-authoritative and do not replace `RUN_CONTEXT.md`.

## 4. Path Authority Rule

Child folders are not globally unique outside their parent bucket.

Selector rule:

- use an explicit path
- or use parent-bucket-scoped child resolution

Forbidden selector behavior:

- treating `01_stage2` as globally unique without its bucket
- recency inference
- lexical-sort inference
- parent fallback
- glob-first matching

Current repository authority remains unchanged:

- `data/results/ACTIVE_RUN.json` is still the only authority fallback for
  current `data/results` workflows

## 5. Legacy Coexistence Rule

Historical `run_*` roots remain frozen legacy surfaces.

This writer contract applies only to future v2 runs.

Legacy compatibility rules remain in force:

- do not rename historical run roots
- do not move historical run roots only to satisfy v2 naming
- preserve legacy reads for historical references

## 6. Helper Contract

The minimal helper layer must provide these governed operations without
activating the v2 layout by default:

1. determine the canonical bucket path from a root, date, and short hash
2. validate that a bucket path has a valid bucket name and parent root
3. normalize a cue token
4. determine the next canonical child folder name inside a valid bucket
5. validate that a child path belongs to a valid bucket

The helper layer must reject at least these invalid direct-creation attempts:

- malformed bucket name
- malformed child name
- a child folder placed outside a valid bucket
- a bucket folder placed outside the declared root when root validation is
  requested

## 7. Non-Rollout Boundary

This contract alone does not enable production v2 writes.

Before first real v2 run creation is safe, the repository still needs:

- maintained entrypoint adoption of the writer helper contract
- guarded end-to-end synthetic validation
- explicit decisions about any future non-legacy selector UX beyond explicit
  path
