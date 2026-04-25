# Proposed Run Naming v2 Spec

Scope: future runs only. Historical paths remain readable and may stay frozen.

## Naming grammar

### Top-level bucket

- Grammar: `YYYYMMDD_<short_hash>`
- Example: `20260329_63b0c8d`
- `YYYYMMDD` is the local execution date for the lineage bucket.
- `<short_hash>` is the short commit or other repo-governed short revision token.
- No `run_` prefix.
- No time-of-day in the bucket name.
- Different dates must always use different top-level buckets.

### Child execution folder

- Grammar: `<ordinal>_<cue>`
- `<ordinal>` is two or three digits, zero-padded.
- `<cue>` is one short lowercase word using `[a-z0-9]+`.
- Examples:
  - `01_stage2`
  - `02_relation`
  - `03_compare`
  - `104_merge`

## Allowed depth

- Depth budget is counted from the top-level bucket.
- Maximum allowed depth is 4.
- Allowed pattern:
  - bucket
  - child execution
  - functional artifact folder
  - files or one more functional subfolder if needed
- Forbidden:
  - nested `run_*` folders
  - recursive `lineage/children/.../lineage/children/...`
  - path meaning encoded through many stacked cue folders

## Child folder grammar

- Child folders must not contain:
  - full historical run IDs
  - timestamps
  - hash fragments outside the top-level bucket
  - multiword descriptive phrases separated by many underscores
- Rich meaning belongs in the explanation files, not the folder name.

## Mandatory explanation file names

### Bucket root

- `LINEAGE.md`
  - required
  - records the bucket purpose, date/hash identity, and ordered child index

### Child execution root

- `RUN_CONTEXT.md`
  - required
  - remains the reproducibility-grade execution spec
- `ARTIFACTS.tsv`
  - optional but recommended
  - lists exact important outputs when the child emits many files

## Active authority rule

- `data/results/ACTIVE_RUN.json` remains the machine authority.
- It must pin:
  - the active bucket path
  - the active child path when relevant
  - the exact authoritative terminal files
- No workflow may infer authority from recency, lexical sort, or child ordinal alone.

## Lineage rule

- One top-level bucket represents one `(date, short_hash)` family.
- Same-day same-hash related work is ordered within that bucket by child ordinal.
- If the date changes, a new top-level bucket is required.
- Same-day retries or follow-up passes must not create new top-level folders when they belong to the same governed family.

## Archival rule

- Historical non-compliant runs may remain under:
  - existing frozen locations
  - `data/results/historical_non_compliant_runs/`
- Archived history must not be renamed solely to satisfy v2.
- New v2 rules apply only to future runs unless an explicit migration plan authorizes normalization.

## Backward compatibility rule

- Readers must continue to recognize historical old-style paths.
- Writers for future runs must emit v2 buckets and v2 child folders only.
- Compatibility utilities may translate historical old-style names into metadata fields, but must not require future buckets to masquerade as old-style `run_*` IDs.
