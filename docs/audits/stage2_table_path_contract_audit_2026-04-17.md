# Stage2 Table-Path Contract Audit

Date: 2026-04-17

## Scope

This audit examines the maintained Stage2 `table_dir` contract for the current
DEV15 no-marker path, focusing on:

1. where `table_dir` is copied into `targeted_manifest.tsv`
2. how Stage2 validates `table_available` and `table_dir`
3. what the intended authoritative source of table assets is

## Facts

### 1. `table_dir` is copied into `targeted_manifest.tsv` by verbatim row copy

File:
- [src/stage2_sampling_labels/run_stage2_composite_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py)

Relevant lines:
- [run_stage2_composite_v1.py:251-260](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:251)
  select manifest rows for the declared scope
- [run_stage2_composite_v1.py:280-282](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:280)
  write `targeted_manifest.tsv` directly from those selected rows

Observed fact:
- Before this audit, there was no Stage2-side refresh of `table_dir`.
- Whatever `table_dir` and `table_available` are present in the incoming
  manifest row become the values in `targeted_manifest.tsv`.

### 2. DEV15 manifest rows currently carry stale Windows-style `table_dir` values

File:
- [data/cleaned/index/manifest_current.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/index/manifest_current.tsv)

Observed DEV15 state:
- every DEV15 row has `dataset_id=goren_2025`
- every DEV15 row has `table_available=yes`
- DEV15 rows carry `table_dir` values like
  `data\\cleaned\\goren_2025\\tables\\5GIF3D8W`

These strings are stale on macOS because Stage2 later interprets them as local
filesystem paths.

### 3. The maintained extractor validates `table_available` and `table_dir` before any LLM call

File:
- [src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)

Relevant lines:
- [extract_semantic_stage2_objects_v2.py:934-947](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:934)

Behavior:
1. read `table_dir` and `table_available` from the manifest record
2. if `table_dir` is present, resolve it relative to `PROJECT_ROOT`
3. if that directory exists, use it
4. if it does not exist and `table_available=yes`, raise
   `FileNotFoundError`
5. only if `table_available` is explicitly false does it skip tables
6. otherwise it falls back to heuristic resolution from `text_path`

Observed fact:
- This validation happens before semantic extraction.
- The current failure for `5GIF3D8W` is therefore a pre-LLM Stage2 contract
  failure, not an LLM or parsing failure.

### 4. Stage2 currently intends to trust manifest `table_dir` directly

Relevant lines:
- [extract_semantic_stage2_objects_v2.py:937-944](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:937)

Observed fact:
- If a manifest row explicitly declares `table_dir`, Stage2 treats that as the
  authoritative table binding for execution.
- The fallback `resolve_tables_dir(text_path, key)` path only runs when there
  is no valid explicit manifest declaration.

### 5. Stage1 hydration already defines the intended authoritative table asset root

Files:
- [src/stage1_cleaning/hydrate_manifest_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py)
- [src/utils/paths.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/paths.py)

Relevant lines:
- [hydrate_manifest_v1.py:238-250](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py:238)
  builds `dataset_tables_map`
- [hydrate_manifest_v1.py:330-333](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py:330)
  sets `table_dir = dataset_tables_root(dataset_id) / key` and marks
  `table_available=yes` if that directory exists
- [paths.py:91-93](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/paths.py:91)
  defines `dataset_tables_root(dataset_id)` as
  `data/cleaned/<dataset_id>/tables`

Observed fact:
- The intended maintained local table authority surface is
  `data/cleaned/<dataset_id>/tables/<paper_key>/`
- For DEV15, that means
  `data/cleaned/goren_2025/tables/<paper_key>/`

### 6. The actual local DEV15 table surface exists

Observed on disk:
- `data/cleaned/goren_2025/tables/5GIF3D8W`
- `data/cleaned/goren_2025/tables/5ZXYABSU`
- `data/cleaned/goren_2025/tables/7ZS858NS`
- `data/cleaned/goren_2025/tables/BB3JUVW7`
- and the rest of the DEV15 keys

So the problem is not that local tables are missing in general. The problem is
that Stage2 is being handed stale `table_dir` strings.

## Inferred Intended Contract

Inference from governance and code:

- `manifest_current.tsv` remains the authority for row universe, dataset
  membership, and scope tags
- Stage1 hydration defines the maintained table asset binding rule through
  `dataset_tables_root(dataset_id) / key`
- Stage2 is intended to trust manifest `table_dir` directly, but only when the
  run-scoped manifest it receives is aligned with the maintained table surface

This mirrors the already-confirmed text-path contract:
- Stage2 should not invent a new table authority
- it should consume a scope manifest whose `table_dir` matches the current
  maintained Stage1 table surface

## Where The Contract Is Broken

The break is at Stage2 scope materialization:

1. `run_stage2_composite_v1.py` copies stale `table_dir` values from the input
   manifest into `targeted_manifest.tsv`
2. `extract_semantic_stage2_objects_v2.py` then lawfully trusts that explicit
   `table_dir`
3. because the copied path is stale Windows-style text, Stage2 raises
   `FileNotFoundError` before any LLM call

## Fix-Layer Assessment

### A. Hydrate `table_dir` when building `targeted_manifest.tsv`

Pros:
- symmetric with the successful `text_path` repair
- keeps `targeted_manifest.tsv` truthful
- preserves extractor behavior
- uses the same Stage1 table-binding rule already encoded in
  `hydrate_manifest_v1.py`

Assessment:
- best fit

### B. Resolve `table_dir` dynamically in extractor

Pros:
- small code footprint

Cons:
- leaves `targeted_manifest.tsv` stale
- makes extractor behavior diverge from the manifest it was handed
- less transparent than the text-path repair

Assessment:
- possible but less clean

## Conclusion

The `table_dir` contract is broken in the same place the `text_path` contract
was broken: the composite Stage2 wrapper writes a run-scoped manifest that
copies stale path bindings unchanged. The narrowest governed repair point is
therefore the Stage2 targeted-manifest generation layer, using the maintained
Stage1 rule:

- `table_dir = data/cleaned/<dataset_id>/tables/<paper_key>`
- `table_available=yes` only if that directory exists
