# Stage2 Text-Path Contract Audit

Date: 2026-04-17

## Scope

This audit examines the maintained Stage2 text-path contract for the current
DEV15 no-marker baseline path, with focus on:

1. where `targeted_manifest.tsv` is created
2. which code path populates its `text_path`
3. which code path later consumes `text_path`
4. whether current governed design intends Stage2 to trust manifest `text_path`
   directly, or to re-resolve through `key2txt.tsv`
5. where the narrowest governed fix should land

## Facts

### 1. `targeted_manifest.tsv` is created inside the maintained composite Stage2 wrapper

File:
- [src/stage2_sampling_labels/run_stage2_composite_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py)

Relevant lines:
- [run_stage2_composite_v1.py:247](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:247)
  loads the input manifest passed by `--manifest-tsv`
- [run_stage2_composite_v1.py:251-260](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:251)
  selects rows by explicit paper keys or uses all manifest rows
- [run_stage2_composite_v1.py:280-282](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:280)
  writes `run_dir / "targeted_manifest.tsv"` by copying `selected_rows`
  directly

Observed fact:
- `targeted_manifest.tsv` is not built by a separate helper.
- The wrapper writes the selected manifest rows verbatim.

### 2. Current `text_path` in `targeted_manifest.tsv` comes directly from the incoming manifest rows

File:
- [src/stage2_sampling_labels/run_stage2_composite_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py)

Relevant lines:
- [run_stage2_composite_v1.py:251-282](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:251)

Observed fact:
- There is no Stage2-side normalization, hydration, or overlay of `text_path`
  before `targeted_manifest.tsv` is written.
- Whatever `text_path` is present in the incoming manifest row becomes the
  `text_path` in the run-scoped `targeted_manifest.tsv`.

Concrete repo state:
- DEV15 rows in
  [manifest_current.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/index/manifest_current.tsv)
  still carry legacy values such as
  `data\\cleaned\\content_goren_2025\\text\\5GIF3D8W.pdf.txt`
- DEV15 rows in
  [key2txt.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/index/key2txt.tsv)
  now point to the repaired maintained clean-text surface such as
  `data/cleaned/content/text/5GIF3D8W.pdf.txt`

### 3. The maintained extractor later consumes `text_path` directly from the Stage2 manifest

File:
- [src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)

Relevant lines:
- [extract_semantic_stage2_objects_v2.py:5896](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:5896)
  declares that `--manifest-tsv` must contain `key/doi/title/text_path`
  columns
- [extract_semantic_stage2_objects_v2.py:5932](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:5932)
  reads the manifest TSV rows
- [extract_semantic_stage2_objects_v2.py:5981-5990](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:5981)
  requires `record["text_path"]`, resolves it relative to `PROJECT_ROOT`, and
  hard-fails if the file does not exist

Observed fact:
- The live extractor does not consult `key2txt.tsv`.
- It trusts `record["text_path"]` from the supplied manifest as the execution
  clean-text path.

### 4. Replay and live normalization paths also trust `record["text_path"]`

File:
- [src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)

Relevant lines:
- [extract_semantic_stage2_objects_v2.py:5455-5458](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:5455)
  legacy replay normalization
- [extract_semantic_stage2_objects_v2.py:5602-5605](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:5602)
  live/raw-response normalization

Observed fact:
- Both replay and live document-normalization code paths derive
  `source_text_path` from `record["text_path"]`.
- This confirms that Stage2 currently treats manifest `text_path` as its
  execution contract surface.

### 5. Stage1 hydration already defines `key2txt.tsv` as the authoritative text-binding surface

File:
- [src/stage1_cleaning/hydrate_manifest_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py)

Relevant lines:
- [hydrate_manifest_v1.py:77-90](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py:77)
  loads `key2txt.tsv`
- [hydrate_manifest_v1.py:129-132](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py:129)
  describes `key2txt.tsv` as the authoritative key-to-text mapping used for
  text hydration
- [hydrate_manifest_v1.py:319-322](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage1_cleaning/hydrate_manifest_v1.py:319)
  populates `text_path`, `text_source_type`, and `text_available` from the
  loaded `key2txt` map

Observed fact:
- In current repo design, `key2txt.tsv` is the maintained clean-text authority
  surface.
- `manifest_current.tsv` is supposed to be hydrated from that surface.

## Inferred Intended Contract

### Stage2 scope contract

Governance evidence:
- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md)
  defines Stage1 as the owner of cleaned-content and manifest asset binding,
  and Stage2 `S2-1` as the stage that resolves declared manifest scope and
  cleaned assets for current execution.
- [project/ACTIVE_PIPELINE_FLOW.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)
  defines Stage2 as consuming cleaned assets, not rebuilding Stage1 authority.
- [docs/maintained_script_surface.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/docs/maintained_script_surface.tsv)
  marks `run_stage2_composite_v1.py` as the maintained coarse-grained Stage2
  entrypoint and `extract_semantic_stage2_objects_v2.py` as the internal Stage2
  extractor.

Inference:
- The maintained extractor is intended to trust manifest `text_path` directly.
- That trust is lawful only when the manifest presented to Stage2 is already
  aligned to the current maintained clean-text authority surface.
- In other words, Stage2 is not intended to invent a second clean-text
  authority. It is intended to consume a scope manifest whose `text_path`
  fields already reflect the authoritative Stage1 text binding.

## Where The Contract Is Broken

The break is between:

1. Stage1 authority:
   - `key2txt.tsv` now points DEV15 papers at repaired local clean text under
     `data/cleaned/content/text/`
2. Stage2 scope materialization:
   - `run_stage2_composite_v1.py` copies stale `text_path` values from the
     incoming manifest rows into `targeted_manifest.tsv` without re-binding them
     to the authoritative `key2txt.tsv` surface
3. Stage2 execution:
   - `extract_semantic_stage2_objects_v2.py` trusts that stale
     `targeted_manifest.tsv` `text_path` directly and fails before the first
     live LLM call

This is why the wrapper-generated
[targeted_manifest.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260417_385b6e1/04_dev15_full_baseline_no_marker_run/targeted_manifest.tsv)
contained broken legacy values even though the repaired clean text existed and
was discoverable through
[key2txt.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/index/key2txt.tsv).

## Fix-Layer Assessment

### Manifest hydration boundary

Pros:
- Most upstream-canonical place to keep `manifest_current.tsv` aligned with
  `key2txt.tsv`

Cons:
- Broader than necessary for this bug
- Reaches back into Stage1 authority maintenance
- User explicitly asked for a narrow Stage2 contract repair, not a Stage1
  redesign

Assessment:
- Too broad for the current repair task

### Stage2 targeted-manifest generation

Pros:
- Exact point where stale `text_path` is propagated into Stage2 execution
- Keeps scope selection manifest/tag/filter driven
- Allows Stage2 to consume the maintained clean-text authority surface without
  redesigning manifest authority
- Produces a run-scoped execution manifest whose `text_path` matches the
  current maintained text surface

Cons:
- Adds a small Stage2-side overlay step

Assessment:
- Best fit for a narrow governed repair

### Extractor text-resolution layer

Pros:
- Also capable of repairing execution

Cons:
- Leaves `targeted_manifest.tsv` stale and misleading
- Makes extractor behavior diverge from the manifest it is given
- Does not satisfy the operational expectation that the generated
  `targeted_manifest.tsv` itself should no longer point at broken paths

Assessment:
- Narrow in code footprint, but less transparent and less consistent with the
  current Stage2 manifest-driven execution contract

## Conclusion

The Stage2 text-path contract is broken at Stage2 scope materialization:

- `run_stage2_composite_v1.py` writes `targeted_manifest.tsv` by verbatim row
  copy
- the copied `text_path` values are stale relative to `key2txt.tsv`
- `extract_semantic_stage2_objects_v2.py` then lawfully trusts those stale
  paths and fails

The narrowest governed repair point is Stage2 targeted-manifest generation,
with a run-scoped overlay from the authoritative maintained `key2txt.tsv`
surface before `targeted_manifest.tsv` is written.
