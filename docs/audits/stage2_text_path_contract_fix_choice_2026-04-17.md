# Stage2 Text-Path Contract Fix Choice

Date: 2026-04-17

## Chosen Fix

Chosen option: `A`

- normalize / hydrate `text_path` correctly when Stage2 builds
  `targeted_manifest.tsv`

## Why This Option

### Smallest

- The break happens exactly when
  `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  copies selected manifest rows into `targeted_manifest.tsv`.
- Repairing `text_path` there avoids broader Stage1 changes and avoids a more
  invasive extractor refactor.
- The extractor can keep its current maintained contract of reading
  `record["text_path"]` from the manifest it is given.

### Safest

- `manifest_current.tsv` remains the single authority for row universe and
  scope metadata.
- `key2txt.tsv` remains the maintained clean-text authority surface.
- The fix only overlays the run-scoped execution manifest with the current
  authoritative clean-text binding for the selected keys.
- No Mac-specific path is hardcoded into Stage2.
- No live LLM call is required to validate the repair.

### Most Consistent With Repo Governance

Governance already separates:

- manifest authority and scope overlays in Stage1
- execution scope resolution in Stage2 `S2-1`

This fix preserves that split:

- Stage2 still selects scope by manifest tags and governed filters
- Stage2 does not invent a new manifest authority
- Stage2 simply resolves the current maintained clean-text binding for the
  selected rows before execution

## Why Not Option B

Option `B` would keep `targeted_manifest.tsv` stale and make the extractor
silently override it by consulting `key2txt.tsv` at runtime.

That is less transparent because:

- the run-scoped manifest would still advertise broken legacy paths
- the extractor would behave differently from the manifest it was handed
- verification of the Stage2 scope surface would be harder

## Plain-Language Contract Statement

Stage2 should continue to run from a manifest-driven execution surface, but the
run-scoped `targeted_manifest.tsv` must carry the current maintained clean-text
binding from `key2txt.tsv`, not stale legacy `text_path` strings copied
unchanged from older manifest rows.
