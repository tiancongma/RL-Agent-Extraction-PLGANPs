# DEV15 Local Source Mapping Validation

## 1. `manifest_current.tsv` is still the only authority
- Maintained governance documents and `src/stage1_cleaning/derive_target_manifest_v1.py` define `data/cleaned/index/manifest_current.tsv` as the single master authority.
- No new canonical manifest was created or substituted. The Stage1 run executed directly against `data/cleaned/index/manifest_current.tsv`.

## 2. Local path mapping is additive, not a second authority
- The local-source recovery layer uses the pre-existing Stage1 compatibility hook in `src/stage1_cleaning/pdf2clean.py` plus a workstation-local config file `.local_stage1_paths.json`.
- The config only remaps the legacy Zotero storage root to this Mac's Zotero storage root at runtime. It does not redefine manifest contents, scope, or source authority.
- A narrow cross-platform fix was applied in `resolve_stage1_source_path()` so the maintained remap contract works when authoritative manifest paths are Windows-style and Stage1 runs on macOS.

## 3. Runtime scope selection still happens via manifest tags / parameters / governed filters
- Current repo governance and `derive_target_manifest_v1.py` route scope selection through manifest tags and CLI predicates, not directory naming.
- This recovery task did not introduce a new directory-based selector or alternate scope authority.
- DEV15 membership remains declared by the manifest `benchmark_tag=DEV15` rows.

## 4. PDF and HTML preservation / conversion status
- The maintained cleaner already supports dual-source processing: when both `html` and `pdf` resolve and `--single-output` is not supplied, `pdf2clean.process_row()` emits one cleaned text per available source family.
- For DEV15 on this Mac, the local Zotero audit found 5 papers with HTML, 10 with PDF, and 0 with both in the same local storage directory.
- Because no DEV15 paper had both local source families present, no DEV15 row required dual-family conversion in this run. The maintained Stage1 behavior still preserves dual-family conversion for any row where both resolve.

## 5. No Stage2 logic was moved into Stage1
- This work only repaired Stage1 local source resolution and reran the maintained Stage1 cleaner.
- No LLM calls were made.
- No Stage2 sampling, labeling, or semantic extraction logic was invoked or moved upstream.

## Result
- DEV15 papers with at least one valid clean text asset after recovery: 15/15.
- DEV15 papers with both HTML-derived and PDF-derived clean text assets after recovery: 0/15.
- Unresolved DEV15 papers: none.
