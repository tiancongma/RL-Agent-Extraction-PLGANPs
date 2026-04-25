# DEV15 Clean Text Recovery Report

## Summary
- DEV15 rows in authoritative manifest: 15.
- DEV15 papers with at least one valid clean text asset: 15.
- DEV15 papers with both HTML-derived and PDF-derived clean text assets: 0.
- DEV15 papers newly recovered in this Stage1 run: 13.
- DEV15 papers preserved from pre-existing clean text: 2.
- DEV15 papers unresolved after recovery: 0.

## Local source recovery method
- Used the maintained Stage1 cleaner `src/stage1_cleaning/clean_manifest_to_text.py` against the authoritative manifest.
- Local source resolution was handled through the existing `.local_stage1_paths.json` compatibility contract in `src/stage1_cleaning/pdf2clean.py`.
- A narrow portability fix was added so Windows-style manifest paths remap correctly on macOS/Linux.

## DEV15 outcome
- All 15 DEV15 papers now have at least one maintained clean text asset under `data/cleaned/content/text/`.
- DEV15 local Zotero storage exposed one source family per paper in this checkout: either HTML or PDF, but not both for the same paper.
- `data/cleaned/index/key2txt.tsv` now points DEV15 rows at the maintained `data/cleaned/content/text/...` outputs.

## Unresolved papers
- None.

## Notes
- Stage1 ran under the system `python3` environment on this Mac, which emitted the existing warning that Marker was not used because Python 3.13 is not provisioned here.
- DEV15 recovery nevertheless completed successfully using the maintained cleaner and fallback parsers (`pymupdf_fallback` for PDFs, `beautifulsoup_fallback` for HTML).
