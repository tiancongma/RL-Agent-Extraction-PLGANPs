# DEV15 Stage2 Text-Path Verification

## Scope selection
- Source manifest: `data/cleaned/index/manifest_current.tsv`
- Selection rule: `benchmark_tag=DEV15`
- Selected papers: `15`
- No LLM call was made in this verification.

## Result
- Rows refreshed through maintained Stage2 scope-resolution overlay: `15` / `15`
- Rows resolving to existing local clean text: `15` / `15`
- Verification status: `pass`

## Sample papers
- `5GIF3D8W`: `data\cleaned\content_goren_2025\text\5GIF3D8W.pdf.txt` -> `data/cleaned/content/text/5GIF3D8W.pdf.txt` (exists=`yes`)
- `5ZXYABSU`: `data\cleaned\content_goren_2025\text\5ZXYABSU.pdf.txt` -> `data/cleaned/content/text/5ZXYABSU.pdf.txt` (exists=`yes`)
- `BB3JUVW7`: `data\cleaned\content_goren_2025\text\BB3JUVW7.html.txt` -> `data/cleaned/content/text/BB3JUVW7.html.txt` (exists=`yes`)
- `WIVUCMYG`: `data\cleaned\content_goren_2025\text\WIVUCMYG.html.txt` -> `data/cleaned/content/text/WIVUCMYG.html.txt` (exists=`yes`)

## Contract statement
- Runtime scope selection still comes from manifest tags.
- Stage2 execution text binding now resolves from the maintained `key2txt.tsv` surface before `targeted_manifest.tsv` is written.
- This verification exercised the same Stage2 helper used by `run_stage2_composite_v1.py` and required no live backend call.
