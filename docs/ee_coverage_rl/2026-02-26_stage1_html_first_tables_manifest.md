# Stage1 HTML-First Table Selection Manifest Update (2026-02-26)

## Scope
- Task type: Type A (general extraction upgrade)
- Layer: Stage1 table extraction output contract only
- No EE-specific logic added

## Code Changes
- Updated `src/stage1_cleaning/extract_tables_for_keys_v1.py` (additive, backward compatible)
- Added manifest fields in per-key `tables_manifest.json`:
  - `preferred_table_source`
  - `selected_table_files`
  - `fallback_table_files`
  - `selection_rule`

Selection rule implemented:
- `html_first_if_nonempty_else_pdf`
- If `n_tables_html_extracted > 0`:
  - `preferred_table_source = "html"`
  - `selected_table_files = *__html_table.csv`
  - `fallback_table_files = *__pdf_table.csv`
- Else if `n_tables_pdf_extracted > 0`:
  - `preferred_table_source = "pdf"`
  - `selected_table_files = *__pdf_table.csv`
  - `fallback_table_files = []`
- Else:
  - `preferred_table_source = "none"`
  - both lists empty

## Documentation Change
- Updated `project/6_AGENT_RUNBOOK.md` under HTML Extraction Invariant:
  - Added note that table selection is HTML-first with PDF fallback-only unless HTML is empty.

## Validation

### A) WIVUCMYG
Command:
`python src/stage1_cleaning/extract_tables_for_keys_v1.py --keys WIVUCMYG --run-id run_debug_html_first_manifest_v1`

Observed (`data/cleaned/content_goren_2025/tables/WIVUCMYG/tables_manifest.json`):
- `preferred_table_source = "html"`
- `n_tables_html_extracted = 6`
- `n_tables_pdf_extracted = 11`
- `selected_table_files` count = 6
- `fallback_table_files` count = 11
- `selection_rule = "html_first_if_nonempty_else_pdf"`

### B) PDF fallback case (HTML=0, PDF>0)
Command:
`python src/stage1_cleaning/extract_tables_for_keys_v1.py --keys 4L3PHAZA --run-id run_debug_html_first_manifest_v1`

Observed (`data/cleaned/content_goren_2025/tables/4L3PHAZA/tables_manifest.json`):
- `preferred_table_source = "pdf"`
- `n_tables_html_extracted = 0`
- `n_tables_pdf_extracted = 19`
- `html_table_reason = "no_html_table_tags"`
- `selected_table_files` are pdf table files
- `fallback_table_files` is empty

## Compatibility
- Existing fields remain intact.
- No path/layout changes.
- No file deletion/renaming.
