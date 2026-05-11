# Stage1 Parser / HTML Native Tables / Cell-Level Sidecar Repair Plan

> **For Hermes:** Use `subagent-driven-development` only for execution after user approval. Current document is a design plan only; do not run live LLM calls and do not connect Weixin/WeChat.

**Goal:** Determine whether Marker materially improves **PDF** clean-text/table structure versus the current PyMuPDF fallback, strengthen HTML-native table extraction as a separate non-Marker path, and add a Stage1 cell-level table sidecar that preserves row/header/cell geometry for downstream Stage2 authority construction.

**Architecture:** Treat Stage1 as source-structure preservation, not semantic extraction. Marker is a **PDF replacement candidate only**; it must not be judged against HTML-native extraction and must not be used for HTML sources. HTML papers use native DOM table extraction. Parser bakeoff outputs must be diagnostic-only until promoted. Runtime parser changes must be selected by measurable PDF structure/anchor improvements, not by parser reputation. The new cell-level sidecar is additive and source-faithful: it may preserve geometry and lineage, but it must not infer formulation semantics, row identity, or benchmark values.

**Tech Stack / Candidate Tools:** current `pdf2clean.py` (`PyMuPDF`, `trafilatura`, `BeautifulSoup`, optional `marker`), candidate `marker-pdf`, candidate `pymupdf4llm`/`pdfplumber`/`docling` only as controlled follow-ups if Marker is insufficient.

**Governance:**
- Do not modify files in `project/` for this diagnostic plan.
- Do not change `data/results/ACTIVE_RUN.json`.
- Do not register a new maintained entrypoint until diagnostic evidence and review pass.
- All bakeoff outputs write under a new explicit run directory in `data/results/<run_id>/...` with `RUN_CONTEXT.md` marking `diagnostic_only: yes` and `benchmark_valid: no`.
- Use explicit input paths from `data/cleaned/index/manifest_current.tsv` and DEV15 scope files; never latest-by-sort, mtime, glob-first, or parent fallback.
- No live LLM calls are required for Stage1 parser bakeoff.

---

## Acceptance Criteria

### A. Marker PDF bakeoff acceptance

Marker is considered beneficial only as a **PDF parser replacement candidate** if it improves Stage1 PDF structure metrics on DEV15/problem PDF anchors without creating major regressions. HTML rows are excluded from Marker-vs-current scoring; HTML is evaluated separately through native DOM extraction.

Required metrics:

1. **PDF clean-text anchor visibility**
   - exact fragment full/partial/absent counts using `audit_source_anchor_cleantext_visibility_v1.py`
   - numeric-token fallback count must not be treated as full recovery

2. **PDF table authority visibility**
   - exact/partial/absent counts using `audit_source_anchor_table_authority_visibility_v1.py`
   - reduce `payload_exists_but_row_header_geometry_degraded`
   - reduce PDF table-authority `absent` papers

3. **Marker PDF structured-output metrics**
   - Marker markdown/table block count, not just paragraph count
   - parsed markdown table count
   - cell count per parsed PDF table
   - non-empty row count / column count
   - markdown/header separator recovery
   - non-empty header-path coverage
   - row-label/column-label coverage
   - caption-to-table binding presence
   - page/bbox availability where Marker supports it

4. **Downstream PDF diagnostic smoke metrics**
   - S2-2 normalized table payload count for selected papers
   - selector authority registry candidate count and violations
   - prompt summary adequacy probe for formulation/design/optimization blocks

5. **Regression guards**
   - no drop in clean-text full visibility papers
   - no new absent source anchors
   - no semantic fields emitted by Stage1
   - no paper-key-specific runtime branches

### B. HTML-native table extraction acceptance

HTML-native extraction is accepted if it preserves DOM table geometry better than current text serialization:

- parse `thead`, `tbody`, `tfoot`, `caption`, `th`, `td`
- preserve `rowspan`, `colspan`, original row/column index, raw cell text, normalized cell text
- preserve blank cells and header-only cells
- bind table caption/nearby label where available
- serialize both a human-readable table block and a machine-readable cell sidecar

### C. Cell-level Stage1 sidecar acceptance

The Stage1 cell sidecar is accepted when every table block emitted by Stage1 can be linked to zero or more cell rows with source lineage:

Required output schema:

```text
paper_key
source_type
source_path
parser
parser_variant
table_id
table_source_kind
page
bbox_json
caption
row_index
col_index
rowspan
colspan
raw_cell_text
normalized_cell_text
is_header_cell
header_scope
header_path_json
row_label_text
column_label_text
source_block_id
source_hash
warnings_json
```

The sidecar must be additive: current text files and `key2txt.tsv` remain stable unless explicitly approved.

---

## Task 1: Freeze baseline Stage1 visibility metrics

**Objective:** Capture current parser behavior and current clean-text/table-authority failures before any parser experiment.

**Files:**
- Read: `data/cleaned/index/manifest_current.tsv`
- Read: DEV15 scope/anchor files already used by existing diagnostics
- Output: `data/results/<run_id>/01_stage1_current_parser_baseline/`

**Steps:**
1. Run current clean-text visibility audit on DEV15.
2. Run current table-authority visibility audit on DEV15.
3. Run selector anchor recall and prompt semantic adequacy diagnostics only if they consume existing artifacts without live LLM.
4. Write a `RUN_CONTEXT.md` recording exact manifest, scope, scripts, and output paths.
5. Verify summary contains: clean-text full/partial/absent, table-authority full/partial/absent, selector violations.

**Expected:** Reproduce current known shape: clean text not fully absent, table authority still weaker than clean text, selector authority violations zero or explainable.

---

## Task 2: Add parser-bakeoff diagnostic script without changing runtime parser

**Objective:** Compare current parser and Marker on the same source files without changing `clean_manifest_to_text.py` active behavior.

**Files:**
- Create: `src/stage1_cleaning/run_stage1_parser_bakeoff_v1.py`
- Test: `tests/test_stage1_parser_bakeoff_v1.py`
- Output: `data/results/<run_id>/02_stage1_parser_bakeoff/`

**Design:**
- Input must be explicit: `--manifest`, `--scope-keys`, `--out-dir`, `--parser current|marker|all`.
- For each paper, resolve source path using the same manifest/path-remap logic as `pdf2clean.py`.
- Run current extraction and Marker extraction into separate candidate artifacts.
- Do not overwrite `data/cleaned/content/text/`.
- Emit:
  - `parser_bakeoff_summary_v1.tsv`
  - `parser_bakeoff_blocks_v1.jsonl`
  - `parser_bakeoff_tables_v1.jsonl`
  - `parser_bakeoff_cells_v1.jsonl`
  - `parser_bakeoff_warnings_v1.tsv`

**Verification:**
- If Marker is not installed, script exits with a clear diagnostic status for Marker rows, not a crash.
- Current parser rows still emit successfully.
- Tests mock missing Marker and confirm no runtime pipeline behavior changes.

---

## Task 3: Install/enable Marker only inside bakeoff path

**Objective:** Make Marker testable without making it the default Stage1 runtime parser.

**Files:**
- Modify only if needed: dependency docs or local environment notes, not governance.
- Do not modify active parser selection in `clean_manifest_to_text.py` yet.

**Steps:**
1. Check whether `marker` is available in the active environment.
2. If absent, install in a controlled env or document blocked dependency status.
3. Run bakeoff for DEV15 PDFs only.
4. Record Marker availability, version, and any failures in `RUN_CONTEXT.md`.

**Verification:**
- Marker outputs are isolated under the diagnostic run directory.
- Current Stage1 assets remain unchanged.

---

## Task 4: Define structure-quality scorer

**Objective:** Convert parser output differences into objective metrics.

**Files:**
- Create: `src/stage1_cleaning/score_stage1_structure_quality_v1.py`
- Test: `tests/test_stage1_structure_quality_v1.py`

**Metrics:**
- `text_chars`, `word_count`, `body_block_count`
- `table_count`, `cell_count`, `nonempty_cell_count`
- `max_columns`, `median_columns`
- `header_cell_count`, `header_path_coverage`
- `blank_cell_count`
- `caption_bound_table_count`
- `page_locator_coverage`
- `bbox_coverage`
- `anchor_exact_hits`, `anchor_numeric_fallback_hits`, `anchor_missing_fragments`
- `geometry_warning_count`

**Output:**
- `stage1_structure_quality_by_paper_v1.tsv`
- `stage1_structure_quality_by_table_v1.tsv`
- `stage1_structure_quality_delta_v1.tsv`

**Verification:**
- Unit tests cover multi-row header, blank cell, rowspan/colspan, caption, and numeric fallback not counted as exact recovery.

---

## Task 5: Run Marker-vs-current bakeoff on DEV15 problem PDFs

**Objective:** Decide whether Marker materially improves the actual **PDF** failure set. This task excludes HTML rows from Marker comparison by design.

**Scope:**
Prioritize PDF anchors:

```text
INMUTV7L
L3H2RS2H
PA3SPZ28
5GIF3D8W
```

HTML anchors such as `BB3JUVW7`, `BXCV5XWB`, `WIVUCMYG`, and `YGA8VQKU` are evaluated only by Task 6 HTML-native extraction.

**Steps:**
1. Run bakeoff for current PyMuPDF fallback and Marker on explicit PDF source paths only.
2. Parse Marker markdown table regions into `marker_markdown_table` records and cell rows.
3. Run structure-quality scorer on PDF-only outputs.
4. Re-run clean-text anchor visibility over candidate PDF clean text outputs.
5. Re-run table-authority visibility over candidate PDF table/cell outputs.
6. Write `marker_pdf_bakeoff_decision_v1.md`.

**Decision rule:**
- Promote Marker to PDF runtime candidate only if it improves PDF table-authority or structure-quality metrics on multiple failing PDFs and does not regress PDF clean-text anchor visibility.
- If Marker improves reading order/text but not cells, keep it as PDF text/layout candidate only and continue Marker-table-adapter/Docling/pdfplumber table bakeoff later.
- Never compare Marker against HTML-native extraction; they are separate source-type paths.

---

## Task 6: Implement HTML-native table extraction as an additive extractor

**Objective:** Preserve publisher HTML table geometry directly from DOM instead of flattening to text too early.

**Files:**
- Modify: `src/stage1_cleaning/pdf2clean.py`
- Test: `tests/test_stage1_html_native_table_extraction_v1.py`

**Implementation contract:**
Add a function such as:

```python
def extract_html_native_table_cells(html_path: Path, doc_key: str) -> list[dict[str, Any]]:
    ...
```

It must:
- walk `<table>` elements in DOM order;
- extract caption text from `<caption>` or nearby table label;
- preserve `<thead>`, `<tbody>`, `<tfoot>` section type;
- preserve `rowspan` and `colspan` integers;
- emit one cell record per physical source cell;
- preserve blank cells instead of dropping them;
- mark header cells from `<th>` or section/header position;
- not infer semantic variable roles.

**Verification:**
- Unit test synthetic HTML table with rowspan/colspan/multi-row header/blank cells.
- Existing clean text projection remains stable or only additively includes better table blocks.
- Sidecar validates no forbidden semantic fields.

---

## Task 7: Add Stage1 cell-level sidecar writer

**Objective:** Emit a stable machine-readable table cell sidecar for Stage2 S2-2 to consume later.

**Files:**
- Modify: `src/stage1_cleaning/pdf2clean.py`
- Modify: `src/stage1_cleaning/clean_manifest_to_text.py`
- Test: `tests/test_stage1_cell_sidecar_v1.py`
- Output under active cleaned content: `data/cleaned/content/tables_cell_sidecar/<paper_key>/stage1_table_cells_v1.jsonl` or run-scoped diagnostic equivalent first.

**Design:**
- During diagnostic phase, write run-scoped candidate sidecars first.
- Only after review, write active cleaned sidecars under `data/cleaned/content/`.
- Preserve text files and `key2txt.tsv` unchanged unless explicit promotion is approved.
- Add `key2structure.tsv` or equivalent additive mapping row only after acceptance.

**Verification:**
- Every cell row links to `paper_key`, `table_id`, parser, source path, and source hash.
- Every table block in sidecar has matching table metadata.
- No semantic fields are emitted.

---

## Task 8: Wire Stage2 S2-2 to optionally read Stage1 cell sidecars

**Objective:** Let Stage2 table authority construction consume cell-level Stage1 sidecars without making them mandatory yet.

**Files:**
- Modify: Stage2 S2-2 table authority construction code after locating exact current function.
- Test: add/extend Stage2 authority tests.

**Contract:**
- Feature flag or explicit input path required.
- If sidecar exists, build `normalized_table_payloads_v1.json` and `table_cell_grid_v1.tsv/jsonl` from source-faithful cells.
- If sidecar is absent, preserve current behavior.
- Do not let deterministic Stage2 sidecar define candidate universe without LLM semantic authorization.

**Verification:**
- Re-run table-authority visibility diagnostics.
- Confirm selector remains ranker, not authority veto.
- Confirm prompt summary uses compact semantic surface while numeric authority remains in payload/grid.

---

## Task 9: End-to-end diagnostic replay on DEV15 subset

**Objective:** Prove Stage1 structure improvements survive through S2-2/S2-7 without requiring Stage5/S5-3 compensation.

**Steps:**
1. Generate candidate Stage1 sidecars for selected papers.
2. Run S2-2/S2-7 diagnostic replay without live LLM if frozen raw responses are valid and semantic task/prompt unchanged.
3. Run Stage3 and Stage5 no-S5-3 only on explicit candidate artifacts if boundary legality is satisfied.
4. Run diagnostic compare only against fixed GT authority with all paths printed.

**Required labels:**
- `diagnostic_only: yes`
- `benchmark_valid: no`

**Expected evidence:**
- fewer table-authority first failures;
- fewer S5-2 mechanical missing gaps for fields recoverable from row-local table cells;
- S5-3 candidate scope shrinks to true residual source-evidenced gaps.

---

## Task 10: Review and promotion gate

**Objective:** Decide whether to promote Marker, HTML-native extraction, and Stage1 sidecar into maintained runtime.

**Review checklist:**
- Tests pass.
- No paper-specific branches.
- No semantic inference in Stage1.
- No active-run pointer changes.
- Sidecar schema documented in existing docs, not new governance docs.
- Maintained script registry updated only if a new runtime entrypoint is approved.
- Parser choice backed by bakeoff metrics, not intuition.

**Promotion options:**
1. Promote HTML-native table cells only.
2. Promote Marker as PDF parser candidate/fallback.
3. Promote Stage1 cell sidecar as additive output, while leaving current text projection unchanged.
4. Defer Marker and test Docling/pdfplumber if Marker does not improve table geometry.

---

## Progress Ledger

Use:

```text
docs/plans/2026-05-06-stage1-parser-table-sidecar-progress.tsv
```

Columns:

```text
task_id	status	artifact_path	test_status	decision	notes	updated_at
```

Execution rule: execute exactly the next pending task, update the ledger, run targeted tests, and stop if blocked after one focused fix attempt.
