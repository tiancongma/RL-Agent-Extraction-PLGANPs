# Marker Table Authority B-Phase Repair Plan

> **For Hermes:** Execute task-by-task with TDD. Keep changes generic, additive, diagnostic-only, and governance-compatible.

**Goal:** Convert Marker's PDF structured table/cell advantage into downstream S2-2 table authority and S5-2 materialization readiness without replacing the legacy clean-text path or changing Stage2 semantic authority.

**Architecture:** Stage1 preserves parser/table/cell provenance and repairs only parser-structure defects that are source-observable. Stage2 consumes Marker-derived structure/cell authority as optional execution metadata and table payload input, while LLM semantic discovery remains the formulation/candidate authority. Stage5/S5-2 should receive better deterministic row-local table payloads so S5-3 remains residual source-evidenced gap filling.

**Scope:** B-phase improvement layer only. This is not Marker full replacement/promotion and not benchmark-valid reporting.

---

## Governance constraints

- Do not modify `project/` governance files.
- Do not update `data/results/ACTIVE_RUN.json`.
- Do not call live LLMs without explicit batch-level approval.
- Do not compare Marker PDF against HTML-native extraction.
- Do not use Marker/sidecar blocks to define Stage2 semantic candidate universe.
- Do not hard-code paper keys, snippets, field maps, or excerpt-specific repairs.
- Preserve `text_path`/flatten text as backward-compatible surface.
- Treat generated replay/diagnostic outputs as diagnostic-only unless a full governed pipeline and compare is explicitly run.

## Acceptance criteria

B-phase is complete when all five contracts below are implemented with tests and diagnostic replay evidence:

1. **Caption/table identity binding**
   - Marker markdown tables get stable captions from nearby source-observable caption lines.
   - Captions are propagated into table records and cell records.
   - Binding is conservative and records match rule/warnings.

2. **Continuation / multi-panel stitching metadata**
   - Consecutive Marker markdown tables that are source-contiguous and share compatible headers/caption context get grouped with `continuation_group_id` metadata.
   - Physical table fragments remain separately addressable; stitching metadata is additive.

3. **Noise table classification**
   - Publisher/article metadata and submission/chrome tables are tagged as `confirmed_noise` or `suppressible_noise` only with explicit source-observable cues.
   - Ambiguous tables stay preserved.

4. **Symbol normalization + provenance**
   - Cell sidecar exposes `normalized_cell_text` suitable for downstream matching while preserving `raw_cell_text` exactly.
   - Normalization handles common scientific symbols: `±`, `μ`, `−`, `ζ`, superscripts/subscripts where source-observable.
   - Warnings/provenance record transformations.

5. **S2-2 table authority bridge**
   - Stage2 normalized table payloads can consume selected Stage1/Marker cell sidecar tables as execution-grade table payloads.
   - Prompt summaries remain compact; numeric/table authority stays in full payload/grid artifacts.
   - Selector cannot irreversibly drop preserved non-noise tables.

---

## Task sequence

### Task B01: Baseline code and artifact inspection

**Objective:** Identify current Marker table/cell sidecar functions, existing tests, and artifact fields.

**Files:**
- Inspect: `src/stage1_cleaning/pdf2clean.py`
- Inspect: `src/stage1_cleaning/run_stage1_parser_bakeoff_v1.py`
- Inspect: `tests/test_stage1_parser_bakeoff_v1.py`
- Inspect: `tests/test_stage2_stage1_sidecar_consumption_v1.py`

**Verification:** No code change. Record findings in progress TSV.

### Task B02: Caption binding TDD

**Objective:** Add source-observable caption binding for Marker markdown tables.

**Test first:** Add a test where a caption line immediately precedes a markdown table and assert table/cells carry caption plus `caption_binding_rule`.

**Implementation target:** `extract_marker_markdown_tables` / nearby helpers in `src/stage1_cleaning/pdf2clean.py`.

**Commands:**
```bash
PYTHONPATH=. python3 -m unittest tests.test_stage1_parser_bakeoff_v1
python3 -m py_compile src/stage1_cleaning/pdf2clean.py
```

### Task B03: Continuation group metadata TDD

**Objective:** Add conservative continuation metadata for adjacent table fragments with compatible headers/caption context.

**Test first:** Two adjacent markdown table fragments with repeated/compatible headers should retain separate table IDs but share a continuation group.

**Implementation target:** table extraction helper only; no Stage2 semantic changes.

### Task B04: Noise table tagging TDD

**Objective:** Tag obvious publisher/article chrome tables without deleting ambiguous scientific tables.

**Test first:** A publisher metadata markdown table is tagged noise; a sparse scientific table remains preserved.

**Implementation target:** Stage1 table sidecar metadata/warnings.

### Task B05: Symbol normalization/provenance TDD

**Objective:** Normalize common scientific symbols while preserving raw cell text.

**Test first:** Cells containing `12 ± 3 μm`, `−25 mV`, `ζ-potential` keep raw text and expose normalized text/provenance warnings.

**Implementation target:** cell extraction/normalization helpers in `pdf2clean.py`.

### Task B06: S2-2 table authority bridge audit and minimal wiring

**Objective:** Confirm Stage2 consumes Stage1 table-cell sidecar rows only for already selected/preserved table candidates and writes execution-grade payload/grid surfaces.

**Test first:** Existing selected candidate consumes Marker/Stage1 sidecar cells; unselected/noise sidecar table does not create semantic candidates.

**Implementation target:** `extract_semantic_stage2_objects_v2.py` only if current bridge is insufficient.

### Task B07: Bounded PDF-only diagnostic replay

**Objective:** Re-run PDF-only DEV anchors through Stage1/Stage2 stop-before-live-call surfaces to verify table/cell authority improvements.

**Constraints:** No live LLM. No ACTIVE_RUN update. Diagnostic-only.

### Task B08: Review and promotion gate recommendation

**Objective:** Summarize whether Marker is ready for broader PDF primary-structured-surface use, and what remains before full replacement.

**Output:** A concise review under `docs/plans/` or append progress TSV notes; no governance promotion.
