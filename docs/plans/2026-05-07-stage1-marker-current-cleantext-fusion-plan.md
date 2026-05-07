# Stage1 Marker/Current Clean Text Fusion Implementation Plan

> **For Hermes:** Use subagent-driven-development skill only for later multi-agent execution; this initial pass is a bounded in-chat implementation/audit and must keep writes serial.

**Goal:** Add a provenance-preserving clean-text structure surface that lets Stage1 expose Marker headings/sections/noise tags while preserving the existing flatten text path consumed by Stage2.

**Architecture:** Stage1 remains the only parser/clean-text owner. The current `text/*.txt` output and `key2txt.tsv` stay backward compatible. Additive structure is stored in Stage1 sidecar JSON under `data/cleaned/content/structure/` through existing `key2structure.tsv`; downstream Stage2 can later consume the section model as optional metadata for evidence ordering/denoising, not as a semantic authority or irreversible source filter.

**Tech Stack:** Python stdlib, existing `src/stage1_cleaning/pdf2clean.py`, existing Stage2 S2-2 candidate/evidence construction in `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`, `unittest` tests.

---

## Governance constraints

- Mandatory context read before edits: `AGENTS.md`, `project/0_PROJECT_CHARTER.md`, `project/1_REQUIREMENTS.md`, `project/2_ARCHITECTURE.md`, `project/PIPELINE_SCRIPT_MAP.md`, `project/ACTIVE_PIPELINE_FLOW.md`, `project/ACTIVE_PIPELINE_RUNBOOK.md`, `project/FILE_NAMING_AND_VERSIONING.md`.
- Do not create files in `project/`; this plan lives under `docs/plans/`.
- Do not modify `data/results/ACTIVE_RUN.json`.
- Do not make benchmark claims from diagnostic clean-text or parser bakeoff outputs.
- Do not use live LLM calls for this work.
- Do not connect Weixin/WeChat.
- Stage1 may preserve structure and parser provenance; it must not infer PLGA formulation semantics or define the Stage2 candidate universe.
- The selector can use section/noise tags only as ranking/denoising metadata. It must not delete non-noise tables or source evidence irreversibly.

## Current inspected state

- Stage1 active entrypoint: `src/stage1_cleaning/clean_manifest_to_text.py` delegates parsing to `src/stage1_cleaning/pdf2clean.py` and promotes `key2txt.tsv` plus additive `key2structure.tsv`.
- `pdf2clean.py` already writes sidecar JSON with `blocks`, `tables`, `metadata`, and `reading_order_source`.
- Marker PDF adapter currently renders chunks as generic paragraphs; it does not preserve markdown headings as heading blocks.
- Existing sidecar blocks lack stable `section_id`, heading hierarchy, `section_label`, `section_kind`, or `noise_class`.
- Stage2 S2-2 currently reads only `text_path.read_text(...)` in `build_candidate_segmentation_artifact()` and `build_evidence_blocks_artifact()`, then re-infers section labels from flattened paragraphs.
- Stage2 already contains section/noise heuristics (`extract_section_label`, `infer_section_kind`, `should_drop_segment`, candidate `noise_flags`), but these are downstream heuristics over flattened text rather than parser-provenance-preserving Stage1 structure.

## Clean text contract to implement

Stage1 sidecar JSON will remain additive and include:

```json
{
  "doc_key": "...",
  "source_type": "PDF|HTML",
  "txt_path": "data/cleaned/content/text/KEY.pdf.txt",
  "txt_hash": "sha1:...",
  "reading_order_source": "marker_native|fallback_linear|...",
  "blocks": [
    {
      "block_id": "b0001",
      "type": "heading|paragraph|table|caption|list",
      "order": 1,
      "text": "...",
      "section_id": "sec0001",
      "section_label": "2. Materials and methods",
      "section_level": 2,
      "section_kind": "methods|results|references|front_matter|...",
      "section_path_json": "[...]",
      "noise_class": "keep|soft_noise|suppressible_noise|terminal_noise",
      "noise_reason": []
    }
  ],
  "sections": [
    {
      "section_id": "sec0001",
      "section_label": "2. Materials and methods",
      "section_level": 2,
      "section_kind": "methods",
      "start_block_id": "b0003",
      "end_block_id": "b0014",
      "block_count": 12,
      "noise_class": "keep",
      "noise_reason": []
    }
  ],
  "metadata": {
    "parser": "marker|pymupdf_fallback|trafilatura_native|beautifulsoup_fallback",
    "section_model_version": "stage1_section_model_v1"
  }
}
```

## Flow contract

1. Stage1 parser output keeps two surfaces:
   - `text/*.txt`: backward-compatible flatten text for existing Stage2.
   - `structure/*.json`: additive structured clean text, section/noise model, parser provenance, tables/cells.
2. Marker PDF path should preserve markdown headings as `heading` blocks before block finalization.
3. Current PyMuPDF fallback remains legal and gets best-effort section tags from its paragraph stream.
4. HTML path keeps existing trafilatura/BS4 behavior and receives the same section tagging after finalization.
5. Stage2 wiring is a later task: read optional sidecar via `key2structure.tsv`/path convention and use section tags as metadata for evidence ordering and suppressible-noise handling. It must preserve table authority and cannot become deterministic semantic discovery.

## Task list

### Task 1: Add failing Stage1 section-model tests

**Objective:** Prove the new Stage1 section model preserves headings, carries section labels to paragraphs, and tags references as suppressible/terminal noise without deleting text.

**Files:**
- Modify: `tests/test_stage1_parser_bakeoff_v1.py`
- Modify: `src/stage1_cleaning/pdf2clean.py`

**Steps:**
1. Add unit tests for `extract_marker_markdown_blocks()` and `annotate_blocks_with_sections()`.
2. Verify tests fail before implementation.
3. Implement minimal generic heading extraction and section annotation.
4. Run targeted tests.

### Task 2: Implement Stage1 section model

**Objective:** Add parser-independent section annotation to sidecar blocks and `sections[]`.

**Files:**
- Modify: `src/stage1_cleaning/pdf2clean.py`

**Steps:**
1. Add heading detection helpers for Markdown headings and short numbered headings.
2. Add `infer_stage1_section_kind()` and `classify_stage1_noise()` with generic non-semantic labels.
3. Add `annotate_blocks_with_sections()` returning `(blocks, sections)`.
4. Attach annotations inside `build_sidecar_payload()`.
5. Ensure `validate_sidecar_payload()` rejects malformed section references but allows absent old sidecars.

### Task 3: Preserve Marker headings

**Objective:** Ensure Marker PDF output carries headings instead of flattening every chunk to paragraph.

**Files:**
- Modify: `src/stage1_cleaning/pdf2clean.py`

**Steps:**
1. Add `extract_marker_markdown_blocks(rendered_text)`.
2. Use it in `extract_marker_pdf_blocks()`.
3. Keep existing Marker table markdown extraction unchanged.
4. Preserve all text in projected clean text; no deletion.

### Task 4: Stage2 consumption design checkpoint

**Objective:** Document but do not yet force Stage2 use of the sidecar.

**Files:**
- Append/update: this plan and progress ledger only unless user approves Stage2 wiring.

**Steps:**
1. Record exact downstream entry points to update later.
2. State that Stage2 will remain compatible with `key2txt.tsv` and missing sidecars.
3. Record required future tests: optional sidecar load, section metadata copied into candidate blocks, terminal-noise sections suppressed only when no table/evidence authority is lost.

### Task 5: Validation

**Objective:** Verify generic tests and no syntax errors.

**Commands:**
```bash
PYTHONPATH=. python3 -m unittest tests.test_stage1_parser_bakeoff_v1
python3 -m py_compile src/stage1_cleaning/pdf2clean.py
```

Expected: tests pass; compile exits 0.

## Stop conditions

- If section tags require paper-specific text, stop and record blocker.
- If Stage2 needs live LLM or benchmark comparison, stop; that is outside this clean-text step.
- If a test failure suggests broad table-authority behavior changed, stop and inspect before further edits.

## Promotion criteria

This step is complete when:

- Stage1 sidecar JSON emits `sections[]` and per-block section/noise metadata.
- Marker headings are preserved as heading blocks when rendered markdown has headings.
- Existing clean text output remains backward compatible.
- Targeted Stage1 tests pass.
- Progress ledger is updated with tested status.
