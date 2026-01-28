# Source Code Overview

This directory contains all executable scripts for the PLGA LLM-extraction pipeline.
Scripts are organized by **pipeline stage**, not by implementation detail.

If you are unsure which script to run, start from the **main entry scripts**
listed under each stage.

---

## Stage 0 — Relevance Filtering (Zotero / Metadata)

Location: `src/stage0_relevance/`

Purpose:
Identify candidate papers relevant to PLGA nanoparticle formulation.

Main entry scripts:
- `prefilter_regex.py` — Regex-based pre-filtering on metadata
- `classify_gemini_grouped.py` — LLM-based relevance classification
- `zotero_fetch_llm_relevant_pdfs.py` — Fetch PDFs/HTML for relevant papers

Supporting scripts:
- `auto_tag_plga_gemini.py`
- `auto_tag_plga_openai.py`
- `zotero_tag_sync.py`
- `zotero_llm_relevant_interactive.py`
- `fill_missing_snapshots.py`

---

## Stage 1 — Cleaning and Manifest Generation

Location: `src/stage1_cleaning/`

Purpose:
Convert HTML/PDF documents into cleaned, structured text and generate manifests.

Main entry scripts:
- `csv2clean_manifest.py` — Generate manifest and trigger cleaning
- `pdf2clean.py` — PDF fallback cleaner

Supporting scripts:
- `html_parser.py` — Shared HTML parsing utilities

---

## Stage 2 — Sampling and Weak Label Extraction

Location: `src/stage2_sampling_labels/`

Purpose:
Define experimental samples and generate weak labels using LLMs.

Main entry scripts:
- `sample_from_manifest_html_first.py` — Sample selection
- `build_key2txt_from_sample_manifest.py` — Build key2txt index
- `auto_extract_weak_labels.py` — Main weak label extraction entry (current)

Versioned / alternative scripts:
- `auto_extract_weak_labels_v3.py` (legacy)
- `auto_extract_weak_labels_v4.py` (current generation logic)

---

## Stage 3 — Manual Ground Truth Annotation

Location: `src/stage3_gt/`

Purpose:
Create partial human-annotated ground truth labels.

Main entry script:
- `gt_tool.py`

Alternative versions:
- `gt_tool_v3.py` (legacy)

---

## Stage 4 — Multi-model Extraction and QC

Location: `src/stage4_eval/`

Purpose:
Run multi-model extraction and merge/QC results.

Main entry script:
- `auto_extract_multimodel.py`

Supporting scripts:
- `multi_model_extract_tier1.py`
- `multi_model_extract_tier2.py`
- `multi_model_merge_qc.py`

---

## Stage 5 — Merge and Publication

Location: `src/stage5_merge_publish/`

Purpose:
Merge outputs and prepare publishable datasets.

Main entry script:
- `merge_results.py`

---

## Legacy Scripts

Location: `src/legacy/`

Purpose:
Preserve historical scripts that are no longer part of the active pipeline.
These scripts should not be used unless explicitly referenced in documentation.

---

## Notes

- Script versioning is handled by git commits and run_id, not file names.
- If a script’s role is unclear, consult:
  - `project/PIPELINE_SCRIPT_MAP.md`
  - `project/4_DECISIONS_LOG.md`
