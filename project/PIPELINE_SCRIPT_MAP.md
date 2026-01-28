# Pipeline Script Map

This document maps **pipeline stages to executable scripts**, clarifying
their purpose, inputs, outputs, and status.

Status legend:
- ACTIVE — part of current main pipeline
- LEGACY — preserved for reference only

---

## Stage 0 — Relevance Filtering

| Script | Purpose | Status |
|------|--------|--------|
| prefilter_regex.py | Regex-based metadata pre-filtering | ACTIVE |
| classify_gemini_grouped.py | LLM relevance classification | ACTIVE |
| auto_tag_plga_gemini.py | Auto-tagging using Gemini | ACTIVE |
| auto_tag_plga_openai.py | Auto-tagging using OpenAI | ACTIVE |
| zotero_tag_sync.py | Sync tags to Zotero | ACTIVE |
| zotero_llm_relevant_interactive.py | Interactive relevance review | ACTIVE |
| zotero_fetch_llm_relevant_pdfs.py | Fetch PDFs/HTML | ACTIVE |
| fill_missing_snapshots.py | Fill missing HTML snapshots | ACTIVE |

---

## Stage 1 — Cleaning and Manifest

| Script | Purpose | Status |
|------|--------|--------|
| csv2clean_manifest.py | Generate manifest and trigger cleaning | ACTIVE |
| pdf2clean.py | PDF text cleaner | ACTIVE |
| html_parser.py | Shared HTML parsing utilities | ACTIVE |

---

## Stage 2 — Sampling and Weak Labels

| Script | Purpose | Status |
|------|--------|--------|
| sample_from_manifest_html_first.py | Sample selection | ACTIVE |
| sample10_from_zotero_manifest.py | Sample10 generation | ACTIVE |
| build_key2txt_from_sample_manifest.py | Build key2txt index | ACTIVE |
| auto_extract_weak_labels.py | Weak label extraction (entry) | ACTIVE |
| auto_extract_weak_labels_v4.py | Weak label logic v4 | ACTIVE |
| auto_extract_weak_labels_v3.py | Weak label logic v3 | LEGACY |

---

## Stage 3 — Manual Ground Truth

| Script | Purpose | Status |
|------|--------|--------|
| gt_tool.py | Manual GT annotation tool | ACTIVE |
| gt_tool_v3.py | Older GT tool | LEGACY |

---

## Stage 4 — Multi-model Extraction and QC

| Script | Purpose | Status |
|------|--------|--------|
| auto_extract_multimodel.py | Multi-model extraction entry | ACTIVE |
| multi_model_extract_tier1.py | Tier-1 extraction | ACTIVE |
| multi_model_extract_tier2.py | Tier-2 extraction | ACTIVE |
| multi_model_merge_qc.py | Merge and QC | ACTIVE |

---

## Stage 5 — Merge and Publish

| Script | Purpose | Status |
|------|--------|--------|
| merge_results.py | Merge final outputs | ACTIVE |

---

## Maintenance Rules

- Only scripts marked ACTIVE may be used in the main pipeline.
- LEGACY scripts must not be modified.
- Any status change must be logged in `project/4_DECISIONS_LOG.md`.
