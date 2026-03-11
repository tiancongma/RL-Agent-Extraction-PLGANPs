# Pipeline Script Map

This document maps **pipeline stages to executable scripts**, clarifying
their purpose, inputs, outputs, and status.

Status legend:
- ACTIVE 鈥?part of current main pipeline
- SUPPORTING 鈥?optional utility for targeted manual review, benchmark maintenance, or conflict arbitration; not part of the default mainline path
- LEGACY 鈥?preserved for reference only

---

## Stage 0 鈥?Relevance Filtering

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

## Stage 1 鈥?Cleaning and Manifest

| Script | Purpose | Status |
|------|--------|--------|
| csv2clean_manifest.py | Generate manifest and trigger cleaning | ACTIVE |
| pdf2clean.py | PDF text cleaner | ACTIVE |
| html_parser.py | Shared HTML parsing utilities | ACTIVE |

---

## Stage 2 — Semantic Extraction and Sampling (LLM)

| Script | Purpose | Status |
|------|--------|--------|
| sample_from_manifest_html_first.py | Sample selection and reproducible subset definition | ACTIVE |
| sample10_from_zotero_manifest.py | Historical sample10 generation utility | ACTIVE |
| build_key2txt_from_sample_manifest.py | Build sample-local key2txt index | ACTIVE |
| auto_extract_weak_labels.py | LLM extraction entry for semantic candidates | ACTIVE |
| auto_extract_weak_labels_v6.py | Mainline semantic extraction baseline | ACTIVE |
| auto_extract_weak_labels_v4.py | Older extraction baseline retained for comparison | ACTIVE |
| auto_extract_weak_labels_v3.py | Weak label logic v3 | LEGACY |

---

## Stage 3 — Formulation Hypothesis and Inheritance Resolution

| Script | Purpose | Status |
|------|--------|--------|
| build_evidence_bundle_for_keys_v1.py | Build deterministic evidence packages from cleaned artifacts | ACTIVE |
| apply_formulation_grouping_v1.py | Group semantic candidates into formulation hypotheses | ACTIVE |
| apply_global_baseline_inheritance_and_rerun_alignment_v1.py | Resolve shared/inherited conditions during hypothesis consolidation | ACTIVE |

---

## Stage 4 — Formulation Assembly and Formulation-level Audit

| Script | Purpose | Status |
|------|--------|--------|
| compute_formulation_alignment_v1.py | Deterministic formulation-level alignment and assembly checks | ACTIVE |
| run_alignment_v3_surfactant_drugnorm.py | Deterministic normalization/alignment pass for assembly stability | ACTIVE |
| build_boundary_alignment_diagnostics_pack_v1.py | Boundary and grouping diagnostics for formulation-level audit | ACTIVE |
| export_dev15_formulation_view_xlsx_v1.py | Human-auditable formulation-level view export | ACTIVE |
| export_evidence_bundle_audit_xlsx_v1.py | Evidence-bundle audit export for targeted review | ACTIVE |

### Planned Stage4 Reconciliation Note

- The next DoE checkpoint / validation reconciliation rule should be implemented first in `src/stage4_eval/eval_weak_labels_v7pilot3.py` because that script currently converts predicted instance rows directly into benchmark paper-level formulation counts.
- Minimum matching key for this rule: factor-level coordinate signature (prefer decoded factor values; fallback to coded factor levels plus other coordinate-defining synthesis variables), not predicted-vs-observed measurement values.
- Stage5 schema/core builders should later mirror the same logic so `formulation_core` outputs do not drift from Stage4 benchmark reconciliation on DoE papers.

### Supporting GT / Annotation Utilities

| Script | Purpose | Status |
|------|--------|--------|
| build_gt_template_from_conflict_queue.py | Build templates for selected conflict cases requiring targeted manual review | SUPPORTING |
| export_gt_annotation_view.py | Export annotation views for targeted manual review and conflict arbitration | SUPPORTING |
| merge_gt_from_annotation_view.py | Merge reviewed annotations for GT maintenance and benchmark support | SUPPORTING |
| gt_summary_report.py | Summarize review decisions for benchmark maintenance and audit tracking | SUPPORTING |
| gt_tool.py | Legacy manual GT annotation tool | LEGACY |
| gt_tool_v3.py | Older GT tool | LEGACY |

These scripts support targeted manual review and benchmark maintenance, but they are not part of the default primary formulation-reconstruction path.

---

## Stage 5 — Final Tabular Export

| Script | Purpose | Status |
|------|--------|--------|
| merge_results.py | Merge verified formulation records into final tabular outputs | ACTIVE |

---

## Maintenance Rules

- Only scripts marked ACTIVE may be used in the main pipeline.
- LEGACY scripts must not be modified.
- Any status change must be logged in `project/4_DECISIONS_LOG.md`.

---

## Primary Path vs Historical/Baseline Path
Stage directory names are retained for implementation stability; the current architectural interpretation is defined by this script map rather than by directory names alone.


- Current primary path is formulation reconstruction:
  document preprocessing -> semantic extraction (LLM) -> formulation hypothesis -> evidence binding -> formulation assembly -> formulation-level audit -> final tabular export.
- GT/conflict annotation utilities are supporting tools for selected review scenarios, not architectural centerpieces of the default mainline path.
- Multi-model extraction/consensus scripts remain in the repository as historical baselines or supporting utilities for selective verification and diagnostics.
- Multi-model consensus is not the architectural center of the current mainline pipeline.

---

## Current DEV-15 Formulation-Instance Execution Clarification

- For the current DEV-15 formulation-instance path, treat `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` as the Stage2 default extractor.
- Treat `src/stage4_eval/eval_weak_labels_v7pilot3.py` as the current Stage4 DEV evaluator and count-reconciliation seam.
- Treat `src/stage4_eval/build_dev15_review_workbook_v1.py` as a supporting reviewer-facing export, not as the evaluator itself.
- The validated DoE coordinate reconciliation now lives in `src/stage4_eval/eval_weak_labels_v7pilot3.py`.
- `src/stage4_eval/test_doe_coordinate_reconciliation_v1.py` remains an experimental validation script, not the default entrypoint.
- The current full DEV-15 reconciled combined count view is the checked-in artifact `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled.tsv`.
- There is not yet a dedicated checked-in canonical builder script for the full combined DEV-15 TSV; agents should not guess one from filename similarity.

