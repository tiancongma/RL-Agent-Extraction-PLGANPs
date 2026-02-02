## 2026-01-28

Decision: Promote `manifest_html10.tsv` as `manifest_current.tsv`  
Reason: Stable HTML-first manifest used for sample10 baseline  
Alternatives: `manifest_html10_bad.tsv` archived due to known path issues  
Impact: Downstream sampling and extraction now depend on this manifest

## 2026-01-30

Decision: Freeze the repository directory structure as a stable interface (no renames or relocations)
Reason: Prevent recurring breakage from hard-coded paths and reduce refactor churn; improve long-term reproducibility and maintainability
Scope: Top-level directories src/, data/, runs/, project/ and root files README.md, requirements.txt, .gitignore are frozen in name and location
Allowed: Add new files/subfolders within these directories; add new stages under src/ (e.g., stage6_*); add new top-level directories only if existing ones are not moved/renamed
Disallowed: Renaming or relocating any frozen directory; moving outputs to alternative roots (e.g., data/results replacing runs/); restructuring that invalidates existing paths
Impact: From this date forward, all code and documentation must assume these paths are stable; future changes should be additive rather than structural

## 2026-01-31

Added stratified sample20 (nano/micro × O/W/W/O/W × table/text) for arXiv methodology validation.
Sampling treated as data-prep step, not run-scoped.
Finalized stratified20 sampling using rule-based strata_tags.tsv with soft HTML preference (html-bias=0.7).
Resulting sample: 20 papers (15 HTML, 5 PDF).
Missing (*,*,text) strata reflect reporting-style distribution, not pipeline error.

Input length cap during v5 weak-label extraction

In the v5 version of auto_extract_weak_labels, the input text passed to LLMs is intentionally capped at 60,000 characters via a --max-chars parameter. This cap is applied before the LLM call, during section-based text assembly or full-text fallback, and is not a model-imposed context limit.

The purpose of this cap is to stabilize cost, latency, and experimental conditions during early-stage validation of the evidence-aware extraction pipeline. As a result, all evidence spans are guaranteed to lie within the retained text window. Information appearing beyond this limit is intentionally excluded at this stage and may be addressed in future iterations through section-aware budgeting or table-first extraction strategies.

This decision is treated as an explicit experimental parameter rather than a limitation of the underlying LLMs.

Due to Gemini 2.5 Free Tier RPD limits, we switch to Gemini 3 Flash and Gemma 3 12B as the primary dual-model setup for batch weak-label extraction.

## 2026-02-01

Created run_20260201_0927_bb13267_sample20 as the first quota-aware, single-model split execution to validate engineering stability and merge/QC compatibility.
This run is not considered a final extraction run for publication.

## 2026-02-02 
- Ground Truth Annotation Workflow

**Decision**  
Manual GT annotation shall not directly edit authoritative TSV files.  
Instead, a two-step workflow is adopted:

1. Export a read-only TSV into an annotation-friendly Excel view.
2. Merge human annotations back into a new authoritative GT TSV via script.

**Rationale**  
Authoritative TSV files may contain multiline fields and quoted evidence text, making direct editing in Excel or IDE CSV editors unsafe. Separating human annotation (Excel UI) from machine-written TSV outputs ensures row integrity, reproducibility, and auditability.

**GT Decision Schema**  
`gt_decision ∈ {accept_model1, accept_model2, override, unclear}`  
- `gt_value_text` must be provided iff `gt_decision = override`.

**Scope**  
This decision affects only Stage 3 (Manual Annotation) and does not modify upstream extraction or downstream evaluation logic.

- Handling of `note` Field in GT

**Decision**  
The `note` field is not treated as a structured, extractable attribute in the current phase.

**Rationale**  
`note` content is highly free-form, lacks a stable textual unit for extraction,
and does not support consistent ground truth adjudication.

**Policy**  
- All `note` entries are labeled as `unclear` during GT annotation.
- The `note` field is excluded from quantitative evaluation and statistics.

**Scope**  
This decision applies to GT and evaluation only and does not affect other structured fields.
