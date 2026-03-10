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

Added stratified sample20 (nano/micro 脳 O/W/W/O/W 脳 table/text) for arXiv methodology validation.
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
`gt_decision 鈭?{accept_model1, accept_model2, override, unclear}`  
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

## 2026-03-06

### Decision: Clarify LLM semantic responsibilities, deterministic arbitration responsibilities, and audit boundary

Decision
- LLM extraction is responsible for semantic structure: instance boundaries, field-role assignment, and shared-vs-instance-specific interpretation.
- Deterministic layers own numeric evidence binding, derivation, schema assembly/export, and QC gating.
- Semantic repair rules in downstream stages must not grow indefinitely; they are tracked as candidates for future upstream schema redesign.
- The PLGA-only database standard remains a database-layer filter and release contract, not an LLM-only decision.

Reason
- The pipeline responsibility audit shows that several downstream semantic-repair rule families are compensating for missing upstream structure.
- Keeping semantic interpretation and deterministic arbitration distinct improves reproducibility, debuggability, and long-term maintainability.

Impact
- Immediate implementation remains layered: LLM extraction -> deterministic arbitration -> audit/release.
- Downstream deterministic rule families remain active for release stability.
- Semantic-repair heavy areas are now explicitly treated as redesign backlog for extraction schema evolution.

### Decision: Set weak_labels_v7 as next target schema architecture step (not implemented)

Decision
- weak_labels_v7 is adopted as the next extraction schema target to strengthen LLM-side semantic structure and reduce downstream semantic repair.
- This decision defines architecture direction only; runtime extraction scripts remain unchanged at this time.

Reason
- Current v6 schema lacks explicit semantic typing for scope, field membership confidence, and evidence region type.
- Downstream semantic repair growth should be replaced by stronger upstream schema contracts where appropriate.

Impact
- Future implementation work should prioritize v7-compatible extraction outputs and staged downstream adoption.
- Deterministic arbitration, derivation, export, and QC responsibilities remain unchanged.

## 2026-03-08

### Decision: Transition from field-first extraction to formulation-level extraction assembly

Decision
- Move formulation grouping earlier into the LLM stage as part of semantic extraction.
- Let the LLM emit formulation hypotheses (instance-level candidate records) rather than extracting isolated fields first and assembling instances only in late deterministic grouping.
- Keep deterministic stages focused on evidence binding, normalization, verification, and export.

Problem discovered
- In multi-formulation papers, field-first extraction followed by late grouping produces recurrent instance-boundary errors.
- Shared procedural descriptions and cross-sentence references cause wrong field-to-instance assignment when grouping is deferred.

Newly discovered issue
- Inheritance-style reporting such as "F2 was prepared similarly to F1 except ..." cannot be handled reliably by purely rule-based late grouping.
- Correct interpretation requires upstream semantic resolution of what is inherited vs what is overridden at the formulation level.

Impact
- The pipeline now explicitly models formulation assembly via a formulation hypothesis layer before deterministic verification.
- Rule-based logic remains deterministic and auditable, but no longer carries primary responsibility for semantic instance reconstruction.
- Final release artifacts remain tabular (one row per formulation), with richer intermediate structures retained for traceability and audit.

### Decision: Retain stage directory names and align architecture via documentation (no directory renaming or code relocation)

Decision
- Stage directory names are retained for implementation stability.
- Current architecture interpretation is maintained through documentation in project_specification.txt, project/2_ARCHITECTURE.md, and project/PIPELINE_SCRIPT_MAP.md.
- No script relocation is performed at this stage because no move is clearly justified as both semantically necessary and low-risk across imports, CLI paths, launch profiles, and docs.

Reason
- Directory/path stability remains a hard reproducibility constraint.
- Several utilities in src/stage4_eval/ and src/stage5_benchmark/ span audit/benchmark support boundaries; moving them now would create avoidable path churn with limited architectural benefit.

Impact
- Semantic stage alignment is enforced through script-map interpretation rather than folder renaming.
- Existing stage folders and code locations remain unchanged in this decision.

## 2026-03-10

### Decision: Compress formulation-instance routing enums for pilot extraction and preserve formulation-centric routing

Decision
- The primary formulation-instance enum set is fixed to:
  - `new_formulation`
  - `variant_formulation`
  - `candidate_non_formulation`
  - `unclear`
- The primary change-role enum set is fixed to:
  - `synthesis_defining`
  - `non_synthesis`
  - `unclear`
- Older larger routing enums such as `doe_run`, `parameter_sweep_variant`, `post_processing_variant`, `test_condition_variant`, `measurement_only`, `post_processing_change`, `test_condition_change`, and `measurement_context_change` are retired as primary routing values.
- Optional auxiliary tags remain allowed through `instance_context_tags` and `change_context_tags` (for example `doe`, `sweep`, `post_processing`, `test_condition`, `measurement_context`, `optimized`, `control`), but these tags must not replace the primary enum sets.

Reason
- The formulation-instance layer needs a minimal operational ontology that keeps formulation identity decisions upstream in the extraction layer while avoiding taxonomy sprawl.
- Post-processing/test/storage/measurement differences must be suppressible without reintroducing a scattered-field-first grouping architecture.

Impact
- Pilot extraction outputs now carry formulation-centric instance metadata with compressed enums, parent links, change descriptions, and evidence refs.
- Distinct formulation rows continue to be defined by synthesis/design changes, including changes outside the initial core field list when the paper makes them identity-defining.
- Controlled pilot comparison is frozen to the previously reused 3-paper DEV15 subset:
  - `5ZXYABSU` / `10.2147/ijn.s130908`
  - `L3H2RS2H` / `10.1016/j.ejpb.2004.09.002`
  - `WIVUCMYG` / `10.1002/jps.24101`
