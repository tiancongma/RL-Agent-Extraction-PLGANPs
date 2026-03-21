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

### Decision: Reconcile DoE checkpoint / validation rows by factor-level coordinate signature, not table position

Decision
- For DoE-style papers, formulation-core identity must be determined primarily by factor-level coordinate signature, not by whether a row appears in a later checkpoint or validation table.
- If a checkpoint / validation row matches an existing design-matrix coordinate, it must not create a new formulation core.
- If a checkpoint / validation row introduces a new coordinate outside the original design matrix, it must create a new formulation core.
- Predicted vs observed values belong to measurement-level representation and must not create separate formulation rows by themselves.

Reason
- Later checkpoint / validation tables can repeat existing experimental coordinates while adding new measured outcomes, which causes false formulation over-counting if row identity is inferred from table occurrence alone.
- The correct identity boundary in DoE papers is the coordinate-defining factor combination, not the reporting location or whether the row is tagged as predicted / observed / validation.

Consequence
- Stage4 benchmark reconciliation must collapse repeated checkpoint rows onto an existing formulation core when the factor-level coordinate signature matches.
- Stage5 core/schema projection should mirror the same coordinate-aware rule so benchmark-facing core counts and database-facing core counts stay consistent.
- Predicted vs observed values should remain attached to measurement-level outputs, not split formulation-core outputs.

Case reference
- `WFDTQ4VX` / `10.1080/10717544.2016.1199605`

Resolved interpretation
- Correct formulation-core count = `29`.

Implementation note
- On `2026-03-10`, the validated `WFDTQ4VX` coordinate-signature merge was integrated into `src/stage4_eval/eval_weak_labels_v7pilot3.py` for Stage4 DEV counting.
- The Stage4 summary now preserves both the raw predicted formulation row count and the reconciled formulation-core count for auditability.

## 2026-03-18

### Decision: Add deterministic `preparation_method` and `emulsion_structure` as schema-only enrichment fields

Decision
- Add two descriptive fields, `preparation_method` and `emulsion_structure`, as deterministic enrichment outputs derived only from existing structured values and stored evidence text.
- Keep the enrichment downstream of formulation identity decisions so it does not participate in Stage2 candidate creation, Stage5 retention, Stage5 collapse, GT compare semantics, or reviewer GT authority.

Reason
- The active schema preserved emulsion-specific method fields but lacked a generalized preparation-method surface for non-emulsion routes such as nanoprecipitation and solvent displacement.
- The enrichment is needed for better method representation without changing benchmark behavior.

Impact
- Stage2 TSV exports and Stage5 final-table exports can now carry a generalized preparation-method field pair.
- DEV15 Layer 1 DOI counts and the reviewed-boundary Layer 2 comparison must remain unchanged; any count change is treated as a regression.

### GT correction example: PA3SPZ28 manual undercount, not a new pipeline rule category

Interpretation
- `PA3SPZ28` / `10.1038/s41598-017-00696-6` should be treated as a GT correction case, not as evidence for a new system-level reconciliation rule.
- The system prediction of `5` formulations is likely structurally correct for this paper.
- The earlier GT count of `3` was a manual annotation undercount.

Why this is not a new rule family
- This case does not show extraction bias.
- This case does not show Stage4 suppression bias.
- This case does not justify a new EE-centered pipeline rule.
- It is an annotation reminder: independently prepared blank / FITC / control-style nanoparticle formulations may be real formulation instances even when the original manual count missed them.

Resolved interpretation
- Correct formulation-core count = `5`.

## 2026-03-11

### Case decision: 5GIF3D8W / 10.1080/10717540802174662 remains an open Stage2 under-enumeration case

Case reference
- `5GIF3D8W` / `10.1080/10717540802174662`
- Title: `Etoposide-Loaded PLGA and PCL Nanoparticles I: Preparation and Effect of Formulation Variables`

Problem statement
- Earlier formulation-skeleton enumeration for this paper was exact at `32` candidate rows against `32` GT rows.
- The later active DEV-15 formulation-instance evaluation on `2026-03-10` reported `predicted_count = 6`, `GT_count = 32`, and under-segmentation of `26`.

Diagnosis summary
- Current repo evidence localizes the loss before Stage4 counting:
  - `data/results/run_20260310_dev15_remaining12_synthmethod_merged/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv` contains only `6` rows for key `5GIF3D8W`.
  - `data/cleaned/labels/manual/formulation_instance_remaining12_eval_2026-03-10/predicted_instance_rows.tsv` also contains the same `6` predicted instances, so Stage4 is reporting Stage2 output rather than collapsing a larger extracted set.
- The surviving six rows are only the high-confidence PLGA variants (`PLGA 50/50`, `75/25`, `85/15`, empty/drug-loaded), while the earlier 32-row formulation-skeleton candidate file includes additional PCL and parameter-sweep formulations.
- Table assets for `5GIF3D8W` exist under `data/cleaned/goren_2025/tables/5GIF3D8W/`, and prior audit artifacts show table-value matches for this DOI, so the current evidence does not support classifying this as a Stage4 regression or a missing-table parse failure.
- Current best interpretation is an open Stage2 formulation-enumeration regression, most likely at the input-assembly / evidence-packing boundary: the current extractor still uses raw-text front-slice packing with prompt-side table-heavy hints only, so this paper's row-level sweep structure is still being abstracted away or not surfaced to the model.

Decision
- Treat this paper as an open diagnostic case, not a resolved rule case.
- Repository interpretation: the correct benchmark target for this DOI remains `32` formulation rows, and the current `6`-row output is a known under-enumeration failure of the active Stage2 path.
- This case is fix-ready only after a validated Stage2 change demonstrates recovery on this DOI without destabilizing the active DEV-15 path.

Implementation status
- No code change was merged for this DOI-specific case in the current repo state.

## 2026-03-18

### Decision: Restore relation-first Stage 3 -> Stage 5 materialization for descriptive synthesis fields

Decision
- Stage 3 now emits `resolved_relation_fields_v1.tsv` as an explicit deterministic contract for relation-backed descriptive synthesis field materialization.
- The initial resolved field set is limited to:
  - `plga_mw_kDa`
  - `surfactant_name`
  - `organic_solvent`
  - `preparation_method`
- Stage 5 must consume both:
  - `formulation_relation_records_v1.tsv`
  - `resolved_relation_fields_v1.tsv`
- Stage 5 must fail fast if either artifact is missing.
- Stage 5 is a materialization layer and must not perform semantic inheritance inference or same-paper donor search.

Reason
- Active DEV15 runs had drifted into a Stage 2 -> Stage 5 path with silent Stage 3 bypass and export-time donor-fill heuristics.
- Relation and inheritance control must be governed by an explicit upstream contract rather than repaired during final export.

Impact
- Current relation logic remains deterministic and stage-local to Stage 3.
- Stage 5 row identity, row counts, and conservative closure policy remain unchanged.
- Proof-case validation for `5GIF3D8W` shows that sparse PLGA sweep rows can recover descriptive synthesis fields from relation-backed branch subgraph closure while measurement outputs remain blank.
- No new script was created for this case.
- The currently documented Stage2 table-heavy row-enumeration prompt rule improved other papers but did not resolve `5GIF3D8W`.

Consequence
- The active pipeline did not change as a result of this case record.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` remains the active Stage2 entrypoint and `src/stage4_eval/eval_weak_labels_v7pilot3.py` remains the active Stage4 evaluator.
- Future similar cases should be classified the same way when weak-label TSV row loss is already present before evaluation: record them as Stage2 under-enumeration / evidence-packing diagnostics rather than as Stage4 reconciliation issues.

## 2026-03-12

### Formulation ontology and extraction scope rules

Case reference
- `5GIF3D8W` / `10.1080/10717540802174662`

Rule 1
- Formulation existence is defined by formulation-defining variables (`polymer identity`, `stabilizer concentration`, `drug/polymer ratio`, `phase ratio`, `solvent`, etc.), not by the current automatic extraction capability.

Rule 2
- Extraction scope must not apply material filtering.
- Formulations containing polymers outside the current modeling scope (for example `PCL` in a PLGA-focused study) must still be extracted and labeled.

Rule 3
- Figure-derived formulation-variable sweeps represent real formulation instances when the sweep variable is a formulation parameter (for example `stabilizer concentration`).
- These must be counted in the formulation target set even if current extraction cannot fully recover them.

Decision
- The correct formulation ontology for `5GIF3D8W` is `32` formulations:
  - `8` table rows
  - `24` figure-derived sweeps

Implementation note
- On `2026-03-12`, the active Stage2 extractor `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` was minimally extended to append low-confidence `candidate_source = "figure_variable_sweep"` candidates when a paper text explicitly declares multi-level formulation-variable sweeps.

### Follow-up: 5GIF3D8W residual 2-row gap traced to PCL table-row omission and fixed in Stage2

Case reference
- `5GIF3D8W` / `10.1080/10717540802174662`

Validated diagnosis
- After the figure-sweep recovery, the remaining gap from `30` to `32` was the missing `PCL Empty` and `PCL Drug loaded` optimized Table 1 rows.
- The loss occurred in the LLM extraction step, not in canonicalization, deduplication, or Stage4:
  - the raw Stage2 response explicitly stated that only PLGA formulations were extracted,
  - the cleaned full-text Table 1 segment still contained the PCL optimized rows and values.

Decision
- Stage2 extraction scope must not bias the model toward PLGA-only row enumeration when the paper reports explicit non-PLGA formulation rows.
- The active extractor prompt was corrected to remove PLGA-only wording and to state the no-material-filtering rule explicitly.

Validation
- Re-running `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` for `5GIF3D8W` recovered the two missing PCL table rows.
- Validated candidate counts after fix:
  - total candidates = `32`
  - `llm_extracted` / table-like = `8`
  - `figure_variable_sweep` = `24`

Consequence
- No workflow change was introduced.
- No Stage4 script changed.

### Explicit polymer identity added as a general extraction-layer field

Decision
- Extraction scope must preserve polymer identity explicitly at the extraction layer.
- Polymer filtering belongs to downstream modeling and release logic, not to Stage2 extraction.

Implementation
- The active Stage2 extractor now adds an explicit additive polymer field layer:
  - `polymer_identity`
  - `polymer_name_raw`
- Existing PLGA-specific fields remain in place for compatibility:
  - `la_ga_ratio_*`
  - `plga_mw_kDa_*`
  - `plga_mass_mg_*`

Consequence
- Mixed-polymer papers can retain non-PLGA rows without relying on implicit interpretation from PLGA-family fields.
- Downstream steps may later filter to `polymer_identity = PLGA` when required, without changing the extraction-layer scope.

### Decision: Apply a minimal Stage2 fix for the confirmed 5GIF3D8W sweep-structure seam

Case reference
- `5GIF3D8W` / `10.1080/10717540802174662`

Confirmed seams
- Duplicate seam:
  - explicit `llm_extracted` semantic variant rows and `figure_variable_sweep` rows were being kept together because synthetic sweep dedup was label-driven (`seen_labels`) rather than condition-signature-driven.
- Shared-section omission seam:
  - `_infer_section_identities(...)` collapsed generic `PLGA-copolymers` sweep sections to the first PLGA identity, which omitted polymer-amount sweep rows for `PLGA 75/25` and `PLGA 85/15`.

Adopted minimal fix strategy
- Keep the active pipeline path unchanged and patch only the active Stage2 extractor.
- Expand shared PLGA section inference conservatively so generic PLGA-copolymer sweep sections can enumerate all already-known PLGA identities for the paper.
- Add a narrow post-generation overlap dedup after synthetic sweep rows are appended.
- Preferred representation policy for overlapping sweep conditions:
  - keep baseline optimized table rows,
  - keep `figure_variable_sweep` rows for single-variable sweep conditions,
  - suppress overlapping `llm_extracted` semantic variant rows when they only restate the same polymer-specific sweep axis and level.

Validation outcome
- Post-fix targeted regression rerun:
  - previous run: `run_20260312_1031_455ac37_targeted5_stage2_regression_v1`
  - new run: `run_20260312_1253_455ac37_targeted5_stage2_regression_v1`
- `5GIF3D8W` changed from `38` rows to `44` rows.
- The confirmed duplicate overlap groups were removed (`6 -> 0`).
- The previously missing polymer-amount sweep rows for `PLGA 75/25` and `PLGA 85/15` were restored (`6 missing -> 0 missing`).
- The paper still over-expands because drug-amount sweep rows now span all four polymer groups, so the full `32`-target structure is still not restored.

Consequence
- This fix is retained as a structural improvement because it removes the confirmed overlap seam and restores the confirmed missing polymer-amount conditions.
- Full DEV15 rerun remains blocked until the remaining `5GIF3D8W` over-expansion seam is resolved.

### Decision: Apply a narrow axis-scoping fix for the confirmed 5GIF3D8W drug-amount over-expansion seam

Case reference
- `5GIF3D8W` / `10.1080/10717540802174662`

Confirmed issue
- After the earlier overlap-dedup and shared-PLGA polymer-amount fix, `5GIF3D8W` still over-expanded from `38` to `44` rows.
- Root-cause diagnostics confirmed that applicability differs by sweep axis for this paper:
  - `stabilizer_concentration`: supported across `PLGA 50/50`, `PLGA 75/25`, `PLGA 85/15`, and `PCL`
  - `polymer_amount`: supported across `PLGA 50/50`, `PLGA 75/25`, `PLGA 85/15`, and `PCL`
  - `drug_amount`: directly supported only for `PLGA 50/50` and `PCL`
- The unsupported rows were the six `drug_amount` sweep rows for `PLGA 75/25` and `PLGA 85/15`.

Adopted minimal fix strategy
- Keep the active Stage2 extractor as the only edited runtime component.
- Tighten `drug_feed_amount_text` sweep generation by:
  - anchoring the section window to the actual `Amount of Drug` heading,
  - preventing generic PLGA-family wording alone from widening `drug_amount` applicability to all known PLGA identities.
- Leave the shared-section widening logic for `stabilizer_concentration` and `polymer_amount` unchanged.

Validation outcome
- Targeted regression rerun:
  - prior post-fix run: `run_20260312_1253_455ac37_targeted5_stage2_regression_v1`
  - post-axis-scoping run: `run_20260312_1321_455ac37_targeted5_stage2_regression_v1`
- `5GIF3D8W` changed from `44` rows to `38` rows.
- The six unsupported `drug_amount` rows for `PLGA 75/25` and `PLGA 85/15` were removed.
- Polymer-amount sweep completeness for `PLGA 75/25` and `PLGA 85/15` was preserved.
- The earlier duplicate overlap seam did not return.

Consequence
- This axis-scoping fix is retained because it resolves the confirmed `drug_amount` applicability error without reintroducing the earlier seams.
- Full DEV15 rerun remains blocked pending additional targeted stability work because `5GIF3D8W` still sits above the intended `32`-structure target and other targeted5 warnings remain.

### Decision: Formalize the formulation-instance system architecture and current benchmark contract

Decision
- The repository is now formally documented as a formulation-instance reconstruction system rather than a pure field-extraction pipeline.
- Stage2 owns candidate formulation-instance extraction with high recall and explicit evidence/provenance, not guaranteed final formulation-table closure.
- Instance-level evaluation is a separate layer that measures candidate-graph behavior such as under-segmentation, over-segmentation, and benchmark count mismatch.
- Final formulation-table semantics, precision recovery, and any generic collapse by core formulation parameters belong to downstream guardrail/normalization ownership, not implicitly to Stage2.

Current active-path contract
- The active DEV-15 comparison path currently compares Stage2 candidate formulation-instance rows directly against the fixed DEV15 skeleton workbook.
- No generic normalization or formulation-core collapse layer is wired between the active Stage2 extractor and the active Stage4 evaluator.
- Only explicitly documented reconciliation logic, such as the integrated `WFDTQ4VX` DoE rule, should alter that direct candidate-layer comparison.

Historical-path clarification
- The DEV15 skeleton bootstrap family under `archive/code/dev15_skeleton_bootstrap/` is a historical benchmark-preparation workflow.
- It must not be described as if it were the active normalization layer for current DEV-15 evaluation.
- Stage5 schema/core tools remain downstream/supporting families for modeling-oriented outputs and benchmark/schema analysis, not the default current DEV-15 counting seam.

Consequence
- Baseline / optimized wording and similar provenance labels should not be treated as permanent blockers to later comparison by core formulation parameters.
- If the repository later intends to compare predictions against final formulation-table targets rather than candidate-instance targets, an explicit downstream guardrail/normalization contract must be defined and wired into the active path.

### Decision: Forbid official GT benchmark reporting from partial or intermediate pipeline layers

Decision
- The repository now forbids treating single-layer or partial-path outputs as official final GT comparison results.
- Formal GT comparison and benchmark reporting may be claimed only from the complete intended pipeline final-output layer.
- Intermediate-layer outputs, including Stage2 candidate rows, candidate graphs, packed evidence views, and component-only evaluation artifacts, may be compared to GT only for debugging, regression localization, and ablation.

Reason
- Repeated direct comparison of partial-layer outputs against final GT created contract confusion and distorted iteration priorities.
- Candidate-instance artifacts and final formulation-table artifacts do not have the same semantics, so reporting them under one benchmark label is misleading even when the counts are numerically comparable.

Policy effect
- If downstream guardrail, normalization, collapse, or final filtering layers are part of the intended system, they must be executed before benchmark-valid GT reporting is made.
- If those layers are not yet wired, the correct label for partial-path GT comparison is `diagnostic-only, not benchmark-valid final output`.
- Future work must not iterate indefinitely on one isolated layer while treating direct GT comparison as if it were the final benchmark contract.

### Decision: Enforce explicit script registry ownership and reproducible run specifications

Decision
- Every `src/*.py` script must now have an explicit recorded identity and I/O contract.
- The minimum recorded metadata is:
  - script path
  - status
  - architecture layer
  - function summary
  - primary inputs
  - primary outputs
  - upstream dependencies
  - downstream consumers
  - current pipeline role
- Scripts that cannot be classified clearly enough to record this metadata with defensible confidence are non-compliant and must be treated as archive/delete candidates rather than as active engineering assets.

Run reproducibility rule
- Every `data/results/run_*` directory must contain a reproducibility-grade run specification in the run root.
- That specification must record run purpose, run type, starting inputs, script execution order, script paths used, intermediate artifacts, final outputs, and benchmark-valid versus diagnostic-only status.
- A run directory lacking this specification is non-compliant.

Reason
- The repository had accumulated scripts whose role could not be inferred quickly enough from code or docs, and historical runs whose execution order was no longer reproducible by inspection.
- That ambiguity is now treated as a governance failure because it blocks reuse, archive decisions, and reproducible benchmarking.

Consequence
- `docs/src_script_registry.tsv`, `docs/src_script_compliance_report.tsv`, and `docs/run_directory_compliance_report.tsv` are now part of the repository governance scaffolding.
- `docs/run_spec_template.md` defines the minimum acceptable run-spec contract for future runs.

### Decision: Cleanup wave 1 moved legacy scripts to archive and quarantined delete candidates

Decision
- Cleanup wave 1 is now executed as a conservative repository-noise reduction pass, not as a pipeline redesign.
- Pre-reduction legacy scripts formerly under `src/legacy/` were moved into `archive/code/pre_reduction_legacy/`.
- Scripts already classified as `delete_candidate_after_confirmation` were moved into `archive/delete_candidates_pending_confirmation/` so they no longer read as normal reusable assets.

Run-history handling
- Non-compliant `data/results/run_*` directories were not physically relocated in wave 1 because many investigations and historical notes still reference their current paths.
- Instead, wave 1 added explicit top-level segregation indexes under `data/results/` to distinguish current engineering runs from historical non-compliant runs.

Consequence
- Wave 1 reduces active-repo ambiguity without breaking the documented active path.
- Additional pruning, run-history compression, and final deletion decisions remain follow-up work for later cleanup waves.


### Decision: Cleanup wave 2 physically removed archive-only code from `src/` and separated historical run noise

Decision
- Cleanup wave 2 is a stricter physical boundary pass, not a pipeline redesign.
- Archive-only code was moved out of `src/` into `archive/code/`.
- Delete candidates pending confirmation were moved out of `src/` into `archive/delete_candidates_pending_confirmation/`.
- `src/` is now reserved for live mainline or branch-active engineering code only.

Run-history handling
- Historical non-compliant runs that were not still named directly in current authoritative docs were physically moved into `data/results/historical_non_compliant_runs/`.
- The small set of non-compliant runs left in `data/results/` remains only because current authoritative docs still point to those exact paths.
- The four current engineering runs were backfilled to reproducibility-grade run specs and now represent the only current engineering runs at the top level of `data/results/`.

Consequence
- The active engineering surface is materially smaller and easier to interpret before any final-output layer design work.
- Remaining cleanup should focus on later pruning/compression, not on keeping archive-only code inside `src/`.

### Decision: Define the minimal final-output layer contract before implementation

Decision
- The repository now formally defines a missing minimal final-output layer as the required downstream contract for future benchmark-valid runs.
- The durable design record lives at `project/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md`, with supporting contract tables in `project/design/`.
- This layer is planned to consume candidate-instance outputs from the active Stage2 path, apply limited final non-formulation filtering plus narrow core-parameter-based collapse, and emit a benchmark-valid one-row-per-formulation table with decision-trace artifacts.

Current-state clarification
- The layer is not yet implemented in the active path.
- Current Stage2 -> Stage4 DEV-15 runs remain candidate-instance diagnostic runs, not full-pipeline benchmark runs.
- Historical Stage5 benchmark scripts and archived skeleton-bootstrap scripts remain reference assets only unless explicitly adapted into the future final-output implementation.

Reason
- The repository now forbids benchmark claims from partial layers, so the missing final-output contract can no longer remain implicit.
- A durable written design is required before implementation so the next architectural step does not collapse into ad hoc script growth or historical-path confusion.

Consequence
- Future implementation work should follow the minimal contract first rather than reactivating broad rule-heavy reconstruction families.
- The benchmark-valid run type remains reserved until this layer and its downstream benchmark comparison step are implemented.

### Decision: Implement phase 1 of the minimal final-output layer with conservative closure rules

Decision
- Phase 1 of the minimal final-output layer is now implemented under:
  - `src/stage6_final_output/build_minimal_final_output_v1.py`
  - `src/stage6_final_output/run_minimal_final_output_v1.py`
- Phase 1 is intentionally narrow. It:
  - filters explicit non-formulation rows,
  - computes a conservative core signature from current candidate-instance fields,
  - collapses rows only when a clear mixed-source redundancy signal is present,
  - emits `final_formulation_table_v1.tsv`, `final_output_decision_trace_v1.tsv`, and `final_output_summary_v1.md`.

Implemented scope
- The implemented filtering contract is conservative and explicit:
  - `candidate_non_formulation`
  - characterization-only post-processing rows
- The implemented collapse contract is conservative and explicit:
  - only rows with known polymer identity and loaded state,
  - only rows without excluded tags such as `doe`, `checkpoint_validation`, `center_point`, or `post_processing`,
  - only rows with sufficiently populated core fields,
  - only rows with a clear mixed-source overlap signal (`llm_extracted` plus `figure_variable_sweep`) for the same signature.

Deliberately not implemented
- broad rule-heavy reconstruction
- generalized DOE collapse
- benchmark comparison over the final formulation table

Validated run
- `run_20260312_1636_455ac37_minimal_final_output_v1`
- input candidate artifact:
  - `data/results/run_20260312_1321_455ac37_targeted5_stage2_regression_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
- outcome:
  - `127` input rows
  - `124` final rows
  - `3` filtered explicit non-formulation rows
  - `0` collapsed rows under the conservative phase-1 rules

Consequence
- The repository now has a working first implementation of Layer 7 final-output closure.
- This is still not a complete benchmark-valid path because the downstream GT comparison step over `final_formulation_table_v1.tsv` has not yet been implemented.

### Decision: Implement the first complete full-pipeline benchmark path for the targeted5 controlled scope

Decision
- The repository now has one checked-in complete benchmark-valid path for the declared targeted5 controlled scope.
- That path runs:
  - Stage2 extraction with `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  - Stage6 final-output closure with `src/stage6_final_output/build_minimal_final_output_v1.py`
  - final-table GT comparison with `src/stage7_full_pipeline/compare_final_table_to_gt_v1.py`
  - orchestration through `src/stage7_full_pipeline/run_full_pipeline_benchmark_v1.py`

Benchmark contract
- Official GT reporting in this path is attached only to `final_formulation_table_v1.tsv`.
- No Stage2 candidate-instance output is permitted to serve as the reported system result.
- Any debugging loop must start from the final-table comparison artifacts and then trace backward into Stage6 or Stage2 only after a final-table mismatch is established.

Scope and limitations
- The implemented benchmark-valid scope is currently the controlled targeted5 engineering manifest.
- The first complete comparison step currently supports per-paper final-formulation count comparison against the authoritative fixed DEV15 skeleton workbook.
- Structured EE-subset benchmark comparison is still unsupported because the authoritative fixed workbook does not expose structured EE GT fields for this scope.

Consequence
- `full_pipeline_benchmark_run` is no longer a purely reserved label; it is now available when the declared targeted5 full runner is executed end to end.
- Outside that declared complete path, partial-layer and candidate-instance runs remain diagnostic only.

### Decision: Restore the canonical active path to manual Stage 0 to Stage 5 reproduction only

Decision
- The official active stage naming is restored to exactly `Stage 0`, `Stage 1`, `Stage 2`, `Stage 3`, `Stage 4`, and `Stage 5`.
- There is only one active Stage 5 namespace:
  - `src/stage5_benchmark/`
- The phase-1 final-output builder and the final-table benchmark comparison step are retained, but they now live under the single Stage 5 namespace:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `src/stage5_benchmark/run_minimal_final_output_v1.py`
  - `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Repository cleanup
- The one-click full rerun wrapper `src/stage7_full_pipeline/run_full_pipeline_benchmark_v1.py` is removed from the active repository surface.
- The former duplicate active namespaces are retired from `src/`:
  - `src/stage5_merge_publish/`
  - `src/stage6_final_output/`
  - `src/stage7_full_pipeline/`
- Historical merge-only code is retained as archive reference under:
  - `archive/code/stage5_merge_publish/merge_results.py`

Canonical execution contract
- The canonical pipeline is now defined only as the explicit manual Stage 0 to Stage 5 path documented in `project/ACTIVE_PIPELINE_FLOW.md`.
- `project/ACTIVE_PIPELINE_FLOW.md` is the single authoritative manual reproduction document.
- Complete pipeline means full traceable stage coverage from raw Zotero-derived inputs to the Stage 5 final formulation table and Stage 5 GT-comparison outputs.
- Complete pipeline does not mean forced full recomputation.

Forbidden path behavior
- No hidden Python orchestrator defines the canonical path.
- No intermediate-layer artifact may be reported as the system result against GT.
- Stage 4 remains a diagnostic and reviewer-facing layer.
- Benchmark-valid reporting remains attached only to the Stage 5 final formulation table and its Stage 5 comparison outputs.

Consequence
- Governance docs, script maps, and source-tree boundaries now describe one active Stage 0 to Stage 5 pipeline only.
- Future debugging must follow the documented stage sequence and trace backward from Stage 5 mismatches rather than introducing new full-rerun wrappers.

## 2026-03-13

### Decision: Add an explicit deterministic Stage 3 formulation relation layer

Decision
- The active pipeline now includes a checked-in deterministic Stage 3 runtime layer under:
  - `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
  - `src/stage3_relation/run_formulation_relation_artifacts_v1.py`
- This layer sits after Stage 2 weak-label extraction and before Stage 5 final formulation closure.
- It must not call any LLM or external API.
- It materializes explicit paper-level relation artifacts:
  - `formulation_relation_records_v1.tsv`
  - `formulation_logic_graph_v1.jsonl`
  - `formulation_relation_summary_v1.tsv`

Reason
- Stage 2 candidate rows already contain semantic formulation hypotheses, but their relation structure was implicit and hard to audit.
- A deterministic intermediate layer is needed to make method grouping, shared fields, variation axes, parent-child links, and candidate membership explicit before final flattening.
- This separates relation reasoning from benchmark-facing closure logic and reduces pressure to hide reconstruction semantics inside Stage 5.

Impact
- Stage 3 is no longer only a documented contract; it now has a dedicated active runtime entrypoint.
- Stage 5 may consume Stage 3 relation records as explicit provenance input.
- Stage 4 remains diagnostic only and manual GT remains a reference input surface rather than the meaning of Stage 3.

### Decision: Keep Stage 5 closure conservative in phase 1 while exposing Stage 3 provenance

Decision
- The current phase-1 Stage 5 builder continues to use the Stage 2 candidate TSV as its primary closure input.
- Stage 3 relation artifacts are accepted as optional deterministic provenance and are attached to retained final rows when supplied.
- Stage 3 relation artifacts do not yet fully drive Stage 5 keep/drop/collapse decisions.

Reason
- The first implementation target is explicit auditability, not immediate replacement of all existing closure heuristics.
- This preserves current benchmark behavior while creating a governed insertion point for later deterministic relation-aware closure work.

Impact
- The active pipeline now has a visible and reproducible intermediate relation layer.
- Remaining work is clearly bounded: future iterations may promote Stage 3 relations from provenance support to stronger closure control without changing the stage layout again.

### Decision: Record UFXX9WXE as a confirmed Stage2 DOE table under-enumeration failure

Case reference
- `UFXX9WXE` / `10.1155/2014/156010`
- Paper type: DOE-style optimization paper with explicitly numbered formulation rows in a source-paper table

Problem statement
- Manual paper review now confirms that the source paper contains a table with `26` explicitly numbered formulations.
- The current active DEV15 lineage extracted only about `5` Stage2 candidate formulations for this paper and produced `4` benchmark-facing Stage5 final rows.
- The current GT row count for this paper remains `26`.

Diagnosis summary
- The failure is confirmed upstream of Stage3 and Stage5:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/analysis/paper_diagnostic_summary.tsv` records `stage2_candidate_count = 5`, `stage5_final_row_count = 4`, and `gt_row_count = 26` for `UFXX9WXE`.
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/analysis/paper_audit_pack.md` records this paper as a DOE-style table case with only a small subset of numbered anchors extracted.
- This is not primarily a Stage3 relation failure:
  - Stage3 already forms one coherent method group and identifies shared parameters and variation axes from the rows that Stage2 did emit.
  - Stage3 cannot infer or materialize formulation rows that were never enumerated by Stage2.
- This is not primarily a Stage5 collapse failure:
  - Stage5 can only retain, collapse, or filter candidate rows that exist upstream.
  - When numbered DOE rows are absent from Stage2 weak labels, Stage5 cannot reconstruct them from relation structure alone.
- The confirmed engineering interpretation is therefore: the active Stage2 path is under-enumerating numbered DOE table rows for this paper.

Decision
- Record `UFXX9WXE` as a confirmed Stage2 under-enumeration failure for numbered DOE-style table rows.
- Repository interpretation: downstream relation inference and closure are not the primary root cause for this paper's `26 -> 5 -> 4` loss pattern.
- Future runs and future Codex analysis must treat similar cases the same way when the source paper contains explicitly numbered DOE/design rows that are missing from Stage2 weak-label output.

Implementation status
- No pipeline behavior change is merged by this record.
- No new Stage3 or Stage5 recovery rule is authorized from this case record alone.
- Any future fix must target the upstream DOE row-enumeration gap rather than treating this as a downstream reconstruction problem.

Consequence
- `UFXX9WXE` is now a confirmed reference case for Stage2 DOE table under-enumeration.
- Downstream layers remain diagnostic for this failure class; they may localize the loss, but they must not be described as capable of recovering rows that Stage2 never emitted.

### Decision: Add deterministic numbered DOE row enumeration at the Stage2 boundary

Case reference
- Primary regression target: `UFXX9WXE` / `10.1155/2014/156010`

Decision
- The active Stage2 boundary now includes a deterministic numbered DOE table-row enumerator implemented in:
  - `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`
- The active Stage2 extractor `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` now additively calls this enumerator by default after LLM extraction and before writing the final Stage2 weak-label artifact.
- The enumerator reads existing Stage1 table assets and emits explicit augmentation artifacts:
  - `numbered_doe_row_candidates_v1.tsv`
  - `numbered_doe_row_candidates_summary_v1.tsv`

Reason
- `UFXX9WXE` confirmed that prompt-only DOE instructions are insufficient for explicit numbered design tables.
- Stage3 and Stage5 cannot reconstruct formulation rows that never existed in Stage2.
- Existing DOE-specific downstream logic is either paper-specific (`WFDTQ4VX`) or branch-active downstream derivation support; it does not solve the upstream enumeration gap in the current canonical path.

Scope
- This implementation is intentionally minimal.
- It targets explicit numbered DOE or design-table rows preserved in Stage1 table CSV assets.
- It preserves non-core varying factors in explicit JSON columns inside the augmentation artifact and in Stage2 `change_descriptions`.
- It does not yet attempt generalized coded-level DOE decoding or broad design-matrix reconstruction from prose alone.

Regression protection
- The enumerator CLI supports `--expected-min-recovered` and exits nonzero if the expected recovery threshold is not met.
- `UFXX9WXE` is the primary regression paper for this capability because the source table contains `26` explicitly numbered formulations while the prior active Stage2 path extracted only about `5` candidates.

Consequence
- Numbered DOE row recovery is now an explicit upstream deterministic responsibility at the Stage2 boundary.
- This is not a downstream patch and does not alter Stage3 or Stage5 ownership boundaries.

### Decision: Enforce single-parent run-lineage containment for same-lineage retries and repair steps

Decision
- One top-level `data/results/run_*` directory now represents one benchmark or experiment lineage.
- Stage-local retries, partial reruns, completion merges, deterministic refreshes, and repair steps for that same lineage must be nested as child executions under the parent lineage directory instead of remaining as sibling top-level runs.
- The recommended child location is:
  - `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`

Reason
- The recent DEV15 DOE rebuild accumulated multiple sibling top-level runs that shared the same timestamp and git short hash but differed only by retry or stage suffix.
- That pattern preserved provenance but created human-facing sprawl, weakened run containment, and made it harder to understand one benchmark lineage by opening a single directory.

Policy effect
- A new top-level run is allowed only when the objective, scope, or benchmark contract is independent enough to stand alone.
- Internal retries or recovery steps must stay under the existing lineage parent.
- If same-lineage runs are reorganized after creation, the parent lineage directory must record old-to-new path mapping and child-step roles explicitly.

Implementation note
- The DEV15 DOE rebuild lineage rooted at `run_20260313_1235_f4912f3_dev15_current_merged_benchmark_doe_v1` was reorganized under this policy.
- The repo now includes `src/utils/audit_run_lineage_layout_v1.py` as a deterministic sprawl-detection utility for top-level run directories.

### Decision: In strong numbered DOE tables, deterministic Stage2 enumeration is primary and the LLM acts as judge rather than row counter

Case reference
- `UFXX9WXE` / `10.1155/2014/156010`

Confirmed repository evidence
- The baseline miss was not caused by missing source data:
  - `data/cleaned/goren_2025/tables/UFXX9WXE/tables_manifest.json` records `html_found = false`, `pdf_found = true`, `n_tables_pdf_extracted = 18`, and `preferred_table_source = pdf`.
  - `data/cleaned/goren_2025/tables/UFXX9WXE/UFXX9WXE__table_13__pdf_table.csv` preserves the explicit numbered DOE structure with rows `1.` through `26.`.
- The critical numbered structure was already present before Stage2:
  - `data/cleaned/goren_2025/text/UFXX9WXE/UFXX9WXE.pdf.txt` preserves the same numbered table content in cleaned text.
- The primary baseline failure occurred at Stage2 interpretation time:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/08_UFXX9WXE_10.1155_2014_156010.txt` shows the model describing only three specific design rows plus the optimized formulation from a truncated full-text window.
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/analysis/paper_diagnostic_summary.tsv` records `stage2_candidate_count = 5` and `stage5_final_row_count = 4` against `gt_row_count = 26`.
- DOE recovery succeeded by deterministic enumeration over the already-available Stage1 PDF table asset:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/children/run_20260313_1157_f4912f3_ufxx9wxe_doe_row_recovery_v5/numbered_doe_row_candidates_v1/numbered_doe_row_candidates_summary_v1.tsv` records `numbered_rows_found = 26`, `existing_stage2_numeric_rows = 3`, and `new_candidates_emitted = 23`.

Decision
- In strong DOE-style numbered tables, deterministic row enumeration is the primary row-discovery mechanism at the Stage2 boundary.
- The LLM must not be treated as the primary mechanism for row counting or row discovery when a structured Stage1 table asset already exposes stable numbered row anchors.
- In these strong-structure cases, the LLM acts as judge, not counter.

Stage2 gating strategy
- `deterministic_enumeration`
  - use when a structured Stage1 table asset exists
  - use when the table exposes stable row anchors such as numbered rows
  - use when the table shows row-wise formulation or design structure
  - use only when safety guards do not indicate obvious misfire risk
- `hybrid_enumeration_review`
  - use when deterministic row anchors exist but the table has enough irregularity that enumerated rows should be emitted with explicit review-oriented provenance
- `llm_discovery_only`
  - use when no reliable structured table asset exists or the paper does not expose stable row-wise formulation structure suitable for deterministic enumeration

LLM role in strong-structure cases
- confirm that a detected table is actually a formulation or design table
- validate whether enumerated rows are true formulation rows
- map column semantics to Stage2 schema fields
- identify exceptional rows such as optimized, control, summary, validation, or non-formulation rows

Scope decision for next engineering work
- The next justified optimization target is the Stage2 boundary for `UFXX9WXE`-class papers.
- The immediate goal is not a broad DEV15-wide DOE rollout.
- The priority is a bounded Stage2 fix and validation path for strong numbered DOE tables.
- The current full DEV15 DOE rebuild is not baseline-ready because broader regressions remain outside the confirmed UFXX recovery case.

Consequence
- Similar future cases should be classified first by source sufficiency and row-anchor strength, not by defaulting to more prompt-side row discovery.
- Broad benchmark rollout remains downstream of bounded Stage2 validation for strong numbered DOE tables.

### Decision: Prefer deterministic numbered-table rows over overlapping LLM numeric rows in the active Stage2 merge path, validated on UFXX9WXE

Case reference
- `UFXX9WXE` / `10.1155/2014/156010`

Patched code path
- `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Decision
- When the deterministic numbered DOE enumerator emits a strong structured-table row whose numeric label overlaps an existing `llm_extracted` row, the structured-table row is preferred.
- The overlapping `llm_extracted` numeric row is removed from the Stage2 candidate set instead of blocking deterministic emission.

Reason
- Before this patch, the active enumerator still suppressed deterministic rows `1`, `2`, and `3` for `UFXX9WXE` because the LLM had already emitted overlapping numeric labels from a `full_text_window` surface.
- That behavior left the active path partially dependent on opaque LLM row counting even in a paper where a structured Stage1 PDF table already preserved the full numbered DOE design.

Bounded validation
- UFXX-only replay run:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/children/run_20260313_1526_f4912f3_ufxx_only_stage2_doe_validation_v1`
- Observed result:
  - baseline Stage2 candidate count: `5`
  - validation Stage2 candidate count: `28`
  - deterministic numbered DOE candidates: `26`
  - baseline final UFXX count: `4`
  - validation final UFXX count: `28`
  - GT count: `26`

Interpretation

## 2026-03-21

### Decision: Keep Stage2 frozen for the BXCV5XWB helper-descendant regression and tighten Stage5 helper-descendant governance

Decision
- Keep Stage2 frozen for the `BXCV5XWB` over-retention class.
- Tighten Stage5 downstream governance in
  `src/stage5_benchmark/build_minimal_final_output_v1.py` so parent-linked
  helper/control/assay descendants are filtered when existing downstream
  signals already show that they are not independent benchmark-facing
  formulation identities.

Reason
- The raw Stage2 sufficiency audit for `BXCV5XWB` confirmed that raw extraction
  already preserved enough identity-relevant content for the six extra helper
  rows:
  - blank/no-drug control semantics
  - FITC model-drug substitution semantics
  - parent linkage to the benchmark-facing KGN rows
  - helper/control/characterization semantics in labels or descriptions
- The drifted Stage2 shaping regressed some primary routing tags, but it did
  not remove the downstream-recoverable helper semantics.
- Under the current freeze rule, this does not justify reopening Stage2.

Exact rule change
- Stage5 previously filtered parent-linked descendants mainly when
  `change_role == non_synthesis`.
- Stage5 now also filters parent-linked helper descendants when a combination
  of preserved downstream-visible signals indicates helper semantics, including:
  - helper payload states such as blank-control or FITC assay substitution
  - helper/control/model-drug-substitution context tags
  - helper/control/substitution text in labels or change descriptions
  - formulation-role evidence such as control/characterization-only

Why the previous rule was too narrow
- It depended too heavily on one upstream routing field and could miss rows
  whose helper semantics were still clearly recoverable downstream.
- This caused benchmark-facing over-retention of blank and assay/helper
  descendants in papers such as `BXCV5XWB`.

Intended behavior
- Keep Stage2 frozen.
- Use downstream governance to suppress clear helper descendants that do not
  define independent synthesis identities.
- Preserve legitimate synthesis variants and sweep-style benchmark-facing rows.

Regression protection
- Extended deterministic no-LLM checker:
  `src/stage5_benchmark/validate_stage5_descendant_filter_regression_v1.py`
- Coverage now asserts:
  - `BB3JUVW7` still retains all 12 benchmark-facing rows
  - `BXCV5XWB` retains only the 3 KGN benchmark-facing rows
  - `RHMJWZX8` drops its parent-linked empty-control helper row
  - blocker-material descendant rows remain filtered
  - `WIVUCMYG` remains stable
- The bounded validation succeeded for the intended patch target.
- The active Stage2 path now explicitly uses the structured Stage1 PDF table asset for the numbered DOE rows and replaces overlapping numeric `llm_extracted` rows with `doe_numbered_table_row` rows.
- A residual `+2` remains at final output because two non-table LLM rows were still retained, so the patch is not yet a broad DOE rollout authorization.

### Decision: Reuse existing raw LLM outputs for validation and replay whenever the LLM-facing input has not changed

Decision
- Validation or replay runs must reuse existing raw LLM outputs whenever the code change does not alter the LLM-facing input.
- Fresh LLM calls are required only when the LLM-facing input changes in a way that can change model output.

LLM-facing input definition
- prompt text
- source window or context selection
- table/text evidence actually sent to the model
- model name or version
- sampling or generation configuration
- any upstream logic that changes what is sent to the model, even if downstream schemas stay the same

Reuse-eligible change classes
- weak-label parsing only
- deterministic candidate generation
- merge, overlap, or dedup logic
- relation or provenance processing
- final table generation
- audit export
- confidence or review export

Engineering rationale
- preserve reproducibility by holding raw model outputs constant when the model-facing input is unchanged
- control cost by avoiding unnecessary regeneration
- keep baseline comparisons fair by separating downstream deterministic changes from new model behavior
- accelerate bounded validation cycles by replaying from existing raw outputs where possible

Naming guidance
- Runs that reuse existing raw LLM outputs must be described as replay or reuse-LLM validation runs.
- They must not be described as fresh full-regeneration runs.

Conservative rule
- If the repository cannot prove that the full LLM-facing input is unchanged because model/config or evidence-window provenance is missing, the run should be treated as `unproven for strict raw-output reuse equivalence`.
- In that case, documentation may still describe a pragmatic replay, but it must not claim strict LLM-input identity without evidence.

### Decision: Upgrade the active Stage5 final-output path from narrow duplicate patches to a controlled duplicate / variant governance layer

Decision
- The active Stage5 final-output path in `src/stage5_benchmark/build_minimal_final_output_v1.py` is upgraded from narrow one-off duplicate-collapse patches to a controlled duplicate / variant governance layer.
- Duplicate / variant governance now happens inside Stage5 final-output handling.
- Provenance labels alone do not define final formulation identity.
- Auto-collapse remains conservative and auditable and is allowed only when Stage5 finds one explicit safe target for the same formulation identity.

Supported Stage5 variant classes
- `duplicate_representation`
- `optimized_variant`
- `checkpoint_or_validation_variant`
- `post_processing_or_measurement_variant`
- `uncertain_variant`

Traceability impact
- The Stage5 decision trace now records:
  - `variant_class`
  - `variant_signal`
  - `equivalence_group_id`
  - `retention_reason`
  - `collapse_reason`
  - `review_needed`
- Collapsed-variant membership is now carried into the final formulation table and downstream audit-ready export surface.

Bounded replay validation
- Validation run:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/children/04_stage5_variant_governance_replay/run_20260313_2002_c4eccc8_dev15_stage5_variant_governance_replay_v1`
- Confirmed outcomes:
  - `UFXX9WXE` remained `27` vs GT `26`, so the DOE recovery benefit was preserved.
  - `INMUTV7L` remained corrected at `12` vs GT `12`.
  - `BXCV5XWB` changed from `9` to `7` through two conservative `post_processing_or_measurement_variant` same-core collapses.
  - No other papers changed relative to the previous replay.
  - No fresh LLM calls were made.

Current limits
- `WFDTQ4VX` checkpoint / validation reconciliation is now classified in Stage5 but is not auto-collapsed without a unique deterministic target.
- Optimized / baseline handling remains unique-target-only.
- Parent / variant inheritance is still not relation-driven in Stage5.
- Ambiguous cases remain `uncertain_variant` with `review_needed = yes`.

### Decision: Mirror the validated WFDTQ4VX checkpoint coordinate rule into the benchmark-valid Stage5 path

Decision
- Integrate one narrow paper-specific collapse rule into `src/stage5_benchmark/build_minimal_final_output_v1.py` for `WFDTQ4VX` / `10.1080/10717544.2016.1199605`.
- The rule collapses a checkpoint / validation row only when the validated coordinate-signature mapping shows that the checkpoint batch matches exactly one existing design-row formulation identity.
- This mirrors the previously validated Stage4 diagnostic rule for the same paper.

Reason
- The repository already recorded that Stage5 benchmark-facing closure should mirror the validated `WFDTQ4VX` coordinate-aware identity rule.
- Leaving that rule only in Stage4 would make blocker-gate evaluation depend on a knowingly incomplete mainline final-output path.

Scope control
- This change does not add generalized DOE closure.
- This change does not alter Stage2 extraction semantics.
- This change does not broaden optimization closure, family closure, or relation-driven inheritance logic.
- If the paper-local source-text parse fails or the checkpoint row does not match exactly one design-row identity, Stage5 falls back to the prior conservative behavior.

Consequence
- `WFDTQ4VX` checkpoint / validation identity closure is now on the benchmark-valid mainline path.
- Broader DOE coordinate closure remains future work and must not be inferred from this narrow integration.

### Decision: version the DEV15 GT authority when benchmark object alignment changes, and preserve prior GT workbooks unchanged

Decision
- DEV15 GT authority updates must be versioned as new workbook artifacts rather than in-place overwrites.
- The preserved prior authority for the original DEV15 skeleton count comparison remains:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
- The variant-aware follow-on authority is:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`

Why this change was required
- The current benchmark-valid object is the Stage5 retained final formulation table, which is variant-aware.
- For `BXCV5XWB`, the retained Stage5 final output contains `9` benchmark-facing rows.
- The older DEV15 skeleton workbook counted that paper as `3` family-like rows, which is no longer definitionally aligned with the active benchmark object.

Authority-switch rule
- Future DEV15 GT count comparisons that are intended to match the active Stage5 benchmark object must use:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`
- The preserved `v1_fixed` workbook remains part of project provenance and must not be silently replaced or deleted.

Reproducibility note
- The versioned GT update is produced by deterministic reuse of the existing Stage5 final formulation table and does not require any fresh Stage0-Stage5 upstream rerun.

## 2026-03-16

### Decision: Add the first active Layer 2 boundary-GT review export surface

Decision
- Add `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py` as the first active engineering surface for Second-Layer GT.
- The script exports a run-scoped XLSX workbook seeded from `final_formulation_table_v1.tsv`, with optional `final_output_decision_trace_v1.tsv`, optional `formulation_relation_records_v1.tsv`, and optional scope-manifest metadata as reference inputs.
- Prediction-reference columns are locked and visually separated from reviewer-editable GT columns.
- Core reviewer actions use dropdown-backed enums instead of free-text decisions where possible.
- The workbook also includes manual-addition template rows so missing GT formulation instances can be added without changing the workbook schema.

Reason
- The repository already has a stable Layer 1 pattern for skeleton workbook generation and review, but no active Layer 2 boundary-GT review surface in `src/`.
- The benchmark object is now the Stage 5 final formulation table, so the review seed should come from Stage 5 rather than from Stage 2 candidate rows.
- Boundary GT needs a human-reviewable but machine-validatable surface before row-level alignment comparison can be implemented safely.

Impact
- The active repo now exposes a repository-native workbook export path for boundary GT review without reviving archived skeleton-bootstrap code as runtime authority.
- This tool is a supporting review surface only; it does not change the canonical production endpoint or benchmark-valid reporting rule.
- Future Layer 2 validation/export and alignment-compare work should build on this workbook schema rather than inventing a separate review format.

### Decision: Correct the Layer 1 GT count for 5GIF3D8W to 26 and formalize design-vs-instance counting

Decision
- Revise the Layer 1 DEV15 GT authority for `5GIF3D8W` / `10.1080/10717540802174662` from `32` formulation rows to `26`.
- The corrected count is:
  - `8` baseline table rows
  - `4` drug-amount sweep formulation instances
  - `2` polymer-content sweep formulation instances
  - `12` stabilizer-concentration sweep formulation instances
- The previous `32`-row count incorrectly included design-only combinations that were mentioned in the sweep structure but were not supported as reported formulation instances with instance-level evidence.

Reason
- Layer 1 GT is a formulation-instance count, not a full design-space count.
- A condition belongs in GT only when the paper presents it as a reported experimental instance, for example as a table row or a condition explicitly tied to results.
- Conditions that appear only as possible combinations or implied sweep coordinates, without row-level or result-level evidence, must be excluded from GT even if they are part of the described experimental design.

Instance-counting rule
- Include a formulation in Layer 1 GT only if it is a reported experimental instance.
- Include when:
  - it appears as a table row
  - it has explicit experimental conditions tied to results
  - the paper clearly treats it as a realized batch or formulation instance
- Exclude when:
  - it is mentioned only as part of a methods design space or possible combination set
  - it has no table row, no result, and no instance-level evidence
  - it exists only as a variable-design description rather than a reported realized formulation

Traceability
- The authoritative workbook `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx` was updated so the excluded `5GIF3D8W` rows are marked non-GT with an explicit justification note.
- The checked export `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_gt_v2_variantaware.tsv` was refreshed to the corrected `26`-row authority for this DOI.

### Decision: Exclude assay-only derivative particles from Layer 1 GT unless independently reported

Decision
- Revise the Layer 1 DEV15 GT authority for `BXCV5XWB` / `10.1007/s10439-019-02430-x` from `9` rows to `3`.
- Retain only the three drug-loaded benchmark formulation instances.
- Exclude the three FITC-labeled particle rows and the three blank-particle rows from Layer 1 GT.

Reason
- Layer 1 GT counts benchmark formulation instances, not assay-only derivatives.
- In this paper, the retained benchmark instances are the reported drug-loaded nanoparticle formulations with formulation-level characterization and benchmark-relevant results.
- The FITC-labeled particles were used only for assay context and were not reported as independent formulation-level benchmark rows.
- The blank particles were used as controls and were not reported as independent formulation-level benchmark rows with benchmark-relevant characterization/results.

Assay-only derivative rule
- Assay-only derivative particles, such as blank controls or FITC-labeled particles used only for imaging or cell experiments, are excluded from Layer 1 GT unless the paper reports them as independent formulation instances.
- If a derivative particle is present only as a control, imaging aid, uptake probe, or assay-specific variant without independent formulation-level reporting, mark it non-GT in the review workbook rather than counting it as a benchmark formulation.

Traceability
- In `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`, `BXCV5XWB_F04` through `BXCV5XWB_F09` were flipped from GT `yes` to `no` and annotated with the assay-only derivative exclusion note.
- The checked export `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_gt_v2_variantaware.tsv` was refreshed so `BXCV5XWB` now contributes only the three retained drug-loaded rows.

## 2026-03-18

### Decision: Restrict Stage 2 sweep expansion to explicitly evidenced formulation instances for narrative-only sections

Decision
- Update the active Stage 2 deterministic sweep-expansion logic so narrative-only variable sweep sections do not emit one formulation row per declared design level unless the section contains explicit formulation-level identity support.
- Keep figure-backed sweep expansion unchanged when the source section has series/axis support indicating reported per-level results.
- For narrative-only sweep sections, emit rows only for identity-level pairs that are explicitly evidenced in the narrative text.

Reason
- Layer 1 and reviewed-boundary Layer 2 both count observable formulation instances, not the full planned design space.
- The previous Stage 2 logic could over-expand methods-declared sweep levels into synthetic formulation rows even when the results text only reported trends or a smaller subset of realized levels.
- `5GIF3D8W` exposed this failure mode: the paper reports stabilizer sweep results per level, but its polymer-content and drug-amount sections contain narrative support for only a subset of the described design levels.

Impact
- The active Stage 2 path now preserves figure-backed sweep coverage while avoiding narrative-only design-space inflation.
- In the `5GIF3D8W` child regression run, Stage 2 and Stage 5 counts for that paper dropped from `38` to `26`, matching the reviewed-boundary authority for this DOI.
- Non-target DEV15 papers preserved their Stage 2 and Stage 5 row counts in the regression run.

Traceability
- The deterministic implementation lives in `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`.
- Regression evidence is recorded in:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/09_5gif3d8w_reported_formulation_filter/run_20260318_1324_ae5599d_dev15_5gif3d8w_reported_formulation_filter_no_llm_v1/RUN_CONTEXT.md`
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/09_5gif3d8w_reported_formulation_filter/run_20260318_1324_ae5599d_dev15_5gif3d8w_reported_formulation_filter_no_llm_v1/analysis/5gif3d8w_reported_formulation_filter_report.md`

### Decision: Exclude external commercial comparator rows from benchmark-facing Stage 5 formulation closure

Decision
- Filter rows from Stage 5 final formulation closure when they are external commercial or marketed comparator references and they do not carry internal preparation identity.
- Keep the Stage 2 extraction surface unchanged so those rows can still appear in candidate-level diagnostics and audit artifacts.

Reason
- The benchmark-facing Stage 5 final formulation table counts internally prepared formulation instances, not marketed reference products.
- `QLYKLPKT` exposed this failure mode: Stage 2 already tagged `Sporanox®` as `formulation_role=comparative` with `commercial` context, but Stage 5 still retained it because the existing filter only excluded rows explicitly marked `candidate_non_formulation`.
- External comparators may have numeric outcomes and still be non-benchmark rows when they are not internally prepared formulation instances.

Rule
- Exclude a row from final formulation closure when all of the following hold:
  - `formulation_role` is `comparative`
  - the row has a commercial or marketed-product signal
  - the row lacks internal preparation identity such as polymer identity, formulation ratio, polymer amount, surfactant identity/concentration, or solvent identity

Impact
- Commercial comparator rows remain visible upstream for auditability but no longer inflate benchmark-facing final-row counts.
- In the `QLYKLPKT` regression run, the commercial comparator row was filtered and the reviewed-boundary DEV15 comparison moved from `14/15` to `15/15`.
- Non-target DEV15 papers preserved their Stage 2 and Stage 5 row counts.

Traceability
- The deterministic implementation lives in `src/stage5_benchmark/build_minimal_final_output_v1.py`.
- Regression evidence is recorded in:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/10_qlyk_commercial_reference_filter/run_20260318_1347_ae5599d_dev15_qlyk_commercial_reference_filter_no_llm_v1/RUN_CONTEXT.md`
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/10_qlyk_commercial_reference_filter/run_20260318_1347_ae5599d_dev15_qlyk_commercial_reference_filter_no_llm_v1/analysis/qlyk_commercial_reference_filter_report.md`

### Decision: Freeze DEV15 Layer 2 at the reviewed-boundary benchmark and use frozen Stage 5 rows as the Layer 3 starting object

Decision
- Freeze the current DEV15 Layer 2 reviewed-boundary benchmark state after the `QLYKLPKT` commercial-comparator exclusion fix.
- Treat the reviewed-boundary-accepted Stage 5 final formulation rows as the only valid starting object for the next GT layer.
- Define the next GT layer as field-level correctness on frozen Layer 2 rows, not a restart of row-boundary review.

Frozen Layer 2 state
- reviewed-boundary GT total rows: `210`
- reviewed-boundary Stage 5 compare result: `15/15` papers matching
- closed benchmark-valid child run:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/10_qlyk_commercial_reference_filter/run_20260318_1347_ae5599d_dev15_qlyk_commercial_reference_filter_no_llm_v1`

Layer 3 direction
- Layer 3 evaluates field correctness only for retained Layer 2 formulation rows.
- Layer 3 should reuse existing deterministic field-level infrastructure where possible:
  - derivation
  - projection
  - alignment evaluation
  - evidence-token QC
  - audit-ready review export
- Historical conflict-based field GT assets remain reference material only and are not the active authority surface.

Traceability
- Layer 2 freeze snapshot:
  - `docs/snapshots/snapshot_2026-03-18_dev15_reviewed_boundary_gt_closed.md`
- Layer 3 protocol and asset triage:
  - `docs/methods/layer3_field_gt_protocol_v1.md`

### Decision: Seed Layer 3 field GT with a compact run-scoped workbook built from the frozen Stage 5 final table

Decision
- Start Layer 3 review from the frozen reviewed-boundary Stage 5 final table
  rather than from weak labels or archived field-conflict tooling.
- Use a run-scoped XLSX workbook as the primary human-review surface, emitted
  under the current lineage's `data/results/run_*/...` structure.
- Keep the workbook compact: one row per `(formulation_id, field_name)` with
  frozen identity columns, reviewer dropdowns, and row-local evidence text.

Current active tool
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

Reason
- Layer 3 authority must stay anchored to frozen Layer 2 formulation rows.
- Existing repo patterns already use run-scoped workbook builders for human GT
  review, especially `build_boundary_gt_review_workbook_v1.py`.
- The field-review workbook needs reviewer usability first: limited columns,
  frozen identity, compact evidence text, and controlled GT status dropdowns.

Field-seed rule
- Seed the workbook only from the frozen Stage 5 final table and deterministic
  row-local enrichments.
- Safe deterministic derivations are allowed only when they use explicit
  final-table inputs from the same frozen formulation row.
- The initial workbook includes `drug_polymer_ratio` only as a safe derived
  seed from `drug_feed_amount_text_value` and `plga_mass_mg_value` when both are
  present.
- The initial workbook does not seed `phase_ratio` because the frozen final
  table does not yet carry a safe explicit phase-ratio field.

Traceability
- Method spec:
  - `docs/methods/layer3_field_gt_protocol_v1.md`
- Script registry:
  - `project/PIPELINE_SCRIPT_MAP.md`
  - `docs/tool_index.md`

### Decision: Enforce strict row-local table evidence or `unresolved_table` in the Layer 3 field review workbook

Decision
- Fix the Layer 3 field-review workbook so a field marked as table-derived no
  longer reuses generic paragraph text.
- Accept table evidence only when the frozen Stage 5 row already carries a
  matching row-local `table_cell` or `table_row` evidence reference.
- If no matching row-local table evidence exists, set
  `evidence_source_type = unresolved_table` and leave `evidence_text` empty.

Reason
- The first Layer 3 workbook seed showed a systemic evidence-binding failure:
  `evidence_source_type = table` was often paired with unrelated paragraph text.
- Root cause: `build_field_gt_review_workbook_v1.py` labeled fields with
  per-field evidence-region metadata but always populated `evidence_text` via
  `resolve_text_evidence()`, causing paragraph reuse across many table rows.
- For Layer 3 human GT work, correctness and auditability are more important
  than evidence recall. Table-derived fields must not silently fall back to
  generic text.

Rule
- For Layer 3 workbook export only:
  - table-derived field -> matching row-local table evidence or `unresolved_table`
  - no paragraph fallback for table-derived fields
  - text-derived field -> prefer sentence-level text containing the extracted value

Traceability
- Updated workbook builder:
  - `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- Authoritative rerun:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/13_layer3_field_gt_evidence_binding_fix_refresh/run_20260318_1457_22e713d_dev15_layer3_field_gt_evidence_binding_fix_refresh_no_llm_v1/RUN_CONTEXT.md`

### Decision: Promote the Layer 3 field-review workbook to v2 for usability and explicit evidence-support status

Decision
- Keep the original Stage 5 `formulation_id` in the Layer 3 workbook, but add
  two reviewer-facing helper identity columns:
  - `formulation_label_stage5`
  - `formulation_label_params`
- Treat this as a material workbook-surface change and emit the refreshed human
  review artifact as `v2`.
- Keep the authoritative artifact inside lineage, but shorten the review path by
  using a shorter child folder and review subdirectory rather than creating a
  detached convenience copy.

Reason
- Reviewers need a readable formulation handle without losing the canonical
  frozen Stage 5 identifier.
- Blank extracted values must not retain inherited evidence text.
- Unsupported extracted values should be flagged explicitly rather than paired
  with misleading evidence spans.

Rule
- For Layer 3 workbook export:
  - blank extracted value -> `evidence_source_type = blank_value` and blank
    `evidence_text`
  - extracted value with no valid text support -> `evidence_source_type = unsupported_text`
  - extracted value with no valid row-local table support -> `evidence_source_type = unresolved_table`
- The workbook remains review-surface-only and does not change benchmark-facing
  Stage 5 rows.

Traceability
- Updated workbook builder:
  - `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- Updated Layer 3 protocol:
  - `docs/methods/layer3_field_gt_protocol_v1.md`
- Authoritative v2 run:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/15_l3gtv2/run_20260318_1521_22e713d_dev15_l3gtv2_v1/RUN_CONTEXT.md`

## 2026-03-19

### Decision: Use conflict-aware Stage 2 instance-kind reconciliation instead of blind trust or blanket downgrade

Decision
- Stage 2 now applies a narrow conflict-aware reconciliation step after basic
  candidate canonicalization and before downstream materialization.
- The reconciler uses existing Stage 2 signals only:
  - raw `instance_kind`
  - `change_role`
  - `formulation_role`
  - `instance_context_tags` / `change_context_tags`
  - `parent_instance_id`
  - populated formulation-identity fields
  - polymer identity / polymer name
- The reconciler supports both directions:
  - rescue GT-valid family-member rows that were typed as
    `candidate_non_formulation` despite strong formulation-family identity
  - downgrade helper/comparative/non-synthesis rows that were typed as
    formulation-facing rows despite weak formulation identity
- No new paper-specific extraction rules are introduced by this decision.

Reason
- Recent reviewed-boundary diagnostics showed three recurring failure patterns:
  - `BXCV5XWB` false exclusion of GT-valid blank / FITC family-member rows
  - `QLYKLPKT` leakage of helper / comparative dosage rows into final output
  - `UFXX9WXE` leakage of characterization/helper formulation rows into final
    output
- Blind trust in raw explicit `instance_kind` is too permissive for helper or
  comparative rows.
- Blanket downgrade based only on `non_synthesis` or
  `characterization_only` would incorrectly suppress legitimate formulation
  family members.

Rule
- Do not auto-demote a row only because it carries `non_synthesis`,
  `characterization_only`, or helper-style tags.
- Do not auto-keep a row only because the raw LLM emitted
  `new_formulation` / `variant_formulation`.
- Reconcile only when the evidence conflict is clear:
  - rescue when strong formulation-family identity is present
  - downgrade when helper/comparative signals are strong and formulation
    identity is weak

Auditability
- Preserve lightweight audit fields in Stage 2 outputs:
  - `instance_kind_raw`
  - `instance_kind_inferred`
  - `instance_kind_reconciliation_note`
- Mark reconciled rows with the `instance_kind_reconciled` context tag rather
  than redesigning the whole schema.

Scope and limitations
- This is a Stage 2-only typing/gating reconciliation step.
- It does not change Stage 3 relation semantics or Stage 5 materialization
  semantics.
- It is intentionally narrow and pattern-based; duplicate alias rows and other
  non-typing failure modes may still require separate work.

### Decision: Canonical Polymer MW Field Migration and Materials-Priority Evidence Packing

Decision
- The canonical molecular-weight field is now `polymer_mw_kDa`.
- `plga_mw_kDa` is retained as a legacy read alias only for transition compatibility.
- This is a canonical naming correction only. The scientific meaning of the field did not change.
- The relation-first architecture is unchanged:
  - Stage 2 remains semantic extraction
  - Stage 3 remains deterministic relation materialization
  - Stage 5 remains materialization-only

Reason
- The active project logic already treats polymer identity as general and LA/GA ratio as conditional.
- Keeping a PLGA-shaped canonical MW name creates avoidable schema and prompt bias for non-PLGA polymers such as PCL.
- The canonical field identity must match the actual project semantics before more DEV15 relation-first review and Layer 3 audit work accumulates.

Impact
- New active outputs should prefer `polymer_mw_kDa` wherever the canonical field is written.
- Old artifacts that still carry `plga_mw_kDa` must remain readable through explicit compatibility mapping.
- No relation-first logic, boundary logic, or Stage 5 semantic behavior changed as part of this naming correction.

### Evidence Packing Adjustment

Decision
- Add `materials_procurement` as an explicit Stage 2 evidence-pack block type.
- The effective evidence-pack priority is now:
  - `metadata`
  - `synthesis_method`
  - `materials_procurement`
  - `table`
  - `caption`
  - `paragraph`

Reason
- Materials/procurement paragraphs often carry shared/global formulation parameters such as polymer identity, molecular weight, grade, and supplier-coded product names.
- Those shared parameters should appear earlier in the LLM input than generic tables, captions, or body paragraphs.

Impact
- This change affects Stage 2 LLM input ordering only.
- It does not change the extraction schema, formulation boundary semantics, relation-first contract, or Stage 5 behavior.

### Explicit Non-Changes

- No donor-fill reintroduction.
- No Stage 5 inference.
- No same-paper donor heuristics.
- No formulation boundary logic changes.

### Decision: Allow narrow Stage 5 descriptive-field inheritance for sparse same-paper sweep rows

Decision
- For Stage 5 final-row assembly, allow a narrow inheritance step for sparse
  rows when a missing descriptive field can be filled from an unambiguous
  compatible donor row elsewhere in the same paper.
- Current inherited field bundles are limited to:
  - `plga_mw_kDa`
  - `surfactant_name`
  - `organic_solvent`
  - `emul_method`

Reason
- Some retained `figure_variable_sweep` rows carry only the varied axis plus
  polymer identity while the same paper already contains richer non-sweep rows
  with the shared polymer-branch metadata.
- The active Stage 5 summary already states that broad scientific inheritance
  repair is intentionally out of scope, so the fix must remain narrow,
  deterministic, and unambiguous.

Rule
- Only fill a field bundle when:
  - the target row is blank for that field bundle
  - the donor is a richer non-sweep row from the same paper
  - polymer branch compatibility is satisfied
  - exactly one normalized donor value exists across the compatible donor set
- Do not fill figure-only numeric outputs such as `size_nm` or
  `encapsulation_efficiency_percent` from trend-only or unsupported evidence.

Traceability
- Updated Stage 5 script:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Focused regression run:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/16_5gif_field_fill/run_20260318_1549_22e713d_5gif3d8w_stage5_field_fill_regression_no_llm_v1/RUN_CONTEXT.md`

### Decision: Stage 5 Identity Contract (DEV15_v2)

Decision
- Stage 5 now applies an explicit identity-constraints layer as part of the
  benchmark-facing `DEV15_v2` final formulation closure contract.
- A benchmark-facing formulation identity is defined as an independently
  reported formulation instance, not merely any row that references a
  formulation family, a shared preparation block, or a comparative context.
- Stage 5 may exclude rows that fail this identity definition when the
  exclusion can be justified using existing deterministic lineage and context
  signals from Stage 2.

Identity definition
- Include a row in the benchmark-facing final formulation table when it
  represents an independently reported formulation instance with its own
  synthesis-defining identity.
- Exclude a row when it only references an already-defined formulation identity
  or only summarizes shared/comparative context without introducing an
  independent formulation instance.

Rule 1: parent-linked non-synthesis descendant suppression
- Exclude a row when all of the following are true:
  - `parent_instance_id` is present
  - `change_role = non_synthesis`
  - `instance_kind = variant_formulation`
  - and the row is explicitly in control, characterization, post-processing, or
    downstream evaluation context
- Conceptual meaning:
  - a parent-linked non-synthesis descendant is a formulation-referencing
    variant, not a new benchmark-facing formulation identity

Rule 2: unparented shared/comparative summary suppression
- Exclude a row when it is unparented and is explicitly marked as either:
  - a shared-condition summary block
  - or a comparative-study summary reference without independent formulation
    identity
- Conceptual meaning:
  - shared preparation summaries and comparative summary references are
    evidence/context surfaces, not independently reported formulation
    instances

Scope
- These are Stage 5 identity constraints, not Stage 2 extraction changes.
- They do not redesign DOE closure, family-boundary closure, or Stage 3
  relation construction.
- They are part of the benchmark-facing `DEV15_v2` contract and should be
  treated as contract-level behavior rather than temporary experiment filters.

## 2026-03-20

### Decision: Layer 2 Risk Stratification Contract For DEV15_v2 GT Review

Decision
- The system now distinguishes benchmark-valid final-output comparison from
  downstream audit-risk stratification.
- A paper-level Layer 2 risk artifact is now part of the supported
  post-comparison metadata layer:
  - `analysis/paper_risk_assessment.tsv`
  - `analysis/paper_risk_assessment_summary.md`
- This risk layer is output-layer metadata only.
- It must not change Stage 2 extraction semantics, Stage 3 relation semantics,
  Stage 5 identity closure, or benchmark-valid final-table counts.

Reason
- Full DEV15_v2 comparison showed that Layer 2 does not need zero residual
  mismatch to allow safe Layer 3 field-level GT audit.
- Small residual differences such as isolated `+1` extra rows can be tolerable
  when they do not imply large-scale identity failure.
- Downstream Layer 3 review still needs a durable way to distinguish
  low-residual papers from papers with likely batch over-generation or
  meaningful missing identities.

Contract
- Risk labels are deterministic and paper-level.
- Required fields include:
  - paper key / DOI
  - matched / extra / missing counts
  - total mismatch
  - mismatch ratio within paper
  - `paper_risk_level`
  - `risk_source`
  - `layer3_inclusion_flag`
  - short rationale
- Initial deterministic policy:
  - `LOW`: `extra_count <= 1` and `missing_count == 0`
  - `HIGH`: `extra_count > 3` or `missing_count >= 2`
  - `MEDIUM`: all remaining non-LOW and non-HIGH papers
  - `INCLUDE` for `LOW`, `REVIEW` for `MEDIUM`, `HOLD` for `HIGH`

Allowed `risk_source` values
- `stage2_over_generation`
- `stage2_under_generation`
- `stage5_over_retention`
- `mixed`
- `unknown`

Usage rule
- This contract is for audit stratification only.
- It must not be used to delete high-risk papers from benchmark outputs.
- It exists so downstream Layer 3 field-level GT interpretation remains valid,
  explainable, and reproducible while pipeline semantics remain frozen.

### Decision: Layer 3 Field GT Workbook Contract Refresh For DEV15_v2

Decision
- The active Layer 3 workbook contract remains anchored to:
  - `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
  - `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- The workbook remains a downstream manual-audit surface built from frozen
  Stage 5 final rows.
- The workbook may now carry Layer 2 paper-risk metadata from
  `analysis/paper_risk_assessment.tsv` so field-level review can distinguish
  low-risk papers from papers that should be reviewed more cautiously.

Reason
- Full DEV15_v2 comparison established that Layer 2 is no longer a zero-
  residual gate for Layer 3 field audit.
- Reviewers still need a durable paper-level warning surface while inspecting
  field-level values, inheritance, normalization, and derivations.
- This metadata belongs in the review layer, not in Stage 2, Stage 3, or Stage
  5 benchmark-valid semantics.

Contract
- The Layer 3 workbook is review support only.
- It must not change formulation identity, row counts, benchmark scores, or
  Stage 5 inclusion decisions.
- Reviewer-facing workbook columns should stay compact and preserve:
  - frozen formulation identity
  - readable helper labels
  - extracted value and evidence support
  - manual GT decision columns
  - optional paper-level risk metadata
- Companion outputs remain:
  - `final_formulation_table_audit_ready_v1.tsv`
  - `field_gt_review_seed_rows_v*.tsv`
  - `field_gt_review_source_summary_v*.tsv`
  - `field_gt_review_workbook_v*.xlsx`

### Decision: run_id Appears Only At Run Root For New Outputs

Decision
- For new outputs and future runs, `run_id` appears exactly once at the run
  root directory.
- Artifact subdirectories below that run root must be functional only and must
  not repeat:
  - the full `run_id`
  - timestamp/hash fragments derived from the `run_id`

Reason
- Repeating run-like identifiers below the run root creates unnecessary path
  inflation and makes artifact paths harder to read, compare, and audit.
- Functional artifact names are sufficient once the enclosing run root already
  provides uniqueness.

Scope
- This rule applies prospectively only.
- Historical run directories and existing lineage layouts are not renamed by
  this decision.
- If a unit needs to be independently rerunnable or lineage-addressable, it
  must be created as a separate run root with its own `run_id`, not as a
  nested artifact folder inside an existing run.

Examples
- Preferred:
  - `data/results/<run_id>/analysis/...`
  - `data/results/<run_id>/audit/...`
  - `data/results/<run_id>/fgt_v3_dev15_v2/...`
- Not allowed for new outputs:
  - `data/results/<run_id>/run_20260320_1317_ab12cd3_dev15_compare/...`
  - `data/results/<run_id>/compare_20260320_1317_ab12cd3/...`

### Decision: Layer 3 Workbook Usability Patch Preserves Frozen Semantics

Decision
- The active Layer 3 field-review workbook and audit-ready export may add
  reviewer-facing usability metadata without changing any frozen Stage 2, Stage
  3, or Stage 5 semantics.
- The current usability patch adds:
  - article-native formulation identifiers as additional reviewer columns
  - separate evidence-anchor carry-through when direct support is missing
  - explicit review warnings for polymer-grade text carried in molecular-weight
    text fields
  - explicit `normalization_pending` markers for raw-mass concentration rows

Reason
- Manual Layer 3 audit needs article-native labels, provenance anchors, and
  normalization warnings to review fields efficiently.
- These aids belong in the workbook/export surface only and must not be
  confused with benchmark-valid identity or benchmark-valid value changes.

Scope
- Canonical system identity remains:
  - `formulation_id`
  - `formulation_label_stage5`
- Article-native identifiers are additive reviewer aids only.
- `evidence_text` remains reserved for direct supporting evidence.
- `evidence_anchor_text`, `evidence_status_detail`, `review_warning`, and
  `normalization_status` are downstream audit metadata only.
- No Stage 5 row membership, Stage 5 identity constraints, benchmark-valid
  counts, or benchmark-valid comparison artifacts change under this decision.

### Decision: Repair the Layer 3 evidence handoff contract without changing frozen semantics

Decision
- The Layer 3 field-review workbook must continue to enforce the strict
  direct-support policy, but it must no longer surface broad row-level
  fallback spans as reviewer evidence anchors when no field-local relationship
  exists.
- Structured fields such as `LA/GA` and `polymer_MW` must use field-aware
  support checks in the workbook/export surface so unrelated numeric text does
  not count as direct support.
- When a frozen final-row field is marked `relation_resolved`, the Layer 3
  seed/reference export must carry the Stage 3 relation-resolution provenance
  as reviewer metadata instead of reusing representative row-level spans as if
  they were direct supporting evidence.

Reason
- The current frozen final table remains benchmark-valid, but the reviewer
  workbook was recomputing evidence support locally from row-level evidence with
  weaker logic than the prior numeric/evidence hardening path.
- This produced two audit-surface failures:
  - misleading `evidence_anchor_text` from broad row spans
  - false support on structured numeric-like fields from unrelated numeric text

Impact
- Stage 2 extraction semantics are unchanged.
- Stage 3 relation semantics are unchanged.
- Stage 5 identity logic, final row membership, and benchmark-valid outputs are
  unchanged.
- The repair applies only to Layer 3 reviewer-facing workbook/export artifacts
  and seed/reference metadata.

### Decision: Promote the Layer 3 Evidence Handoff Contract to a first-class functional unit

Decision
- The Layer 3 Evidence Handoff Contract is now a first-class reviewer-surface
  functional unit in the repository.
- It is defined as a durable contract, not as an ad hoc workbook heuristic.
- It must remain regression-protected so future workbook/export changes cannot
  silently weaken evidence support behavior.

Reason
- Strong evidence-binding safeguards already existed in the repository, but the
  active Layer 3 workbook path previously allowed weaker local heuristics and
  broad row-level anchor carry-through to override that intent.
- That mismatch was subtle and reviewer-facing, which made it vulnerable to
  future silent degradation unless made explicit and testable.

Contractual resolution
- The contract is defined in:
  - `docs/methods/layer3_field_gt_protocol_v1.md`
- Golden regression cases live in:
  - `docs/methods/layer3_evidence_handoff_golden_cases_v1.tsv`
- Minimal validation mechanism:
  - `src/stage5_benchmark/validate_layer3_evidence_contract_v1.py`

Hard rule
- Layer 3 reviewer exports may add reviewer metadata, but they must not:
  - downgrade stronger upstream evidence/QC behavior
  - surface non-local row text as field evidence
  - mark structured fields as supported using generic numeric-token overlap

Impact
- This adds durability and traceability only.
- Stage 2 extraction semantics are unchanged.
- Stage 3 relation semantics are unchanged.
- Stage 5 identity logic, benchmark-valid outputs, and final table membership
  are unchanged.
- Future Layer 3 workbook/export changes must validate against the golden
  cases before they should be considered contract-compliant.

## 2026-03-21

### Decision: Tighten the Stage5 descendant filter so ambiguous sweep-style variants are not auto-suppressed

Decision
- Keep the Stage5 `parent_linked_non_synthesis_descendant_variant` rule for
  obvious non-benchmark descendants.
- Narrow the rule so `post_processing` alone is not sufficient when a
  parent-linked row is still a sweep-style `variant_formulation` carrying
  paper-local formulation-member identity signals.
- Those ambiguous rows must fall through to the existing conservative
  variant-governance path, where they can still be retained as
  `kept_uncertain_variant_no_signature` if no unique safe collapse target is
  found.

Regression being fixed
- `BB3JUVW7` / `10.1016/j.ijpharm.2021.120820`
- Valid benchmark-facing formulation identities `F2.2`, `F2.4`, `F2.5`,
  `F2.6`, and `F2.7` were filtered in Stage5 after the early
  `parent_linked_non_synthesis_descendant_variant` branch was added.

Root cause
- The early Stage5 descendant filter matched:
  - `instance_kind = variant_formulation`
  - non-empty `parent_instance_id`
  - `change_role = non_synthesis`
  - overlap with post-processing-style context tags
- That rule ran before the older uncertain-variant fallback, so rows with
  benchmark-facing sweep identity never reached conservative variant review.
- The prior rule was too broad because `post_processing` was treated as a
  universal exclusion signal even for table-native sweep members that the
  paper still reports as distinct formulations.

New intended behavior
- Still filter parent-linked descendants when they are clearly:
  - `control`
  - `characterization_only`
  - or downstream measurement / PK / in-vivo rows
- Do not auto-filter parent-linked rows solely because `post_processing`
  appears when the row remains a sweep-style `variant` formulation member.

Regression coverage added
- New deterministic regression checker:
  - `src/stage5_benchmark/validate_stage5_descendant_filter_regression_v1.py`
- Coverage includes:
  - `BB3JUVW7` must retain `12` benchmark-facing final rows, including the five
    restored `F2.x` rows
  - known non-benchmark descendant rows in existing blocker material must still
    be filtered
  - `WIVUCMYG` must remain unchanged at the final-count level

Impact
- This is a minimal Stage5 logic repair only.
- Stage2 extraction semantics are unchanged.
- Stage3 relation semantics are unchanged.
- Workbook, audit, and schema-v2 layers are unchanged.
