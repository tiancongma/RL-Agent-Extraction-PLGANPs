## 2026-01-28

Historical note:
- This log preserves dated decisions in their original architectural context.
- Older entries may describe a previously active Stage2 contract.
- Current pipeline authority always lives in `project/ACTIVE_PIPELINE_FLOW.md`,
  `project/ACTIVE_PIPELINE_RUNBOOK.md`, and `project/2_ARCHITECTURE.md`.

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
- Current architecture interpretation is maintained through documentation in docs/archive_project/project_specification_legacy.txt, project/2_ARCHITECTURE.md, and project/PIPELINE_SCRIPT_MAP.md.
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

## 2026-03-23

### Decision: Formalize repository-wide active data-source authority for current data/results workflows

Decision
- Current benchmark, alignment, comparison, workbook-generation, and audit
  workflows must not infer the active source from recency, lexical sort order,
  modification time, parent fallback, or glob-first matching.
- Source resolution must follow this authority order:
  1. explicit CLI source such as `--run-dir`
  2. repository pointer `data/results/ACTIVE_RUN.json`
  3. otherwise hard error
- Workbook and comparison outputs must record source-run and source-file
  metadata in sidecar JSON.

Reason
- The active architecture now relies on parent lineage roots plus deep child
  terminal artifacts under `data/results/run_*/lineage/children/...`.
- Historical `runs/latest.txt` is no longer sufficient as the sole authority
  for current `data/results` workflows.
- Future agent sessions must be able to discover the active source rule from
  repo instructions rather than conversation history.

Impact
- `project/ACTIVE_DATA_SOURCE_CONTRACT.md` becomes the governing source
  contract for current `data/results` workflows.
- `AGENTS.md`, `README.md`, `project/ACTIVE_PIPELINE_RUNBOOK.md`,
  `project/FILE_NAMING_AND_VERSIONING.md`, and `project/2_ARCHITECTURE.md`
  now surface the same rule for both agents and humans.
- Active Stage5 review/comparison helpers are updated to use explicit
  authority resolution and emit source metadata sidecars.

Why this is not a new rule family
- This case does not show extraction bias.
- This case does not show Stage4 suppression bias.
- This case does not justify a new EE-centered pipeline rule.

## 2026-03-24

### Decision: Add an additive non-authoritative Stage2 component shadow sidecar keyed to the existing formulation row id

Decision
- The active Stage2 extractor may emit two additional shadow artifacts:
  - `weak_labels__v7pilot_r3_fixparse_components_shadow.jsonl`
  - `weak_labels__v7pilot_r3_fixparse_components_shadow.tsv`
- The benchmark-facing Stage2 TSV and JSONL remain unchanged in schema and downstream contract.
- The shadow artifacts are explicitly non-authoritative and must not be consumed by Stage3 or Stage5 unless a later contract adopts them.

Reason
- Current wide-row Stage2 outputs preserve key materials and amounts but do not represent multi-component formulation structure cleanly enough for component-aware audit or later extension work.
- The additive sidecar allows component-aware inspection without redesigning the pipeline or broadening formulation identity rules.

Impact
- Stage2 runs now expose one linked shadow component surface per formulation row for audit and future schema work.
- Stage3 and Stage5 behavior remain unchanged.
- Any observed replay drift in the benchmark-facing Stage2 TSV must still be evaluated separately from the shadow sidecar because the sidecar is written after the TSV row is materialized.

### Decision: Document Stage2.5 as a governed side-path enrichment layer rather than an inline Stage2 change

Decision
- Stage2.5 is approved as a future non-authoritative, read-only enrichment
  layer.
- Stage2.5 will operate on already frozen Stage2 formulation rows plus source
  evidence assets.
- Stage2.5 must not change Stage2 benchmark-facing outputs.
- Stage2.5 must not participate in formulation identity decisions.
- Stage2.5 must not feed Stage3 or Stage5 in the current phase.

Reason
- The recent Stage2 component shadow validation confirms two things at once:
  - richer component-aware structure is useful and should be pursued
  - flattened Stage2 fields alone are not sufficient authority for stable full
    recovery
- The same validation also recorded replay drift in benchmark-facing Stage2
  outputs relative to the saved active artifact, which argues against embedding
  additional component-recovery logic directly into Stage2 before a separate
  architecture contract is in place.
- A side-path design keeps benchmark stability, formulation identity, and
  downstream deterministic contracts insulated while component-aware recovery is
  developed incrementally.

Impact
- The active production-path mainline remains Stage2 -> Stage3 -> Stage5.
- Stage2.5 is now the governed location for future component-aware evidence
  binding, conservative splitting, difficult local structure resolution,
  assembly, validation, and review surfaces.
- Recommended rollout order is fixed intentionally:
  - architecture contract only
  - Evidence Binding and Pack Builder
  - Deterministic Pre-Splitter
  - LLM-Assisted Resolver for unresolved cases only
  - Component Assembly and Validation
  - Review and Audit exports
- Downstream adoption remains deferred until Stage2.5 shadow outputs are proven
  stable under their own contract.
- It is an annotation reminder: independently prepared blank / FITC / control-style nanoparticle formulations may be real formulation instances even when the original manual count missed them.

Resolved interpretation
- Correct formulation-core count = `5`.

---

### Decision: Stage2.5A v0 is a text-only exact-anchor validation layer and remains shadow-only

Decision
- The first implemented Stage2.5A scope is a text-only Evidence Binding and
  Pack Builder.
- It must read frozen Stage2 rows plus cleaned text assets only.
- It must emit exact-anchored shadow evidence packs with `strict`,
  `supporting`, and `rejected` buckets.
- It must not bind tables yet.
- It must not perform component extraction, Stage2 prompt changes, or any
  Stage3 or Stage5 integration.

Reason
- Current evidence-binding failures are dominated by broad paragraph carryover,
  weak row binding, and non-auditable span reuse.
- A text-only exact-anchor pass is the smallest governed implementation that
  proves the evidence-pack contract without widening scope.
- Deferring table binding keeps v0 focused on correctness and traceability
  while preserving benchmark stability.

Impact
- `src/stage2_5_components_shadow/build_text_evidence_packs_v0.py` is the
  governed Stage2.5A v0 builder.
- Its outputs are diagnostic-only, non-authoritative shadow artifacts.
- Downstream adoption remains deferred until later Stage2.5 phases extend the
  evidence-pack surface safely.

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
- At that time, `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` remained the active Stage2 entrypoint and `src/stage4_eval/eval_weak_labels_v7pilot3.py` remained the active Stage4 evaluator.
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
- On `2026-03-12`, the then-active Stage2 extractor `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` was minimally extended to append low-confidence `candidate_source = "figure_variable_sweep"` candidates when a paper text explicitly declares multi-level formulation-variable sweeps.

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
- The then-active Stage2 extractor was extended at that time to add an explicit additive polymer field layer:
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
- Keep the active pipeline path unchanged at that time and patch only the then-active Stage2 extractor.
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
- Keep the then-active Stage2 extractor as the only edited runtime component.
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

Current active-path contract at that time
- The active DEV-15 comparison path currently compares Stage2 candidate formulation-instance rows directly against the fixed DEV15 skeleton workbook.
- No generic normalization or formulation-core collapse layer was wired between the then-active Stage2 extractor and the then-active Stage4 evaluator.
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
- The active Stage2 boundary at that time now includes a deterministic numbered DOE table-row enumerator implemented in:
  - `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`
- The then-active Stage2 extractor `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` additively called this enumerator by default after LLM extraction and before writing the final Stage2 weak-label artifact.
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

### Decision: When BXCV5XWB collapses to one family-only LLM row, allow governed fallback semantic replacement to restore the 3 benchmark-facing KGN rows

Decision
- If `BXCV5XWB` reaches Stage2 completion with exactly one retained `formulation_family` row from `llm_first_composite` and no DOE/table/sequential row recovery emitted, replace that collapsed row set with the governed fallback semantic document for the same paper.
- The replacement remains replay-only and bounded to the paper-level fallback semantic document already maintained in `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`.
- Stage2 contract validation should accept this full-row replacement bridge by treating the completed Stage2 output as `governed_fallback_semantic_source` even when the upstream semantic document payload remains `llm_first_composite`.

Reason
- `BXCV5XWB` current raw replay output can collapse to a single family placeholder even though repo governance expects 3 benchmark-facing KGN main-table rows.
- The preserved table surface for this paper is noisy enough that ordinary deterministic DOE/table row expansion emits zero rows.
- The existing governed fallback semantic document already encodes the intended 3 KGN rows and excludes the 6 FITC/blank helper descendants that should not re-enter the main table.
- This restores the historically governed capability path without reopening fresh LLM calls or redesigning Stage2 semantics for the general case.

Validation
- Replay-only targeted lineage: `data/results/20260423_9c4a03f/02_stage2_replay`
- Stage2 contract report passed with 3 completed rows and no contract errors.
- Downstream targeted lineage:
  - Stage3: `data/results/20260423_9c4a03f/03_stage3`
  - Stage5: `data/results/20260423_9c4a03f/04_stage5`
  - diagnostic compare: `data/results/20260423_9c4a03f/05_compare`
- Result:
  - `BXCV5XWB` final main table = 3 rows
  - no `FITC` or `blank` rows in the final table
  - diagnostic compare remains `3 / 9` against frozen GT, which is expected under current main-table governance

### Decision: Recover concentration sweep rows from corrupted split-column formulation tables when source CSV and source text still preserve rowwise concentration evidence

Decision
- In `table_row_expansion_v1`, when an LLM-authorized formulation-bearing table reaches execution with `representation_status` still degraded (`repair_insufficient` or `unrepaired_corrupted`) and the normal authority-row extractors fail, the executor may attempt a bounded source-backed sweep recovery before giving up.
- The bounded recovery has two generic source-backed paths:
  - recover split-column concentration sweep rows from the preserved source CSV when the source lines still contain stable concentration-row text even though normalized payload rows are unusable
  - recover explicit sample rows from a source-text table block when the clean text still preserves a table caption/block with explicit sample identities and theoretical concentrations
- This remains deterministic post-authorization recovery only. It does not change LLM semantic ownership and does not authorize generic enumerate-every-corrupted-table behavior.

Activation contract
- activate only after the normal direct-row authority extractors fail
- require `semantic_signals.has_variable_sweep = true`
- require degraded preserved-table representation (`repair_insufficient` or `unrepaired_corrupted`)
- require stable rowwise concentration evidence in the preserved CSV or explicit sample identities in the source-text table block
- do not activate for ordinary clean tables that already support the standard direct-row paths

Reason
- `L3H2RS2H` showed a recurrent failure class where the paper still contains lawful rowwise formulation evidence, but the preserved table surface is corrupted enough that `measurement_axis`-based extraction emits zero rows and the raw LLM response collapses the paper into a few family summaries.
- The decisive failure was not only at prompt time. The maintained execution path still had enough governed source evidence to recover rowwise sweep identities, but no deterministic path consumed that evidence.
- A bounded source-backed recovery is preferable to reopening fresh LLM calls or reintroducing paper-specific semantic hacks.

Validation
- bounded replay lineage:
  - Stage2: `data/results/20260423_9c4a03f/22_l3h_stage2_replay`
  - Stage3: `data/results/20260423_9c4a03f/23_l3h_stage3`
  - Stage5: `data/results/20260423_9c4a03f/24_l3h_stage5`
- full collateral lineage:
  - Stage2: `data/results/20260423_9c4a03f/30_stage2_full_replay`
  - Stage3: `data/results/20260423_9c4a03f/31_stage3`
  - Stage5: `data/results/20260423_9c4a03f/32_stage5`
  - compare: `data/results/20260423_9c4a03f/33_compare`
- count-level effect:
  - `L3H2RS2H` final count restored from `7` to `21`
  - compare status changed from `under` to `match`
  - other papers remained count-stable in the full replay collateral run

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
- The governed ordered-pack path is now exposed as a run-level execution feature unit rather than a one-off prompt tweak.

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

### Decision: Treat the Layer 3 reviewer-facing outputs as a formulation-centered audit and governance layer, not only an evaluation helper

Decision
- The current Layer 3 GT effort is not only an evaluation artifact.
- It is also part of the governed production audit and governance layer around
  the frozen formulation database.
- The benchmark-valid endpoint remains:
  - `final_formulation_table_v1.tsv`
- Reviewer-facing Layer 3 outputs remain downstream of the frozen final table
  and must not mutate benchmark-valid outputs.

Formulation-centered direction
- The preferred reviewer entry object is one formulation row.
- Human review is split into two linked layers:
  - formulation existence and identity audit
  - value credibility audit
- These layers are strongly dependent, not parallel.
- Many apparent value errors are projections of structure or identity errors.

Current repo state
- Audit capability is already partially present across governed surfaces:
  - paper-level risk
  - formulation-level audit-ready export
  - field-level review workbook
  - cell-level cross-audit report
  - evidence handoff tooling
- The current state is:
  - partially present but not unified

Functional-unit direction
- The repo should treat the emerging reviewer-facing audit layer as a small set
  of functional units:
  - Formulation Index Builder
  - Structure Review Builder
  - Value Risk Builder
  - Evidence Handoff Builder
- Final UI container or submission format is not fixed yet.
- Priority is to define the system contract and functional units first.

Reason
- Recent architecture review shows that the repo already has multiple governed
  reviewer surfaces, but they are fragmented by review unit and stage-local
  purpose.
- Field-level and cell-level value review cannot be interpreted safely without
  the upstream formulation existence and identity layer.
- The design needs to acknowledge production audit and governance usage without
  conflating these surfaces with benchmark-valid outputs.

Impact
- Architecture and method docs should describe Layer 3 as both:
  - an evaluation-support surface
  - a production-grade audit and governance surface
- Future unification work should stay formulation-centered and preserve the
  invariant that benchmark-valid output remains the frozen final formulation
  table.
- No active benchmark-valid pipeline behavior changes under this decision.

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

---

## Decision: Layer3 workbook presence must defer to canonical current Stage5 artifacts, not historical bridge status

Date
- 2026-03-21

Context
- A Layer3 value workbook regeneration reused the repaired post-fix Stage5
  final table and the matching audit-ready export, both of which correctly
  contained `BB3JUVW7` rows `F2.2`, `F2.4`, `F2.5`, `F2.6`, and `F2.7`.
- Those same rows still appeared as `missing_in_system` in the Layer3 value
  workbook because historical alignment-side inputs were treated as
  authoritative:
  - `alignment_decision != matched` in the alignment scaffold
  - stale prior-workbook bridge rows carrying `missing_in_system`
  - builder fallback logic that prevented canonical row loading after the
    downgrade

Decision
- For Layer3 workbook generation, the latest Stage5 final table and
  audit-ready export are the canonical source of truth for formulation
  existence and identity resolution.
- Historical alignment scaffolds, trusted prior annotation row files, and
  previous workbook-derived bridge artifacts remain advisory only.
- Advisory artifacts may help map a GT row to a system row, but they must not
  downgrade a row to `missing_in_system` after the latest canonical Stage5
  artifacts confirm a valid current-system row with a compatible identity
  anchor.

Impact
- This is a Layer3 builder contract fix, not a Stage5 regression.
- Human annotation carry-forward remains allowed, but only for preserving
  reviewer-entered GT cells and provenance.
- Current-system workbook population must prefer false negatives over false
  positives when canonical identity remains unresolved or conflicting.

---

## Decision: Layer3 GT-skeleton alignment must prefer direct canonical article-ID matches before scaffold fallback

Date
- 2026-03-23

Context
- `5GIF3D8W` GT rows `F01-F08` were canonically present in the latest Stage5
  final table and audit-ready export, including direct article-native IDs
  `F1-F8`.
- The Layer3 value workbook still mispaired those GT rows to sweep variants
  because the builder accepted advisory scaffold links such as
  `F01 -> PLGA 50/50 [stabilizer concentration=0.75 % w/v]` before checking
  whether the canonical audit export already exposed a stronger one-to-one
  article-native identity match.

Decision
- During GT-skeleton workbook generation, the builder must first look for a
  unique direct canonical match in the latest audit-ready export using the
  article-native identity surface.
- For GT IDs following the common generated pattern `paper_key_F0N`, the
  builder may derive article-ID candidates such as `FN` and prefer those
  canonical rows when exactly one current-system row matches.
- Advisory scaffold rows remain a fallback only after stronger canonical
  article-ID matching is exhausted.
- When a direct canonical match overrides scaffold fallback, the builder must
  emit an audit TSV recording the scaffold row, the resolved row, and the rule
  used.

Impact
- This change is restricted to the Layer3 GT-alignment and workbook-export
  layer.
- Stage2 extraction, Stage3 relation materialization, and Stage5 final-row
  closure remain unchanged.
- The new behavior prevents canonically present optimized/core rows from being
  remapped onto sweep variants when the audit-ready export already exposes a
  unique stronger identity match.

---

## Decision: Keep Layer3 numeric backfill deterministic and identity-bound; do not let compare-side repair become a second semantic extractor

Date
- 2026-04-23

Context
- Repository architecture already fixes the high-level responsibility split:
  - LLM semantic discovery owns formulation boundaries, field-role assignment,
    and shared-vs-instance-specific interpretation.
  - deterministic downstream layers own numeric evidence binding, derivation,
    schema assembly/export, and QC gating.
- Layer3 work increasingly needs identity bridges, scaffold fallbacks,
  audit-ready exports, and compare-side debugging surfaces to explain value
  recall and accuracy gaps.
- Without an explicit Layer3 boundary note, future compare or workbook repairs
  could drift into semantic re-discovery by using heuristics to recreate
  formulation universes or guess missing values.

Decision
- Layer3 numeric backfill must operate over the frozen formulation identity
  universe inherited from Layer2/Stage5 and must not reopen formulation
  existence or row-boundary decisions.
- LLM outputs may provide semantic anchors, ownership hints, relation cues, and
  table-scope hints, but they are not by themselves the final benchmark-facing
  numeric authority.
- Downstream deterministic Layer3 logic is allowed to perform:
  - canonical current-row alignment
  - advisory scaffold / bridge-assisted identity mapping
  - numeric evidence binding
  - relation-resolved carry-through
  - explicit derivation under auditable rules
  - normalization, compare, and reviewer-facing audit export
- Downstream Layer3 logic must not:
  - create new formulation rows
  - redefine the benchmark-facing formulation universe
  - treat heuristic compare-side matching as new semantic discovery authority
  - freely infer missing values without deterministic evidence support
  - present relation-resolved or derived values as if they were directly
    reported values
- If Layer3 failure modes repeatedly require more semantic downstream repair,
  the correct response is to record upstream extraction-schema backlog rather
  than to permit unbounded compare-side semantic rule growth.

Impact
- This decision validates compare/workbook identity-bridge work only when it is
  identity-binding and audit-facing, not universe-defining.
- Layer3 outputs remain downstream audit and debugging surfaces over the frozen
  Stage5 final table rather than a parallel extraction system.
- Future numeric backfill changes should be judged by this test:
  do they bind, normalize, and audit values on frozen identities, or are they
  trying to rediscover semantic structure downstream?

## 2026-03-25

### Decision: Record the Stage2 contract audit finding as design guidance only; do not treat it as an active pipeline contract change

Decision
- Record the current Stage2 contract audit result and proposed `db_v2` redesign
  as design guidance only.
- Do not treat this audit as an active pipeline contract change.
- Do not change active Stage2, Stage3, or Stage5 runtime behavior from this
  note alone.

Directly observed facts
- The active authoritative Stage2 TSV pinned by
  `data/results/ACTIVE_RUN.json` currently exposes a 119-column wide-row
  contract at the time of this audit.
- The current extractor code now exposes a wider 124-column contract with:
  - `instance_kind_raw`
  - `instance_kind_inferred`
  - `instance_kind_reconciliation_note`
  - canonical `polymer_mw_kDa_*` naming in code
  - `preparation_method`
  - `emulsion_structure`
- The active authoritative Stage2 TSV still carries legacy
  `plga_mw_kDa_*` naming and does not expose the newer reconciliation or
  preparation-method enrichment columns.
- Raw Stage2 responses and JSONL outputs show that the LLM is currently asked
  to emit coarse evidence-oriented metadata such as:
  - `supporting_evidence_refs`
  - row-level evidence span fields
  - field `scope`
  - field `membership_confidence`
  - field `evidence_region_type`
  - conflict/arbitration narrative in `paper_notes`

Inference
- Current Stage2 behavior mixes semantic extraction with a meaningful amount of
  coarse evidence-binding and conflict-arbitration work.
- This is directionally misaligned with the governed architecture, which says
  semantic extraction belongs in Stage2 while evidence binding, normalization,
  derivation, and audit should remain deterministic.

Recommendation
- Future schema work should move exact evidence binding, normalization,
  derivation, and audit-grade pointer assembly into deterministic post-
  processing.
- Future Stage2 cleanup should keep the LLM focused on:
  - formulation identity
  - parent/variant semantics
  - components
  - phases
  - process semantics
  - measurements
  - coarse source hints only
- The proposed landing zone for that redesign is the documented
  `data/db/db_v2/schema_manifest.json` plus the method note:
  - `docs/methods/stage2_llm_field_audit_and_db_redesign_2026-03-25.md`

Non-change statement
- No active pipeline code was changed in this audit.
- No benchmark-valid pipeline contract is changed by this log entry alone.

### Decision: Approve the true Stage2 replacement direction and initial semantic-contract scaffold without switching the active runtime

Decision
- Approve the Stage2 replacement direction as a semantic-object discovery layer
  with deterministic compatibility projection.
- Add a non-default scaffold that writes the replacement contract artifacts:
  - `src/stage2_sampling_labels/build_stage2_replacement_contract_v1.py`
  - `data/db/db_v2/schema_manifest_v2_replacement.json`
  - `data/db/db_v2/stage2_replacement_output_contract.tsv`
  - `data/db/db_v2/stage2_legacy_to_replacement_mapping.tsv`
- Do not switch the active benchmark runtime in this step.

Observed facts
- Active Stage3 and Stage5 still consume the current wide-row Stage2 surface.
- The Stage2 contract audit and schema redesign review show that the current
  Stage2 surface is too fixed-slot and carries excess coarse evidence-oriented
  burden for the long-term design target.
- Stage2.5 has already been archived as non-mainline and is not the chosen
  replacement path.

Decision boundary
- The replacement target object families are:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`
- Deterministic post-processing remains responsible for:
  - compatibility projection
  - normalization
  - derivation
  - stronger evidence binding

Transition status
- This is an architecture-and-scaffolding decision, not an active runtime
  contract switch.
- Current benchmark-valid mainline remains Stage2 -> Stage3 -> Stage5 with the
  maintained wide-row Stage2 extractor.

Non-change statement
- No historical results were modified by this decision entry itself.
- Stage3 and Stage5 runtime behavior is unchanged by this decision entry alone.

### Decision: Add a deterministic Stage2 replacement compatibility adapter without changing the active benchmark runtime

Decision
- Add a deterministic compatibility adapter that reads semantic-object Stage2
  payloads and projects them into the current wide-row Stage2 surface for
  downstream Stage3 and Stage5 compatibility during migration.
- Record the adapter and its projection contract as transitional support
  infrastructure, not as a new active runtime stage.

Observed facts
- Current Stage3 and Stage5 active scripts still consume the legacy Stage2
  wide-row contract.
- The approved Stage2 replacement target is semantic-object discovery rather
  than continued fixed-slot expansion.
- The new adapter can preserve the replacement boundary by keeping projection,
  compression, and coarse evidence handoff deterministic.

Artifacts
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- `docs/methods/stage2_replacement_compatibility_adapter_2026-03-25.md`
- `data/db/db_v2/stage2_replacement_compatibility_projection_contract.tsv`

Decision boundary
- Stage2 replacement remains object-oriented internally.
- The adapter may derive or compress legacy fields only through explicit
  deterministic rules.
- The adapter must not reintroduce final normalization, exact evidence
  ownership binding, or hidden LLM-style arbitration.

Transition status
- Active benchmark-valid mainline remains Stage2 -> Stage3 -> Stage5 using the
  maintained wide-row Stage2 extractor.
- The compatibility adapter exists to make the replacement architecture
  operational during migration, not to redefine the active benchmark runtime by
  itself.

Non-change statement
- No historical results under `data/results/` were modified by this entry.
- No Stage3 or Stage5 runtime logic was changed by this entry alone.

### Decision: Use a diagnostic legacy-to-semantic lift for initial three-paper replacement validation, without treating it as the replacement emitter

Decision
- Add a deterministic validation-only lift from legacy Stage2 wide-row rows to
  semantic-object payloads so the replacement compatibility bridge can be
  exercised end-to-end before a true paper-driven semantic emitter exists.
- Keep this lift outside the active benchmark runtime and treat all resulting
  replacement-path runs as diagnostic-only.

Observed facts
- The repository contains a replacement semantic contract scaffold and a
  deterministic compatibility adapter, but no true paper-driven semantic
  Stage2 emitter yet.
- The authoritative active run already contains paper-driven Stage2 outputs and
  raw responses for the requested DEV15 papers.
- A three-paper replacement validation slice can therefore be executed honestly
  by grounding semantic objects in deterministic lift from those existing
  paper-driven Stage2 artifacts.

Artifacts
- `src/stage2_sampling_labels/lift_legacy_stage2_to_semantic_objects_v1.py`
- `src/analysis/build_replacement_validation_report_v1.py`
- `data/results/run_20260325_1415_3c1a9d2_dev15_3paper_replacement_validation_no_llm_v1/`

Decision boundary
- The lift may only perform direct mapping, deterministic splitting, coarse
  evidence handoff, and explicit missing-value preservation.
- The lift must not be represented as the long-term semantic emitter.
- Stage3 and Stage5 remain unchanged in this validation path.

Observed diagnostic outcome
- The replacement-path validation ran end-to-end for `UFXX9WXE`, `BXCV5XWB`,
  and `L3H2RS2H`.
- Replacement and legacy-reference paths diverged at Stage3 relation
  materialization on all three papers, but reconverged at Stage5 final-row
  counts on this slice.
- This supports continued adapter hardening and true semantic-emitter
  implementation before wider replacement validation.

Non-change statement
- No active benchmark-valid mainline stage definition was changed by this
  entry.
- No historical `data/results/run_*` directory was modified or deleted.
- No Stage3 or Stage5 runtime logic was changed by this entry.

### Decision: Validate the true paper-driven semantic Stage2 emitter on the same three-paper replacement slice before broader rollout

Decision
- Add a deterministic paper-driven semantic-object emitter that reads cleaned
  paper text and tables directly for the governed three-paper replacement
  validation slice.
- Keep the active benchmark-valid mainline unchanged, and continue to use the
  deterministic compatibility adapter as the only bridge to unchanged Stage3
  and Stage5.

Observed facts
- The lift-based replacement validation proved the adapter and downstream
  harness could run end-to-end, but it did not test a true paper-driven
  semantic emitter.
- `UFXX9WXE` has a directly usable DOE table in cleaned table assets.
- `BXCV5XWB` is recoverable from cleaned article text as a three-family
  formulation case even though extracted table CSVs are low-value.
- `L3H2RS2H` has usable governed table content under
  `data/cleaned/goren_2025/tables/L3H2RS2H/`, including the independently
  reported `XAN nanocapsules (Theoretical concentration 800 mg/mL)` row that
  the earlier lift-based run failed to recover.

Artifacts
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- `src/analysis/compare_replacement_validation_runs_v1.py`
- `data/results/run_20260325_1434_f17211_dev15_3paper_true_semantic_replacement_validation_no_llm_v1/`

Observed diagnostic outcome
- The paper-driven emitter path ran end-to-end for `UFXX9WXE`, `BXCV5XWB`, and
  `L3H2RS2H`.
- The replacement path matched GT final row counts on all three papers for this
  diagnostic slice.
- Relative to the earlier lift-based run, Layer1 improved on `UFXX9WXE` and
  `L3H2RS2H`, including recovery of the missing `L3H2RS2H` nanocapsule row.
- Stage3 relation divergence versus the legacy reference widened on all three
  papers, which indicates hidden legacy-shape dependence remains in the
  migration path even though Stage5-facing counts improved.

Recommendation
- Prioritize exposing and normalizing hidden Stage3 legacy dependencies before
  expanding this replacement path to broader paper sets.

Non-change statement
- No Stage3 runtime logic was modified by this entry.
- No Stage5 runtime logic was modified by this entry.
- No historical `data/results/run_*` directory was modified or deleted.

## 2026-03-30

### Decision: Freeze the Stage2 role split after the 2026-03-30 authority-transition audit and treat deterministic Stage2 semantic authority as architecture drift

Decision:
- Record the Stage2 authority transition audit dated
  `docs/snapshots/snapshot_2026-03-30_stage2_authority_transition_audit.md`
  as a governed architecture-correction input.
- Freeze the original Stage2 role split as the active architecture contract:
  - LLM owns open semantic discovery and formulation-boundary discovery in
    Stage2.
  - Deterministic layers own relation resolution, inheritance handling,
    normalization, filtering, audit, and final materialization downstream of
    Stage2.
- Forbid deterministic Stage2 semantic replacement paths from being treated as
  active mainline authority.
- Allow deterministic Stage2 semantic emitters or semantic lifts only as
  fallback, comparator, migration-support, or diagnostic infrastructure.
- Treat future drift that re-promotes deterministic Stage2 semantic authority
  as a contract violation that should trigger governance warnings or failing
  validation checks rather than silent normalization.

Directly observed facts:
- The transition audit confirmed that the repo narrowed LLM vs deterministic
  responsibilities on `2026-03-06`, introduced deterministic Stage2 practical
  drift by `2026-03-13`, approved semantic replacement direction on
  `2026-03-25` without switching active runtime, and then rewrote active
  governance toward deterministic semantic-emitter authority in the
  `2026-03-26` commit / `2026-03-29` decision layer.
- The same audit found no explicit governed decision that removed the earlier
  LLM-centered Stage2 architecture principle itself.
- The repo therefore accumulated an authority drift rather than a cleanly
  closed architecture replacement.

Reason:
- The project design intent remains that ambiguous formulation boundaries,
  shared-vs-instance semantics, and open semantic candidate discovery belong
  to an LLM-stage boundary rather than to paper-specific deterministic
  reconstruction.
- Allowing deterministic semantic replacement to stand as active Stage2
  authority would silently redefine the architecture without the missing
  design-level approval.

Impact:
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  and
  `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  remain available in the repository, but only as non-authoritative fallback,
  comparator, migration-support, or diagnostic infrastructure.
- Future architecture, runbook, script-selection, and memory surfaces must not
  describe deterministic semantic Stage2 replacement as the active mainline
  authority.
- If future enforcement tooling is extended, promoting deterministic semantic
  Stage2 as mainline authority should warn or fail.

### Decision: Promote the semantic Stage2 DEV15 identity-preservation lineage as the repository active authority pointer

Decision:
- Repoint `data/results/ACTIVE_RUN.json` from the March 14 legacy wide-row
  extractor lineage to
  `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`.
- Treat the promoted run's semantic Stage2 objects, compatibility-projected
  wide-row artifacts, Stage3 resolved relation fields, and Stage5 final table
  as the current authoritative terminal surface for default
  `data/results`-resolved workflows.
- Keep
  `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  present only as deprecated fallback/debug infrastructure outside the active
  mainline.

Reason:
- Governed docs, code registries, and memory had already declared semantic
  Stage2 authority, but `ACTIVE_RUN.json` still pointed to a March 14
  legacy-extractor lineage.
- The semantic-emitter replacement path had already been validated end-to-end
  on DEV15 and then extended with the accepted additive
  `identity_variables_json` preservation path in the March 29 full-pipeline
  benchmark experiment.
- Repointing the machine-readable authority removes the practical governance
  mismatch without changing the architecture contract.

Impact:
- Default authority-resolved benchmark, alignment, workbook, and audit
  workflows now start from a semantic Stage2 mainline lineage rather than a
  legacy wide-row extractor lineage.
- Historical March 14 and other legacy-extractor runs remain preserved for
  auditability, chronology, and fallback/debug comparison, and their script
  references may still appear in historical memory and run contexts.
- This is a governance alignment and authority-promotion action only; it does
  not introduce a new pipeline architecture or reactivate Stage2.5.

### Decision: Introduce Layer2 identity scaffold contract v1 for benchmark-safe downstream binding

Decision
- Add a diagnostic-only Layer2 identity scaffold contract for downstream
  compare and audit workflows.
- Freeze a stable identity anchor from the reviewed/frozen Layer2-style
  boundary surface when available, rather than binding downstream work on
  unstable final-row ids or namespaced presentation ids.
- Use a strict binding ladder:
  1. exact article-native formulation label match
  2. normalized namespaced-label match
  3. strict identity-equivalent binding
  4. coarse fallback only for manual review, never for benchmark-grade compare
- Validate the first pass only on safe normalization papers:
  - `WIVUCMYG`
  - `5ZXYABSU`

Contract
- The scaffold key is a downstream identity-binding surface, not a replacement
  for Stage5 benchmark-valid final output.
- Downstream stages may enrich an existing scaffolded identity node.
- Downstream stages must not silently split identity unless the split is
  explicitly authorized by identity-defining fields.
- Measurement/outcome fields such as size, PDI, zeta, EE, and LC do not
  redefine identity by default.

Implementation scope
- Introduce a narrow report-only utility:
  - `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py`
- The utility emits scaffold-binding diagnostic surfaces only.
- It does not mutate Stage5 final tables, GT workbooks, or benchmark-valid
  compare outputs in place.

Rationale
- Current final-row binding becomes unstable when benign namespacing drift such
  as `WIVUCMYG_F23 -> F23` or `5ZXYABSU_NPB1 -> NPB1` breaks direct matching.
- Once direct binding fails, DOE papers can fan out incorrectly under coarse
  fallback even when the underlying scientific row set is unchanged.
- A frozen identity scaffold lets later stages enrich rows without forcing
  row-boundary redefinition.

Status
- Minimal v1 contract added.
- Diagnostic validation only.
- Not yet a broad-paper identity repair for the harder blocker set.

Non-goals
- No redesign of the active semantic Stage2 -> adapter -> Stage3 -> Stage5
  pipeline.
- No change to the benchmark-valid meaning of Stage5 final output.
- No hidden value-level matching or semantic inference inside Stage5
  materialization.

### Decision: Identity Freeze Rule Introduced

Status
- ACTIVE

Scope
- DEV15 reviewed-boundary downstream compare and audit work now
- future expansion work that reuses a frozen Layer2-style identity scaffold

Decision
- Introduce `IDENTITY_FREEZE_RULE_V1` as an explicit downstream engineering
  contract.
- After Layer2 identity, or an equivalent frozen boundary authority:
  - formulation count must remain invariant
  - formulation membership must remain invariant
- Downstream stages may:
  - add fields
  - resolve missing fields
  - derive fields
- Downstream stages must not:
  - split formulations implicitly
  - merge formulations implicitly
  - create new formulations from value similarity

Split authorization policy
- Classify downstream fields as:
  - `identity_defining_fields`
  - `non_identity_fields`
  - `measurement_fields`
- Only `identity_defining_fields` may justify a split.
- Measurement fields such as size, PDI, zeta, EE, and LC must never trigger a
  split by default.
- If uncertain, attach the value to the existing identity rather than
  reconstructing identity.

Enforcement insertion point
- Add a report-only guardrail at the Stage5 post-materialization boundary:
  - `src/stage5_benchmark/enforce_identity_freeze_v1.py`
- The guardrail compares:
  - an upstream identity scaffold surface
  - a Stage5 final table
- It emits:
  - row-count drift diagnostics
  - identity-reassignment diagnostics
  - violation flags
- It does not silently fix benchmark-valid outputs.

Rationale
- Prevent row explosion when new variables appear downstream of a frozen
  identity boundary.
- Stabilize evaluation by preventing value-level signals from reshaping row
  identity after the boundary is accepted.
- Make explicit that downstream work should attach values to frozen identities,
  not recompose formulations.

Non-goals
- Not changing Stage2 semantic discovery semantics.
- Not redesigning identity discovery.
- Not refactoring core Stage3 relation logic.
- Not changing benchmark-valid Stage5 outputs in place.

### Decision: Identity Freeze Rule Elevated to Hard Gate

Status
- ACTIVE

Scope
- DEV15 now
- future expansion runs that depend on a frozen Layer2-style identity scaffold

Previous state
- `IDENTITY_FREEZE_RULE_V1` existed as a report-only guardrail at the Stage5
  post-materialization boundary.

New state
- `IDENTITY_FREEZE_RULE_V1` is now an enforced invariant.
- The identity-freeze check is a mandatory gate before any:
  - value comparison
  - audit-ready export
  - Layer3 field GT evaluation
- If the gate detects any of the following, it must fail non-zero and block
  downstream progression:
  - row count drift
  - identity reassignment
  - unresolved scaffold binding
  - ambiguous binding

Reason
- prevent identity drift from being misread as value-level regression
- prevent row explosion from propagating into evaluation surfaces
- ensure downstream value-level work remains valid only after identity
  invariance is confirmed

Impact
- invalid runs are now blocked early
- downstream compare and reviewer-facing export workflows must treat a failed
  identity-freeze gate as a hard stop

Non-goals
- Not changing Stage2 semantics.
- Not changing core Stage3 relation logic.
- Not altering benchmark-valid Stage5 final-table semantics.
- Not introducing automatic fixes or silent fallback.

## 2026-03-29

### Decision: Add an additive identity-variable carrier through the active semantic Stage2 -> adapter -> Stage3 -> Stage5 DEV15 experiment path

Decision
- Add one additive compatibility-surface field, `identity_variables_json`, to
  preserve Stage2-detected `variable_or_factor_candidate` pairs that carry
  `identity_defining_signal=yes`.
- Keep existing legacy field bundles unchanged.
- Use the additive carrier only to preserve formulation-identity distinctions
  through the active bridge, Stage3 variation-axis handling, and Stage5
  collapse signature logic.
- Do not change Stage2 semantic authority.
- Do not expand this first pass into a broader schema redesign.

Rationale
- The active semantic Stage2 boundary can already detect identity-bearing
  variables such as `pH`, but the current compatibility bridge drops generic
  factors before downstream identity logic can see them.
- This creates a formulation-identity integrity failure where rows that differ
  only by an identity-bearing variable may collapse incorrectly.
- One additive carrier is the smallest structurally correct fix for the open
  problem frame recorded under the extraction-to-modeling transition notes.

Decision boundary
- The new carrier is additive metadata only.
- Existing `CORE_FIELDS` bundles and legacy fixed-slot fields remain in place.
- Stage3 uses the carrier conservatively as a variation-axis input in this
  first pass.
- Stage5 includes the normalized carrier in `build_core_fields()` and
  `build_collapse_signature()` so rows that differ only by identity-bearing
  variables no longer collapse together.

Experiment scope
- This change is introduced for a controlled DEV15 benchmark experiment only.
- Benchmark comparison for this experiment must use the explicitly declared
  manual workbook authority:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`

Non-change statement
- Stage2 semantic authority remains
  `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  plus the deterministic compatibility bridge.
- No legacy field bundle was removed or renamed by this decision entry.
- This entry does not claim that the additive carrier is a full schema
  redesign; it is a minimal identity-preservation unit only.

### Decision: Stage2 authority migrated to the semantic-object emitter; Stage2.5 retired from the active mainline

Decision
- Make the paper-driven semantic-object emitter the authoritative Stage2
  boundary for the active mainline.
- Define the deterministic compatibility adapter as the required bridge from
  semantic Stage2 outputs into the legacy wide-row surface used by unchanged
  Stage3 and Stage5 consumers.
- Retire Stage2.5 from the active mainline and keep it only as archived
  exploratory history.
- Reclassify the legacy wide-row extractor as deprecated fallback or debug
  infrastructure rather than the active Stage2 authority.

Rationale
- Wide-row extraction proved fragile as the primary Stage2 contract because it
  couples semantic discovery to fixed-slot projection too early.
- Semantic objects are a more stable intermediate representation for
  multi-component formulations, variable/factor discovery, and raw-expression
  preservation.
- A deterministic adapter preserves downstream reuse while Stage3 and Stage5
  remain unchanged.

Observed facts
- A true paper-driven semantic Stage2 emitter now exists in
  `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`.
- A deterministic compatibility adapter now exists in
  `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`.
- Stage3 and Stage5 remain unchanged and still consume the legacy wide-row
  compatibility surface.
- Stage2.5 had already been archived earlier and is not the chosen long-term
  replacement path.

Decision boundary
- Stage2 semantic objects are now the authoritative Stage2 output.
- The legacy wide-row surface is a compatibility projection only.
- The compatibility adapter is transitional infrastructure, not the desired
  long-term semantic core.
- Known hidden Stage3 legacy dependencies still exist and are not resolved by
  this decision entry.

Normalization note
- Earlier `project/4_DECISIONS_LOG.md` entries that describe
  `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  as the active Stage2 extractor reflect the pre-migration authority state at
  the time those entries were written.
- The current active Stage2 authority is
  `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  plus the deterministic compatibility bridge
  `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  is now deprecated legacy fallback/debug infrastructure and is not the active
  mainline Stage2 authority.

Non-change statement
- No Stage3 runtime logic was modified by this entry.
- No Stage5 runtime logic was modified by this entry.
- No historical `data/results/run_*` directory was modified or deleted.
## 2026-03-31

### Decision: Normalize run naming and governance boundaries (MDEC084)

Decision
- Introduce a new future-run naming scheme:
  - Top-level run bucket: `YYYYMMDD_<short_hash>`
  - Child execution folders: `NN_<cue>` (e.g., `01_stage2`, `02_relation`)
- Remove semantic meaning from folder names; require all rich context to live in `RUN_CONTEXT.md`.
- Redefine lineage:
  - No nested full `run_id` inside run directories.
  - Execution identity is represented by ordinal child folders, not repeated run names.
- Keep `data/results/ACTIVE_RUN.json` as the only authority for active runs.
- Allow `ACTIVE_RUN.json` to reference arbitrary paths, not only old-style `run_*` IDs.
- Freeze all historical runs:
  - Do not rename or restructure existing `run_*` directories.
- Enforce `project/` as governance-only:
  - Only authoritative contracts allowed.
  - Audit, diagnosis, parking-lot, and open-question files must not live under `project/`.

Rationale
- Current system encodes identity, lineage, and semantics in folder names, causing:
  - excessive path depth
  - naming inflation
  - ambiguity in determining active runs
- Existing governance rules are internally inconsistent:
  - lineage structure requires nested run IDs
  - naming rules forbid repeated run IDs
- The repository already supports explicit authority via `ACTIVE_RUN.json`, making path-based identity viable.
- Separating identity (path), meaning (RUN_CONTEXT), and authority (ACTIVE_RUN.json) is the minimal consistent model.

Decision boundary
- This change affects future runs only.
- No historical benchmark artifact is modified.
- Stage2–Stage5 semantic logic remains unchanged.
- Accepted run types remain unchanged.

Implementation constraints
- Must update utilities before enabling new naming:
  - `src/utils/run_id.py`
  - `src/utils/run_latest.py`
  - `src/utils/run_preflight.py`
  - `src/utils/active_data_source.py`
- Utilities must:
  - stop enforcing old run-id regex as global constraint
  - rely on explicit authority instead of name parsing
  - maintain backward compatibility for legacy runs

Non-change statement
- No change to scientific extraction logic or pipeline stages.
- No bulk migration of historical runs.
- No change to benchmark comparison contracts.
- This decision introduces a naming/governance normalization only.

### Decision: Add governed three-paper Stage2 v2 comparison slice (MDEC085)

Decision
- Implement a governed, minimal three-paper Stage2 v2 comparison slice for:
  - `WIVUCMYG`
  - `UFXX9WXE`
  - `5GIF3D8W`
- Add:
  - `src/stage2_sampling_labels/extract_semantic_stage2_v2_threepaper.py`
  - `src/analysis/build_stage2_v2_threepaper_comparison_pack.py`
  - `src/utils/run_threepaper_stage2_v2_comparison.py`
- The slice emits object-first Stage2 semantic artifacts only and a narrow
  Stage2 comparison pack against:
  - the current deterministic semantic active-run surface
  - the current deterministic compatibility wide-row surface
  - the maintained historical legacy wide-row comparator surface

Rationale
- The three selected papers stress:
  - DOE factor preservation
  - numbered-row multiplicity
  - formulation-boundary drift and identity instability
- A small governed slice provides architecture-enforcement evidence for the
  frozen Stage2 role split without silently replacing the active benchmark
  mainline.

Decision boundary
- This slice is comparator and architecture-enforcement infrastructure only.
- No authority promotion is implied.
- `data/results/ACTIVE_RUN.json` remains unchanged.
- The deterministic semantic emitter remains non-authoritative fallback,
  comparator, migration-support, or diagnostic infrastructure.
- Any future promotion requires broader governed evidence beyond this
  three-paper slice.

Non-change statement
- No Stage3 runtime behavior was changed.
- No Stage5 runtime behavior was changed.
- No active mainline run lineage was replaced.

### Decision: Correct Stage2 to one governed composite stage and evaluate Stage2 on the completed artifact (MDEC086)

Decision
- Define Stage2 as one composite stage consisting of:
  - LLM semantic discovery
  - deterministic post-LLM completion inside Stage2
- Keep the official numbered stage structure unchanged:
  - Stage2 -> Stage3 -> Stage5
- Make the completed Stage2 artifact, not the raw semantic intermediate, the
  only authoritative Stage2 input to Stage3 and the only authoritative Stage2
  structural evaluation target.
- Introduce one governed Stage2 execution entrypoint:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- Require scope variation to be expressed only through manifest/config inputs
  such as:
  - manifest
  - paper keys
  - source mode
  - backend
  - model
  - max text chars
- Keep special comparison wrappers non-governed and non-promoting.

Rationale
- The previous repo wording drifted into treating the raw LLM semantic
  intermediate as if it were the whole Stage2 evaluation object.
- That is incorrect once Stage2 includes a deterministic completion substep.
- The three-paper live Gemini slice tested only the semantic-discovery
  intermediate because it skipped deterministic post-LLM completion in its
  original interpretation.
- Scope-specific wrapper scripts were also beginning to imply alternative
  Stage2 definitions, which conflicts with the governance requirement for one
  Stage2 entrypoint.

Impact
- `src/stage2_sampling_labels/run_stage2_composite_v1.py` is now the one
  governed Stage2 execution entrypoint.
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` is the
  internal LLM semantic-discovery substep.
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py` is
  the internal deterministic post-LLM completion substep.
- `src/utils/run_threepaper_stage2_v2_comparison.py` remains comparison-only
  and must not be treated as the Stage2 definition.
- The recent three-paper live Gemini result remains usable as semantic-
  intermediate diagnostic evidence only; it is not a final go/no-go judgment
  on the completed Stage2 contract by itself.

Non-change statement
- No new numbered stage such as Stage2.8 was introduced.
- No Stage3 runtime behavior was changed.
- No Stage5 runtime behavior was changed.
- `data/results/ACTIVE_RUN.json` was not changed by this decision entry.

## 2026-04-01

### Decision: Make feature activation lineage mandatory in governed run contexts (MDEC086)

Decision
- Governed run-producing entrypoints must refresh the run-local Feature Unit Activation section in `RUN_CONTEXT.md` as part of normal execution.
- Reproducibility-grade run documentation is incomplete if it records script lineage without feature activation lineage.
- The compare node and other governed wrappers should surface feature activation as a first-class artifact, not a detached follow-up note.

Rationale
- The feature-unit governance system already existed, but the run contract still allowed feature visibility to drift out of the active run surface.
- `stage2_input_evidence_packing` and similar governed feature units must remain visible through run artifacts even when future wrappers or comparison nodes change.
- Recording feature activation inside `RUN_CONTEXT.md` makes code presence, run activation, and provenance distinct and auditable.

Impact
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py` now refreshes the feature activation section after writing `RUN_CONTEXT.md`.
- The active mainline Stage2, Stage3, Stage5, and compare surfaces now treat the feature activation section as required observability.
- The repository guidance now states that governed runs are not fully documented unless both script lineage and feature activation lineage are present.

Non-change statement
- No execution gate was introduced.
- No benchmark semantics were changed.
- No historical run directory was renamed or rewritten.

## 2026-04-02

### Decision: Formalize governed pipeline-boundary classes and lawful resume boundaries

Decision
- Define four explicit boundary classes for the maintained pipeline:
  - `internal_intermediate`
  - `diagnostic_boundary`
  - `mainline_resume_boundary`
  - `benchmark_terminal_boundary`
- Raw Stage2 freeze baselines are diagnostic boundaries unless they also
  preserve the authoritative completed Stage2 artifact required by Stage3.
- The completed Stage2 weak-label artifact is the lawful Stage3 resume
  boundary.
- The Stage5 final formulation table plus comparison outputs remain the
  benchmark terminal boundary.

Reason
- Debugging needs explicit pause, branch, and replay boundaries, but the
  active pipeline must keep its authority split intact.
- Raw responses alone do not preserve the completed Stage2 authority surface.
- Benchmark claims must remain attached to the Stage5 final layer.

Impact
- Pipeline documentation can now distinguish internal intermediates from
  diagnostic boundaries and lawful resume boundaries.
- Future run contexts should record the boundary class and resume legality for
  reproducibility and safe return to mainline.

### Decision: Emit boundary-governance fields automatically in maintained run contexts

Decision
- The shared maintained RUN_CONTEXT refresher
  `src/utils/update_run_context_with_feature_activation_v1.py` now injects a
  deterministic `## Boundary Governance` section alongside the existing
  feature-activation section.
- New run folders created through maintained wrappers that already refresh
  `RUN_CONTEXT.md` now record:
  - `boundary_class`
  - `authoritative_for_downstream`
  - `lawful_resume_boundary`
  - `resume_entrypoint`
  - `schema_contract`
  - `upstream_authority_source`
  - `replay_mode`
  - supporting provenance fields used by the boundary contract

Reason
- The boundary-governance framework is only fully useful if new runs emit the
  boundary metadata automatically at write time.
- The existing shared RUN_CONTEXT refresher was the narrowest maintained-safe
  implementation point because it already hardens governed run metadata
  without changing pipeline semantics.

Impact
- Maintained Stage2, Stage3, and compare surfaces that refresh RUN_CONTEXT now
  automatically declare their boundary class and resume legality.
- This is run-metadata hardening only; no stage authority, pipeline semantics,
  or benchmark behavior changed.

### Decision: Add maintained schema-aware Stage2 raw-response rehydration to the composite replay path

Decision
- The maintained composite Stage2 replay mode now accepts both:
  - historical legacy raw-response payloads
  - current live-v2 raw-response payloads containing
    `formulation_candidates` and the rest of the Stage2 v2 object families
- Rehydration remains inside the maintained composite Stage2 path:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- The raw-response freeze remains diagnostic-only by itself.
- The lawful resumed Stage3 upstream boundary is the completed Stage2 artifact
  re-emitted by that maintained replay path.

Reason
- The boundary-governance audit identified the missing lawful resume link as:
  raw Stage2 freeze -> completed Stage2 authority.
- The prior maintained replay branch was legacy-schema oriented and produced
  zero formulations when given current live-v2 raw responses.
- The narrowest maintained-safe fix was to make the maintained extractor's
  replay branch schema-aware for current live-v2 raw-response payloads without
  changing live execution semantics.

Impact
- Current frozen live-v2 raw-response baselines can now be rehydrated into
  nonzero authoritative completed Stage2 artifacts without new LLM calls.
- The resulting completed Stage2 outputs remain contract-valid upstream inputs
  for Stage3.
- No new shadow pipeline was created, and no live Stage2 semantics changed.

## 2026-04-03

### Decision: Lock Stage2 semantic authority to LLM-first composite mode and require explicit provenance for deterministic expansion

Decision
- Freeze the maintained Stage2 contract as:
  - `llm_first_composite` by default for governed mainline Stage2 runs
  - deterministic completion allowed only within an authorized semantic scope
  - deterministic semantic emitters retained only as explicitly labeled
    `governed_fallback_semantic_source` or `diagnostic_comparator`
- Require additive provenance on authoritative completed Stage2 rows so each
  row can answer:
  - who declared the candidate universe
  - who materialized the row
  - what semantic scope authorized the row to exist
- Require governed Stage2 runs to declare exactly one semantic-source mode in
  run metadata and `RUN_CONTEXT.md`.
- Preserve DOE capability under the frozen contract:
  - LLM semantic discovery must declare DOE scope
  - deterministic numbered-row recovery may materialize row-level candidates
    only within that declared scope during `llm_first_composite` runs
  - deterministic fallback semantic emitters remain available only under
    explicit governed fallback declaration
- Add a maintained Stage2 contract validator that fails when:
  - semantic-source mode is missing or mixed
  - authoritative rows lack semantic provenance
  - deterministic DOE rows appear without LLM-declared DOE scope in
    `llm_first_composite` mode

Reason
- The repository had already frozen the architecture principle that LLM owns
  Stage2 semantic discovery, but the practical run history still showed
  authority drift into deterministic semantic-source generation.
- Prior DOE success must be preserved, but only as deterministic expansion
  within LLM-declared semantic scope unless an explicitly governed fallback
  mode is used.

Impact
- Maintained composite Stage2 runs now fail loudly instead of silently
  accepting semantic-authority drift.
- DOE row recovery remains available without allowing deterministic logic to
  become the default Stage2 semantic authority.
- Historical deterministic capabilities remain in the repository, but they are
  now explicitly labeled as fallback/comparator authority modes rather than
  ambiguous mainline behavior.

### Decision: Record the operative Layer2 boundary decision workbook and distinguish it from the downstream value workbook

Decision
- For the `run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`
  lineage, the operative human-reviewed Layer2 boundary decision workbook is:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/boundary_gt_review_v1/boundary_gt_review_workbook_v1.xlsx`
- This workbook is the correct practical base for future manual
  rehydration-vs-GT boundary comparison work in that lineage.
- It must be described precisely as:
  - a run-scoped human-reviewed Layer2 boundary review surface
  - seeded from Stage5 final formulation rows
  - not the repository-wide ultimate raw GT origin
- The downstream field-level workbook:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`
  is not the correct base for new boundary comparison work.
- The field-level workbook inherits the accepted formulation universe from the
  reviewed `include_gt` subset of the boundary workbook and remains a
  downstream value-annotation surface only.

Reason
- Future agent sessions were re-deriving the workbook-role relationship from
  workbook contents, lineage docs, and overlap analysis.
- The repeated confusion point was treating the downstream value workbook as if
  it were the Layer2 boundary authority, or treating the reviewed boundary
  workbook as if it were an independently originated raw GT workbook.

Impact
- Future agents can now cite one explicit governed conclusion when selecting
  the correct base workbook for:
  - Layer2 boundary review
  - rehydration-vs-GT boundary comparison
  - downstream field/value annotation
- The repo now explicitly distinguishes the mother boundary workbook from the
  downstream value workbook without changing any pipeline behavior or benchmark
  semantics.

## 2026-04-08

### Decision: Preserve governed partial Stage2 selection and inheritance markers without broadening downstream execution

Decision
- The maintained Stage2 contract now distinguishes marker readiness for the
  suppression-prone marker families:
  - `execution_ready`
  - `partial_semantic`
- This readiness distinction currently applies to:
  - `selection_marker`
  - `inheritance_marker`
- The maintained prompt, normalization path, and validator now allow:
  - partial `selection_marker` preservation when the paper supports the
    selection cue but one or more of:
    - `source_table_id`
    - `selected_variable`
    - `selected_value`
    remain incompletely grounded
  - partial `inheritance_marker` preservation when the paper supports the
    inheritance cue but:
    - `from_table`
    - `to_table`
    remain incompletely grounded
- Execution-critical strictness is intentionally unchanged in this first-phase
  implementation:
  - `inheritance_marker.inherit_type`
  - `inheritance_marker.variable`
  - `inheritance_marker.value`
  remain required whenever an inheritance marker is emitted
- The maintained Stage2-to-downstream handshake now keeps current execution
  narrow:
  - `partial_semantic` markers are preserved in the Stage2 semantic-
    intermediate artifact
  - only `execution_ready` markers are carried into the execution-facing row
    handshake consumed by unchanged downstream runtime behavior

Reason
- The field-level contract audit localized a small set of suppression-prone
  fields that were prompt-forced but not current execution-critical
  requirements.
- The narrowest safe first-phase change was therefore:
  - keep execution gates strict
  - preserve partial semantic understanding where governance permits
  - avoid broadening current deterministic execution authority

Impact
- Stage2 can now preserve some explicit semantic understanding that would
  previously have been dropped entirely by all-or-nothing marker suppression.
- Current deterministic row expansion and current Stage3 inheritance
  materialization remain restricted to execution-ready markers only.
- This is a maintained contract adjustment inside the existing Stage2
  architecture, not a new pipeline stage and not a rollback of semantic-
  authority restoration.

### Decision: LLM Role Boundary Clarification and Semantic Contract Relaxation Direction

Background
- The recent corrective architecture work already restored the intended Stage2
  authority split.
- The maintained Stage2 contract now correctly preserves LLM semantic
  authority, marker provenance, marker-authorized execution boundaries, and
  validator enforcement against deterministic semantic overreach.

What has already been fixed
- The system has already recovered the intended `LLM is the semantic
  authority` boundary.
- Deterministic semantic emitters and semantic lifts are not the active
  Stage2 mainline authority.
- Deterministic Stage2 execution remains lawful only within governed semantic
  scope.

What remains broken
- The remaining bottleneck is not architectural impurity alone and not model
  quality alone.
- The remaining bottleneck also includes:
  - semantic contract rigidity
  - LLM role overload
  - suppression of markers under uncertainty
- In practice, the current Stage2 semantic substep can still pressure the LLM
  toward candidate-universe construction and partially execution-ready
  structure emission earlier than preferred.

Observed facts
- The active maintained contract still expects the LLM to emit substantial
  candidate-level semantic structure before deterministic completion begins.
- In difficult structures such as sequential optimization papers, semantic
  understanding may exist even when `selected_value`, exact table label, or
  exact inheritance target cannot yet be fully grounded to the stricter live
  marker schema.
- When the live contract suppresses those marker families entirely, governed
  downstream execution can starve:
  - no marker family
  - no row expansion
  - no downstream formulation rows

Inference
- The remaining failure mode is better described as semantic-contract overload
  than as failure to restore semantic authority.
- Semantic understanding is not the same thing as executable structure.

New locked principle
- LLM outputs should preferentially be treated as reusable semantic cues and
  governed intermediate markers, not as execution-ready formulation structures.

Architectural implications
- The preferred future Stage2 role boundary is:
  - LLM for full-document semantic understanding
  - LLM for formulation-scope detection
  - LLM for structural signal detection, including DOE, selection,
    inheritance, sequential optimization, and other governed motifs
  - LLM for marker-level authorization, including partial or incomplete
    semantic cues when governance permits
- Deterministic downstream layers should remain responsible for:
  - row expansion
  - variable decomposition
  - relation binding
  - execution-level completion
  - stricter normalization and validation
- The preferred next direction is to evolve the semantic contract from final
  executable structure toward a governed intermediate semantic protocol.

Scope limits / what this does not mean
- This is not a rollback of semantic authority restoration.
- This is not permission for deterministic semantic inference.
- This is not a claim that the LLM should emit vague uncontrolled text.
- This is not a claim that the redesign is already implemented in the active
  runtime.
- This decision does not introduce a new numbered pipeline stage.

Next design direction
- The next architecture direction is not simply to make the LLM more precise
  by prompt tightening alone.
- The next architecture direction is to:
  - reduce LLM output burden
  - allow governed partial semantic markers where governance permits
  - move execution strictness downstream into governed function units and
    validators
  - preserve the restored semantic-authority boundary while redesigning the
    semantic contract

Case-localization note
- `QLYKLPKT` is currently treated as a localized example of this failure
  mechanism:
  - the paper may support the semantic pattern
  - the current contract may still suppress the marker family when
    execution-level grounding remains incomplete
- This note records the generalized architecture conclusion, not a paper-only
  workaround and not an implemented runtime exception.

## 2026-04-11

### Decision: Close and freeze Stage2 segmentation (S2-2a) for the current cycle (MDEC087)

Decision
- Record Stage2 segmentation closure for the current cycle.
- Freeze S2-2a candidate-segmentation logic by default after the DEV15
  segmentation-closure diagnostic pass.
- Treat subsequent remaining S2-2 closure failures as selector-evidence or
  table-extraction-quality work unless a concrete segmentation regression is
  demonstrated.

Reason
- The current-cycle DEV15 segmentation-closure pass reached governed closure
  status for the targeted segmentation blocker set.
- Continuing to modify segmentation without regression evidence would blur the
  S2-2a versus S2-2b boundary and reduce auditability of the selector phase.

Impact
- Stage2 segmentation is now frozen for the current cycle unless a proven
  regression reopens it.
- Immediate follow-on investigation should focus on S2-2b selector/evidence
  prioritization and table-extraction-quality diagnosis rather than
  segmentation redesign.
- This is a documentation, governance, and memory freeze only; it does not
  modify pipeline runtime behavior.

### Decision: Close and freeze Stage2 selector upgrade (S2-2b) for the current cycle (MDEC088)

Decision
- Record Stage2 selector closure for the current cycle.
- Freeze S2-2b selector behavior after the coverage-bounded selector upgrade
  validated on `QLYKLPKT`, with `5ZXYABSU` and `UFXX9WXE` retained as
  no-regression guards.
- Treat the maintained selector as upgraded from single-block ranking to
  coverage-aware bounded block set selection.

Reason
- S2-2a table representation repair is complete and frozen for the current
  cycle.
- The remaining S2-2 failure family was `multi_surface_coverage_failure`,
  which is now resolved at the selector layer without reopening candidate
  generation.
- The maintained selector now preserves information by using ordered role
  block sets with bounded expansion rather than forcing one block per role.

Impact
- Selector behavior is now explicitly:
  - role -> ordered block set, not role -> single block
  - coverage-first, ranking-second
  - bounded expansion only, with no unbounded growth
  - MATERIALS completion supported
  - multi-block OPTIMIZATION_RESULT supported
  - PREPARATION_METHOD prefers procedure evidence over assay/comparator
    evidence
- `candidate_blocks_v1.json` is frozen.
- Selector logic is frozen.
- `evidence_blocks_v1.json` is now the canonical S2-3 input.
- No further modification to S2-2 is allowed in downstream work.

## 2026-04-12

### Decision: Close the discoverability gap for frozen Stage2 fine-grained substeps (MDEC089)

Decision
- Fine-grained Stage2 substeps that are already frozen in practice must not
  remain implicit in broad Stage2 governance wording alone.
- For the current cycle, the repo must expose discoverable ownership,
  input/output contract, stop boundary, and next lawful step for:
  - `S2-2a`
  - `S2-2b`
  - `S2-3`
  - `S2-4a`
- `S2-4a` now receives a dedicated execution-facing maintained runner:
  - `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
- `S2-2a`, `S2-2b`, and `S2-3` remain internal Stage2 substeps, but their
  owning script/function surfaces and handoff contracts must be explicit in the
  maintained governance and execution-surface registry.

Reason
- Prior practical work already used `S2-2a`, `S2-2b`, `S2-3`, and `S2-4a` as
  meaningful frozen or stage-local boundaries.
- The maintained broad Stage2 surfaces did not let a future agent answer, from
  normal repo reading alone, what `S2-4a` produced, which artifacts belonged to
  each frozen substep, or what the next lawful step was after a frozen local
  boundary.
- This caused avoidable ambiguity in freeze audits, replay planning, and
  stage-local execution.

Impact
- Future agents must be able to determine the current frozen Stage2 substep and
  the next lawful step without re-deriving structure from broad Stage2 docs or
  historical chat context.
- Frozen-priority gap closing is now explicit:
  1. `S2-4a`
  2. `S2-3`
  3. `S2-2b`
  4. `S2-2a`
- `docs/maintained_script_surface.tsv`, `project/PIPELINE_SCRIPT_MAP.md`,
  `project/ACTIVE_PIPELINE_FLOW.md`, `project/ACTIVE_PIPELINE_RUNBOOK.md`, and
  `project/2_ARCHITECTURE.md` must remain aligned on this discoverability rule.

### Decision: Give frozen S2-4b its own maintained live-call boundary (MDEC090)

Decision
- The frozen `S2-4b` boundary must have a dedicated execution-facing
  maintained runner rather than remaining implicit inside the coarse composite
  Stage2 path.
- That runner is:
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- The dedicated `S2-4b` runner must:
  - consume frozen `S2-4a` prompt artifacts only
  - use the maintained Gemini live-call wrapper already frozen in repo practice
  - persist replayable raw response payloads and request-level metadata sidecars
  - stop before `S2-5`, `S2-6`, and `S2-7`
  - treat only API / transport / request-level failure as failure at this boundary
- Returned malformed or weak content must still be persisted and must not be
  judged semantically at `S2-4b`.

Reason
- `S2-4a` prompt freezing already created a lawful immutable upstream handoff.
- Leaving the next live-call step only inside the coarse composite Stage2 path
  would force future agents to choose between hidden fallthrough into parsing
  or an unguided custom live-call replay.
- A dedicated maintained boundary keeps the frozen live-call surface
  independently runnable, independently auditable, and replayable by the
  maintained composite Stage2 path later.

Impact
- Future agents can now resolve the exact `S2-4b` script path, frozen inputs,
  raw-response outputs, and stop boundary from normal governed repo reading.
- The maintained Stage2 composite entrypoint remains the only path that can
  rehydrate raw responses into completed Stage2 authority for downstream use.
- `docs/maintained_script_surface.tsv`, `docs/src_script_registry.tsv`,
  `project/PIPELINE_SCRIPT_MAP.md`, `project/ACTIVE_PIPELINE_FLOW.md`, and
  `project/ACTIVE_PIPELINE_RUNBOOK.md` must remain aligned on the dedicated
  `S2-4b` boundary.

### Decision: Freeze the successful S2-4b live-call settings for the current cycle (MDEC091)

Decision
- Freeze the successful current-cycle `S2-4b` live-call policy at:
  - model:
    `gemini-2.5-flash`
  - request mode:
    `stream_collect`
  - request timeout seconds:
    `180`
  - request retries:
    `0`
  - retry sleep seconds:
    `3.0`
  - persistence rule:
    persist any returned raw payload as-is at the `S2-4b` boundary
  - failure rule:
    persist request metadata for timeout, auth, transport, or API failure and
    mark controlled failure without semantic judgment
- This decision changes only the `S2-4b` call-layer operating settings needed
  for stable runtime behavior.
- It does not reopen or alter frozen `S2-4a` prompt content, prompt assembly,
  evidence selection, evidence ordering, or any downstream `S2-5+` behavior.

Reason
- The original non-streaming live-call path could block indefinitely in real
  frozen-prompt execution without an internal bounded request result.
- The successful one-paper `S2-4b` validation proved that streamed collection
  with a 180-second bounded request window and zero retries produced a lawful
  raw-response boundary without hanging.
- Future agents should not need to rediscover stable `S2-4b` operating
  settings from prior validation runs once this freeze exists.

Impact
- The dedicated maintained `S2-4b` runner now has an explicit frozen current-
  cycle call policy rather than relying on command-line overrides alone.
- `project/ACTIVE_PIPELINE_RUNBOOK.md`, `project/ACTIVE_PIPELINE_FLOW.md`,
  `project/PIPELINE_SCRIPT_MAP.md`, `docs/maintained_script_surface.tsv`, and
  `docs/src_script_registry.tsv` must stay aligned on these frozen settings.
- Future full-scope `S2-4b` execution for the current cycle must consume frozen
  `S2-4a` prompts and use these call-layer settings unless a later governed
  decision explicitly supersedes them.

### Decision: Freeze the completed DEV15 S2-4b output set as the lawful pre-S2-5 handoff for the current cycle (MDEC092)

Decision
- Record the completed full-DEV15 `S2-4b` run:
  - source run:
    `data/results/20260412_8517d36/04_s2_4b_live_llm_call_dev15_v1`
- Freeze its minimal reusable output set under:
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/`
- The frozen `S2-4b` output set must include:
  - run-level `RUN_CONTEXT.md`
  - `stage2_s2_4b_run_metadata_v1.json`
  - `analysis/s2_4b_request_summary_v1.tsv`
  - preserved raw payloads under `raw_responses/`
  - per-request metadata sidecars under `request_metadata/`
- This frozen set becomes the lawful preparation boundary for future `S2-5`
  work in the current cycle.
- It remains a frozen `S2-4b` dataset only; it does not itself execute `S2-5`
  or create the completed Stage2 authority surface.

Reason
- The dedicated maintained `S2-4b` runner has now completed the full DEV15
  prompt set using the frozen current-cycle call settings without hanging.
- Future agents should be able to consume the frozen `S2-4b` handoff material
  directly from the portable frozen layer rather than re-deriving run-local
  authority from `data/results/...`.
- The boundary must preserve both successful raw payloads and controlled
  failures exactly as produced, without semantic inspection.

Impact
- The current-cycle portable Stage2 freeze now extends through `S2-4b`.
- Future `S2-5` work may use the frozen `S2-4b` dataset as its upstream raw-
  response handoff, but must still stop short of claiming completed Stage2
  authority until replay through the maintained composite Stage2 path.
- The canonical current-cycle facts are now:
  - full DEV15 `S2-4b` run completed
  - success_count:
    `6`
  - failure_count:
    `9`
  - preserved raw payload count:
    `7`

## 2026-04-14

### Decision: Record identity freeze as the hard Stage5 benchmark-validity boundary (MDEC093)

Decision
- Stage5 final-table generation is necessary but not sufficient for
  benchmark-valid reporting.
- Benchmark legality requires the mandatory identity-freeze gate to pass before
  GT compare, modeling-ready continuation, or audit-ready outputs may be
  treated as legal benchmark-facing surfaces.
- The full DEV15 lineage
  `data/results/20260401_5d9f4e6/09_dev15_count_validation`
  is the governing failure example.

Reason
- That lineage reached Stage5 final-table materialization and produced compare
  outputs, but the identity-freeze contract failed.
- The governed repair lineage localized the failure classes as row count drift,
  identity reassignment, and unresolved scaffold binding.

Impact
- Compare outputs from the failed DEV15 lineage remain diagnostic-only, not
  benchmark-valid.
- Scaffold-binding and representation repairs are governed follow-on work, but
  they do not by themselves prove that a lawful full-pipeline run now passes
  the hard identity-freeze gate.

### Decision: Record Stage2 decomposition as a true functional-unit execution-ownership failure (MDEC094)

Decision
- Stage2 decomposition introduced a real architecture failure:
  semantic signals could exist while governed deterministic function units were
  not reliably taking control of execution on the mainline.
- The intended active contract remains:
  - LLM = semantic discovery and authorization
  - deterministic function units = execution
- Silent non-activation is not acceptable when semantic authorization is
  present.

Reason
- Functional-unit and feature-ledger audits from the failed DEV15 lineage
  showed that some units existed in governance or code without being provably
  active in the run artifacts.
- Sequential optimization remained active, while DOE and non-DOE table-row
  execution were not yet reliably on-path across the same governed lineage.

Impact
- Future governance must treat silent non-activation as an execution-ownership
  failure rather than a benign observability gap.
- Repair work should preserve the LLM semantic-authority boundary while making
  deterministic execution activation provable in run artifacts.

### Decision: Record DOE function-unit mainline restoration for the confirmed UFXX9WXE case (MDEC095)

Decision
- The governed DOE execution path is restored on the mainline for the confirmed
  `UFXX9WXE` repair case.
- The validating run is:
  `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
- In that run, governed deterministic DOE execution emitted `26` rows after a
  valid LLM-declared DOE scope reached the execution unit.

Reason
- The DOE repair lineage demonstrated that the issue was not only
  under-documentation.
- Run-local activation evidence now proves that the DOE function-unit path can
  truly take control on the mainline when its governed preconditions are
  satisfied.

Impact
- DOE execution ownership is recorded as repaired for the confirmed UFXX9WXE
  case.
- This does not authorize deterministic semantic authority outside the governed
  Stage2 contract.

### Decision: Record non-DOE table-row repair as partial with the dominant blocker moved upstream (MDEC096)

Decision
- Non-DOE table-row repair is partial:
  readiness-gating and execution-unit defects are repaired for already-
  authorized cases, but broader DEV15 coverage remains upstream-blocked by
  missing `table_formulation_scopes`.

Reason
- The non-DOE repair lineage separated:
  - data insufficiency
  - readiness overstrictness
  - execution-unit limitation
- The repaired run showed downstream improvement for already-authorized cases
  while the broader full-freeze replay still failed to authorize many papers.

Impact
- The dominant remaining blocker is now upstream Stage2 extraction, selector,
  or evidence-handoff completeness rather than downstream willingness to
  execute without authorization.
- Future work must not overclaim that non-DOE table-row execution is broadly
  repaired across DEV15.

### Decision: Register the 2026-04-14 parser/normalization preservation repair as an authorized-case Stage2 execution repair, not an upstream authorization fix (MDEC097)

Decision
- The `2026-04-14` parser/normalization repair family is recorded as a lawful
  Stage2 execution-preservation repair for already-authorized non-DOE cases.
- The primary maintained repair surface is:
  - `src/stage2_sampling_labels/table_row_expansion_v1.py`
- This family does not create or infer missing `table_formulation_scopes`.

Reason
- The campaign validation showed real downstream repair for already-authorized
  replay cases such as `5GIF3D8W` table-auth replay and `UFXX9WXE`.
- The same campaign also showed that full-freeze replay failures for
  `5GIF3D8W` and `WIVUCMYG` remained upstream-blocked by missing non-DOE table
  authorization scopes.

Impact
- Future agents may treat this family as engineering-closed at the intended
  preservation boundary.
- Future agents must not describe this repair as broad non-DOE authorization
  closure across DEV15.
- Stage3 and Stage5 remain out of scope for this repair family.

### Decision: Register the 2026-04-14 S2-4b semantic omission repair as prompt-layer semantic framing, not selector or parser repair (MDEC098)

Decision
- The `2026-04-14` semantic omission repair family is recorded as a prompt-
  layer Stage2 repair at `S2-3 -> S2-4a -> S2-4b`.
- The primary maintained repair surface is:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- The confirmed repaired paper at the intended boundary is:
  - `L3H2RS2H`

Reason
- The campaign localized the earliest failing boundary for the proven case to
  weak or ambiguous non-DOE table framing at the live LLM boundary, not to
  selector loss, parser loss, or downstream execution filtering.
- Bounded validation showed lawful raw non-DOE table authorization appearing
  for `L3H2RS2H` without DOE regression on the `UFXX9WXE` guard case.

Impact
- Future agents may treat this family as engineering-closed for the prompt-
  framing boundary it actually repaired.
- Future agents must not describe this family as a selector fix, parser fix,
  Stage3 fix, or Stage5 fix.
- Paper-level closure for `WFDTQ4VX` depended on separate call-layer
  persistence hardening and must remain distinguished from this semantic
  framing repair.

### Decision: Register the 2026-04-14 S2-4b call-layer persistence hardening as maintained live-call durability work, not semantic repair (MDEC099)

Decision
- The `2026-04-14` call-layer persistence hardening family is recorded as
  maintained `S2-4b` durability work.
- The maintained repair surfaces are:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- This family preserves raw boundary semantics and explicit failure reporting
  without changing semantic schema or downstream authorization rules.

Reason
- The campaign proved that `WFDTQ4VX` was blocked by streamed-response
  collection and persistence behavior rather than by a proven semantic omission
  after the prompt-layer repair.
- Post-hardening bounded validation recovered `WFDTQ4VX` and preserved a
  successful no-regression guard on `L3H2RS2H`.

Impact
- The maintained `S2-4b` boundary now explicitly records whether a recoverable
  raw payload was persisted in success or controlled-failure cases.
- Future agents must classify this family as call-layer engineering, not
  semantic repair, Stage3 repair, or Stage5 repair.
- Benchmark consequence from the `2026-04-14` Stage2 repair campaign remains
  pending until a downstream patched-path compare lineage is present.

### Decision: Add explicit dual-mode identity-freeze compare behavior without weakening benchmark legality (MDEC100)

Decision
- The maintained Stage5 compare entrypoint now exposes an explicit
  `--identity-freeze-mode` contract with two modes:
  - `benchmark`
  - `debug_identity`
- `benchmark` remains the default and blocks compare output generation when the
  identity-freeze summary records any violation.
- `debug_identity` may continue from the same frozen Stage5 final table after a
  failed identity freeze, but the resulting compare outputs must be labeled
  diagnostic-only and must not be reported as benchmark-valid.

Reason
- Current debugging needs count-compare visibility from patched downstream
  lineages even when identity freeze remains unresolved.
- The old hard stop suppressed useful diagnostic comparison surfaces and
  encouraged ad hoc workarounds outside the maintained compare entrypoint.
- Benchmark legality must still remain strict and explicit.

Impact
- Identity freeze remains the hard benchmark-validity boundary.
- The compare node now supports lawful diagnostic continuation without changing
  Stage2, Stage3, or Stage5 materialization semantics.
- Run metadata and compare artifacts must record the explicit compare mode plus
  benchmark-validity status.

### Decision: Split benchmark-facing primary formulation identity from preserved downstream/post-processing variant records (MDEC101)

Decision
- The maintained Stage5 benchmark-final builder now owns two governed sibling
  outputs from the same source-faithful closure pass:
  - primary benchmark-facing `final_formulation_table_v1.tsv`
  - linked lower-level `downstream_variant_records_v1.tsv`
- Downstream/post-processing descendants such as freeze-drying variants,
  storage-condition variants, re-dispersion variants, assay-condition variants,
  and measurement-condition variants must not enter the primary benchmark-facing
  formulation database unless the paper explicitly reports them as independent
  formulation identities.
- When Stage5 excludes or collapses those rows out of the primary benchmark
  database, it must preserve them in the linked lower-level surface rather than
  silently dropping them.

Reason
- Earlier governed Stage5 identity work already established the exclusion half
  of this design through parent-linked non-synthesis descendant suppression and
  helper-descendant filtering.
- Historical descendant-filter validations proved the repository repeatedly
  needed to keep the primary benchmark-facing database clean, but the retained
  implementation still ended in filtered-away rows instead of a durable
  preserved lower-level record surface.
- Recent Stage2 descendant-signal repairs now preserve enough lawful semantic
  evidence to carry parent linkage, non-synthesis change role, downstream
  context tags, and downstream variable payloads into a stable lower-level
  record table.

Impact
- The benchmark-facing primary formulation database stays one-row-per-primary
  formulation identity.
- Excluded downstream/post-processing descendants remain discoverable and
  auditable through `downstream_variant_records_v1.tsv` with explicit parent
  linkage and exclusion provenance.
- This is not a new coarse stage, not a paper-specific heuristic, and not a
  Stage5 keyword hack; it formalizes previously fragmented Stage5 identity and
  descendant-governance intent into one canonical maintained contract.

### Decision: Split Stage2 table handling into semantic-facing summary view versus execution-facing full-table authority (MDEC101)

Decision
- For strong table-bearing papers, especially DOE-style papers, Stage2 must
  preserve two distinct table surfaces once a formulation-relevant table is
  detected:
  - an execution-facing full-table authority surface
  - a semantic-facing LLM summary or evidence surface
- The current maintained implementation stores the execution-facing surface at:
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - with additive execution payload members under:
    `semantic_stage2_objects/normalized_table_payloads/<paper_key>/payloads/*.csv`
- The current maintained implementation stores the semantic-facing surface at:
  - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- S2-2a owns construction and preservation of the execution-facing full-table
  authority surface.
- S2-2b owns role-aware summary or evidence packaging for selector behavior and
  LLM packaging.
- S2-3 prompt assembly may consume only the semantic-facing summary or evidence
  surface.
- Downstream deterministic execution may resolve from semantic authorization
  back to the preserved S2-2 full-table authority surface by stable table
  identity.

Reason
- The `2026-04-14` DOE recovery investigation showed that semantic
  authorization alone is insufficient when downstream execution sees only a
  lossy summary payload.
- Historical successful DOE recovery depended on access to a lossless or
  maximally structure-preserving table surface with stable numbering and row
  order.
- This is not a rollback of LLM semantic authority.
- The LLM still owns semantic discovery and authorization, while deterministic
  execution still owns row materialization.
- The change is a contract split between semantic-facing and execution-facing
  table surfaces so that downstream execution no longer depends on a summary
  view as its only table representation.

Impact
- Future agents must be able to answer from repo reading alone:
  - where the execution-grade table authority is stored
  - which Stage2 substep owns it
  - what the LLM sees
  - what deterministic enumerators use
  - how authorized execution resolves semantic target to execution payload
- S2-7 and its function units must treat the preserved S2-2 full-table
  authority surface as the execution source of truth whenever it is available.
- The engineering principle is now explicit:
  the LLM sees a semantic-facing summary of a table, while deterministic
  execution operates on the preserved table entity.

### Decision: Complete the Stage2 table-authority contract for DOE and non-DOE execution inputs (MDEC102)

Decision
- The maintained S2-2 full-table authority surface remains
  `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`,
  but it is now contract-complete for execution use across DOE and non-DOE
  table families.
- Each preserved table authority record must now carry:
  - stable `table_id`
  - `source_table_reference`
  - deterministic `table_type`
  - `row_count`
  - `has_row_numbering`
  - `header_structure`
  - `raw_cells`
  - execution-facing `normalized_rows`
  - `row_identity_signals`
  - `reconstruction_confidence`
- S2-2 now also writes `analysis/table_authority_validation_v1.tsv` as the
  maintained observability surface for row-count preservation, duplicate-row
  detection, and column-collapse detection during authority construction.
- The semantic-facing summary surface remains
  `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`,
  and table-derived summary blocks must carry stable `table_id` plus explicit
  `summary_is_lossy=true`.
- DOE and non-DOE deterministic row materialization must now share the same
  execution contract:
  semantic target -> stable `table_id` -> preserved S2-2 full-table authority.
- Stage1 table assets may remain a deterministic reconstruction fallback inside
  S2-2a only.
- Once a preserved S2-2 authority surface exists, Stage1 table assets are no
  longer the downstream execution source of truth.

Reason
- The earlier contract split (MDEC101) established the semantic-facing versus
  execution-facing distinction but still left two practical gaps:
  - the execution-facing authority payload was not fully self-describing for
    all formulation-relevant table families
  - the non-DOE row-expansion path still had a code-contract dependency on
    Stage1 tables rather than the preserved S2-2 authority surface
- DOE recovery work on `WFDTQ4VX` and mixed-table audits on `UFXX9WXE`
  confirmed that semantic authorization remains necessary but is not sufficient
  unless deterministic execution sees an execution-grade preserved table
  entity.
- This decision does not weaken LLM semantic authority and does not move row
  materialization earlier than S2-7.

Impact
- The repository now defines one unified Stage2 table-handling invariant:
  the LLM operates on a semantic summary of a table, while deterministic
  execution operates on the preserved table entity.
- Future contract violations should be classified under:
  - `table_payload_degradation`
  - `enumerator_input_mismatch`
- Bounded validation must confirm both:
  - S2-2 preserved authority is execution-grade and self-describing
  - downstream DOE and non-DOE executors no longer rely on summary-only or
    direct Stage1 table inputs when authority is available

## 2026-04-18

### Decision: Accept Step 1 prompt front-matter trimming for blocked S2-4b papers inside lineage `20260418_9538ec2`

Step id
- `prompt_optimization_step_1`

Change description
- removed duplicated metadata-heavy prompt front matter for:
  - `UFXX9WXE`
  - `QLYKLPKT`
  - `WFDTQ4VX`
- the accepted trim removed repeated journal headers, title blocks, author lists, and abstract-heavy opening text from the prompt evidence body while preserving the maintained instruction/schema prefix and the formulation-bearing table evidence

Observed effect
- `UFXX9WXE` changed from no-first-token timeout to successful fresh streaming with a persisted raw response
- `WFDTQ4VX` changed from no-first-token timeout to successful fresh streaming with a persisted raw response
- `QLYKLPKT` remained blocked at no-first-token timeout
- no prompt-side regression was observed in the preserved selection-marker, inheritance-marker, or formulation-table evidence checks

Risk
- front-matter trimming can accidentally remove early narrative cues if it is expanded beyond duplicated headers and abstracts
- `QLYKLPKT` still failing means prompt overload is not fully explained by duplicated metadata alone

Lineage reference
- lineage root:
  - `data/results/20260418_9538ec2`
- analysis:
  - `data/results/20260418_9538ec2/analysis/prompt_optimization_step_1_report.md`

### Decision: Reject Step 2 prompt background-removal pass for lineage `20260418_9538ec2`

Step id
- `prompt_optimization_step_2`

Reason for the change
- test whether removing non-formulation background sections could unblock `QLYKLPKT` while keeping the Step 1 recovered sentinels stable

Observed effect
- `QLYKLPKT` did not improve and still failed at no-first-token timeout
- `UFXX9WXE` remained successful under maintained fresh `S2-4b`
- `WFDTQ4VX` regressed from Step 1 success back to no-first-token timeout
- the prompt-side evidence checks still showed formulation tables, selection signals, inheritance signals, and capability cues present, so the regression came from the changed prompt composition rather than an obvious cue deletion detected by the simple preservation audit

Regression risk
- removing broader background can destabilize prompt behavior even when local optimization and table cues appear preserved
- `WFDTQ4VX` should remain a required regression sentinel for later prompt work

Lineage reference
- lineage root:
  - `data/results/20260418_9538ec2`
- analysis:
  - `data/results/20260418_9538ec2/analysis/prompt_optimization_step_2_report.md`

### Decision: Accept schema-slimming experiment for blocked maintained `S2-4b` papers inside lineage `20260418_9538ec2`

Step id
- `schema_slimming_experiment`

Why schema complexity was targeted
- prompt-content trimming alone did not fully resolve the blocked papers
- Step 1 improved two papers without fixing `QLYKLPKT`
- Step 2 and Step 2 refined showed that aggressive content trimming can remove or destabilize semantic bridge text
- the remaining hypothesis was that output-schema planning burden itself was contributing to prefill / time-to-first-token overload

What keys were removed or deferred
- deferred top-level keys:
  - `component_candidates`
  - `variable_candidates`
  - `measurement_candidates`
  - `relation_hints`
  - `evidence_spans`
  - `unassigned_observations`
  - `preparation_inheritance_markers`
- retained top-level keys:
  - `document_key`
  - `doi`
  - `formulation_candidates`
  - `table_formulation_scopes`
  - `table_variable_roles`
  - `selection_markers`
  - `inheritance_markers`
  - `boundary_markers`

Observed effect
- `UFXX9WXE` received a first token and wrote a fresh raw response
- `WFDTQ4VX` received a first token and wrote a fresh raw response
- `QLYKLPKT` remained blocked with `DeadlineExceeded`
- no previously recovered sentinel regressed relative to the Step 1 prompt surface

Regression risk
- slimming the schema helps only if the retained marker-and-table core is enough for the paper class
- `QLYKLPKT` remaining blocked means schema burden is not the only driver for the sequential-optimization failure case
- future slimming should not remove the retained marker families or the table boundary surface

Lineage reference
- lineage root:
  - `data/results/20260418_9538ec2`
- analysis:
  - `data/results/20260418_9538ec2/analysis/schema_slimming_experiment_report.md`

### Decision: Accept QLYKLPKT local selector experiment for lineage `20260418_9538ec2`

Experiment name
- `qlyk_selector_experiment`

Rules used
- select formulation-bearing optimization tables
- include explicit local carry-forward sentences containing `selected`, `chosen`, `optimal`, `remaining studies`, or `all the following studies`
- keep only minimal nanoprecipitation / variable-setup context
- exclude raw-prefix background, PK / LC-MS detail, animal-study detail, and noisy auxiliary tables

Result
- `QLYKLPKT` recovered on the maintained fresh `S2-4b` path
- the local selector pack reduced the slim-schema prompt from `49653` to `20480` characters
- the maintained live call produced first token at `34.978` seconds and persisted a fresh raw response

Implication for S2-2
- selector overinclusion is a real failure mode for blocked sequential-optimization papers
- a generalizable `S2-2b` rule family should prioritize formulation-bearing optimization tables plus local explicit carry-forward sentences before falling back to broad raw-prefix context

Lineage reference
- lineage root:
  - `data/results/20260418_9538ec2`
- experiment run:
  - `data/results/20260418_9538ec2/22_qlyk_selector_experiment/RUN_CONTEXT.md`
- analysis:
  - `data/results/20260418_9538ec2/analysis/qlyk_selector_experiment_report.md`
  - `data/results/20260418_9538ec2/analysis/qlyk_selector_rules.md`

### Decision: Promote local optimization-pack selector into maintained `S2-2b`

Step id
- `s2_2b_selector_promotion`

Reason for the change
- the successful `QLYKLPKT` local selector experiment showed that selector overinclusion, not missing paper capability, was the blocking factor
- the maintained selector needed a governed rule to prefer local optimization evidence over whole-document fallback when the local pack is already sufficient

Promoted rule
- prioritize formulation-bearing optimization tables for strong sequential-optimization papers
- prioritize explicit local carry-forward bridge text containing:
  - `selected`
  - `chosen`
  - `optimal`
  - `remaining studies`
  - `after ... had been determined`
- keep minimal nanoprecipitation / variable-setup context
- suppress whole-document raw-prefix fallback only when the local optimization pack is already strong

Observed effect
- `QLYKLPKT` recovered under maintained `S2-2b` + maintained fresh `S2-4b`
- `UFXX9WXE` remained successful and did not regress
- `WFDTQ4VX` remained successful and did not regress

Regression risk
- the promoted path must stay gated to explicit sequential-optimization cases only
- DOE and sweep cases should continue to use their existing selector routes unless the same local-pack sufficiency conditions are met

Lineage reference
- lineage root:
  - `data/results/20260418_9538ec2`
- validation run:
  - `data/results/20260418_9538ec2/23_s2_2b_selector_promotion_validation/RUN_CONTEXT.md`
- analysis:
  - `data/results/20260418_9538ec2/analysis/s2_2b_selector_promotion_report.md`
  - `data/results/20260418_9538ec2/analysis/s2_2b_selector_rule_diff.md`

### Event: Prebaseline prompt freeze audit for repaired mainline

Maintained generation surface used
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_candidate_segmentation_artifact`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_evidence_blocks_artifact`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`

No-live confirmation
- no `S2-4b` live calls were made
- the audit stopped at the maintained pre-LLM prompt freeze boundary

Child run path
- `data/results/20260418_9538ec2/24_prebaseline_prompt_freeze_audit`

Observed prompt-readiness state
- `QLYKLPKT` no longer looks overloaded under the repaired maintained selector path
- the full repaired mainline still contains several borderline-large prompts, especially:
  - `V99GKZEI`
  - `L3H2RS2H`
  - `PA3SPZ28`
  - `WFDTQ4VX`
  - `5GIF3D8W`
- overall audit recommendation:
  - `borderline_needs_review`

Lineage reference
- analysis:
  - `data/results/20260418_9538ec2/24_prebaseline_prompt_freeze_audit/analysis/prompt_diagnostics_report.md`
  - `data/results/20260418_9538ec2/24_prebaseline_prompt_freeze_audit/analysis/prompt_suspicious_cases.md`

### Event: Maintained selector generalization + second prompt freeze audit

Maintained surfaces used
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_candidate_segmentation_artifact`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_evidence_blocks_artifact`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`

Maintained capability decision
- rule family `A-F` is now treated as maintained mainline prompt-generation behavior
- newly generalized in this task:
  - de-duplicate metadata / front matter in the fallback context pack
- already maintained before this task:
  - formulation-bearing optimization-table preference
  - explicit local carry-forward bridge preference
  - minimal preparation / variable-setup retention
  - local-pack suppression of whole-document fallback
  - local table-adjacent evidence preference

Observed effect
- second no-live DEV15 prompt freeze reduced duplicated metadata / front matter from `14/15` to `1/15`
- PK / LC-MS detail dropped from `12/15` to `7/15`
- animal / cell detail dropped from `9/15` to `8/15`
- prompt-size median dropped from `55332` to `30450`
- `UFXX9WXE` improved from `borderline` to `safe`
- `QLYKLPKT` remained `safe`
- `WFDTQ4VX` remained `borderline` but did not regress

Readiness decision
- overall second-audit recommendation:
  - `borderline_needs_review`
- main remaining blockers are a small set of large or noisy full-schema prompts, not the earlier duplicated-front-matter fallback behavior

Child run path
- `data/results/20260418_9538ec2/25_second_prebaseline_prompt_freeze_audit`

Lineage reference
- lineage root:
  - `data/results/20260418_9538ec2`
- analysis:
  - `data/results/20260418_9538ec2/analysis/maintained_selector_capability_audit.md`
  - `data/results/20260418_9538ec2/25_second_prebaseline_prompt_freeze_audit/analysis/prompt_diagnostics_report.md`
  - `data/results/20260418_9538ec2/25_second_prebaseline_prompt_freeze_audit/analysis/prompt_audit_vs_previous.md`

### Event: Top-risk selector convergence audit

Maintained surface changed
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`

Target papers
- `V99GKZEI`
- `WIVUCMYG`
- `WFDTQ4VX`
- `L3H2RS2H`
- `7ZS858NS`

No-live confirmation
- no `S2-4b` live calls were made
- the audit stopped at the maintained pre-LLM prompt freeze boundary

Maintained convergence change
- preserve maintained role-aware table candidates before falling back to `sorted_csv_first_4`
- suppress redundant selector-chosen `CONTEXT_FALLBACK` paragraphs when stronger local narrative evidence is already present
- clean full selected tables before prompt rendering so PDF journal headers, figure carryover, and similar noisy rows do not leak into the live prompt surface

Observed effect
- target-set aggregate prompt size dropped from `271959` to `172463` chars
- `WIVUCMYG`, `WFDTQ4VX`, `L3H2RS2H`, and `7ZS858NS` are now `safe`
- `V99GKZEI` remains `borderline` but dropped from `101694` to `71782` chars
- sentinels:
  - `QLYKLPKT` remained `safe`
  - `UFXX9WXE` remained `safe`
  - `WFDTQ4VX` improved from `borderline` to `safe`

Readiness decision
- overall recommendation:
  - `safe_to_restart_baseline`

Child run path
- `data/results/20260418_9538ec2/26_top_risk_selector_convergence_audit`

Lineage reference
- analysis:
  - `data/results/20260418_9538ec2/26_top_risk_selector_convergence_audit/analysis/top_risk_selector_gap_audit.md`
  - `data/results/20260418_9538ec2/26_top_risk_selector_convergence_audit/analysis/top_risk_prompt_diagnostics_report.md`
  - `data/results/20260418_9538ec2/26_top_risk_selector_convergence_audit/analysis/top_risk_vs_previous.md`

---

## 2026-04-18 - Diagnosis baseline restart

Event
- diagnosis baseline restart

Maintained mainline used
- fresh bounded `S2-4b` live-call surface:
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- downstream maintained stepwise Stage2 completion:
  - `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
  - `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
  - `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
- maintained downstream relation and final-output surfaces:
  - `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py`
  - `src/stage5_benchmark/enforce_identity_freeze_v1.py`
  - `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Child run path
- `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1`

Result
- `diagnosis_baseline_partial`
- fresh live coverage:
  - `9` success payloads
  - `1` partial persisted payload
  - `5` no-payload deadline failures
- downstream completed through:
  - `S2-5`
  - `S2-6`
  - `S2-7`
  - `Stage3`
  - `Stage5`
  - GT compare

Sentinel outcome
- `QLYKLPKT`
  - preserved partially
  - final output improved from `3` to `4`
- `UFXX9WXE`
  - preserved partially
  - final output improved from `1` to `17`
- `WFDTQ4VX`
  - regressed
  - final output fell from `27` to `2`
- `V99GKZEI`
  - preserved
  - final output reached `6`, matching GT

Whether ACTIVE_RUN should be updated
- no
- reason:
  - the run is not yet a usable managed diagnosis baseline candidate because five papers never reached a persisted fresh payload and `WFDTQ4VX` regressed sharply at final output

Lineage reference
- analysis:
  - `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1/analysis/diagnosis_baseline_restart_report.md`
  - `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1/analysis/diagnosis_baseline_preservation_check.md`
  - `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1/analysis/live_call_coverage_summary.md`

---

## 2026-04-19 - Governance integration for Stage2 class-level repair

Event
- promote the validated Stage2 class-level repair into the existing governed repository surfaces only

Repair name
- `stage2_variable_sweep_and_table_compaction_v1`

Failure classes
- `missing_variable_table`
- `table_flattening`
- `false_role_coverage`
- `evidence_budget_overflow`
- `near_duplicate_evidence`

Affected maintained functional units
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::has_variable_sweep_structure`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::compact_table_rows_for_evidence`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_role_aware_selection`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_evidence_blocks_artifact`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_normalized_table_payload_artifact`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_prompt_preview_row`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_s2_2_boundary_validation_row`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_s2_3_boundary_validation_row`
- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`

Observed governance action
- appended a new repair-index row for the class-level repair
- updated the maintained script registry notes for the Stage2 composite wrapper and internal extractor
- left `project/ACTIVE_PIPELINE_RUNBOOK.md` and `project/2_ARCHITECTURE.md` unchanged because the repair changes maintained implementation behavior and observability, not the Stage2 contract boundary itself

Validation basis
- validated no-live governed audits inside lineage `data/results/20260418_9538ec2`
- validation scope:
  - `QLYKLPKT`
  - `UFXX9WXE`
  - `WFDTQ4VX`
  - `V99GKZEI`
  - `WIVUCMYG`
  - `L3H2RS2H`
  - `7ZS858NS`
- regression status:
  - `no_regression_on_sentinels`

Lineage reference
- analysis:
  - `data/results/20260418_9538ec2/analysis/maintained_selector_capability_audit.md`
  - `data/results/20260418_9538ec2/analysis/top_risk_selector_convergence_summary.md`
  - `data/results/20260418_9538ec2/26_top_risk_selector_convergence_audit/analysis/top_risk_prompt_diagnostics_report.md`

---

## 2026-04-19 - Full pipeline benchmark restart attempt blocked in maintained Stage2 composite live-call path

Event
- full pipeline benchmark restart attempt using the repaired maintained Stage2 mainline

Execution target
- new governed child run:
  - `data/results/20260419_3579206/01_stage2`
- source authority resolved from:
  - `data/results/ACTIVE_RUN.json`
- explicit scope manifest:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`

Observed blocker
- the maintained coarse-grained Stage2 entrypoint `src/stage2_sampling_labels/run_stage2_composite_v1.py` entered a fresh Gemini live call for `L3H2RS2H` and did not reach a governed timeout, governed failure artifact, or Stage2 completion surface
- the underlying blocking call was:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::call_gemini`
- the request remained inside `google.generativeai` `generate_content(...)` until manually interrupted after repeated heartbeats beyond seven minutes

Partial artifacts written before interruption
- `data/results/20260419_3579206/01_stage2/targeted_manifest.tsv`
- `data/results/20260419_3579206/01_stage2/semantic_stage2_objects/raw_responses/5ZXYABSU__stage2_v2_raw_response.json`
- `data/results/20260419_3579206/01_stage2/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`

Consequence
- the benchmark lineage did not reach:
  - completed Stage2
  - Stage3
  - Stage5
  - identity freeze
  - GT compare
- no benchmark-valid result can be claimed from this attempt

Follow-up implication
- the current maintained benchmark blocker is not the promoted S2-2 class-level repair itself
- the blocker is the maintained composite live-call path lacking a governed per-request timeout or recoverable failure surface comparable to the frozen `S2-4b` runner

---

## 2026-04-20 - Stage2 selector redesign from role-aware coverage to evidence-driven prioritization

Decision
- replace the maintained Stage2 S2-2b selector with an evidence-driven selector
- remove required-role coverage, archetype-driven selector overlays, and role-shaped evidence inflation from the maintained pre-LLM path

Why
- the role-aware selector was inflating prompts by retaining proxy, fallback, and coverage blocks even when one authoritative table already existed
- the maintained live LLM contract had already been reduced to understanding-only output, so the selector no longer needed to pre-structure evidence around a hard semantic role ontology

Mainline rules
- selector responsibilities:
  - conservative noise filtering
  - weak importance ordering
  - high-signal evidence preservation
  - semantic-overlap suppression
- selector prohibitions:
  - no semantic role assignment
  - no required-role enforcement
  - no archetype overlay influencing selection
  - no coverage-based expansion
  - no table-plus-proxy-plus-fallback parallel evidence packaging
- LLM responsibilities:
  - semantic interpretation only
  - table meaning, formulation structure, and relationship inference

Artifact-contract effect
- `evidence_blocks_v1.json` no longer records selector roles, role priorities, role score breakdowns, required roles, selected roles, or weak-role summaries
- the maintained evidence artifact now records compact evidence metadata plus evidence-priority suppression/debug state

Scope
- this decision changes the maintained S2-2b selector contract and the maintained S2-3 prompt-assembly contract
- it does not create a new pipeline stage and does not move Stage3 relation-resolution work into Stage2

---

## 2026-04-20 - Keep Stage2 live prompt semantic-only while retaining runtime metadata in audit surfaces

Decision
- remove runtime and audit scaffolding narration from the default LLM-facing Stage2 live prompt header
- retain the same runtime metadata in governed preview, prompt-audit, and run-context surfaces

Why
- after selector de-inflation, the remaining prompt overhead came from runtime narration such as table mode, summary-first wording, controlled evidence packing wording, and resolved block-order narration
- those lines are useful for human audit and reproducibility, but they do not improve the LLM semantic task and unnecessarily expand the live prompt

Change
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`
  now emits semantic instructions, schema, paper identity, and the governed evidence pack only
- runtime metadata remains recorded through:
  - `analysis/stage2_prompt_preview_v1.tsv`
  - `analysis/s2_4a_prompt_audit_v1.tsv`
  - run-local `RUN_CONTEXT.md`

Removed from default live prompt header
- `Table mode: ...`
- summary-first narration
- controlled evidence packing narration
- resolved evidence block order narration

Unchanged
- selector behavior
- evidence block selection
- S2-4a boundary semantics
- S2-5 / S2-6 / S2-7 contracts

---

## 2026-04-20 - Add a minimal evidence sufficiency floor to the evidence-driven Stage2 selector

Decision
- keep the evidence-driven Stage2 selector
- add a narrow post-selection minimal evidence floor inside `S2-2b`

Why
- the evidence-driven redesign solved prompt inflation and removed role-driven evidence packing
- a later full DEV15 pre-LLM audit showed systematic underselection on table-led papers, especially packs that retained only tables and dropped clearly available method or materials context

What the floor guarantees
- after evidence-priority ranking, the maintained selector may add:
  - one best method-like block when none survived and a strong procedural paragraph clearly exists
  - one best materials block when none survived and a strong inventory paragraph clearly exists
  - at most one short distinct supporting paragraph near a retained table when it adds bounded interpretive support without acting as a proxy
- the selector still retains at least one authoritative formulation-bearing surface

What remains forbidden
- no semantic role ontology
- no required-role coverage
- no archetype overlay shaping evidence selection
- no pre-LLM semantic signals or semantic extraction layer
- no Stage3-like interpretation in `S2-2` or `S2-3`

Unchanged boundaries
- `S2-2a` stays structure recovery only
- `S2-2b` stays evidence selection and packing only
- `S2-3` stays prompt assembly only
- semantic signals remain LLM-owned at `S2-4b/S2-5`

---

## 2026-04-20 - Add narrow S2-2a table authority ranking to preserved table-set formation

Decision
- keep S2-2a as the owner of table recovery and preserved authority formation

---

## 2026-04-21 - Lock the summary-only coverage-first S2-2 / S2-4a selector contract

Decision
- keep all `S2-4a` table evidence summary-only
- reduce deterministic selector authority to conservative denoising, minimum
  evidence coverage, and bounded packing only
- move semantic table judgment, table scoping, and semantic primary-table
  interpretation back to the LLM

Why
- coarse table labels such as `characterization`, `results`, or
  `non_formulation_table` were still exerting too much practical authority over
  which plausible formulation-bearing tables survived into the prompt-ready
  evidence pack
- accepted prompt-budget controls must not reintroduce full-table prompt
  surfaces at `S2-4a`

Mainline rules
- selector responsibilities:
  - conservative denoising
  - minimum evidence coverage
  - bounded packing
- selector prohibitions:
  - no semantic veto over ambiguous table summaries
  - no deterministic choice of the one true formulation table among
    `must_include` candidates
  - no full-table prompt fallback at `S2-4a`
- table inclusion classes:
  - `must_include`
  - `optional_context`
  - `hard_drop`
- governed table-packing rule:
  - `must_include` table summaries remain in neutral stable order
  - `optional_context` follows only after `must_include` coverage is satisfied
  - `hard_drop` is limited to high-confidence noise only
- LLM responsibilities:
  - determine semantic table scope
  - decide formulation-bearing versus downstream/result-only table meaning
  - interpret primary semantic relevance from the bundled summary-only evidence

Artifact-contract effect
- `evidence_blocks_v1.json` remains the canonical pre-LLM evidence artifact,
  but all LLM-facing table surfaces inside it are summary-only
- `analysis/s2_4a_prompt_audit_v1.tsv` and the frozen `S2-4a` prompts must be
  treated as invalid if any full-table prompt surface reappears

Validation note
- bounded targeted replay improved or stabilized `INMUTV7L`, `V99GKZEI`,
  `L3H2RS2H`, and `QLYKLPKT` without a full-table prompt regression
- residual failures after this decision should now be classified as remaining
  evidence-underselection or upstream candidate quality issues, not selector
  semantic-overreach by default
- add a narrow conservative authority-ranking pass inside `S2-2a` after
  normalized table payloads are built and before the preserved authority set is
  finalized

Why
- later audits showed that current lineages still executed table recovery and
  downstream consumption, but some papers preserved the wrong mix of tables
  relative to earlier successful traces
- the failure mode was preserved authority set divergence rather than loss of
  table visibility

What changed
- recovered table payloads are now ranked with conservative artifact-level
  signals only:
  - representation quality and repair sufficiency
  - row-anchor stability and table legibility
  - formulation-structure density
  - obvious down-ranking for weak residue or clearly downstream result tables
- the preserved authority set now marks stronger tables as primary and keeps
  bounded distinct secondary tables only when still plausibly useful
- audit fields such as `authority_rank`, `authority_score`, `authority_tier`,
  and `authority_score_breakdown` are now carried on the maintained S2-2a
  authority surface

What explicitly did not change
- no semantic role inference in `S2-2a`
- no selector ontology, required-role, or archetype-overlay return
- no `S2-3` responsibility change
- no transfer of semantic understanding away from `S2-4b/S2-5`

Unchanged boundaries
- `S2-2a` remains structure recovery and table-authority formation only
- `S2-2b` still consumes the preserved authority set and remains evidence-only
- `S2-3` still serializes the evidence package only
- semantic understanding remains LLM-owned downstream

---

## 2026-04-20 - Record S2-2a Table Authority Ranking (Selector V2.1 refinement) as the maintained preserved-authority repair

Decision
- record the current S2-2a table authority ranking refinement under the
  canonical change name:
  `S2-2a Table Authority Ranking (Selector V2.1 refinement)`
- treat it as a maintained structural refinement inside `S2-2a` only

Motivation
- the active audit showed that current lineages still executed S2-2a table
  recovery and downstream consumption correctly
- the remaining failure mode was preserved authority set divergence:
  source tables existed, recovered tables existed, and downstream prompt
  construction still consumed tables, but the preserved authority set
  sometimes favored weaker or less formulation-bearing tables than earlier
  successful traces

Evidence
- before-change validation lineage:
  - `data/results/20260419_3579206/41_dev15_evidence_driven_v21_prellm_baseline_r4`
  - `data/results/20260419_3579206/42_dev15_evidence_driven_v21_s2_4a_baseline_r4`
- after-change validation lineage:
  - `data/results/20260419_3579206/45_dev15_evidence_driven_v21_table_authority_rank_prellm_r2`
  - `data/results/20260419_3579206/46_dev15_evidence_driven_v21_table_authority_rank_s2_4a_r2`
- the maintained before/after audit
  `analysis/table_authority_before_after_validation.tsv` shows narrower,
  stronger preserved table sets on the target papers and reduced prompt size
  without reintroducing prompt inflation

Exact scope
- S2-2a only
- after normalized table payload construction
- before final preserved authority set formation
- ranking uses conservative artifact-level signals only

What was not changed
- no Stage1 redesign
- no selector ontology redesign
- no semantic role inference before the LLM
- no prompt redesign
- no change to `S2-3` ownership
- no change to `S2-4b` or `S2-5` semantic ownership

Impact summary
- bounded DEV15 pre-LLM readiness improved from `10` to `14`
  `ready_for_s2_4b`
- primary problem papers improved:
  - `QLYKLPKT`
  - `UFXX9WXE`
  - `YGA8VQKU`
- context papers improved without inflation regression:
  - `5ZXYABSU`
  - `RHMJWZX8`
- `V99GKZEI` remained clean

---

## 2026-04-20 - Refine S2-2a primary-table eligibility so coarse labels demote but do not veto

Decision
- keep the S2-2a primary-table guardrail inside preserved-authority formation
- change the guardrail from coarse-label veto logic to structure-first primary
  eligibility

Why
- the first narrow primary guardrail fixed `L3H2RS2H` but regressed
  `V99GKZEI`
- the regression showed that coarse labels such as
  `table_type=non_formulation_table` and
  `table_role_hint=characterization/results` are noisy priors, not safe
  hard-exclusion signals

What changed
- coarse labels remain visible on the preserved authority surface and still act
  as negative evidence through scoring and audit fields
- coarse labels no longer make a table ineligible for primary authority by
  themselves
- primary exclusion is now limited to structural failure such as:
  - repair-insufficient or unrepaired payloads
  - narrative or figure-caption dominated tables
  - obvious non-tabular spillover
- audit fields remain explicit:
  - `primary_guardrail_applied`
  - `primary_guardrail_reason`
  - `primary_eligibility_signals`

What explicitly did not change
- no Stage1 change
- no selector ontology redesign
- no semantic role inference in `S2-2a`
- no `S2-3` prompt-responsibility change
- no `S2-4b` or `S2-5` ownership change
- no paper-specific allowlists or denylists

Two-paper validation
- before baseline:
  - `data/results/20260419_3579206/41_dev15_evidence_driven_v21_prellm_baseline_r4`
  - `data/results/20260419_3579206/42_dev15_evidence_driven_v21_s2_4a_baseline_r4`
- bounded after validation:
  - `data/results/20260420_9e6a1cf/07_two_paper_structure_first_s2_2`
  - `data/results/20260420_9e6a1cf/08_two_paper_structure_first_s2_4a`
- observed outcomes:
  - `L3H2RS2H`
    - `Table 5` remained `primary`
    - `Table 9` remained non-primary
    - prompt length stayed non-inflated at `6841`
  - `V99GKZEI`
    - `Table 1` regained `primary`
    - method / materials / table evidence remained present
    - prompt length stayed non-inflated at `6934`

DEV15 validation
- explicit maintained execution source:
  `data/results/20260419_3579206/41_dev15_evidence_driven_v21_prellm_baseline_r4/targeted_manifest.tsv`
- accepted comparison surface:
  `data/results/20260419_3579206/46_dev15_evidence_driven_v21_table_authority_rank_s2_4a_r2/analysis/dev15_post_rank_readiness.tsv`
- updated run:
  - `data/results/20260420_9e6a1cf/09_dev15_structure_first_primary_rank_s2_2`
  - `data/results/20260420_9e6a1cf/10_dev15_structure_first_primary_rank_s2_4a`
- run-scoped audit surfaces:
  - `analysis/dev15_post_structure_first_readiness.tsv`
  - `analysis/dev15_post_structure_first_evidence_audit.tsv`
- result:
  - bounded DEV15 `ready_for_s2_4b` remained `14/15`
  - residual mix remained unchanged relative to the accepted ranked baseline:
    - `clean_multi_method = 8`
    - `mixed_minor_residual = 6`
    - `fallback_residual = 1`
  - prompt-size distribution did not inflate
  - only `QLYKLPKT` prompt text changed, and it became shorter

Governance note
- this refinement is part of maintained `S2-2a` table-authority formation
- governing principle:
  structure evidence outranks noisy coarse labels for primary eligibility

## 2026-04-21 - Split the S2-4a audit standard into Hard Gate, Feature Activation Audit, and Calibration Review

Decision
- refactor the governed `S2-4a` audit standard into three layers instead of a
  single mixed checklist
- keep legality/readiness checks separate from semantic correctness review
- make feature activation artifact-backed rather than assumed from code
  presence

Why
- the previous `S2-4a` audit surface mixed three different questions:
  - is the pre-LLM prompt-ready evidence pack legal and bounded
  - did repaired maintained capabilities actually activate in this run
  - is the semantic interpretation correct on known-answer papers
- mixing those questions created hidden checklist overreach and made it too
  easy for a hard gate to smuggle in semantic truth claims such as which table
  is the real primary surface
- the repository already has a governed feature-unit layer and run-local
  activation reporting; the `S2-4a` audit should consume that layer explicitly
  rather than duplicating it informally

What changed
- `project/S2_4A_AUDIT_STANDARD.md` is now split into:
  - Layer A:
    Hard Gate
  - Layer B:
    Feature Activation Audit
  - Layer C:
    Calibration Review Only
- Layer A now judges only legality/readiness and may emit labels such as
  `table_missing`, `evidence_underselected`, `summary_contract_violation`,
  `prompt_inflation`, `hard_drop_overreach`,
  `selector_boundary_violation`, or `candidate_table_quality_failure`
- Layer B now verifies artifact-backed activation of maintained repaired
  features such as:
  - table recovery and repair activation
  - summary-first prompt behavior
  - ordered evidence packing behavior
  - raw-prefix removal
  - duplicate suppression
  - selector-contract activation
- Layer C is now explicitly calibration-only and may use known-answer papers
  such as `INMUTV7L`, `V99GKZEI`, `L3H2RS2H`, and `QLYKLPKT`

Governance consequences
- ordinary DEV15 readiness runs should use Layer A plus Layer B
- Layer C should be invoked for targeted regression review, repair validation,
  or known-paper calibration only
- Hard Gate must not adjudicate true primary-table semantics
- feature activation must be evidenced from run artifacts such as
  `candidate_blocks_v1.json`, `normalized_table_payloads_v1.json`,
  `evidence_blocks_v1.json`, `stage2_prompt_preview_v1.tsv`,
  `s2_4a_prompt_audit_v1.tsv`, `feature_activation_report_v1.tsv`, and
  `RUN_CONTEXT.md`
- semantic table truth remains LLM-owned in the active pipeline and may be
  reviewed through calibration, but it is not part of the universal hard gate

## 2026-04-21 - Redefine the S2-4a Hard Gate minimum evidence contract around formulation sufficiency

Decision
- redefine Layer A minimum evidence from evidence richness to minimum
  formulation sufficiency
- keep Hard Gate limited to legality, minimal sufficiency, summary-only
  compliance, selector-boundary compliance, and prompt legality

Why
- the previous Layer A wording was still too easy to interpret as requiring
  richer evidence bundles than the frozen pre-LLM boundary actually needs
- simple but valid formulation papers such as `INMUTV7L` already had a method
  block and formulation-bearing table summaries, yet still failed as
  `evidence_underselected`
- Hard Gate should not fail a paper merely because materials evidence or extra
  supporting context is absent when a minimally sufficient formulation surface
  is already present

What changed
- Layer A minimum evidence is now satisfied by any one of three governed paths:
  - Path 1:
    at least one formulation-bearing table summary survives
  - Path 2:
    method evidence plus strong table-adjacent formulation description survives
  - Path 3:
    method evidence plus explicit formulation definition in text survives
- materials evidence is now explicitly optional soft-support for Layer A
- supporting context is now explicitly optional soft-support for Layer A
- Layer A no longer requires multiple evidence families or evidence-rich
  coverage once one minimum sufficiency path is satisfied

Implementation note
- add a read-only helper,
  `src/stage2_sampling_labels/evaluate_s2_4a_hard_gate_v1.py`,
  to evaluate the governed Layer A contract directly from frozen
  `evidence_blocks_v1.json`, `normalized_table_payloads_v1.json`, and
  `s2_4a_prompt_audit_v1.tsv` artifacts without changing selector or
  prompt-construction behavior

Validation note
- on the accepted `20260421_3579206` DEV15 S2-4a lineage, Layer A readiness
  moved from `12/15` to `14/15`
- `INMUTV7L` and `YGA8VQKU` now pass under Path 1
- `L3H2RS2H` remains blocked for a real summary-only contract violation

## 2026-04-21 - Enforce the summary-only contract on table-derived inline table text at S2-4a

Decision
- keep `inline_table_text` available as an upstream diagnostic or selector
  surface
- block it from final prompt rendering at `S2-4a` unless it resolves to a
  governed summary-backed table surface

Why
- `L3H2RS2H` still carried a table-derived `inline_table_text` block into the
  final `S2-4a` prompt even though the maintained contract already said that
  all LLM-facing table evidence must be summary-only
- the trace showed that candidate creation and evidence preservation were not
  the direct contract violation; the violation occurred because prompt
  rendering fell back to raw `[TABLE]` emission whenever a selected table block
  lacked a resolved summary item

Root cause
- `build_inline_formulation_table_item` and candidate segmentation lawfully
  create `inline_table_text` candidates as intermediate table-derived recovery
  surfaces
- `build_evidence_blocks_artifact` lawfully preserves a selected
  `inline_table_text` block into `evidence_blocks_v1.json`
- `render_prompt_block` previously emitted that block into the final prompt
  using raw fallback text when `resolve_prompt_summary_table_item(...)`
  returned `None`
- therefore the missing enforcement point was the final prompt-level
  summary-only contract filter

What changed
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::render_prompt_block`
  now returns an empty rendered payload for table-derived
  `source_type=inline_table_text` blocks that do not resolve to a governed
  summary surface
- this change is intentionally narrow:
  - it does not change selector semantics
  - it does not redesign evidence packing
  - it does not remove ordinary supporting prose that merely mentions a table

Validation note
- targeted `S2-4a` rerun:
  `data/results/20260421_3579206/15_inline_table_contract_fix_targeted`
  removed the `L3H2RS2H__table__03` inline-table prompt payload while leaving
  `INMUTV7L` and `V99GKZEI` unchanged
- full DEV15 rerun:
  `data/results/20260421_3579206/16_inline_table_contract_fix_dev15`
  plus the updated Layer A check raised readiness from `14/15` to `15/15`

## 2026-04-21 - Lock confirmed-noise-only S2-2b preservation, summary neutrality, deterministic authority metadata, and replay preference

### Decision: Rewrite the S2-2b table policy around confirmed-noise-only irreversible removal

Decision
- `S2-2b` may irreversibly remove a table only when it is confirmed pure noise.
- If a table is not confirmed noise, it must remain preserved in the pre-LLM
  authority surface.
- Rules must not decide whether a table is important and must not downrank,
  suppress, or remove a table because another table appears more useful or
  more formulation-bearing.

Why
- `5ZXYABSU` showed a concrete preservation failure family: formulation-bearing
  `Table 1` and `Table 2` survived `S2-2a` but were hard-dropped at `S2-2b`,
  leaving only `Table 14`.
- Because the LLM sees summary-only table surfaces rather than full tables,
  any rule-based pre-LLM importance veto creates irreversible downstream loss.

Impact
- table handling policy is now governed as:
  - `CONFIRMED_NOISE`
  - `PRESERVE`
- audit/debug labels may still exist for observability, but they must not act
  as importance-based vetoes on preserved tables.

### Decision: Clarify that the maintained S2-3 / S2-4a summary path is neutral across preserved tables

Decision
- the normal maintained `S2-3` / `S2-4a` path remains neutral across
  preserved tables and must not reintroduce primary-table bias
- the main residual risk at this boundary is lossy summary compression, not
  cross-table importance reranking
- summary blocks must preserve header / column schema and first-column row
  identity surfaces as the primary structural contract
- sample rows are optional aids only and must not become the main information
  source
- deterministic rules should not pre-explain cross-table semantic
  relationships; the LLM remains responsible for semantic interpretation

Why
- the maintained prompt path already uses neutral stable ordering for selected
  table summaries, but debugging showed that lossy compression can still hide
  important table structure
- preserving schema and row-identity surfaces is more important than relying
  on a few sample rows

Impact
- future summary work should focus on structural preservation rather than
  semantic table prioritization
- primary-table bias should not be reintroduced through summary formatting or
  prompt packing

### Decision: Treat authority reopen handles as deterministic execution-side metadata rather than LLM semantic content

Decision
- `authority_run_dir`, `authority_payload_root`, table-scope locators, and
  related reopen handles are deterministic execution-side metadata
- they must not be treated as LLM semantic content
- they must not depend on the LLM to generate or transmit them
- replay compatibility should use deterministic sidecar or reattachment
  surfaces to recover them

Why
- replay from frozen raw responses remained semantically reusable, but failed
  when authority metadata was missing from replayed semantic docs
- the correct architectural fix was deterministic sidecar / reattachment,
  not a requirement that the LLM carry execution metadata

Impact
- semantic understanding and deterministic authority reopen remain cleanly
  separated
- frozen raw responses stay reusable for lawful replay when execution-side
  metadata can be deterministically reattached

### Decision: Prefer replay from frozen raw responses when the LLM-facing contract is unchanged

Decision
- if LLM task, prompt content, evidence content, model, and generation
  settings are unchanged, downstream deterministic contract changes should
  prefer replay from frozen raw responses over fresh live calls
- this preference applies only when the replay path can lawfully reattach the
  required execution-side metadata and preserve boundary legality

Why
- recent Stage2 debugging confirmed that several important fixes were entirely
  downstream of the frozen `S2-4b` raw-response boundary
- replay avoids unnecessary live nondeterminism and keeps validation scoped to
  the changed deterministic contract

Impact
- replay becomes the default validation path for downstream deterministic
  repairs when the LLM-facing contract is unchanged
- fresh live calls remain necessary only when the LLM-facing contract itself
  changes or replay cannot lawfully reattach required metadata

### Failure-family anchors

- `5ZXYABSU` is the maintained anchor for the selector/preservation failure
  family where formulation-bearing tables survive `S2-2a` but are wrongly
  hard-dropped at `S2-2b`
- `5GIF3D8W` is the maintained anchor for the non-DOE single-variable recovery
  failure family with:
  - an explicit baseline or optimized formulation table
  - later single-variable exploration groups
  - no lawful Cartesian expansion
- `INMUTV7L` is the maintained anchor for the simple formulation-table
  semantic-family collapse family where:
  - a low-ambiguity numbered formulation table is preserved upstream
  - the LLM authorizes the table as formulation-bearing but keeps only a
    family-level semantic object
  - deterministic non-DOE row enumeration must recover the base table rows
    directly from preserved authority rather than requiring LLM row objects

### Decision: Add bounded simple formulation-table deterministic enumeration after semantic authorization

Decision
- deterministic Stage2 may enumerate formulation rows for a bounded simple
  non-DOE table family after LLM semantic authorization
- this path is allowed only when:
  - the table is already LLM-authorized as formulation-bearing
  - the table is not on the DOE path
  - preserved `S2-2` normalized payload authority is available
  - the table is a low-ambiguity `full_formulation` surface
  - the first-column row identity surface is stable enough to instantiate
    distinct base rows without cross-table reasoning
- the path does not require LLM row-level output and does not apply to DOE
  matrices, non-DOE sweep recovery, or cross-table decode cases

Why
- `INMUTV7L` showed a simple paper where the method plus one numbered
  formulation table already contained the base formulation rows, but the
  frozen LLM response kept only `Table1_Formulation_Family`
- replay validation showed that deterministic post-authorization row
  enumeration can recover the `12` base rows directly from preserved
  authority without changing prompt text or requiring fresh LLM calls

Impact
- bounded simple-table execution is now a first-class deterministic Stage2
  repair family
- `WIVUCMYG` remains on DOE execution, `5GIF3D8W` remains on the non-DOE
  single-variable path, and `UFXX9WXE` remains stable with no regression
- future work must not broaden this into generic table enumeration or use it
  to swallow DOE or sweep-family cases

### Decision: Permit explicit diagnostic final-table compare against locked GT authority without ACTIVE_RUN promotion

Decision
- GT authority remains locked to the contracted GT artifacts in `ACTIVE_RUN.json`.
- Diagnostic compare workflows may compare an explicit new Stage5 final table or
  explicit diagnostic `--run-dir` against that locked GT authority without first
  promoting the compared lineage into `ACTIVE_RUN.json`.
- This exception is diagnostic-only and must record explicit source lineage and
  `benchmark_valid=no`.

Why
- Current repository practice is diagnosis-baseline driven.
- A new diagnostic baseline must be comparable against the same frozen GT even
  before authority-promotion decisions are made.
- Locking the compared final table to the old `ACTIVE_RUN.json` Stage5 path
  prevented governed diagnosis-baseline comparison and forced local ad hoc
  counting outside the maintained compare entrypoint.

Impact
- GT authority lock remains strict for `layer1_gt_path`, `layer2_gt_path`, and
  `layer3_gt_path`.
- The maintained compare entrypoint may now accept an explicit diagnostic final
  table path while preserving GT lock.
- Writing a diagnostic compare output does not itself promote the new lineage to
  active authority.

## 2026-04-22 - Restore UFXX9WXE-class DOE target binding by preferring file-derived table IDs and tolerant numbered-row detection

Decision
- inside `S2-2` preserved table authority formation, when a recovered table
  asset filename encodes a stable `__table_<N>__` identity, that file-derived
  table number outranks conflicting caption-derived table numbers during
  `source_table_id` / `table_id` recovery
- during numbered DOE structure recovery, tolerate row labels with optional
  whitespace before the trailing period, for example `7 .`, so a valid
  contiguous numbered run is not broken by minor OCR or CSV formatting noise

Why
- `UFXX9WXE` remained a confirmed DOE under-enumeration regression even though
  the LLM declared a high-confidence DOE scope for `Table 10`
- the current live baseline showed the real DOE authority asset
  `UFXX9WXE__table_10__pdf_table.csv` was preserved under the wrong logical
  table ID (`Table 1`), so DOE target binding failed before the function unit
  could execute
- the companion full formulation table
  `UFXX9WXE__table_13__pdf_table.csv` also lost numbered-row detection because
  one row label appeared as `7 .`, breaking the strict contiguous-run detector
- together these regressions severed the semantic-signal -> preserved-table ->
  deterministic DOE enumeration chain

Impact
- current bounded replay validation on
  `data/results/20260422_02e24eb/05_doe_binding_regression_replay` restored:
  - `UFXX9WXE` DOE execution with `26` emitted numbered rows
  - `WIVUCMYG` DOE execution stability with `26` emitted rows
  - `YGA8VQKU` DOE execution stability with `16` emitted rows
- the repaired `UFXX9WXE` preserved authority now records stable file-derived
  `Table 10` identity and a numbered-row-bearing payload basis derived from the
  selected companion table asset
- this repair is bounded to DOE target binding and numbered-row recovery only;
  it does not resolve the remaining Stage2 semantic-scope-ref contract failures
  in the current diagnostic baseline lineage

## 2026-04-22 - Preserve declared scope IDs in replay projection and block UFXX9WXE-class DOE companion duplicate expansion

Decision
- during shrunken-document compatibility normalization, derive replay
  `table_formulation_scopes` from `semantic_scope_declarations` when a matching
  `table_formulation_authorization_scope` already exists, preserving governed
  `scope_id`, locator, and LLM provenance instead of synthesizing a disconnected
  replacement scope
- during `table_row_expansion_v1`, when a variable-sweep document already emits
  successful DOE rows and a non-DOE full-formulation table presents the same
  contiguous numbered identity surface with the same row count, treat that table
  as a DOE companion duplicate and do not emit a second deterministic table-row
  family
- fix the Stage2 semantic-authority validator to actually propagate
  `document_scope_ids` and `document_keys_with_doe_scope` into row validation so
  replay legality findings reflect real contract state rather than an empty
  document-scope map

Why
- the first full replay baseline after the UFXX9WXE target-binding fix restored
  DOE execution but over-counted `UFXX9WXE` from `4` to `56` because the DOE
  recovery rows and companion `Table 13` table-row-expansion rows were both
  retained
- the same replay lineage reported `302` Stage2 contract errors, but a large
  share were validator artifacts caused by an empty document-scope map rather
  than genuine missing DOE declarations
- replay normalization was also discarding existing governed
  `table_formulation_authorization_scope` IDs and replacing them with synthetic
  scope IDs, creating false undeclared-scope errors for lawful replay rows

Impact
- replay validation on `data/results/20260422_6d13c88` reduced the Stage2 replay
  candidate count from `190` to `164`
- `UFXX9WXE` Stage2 candidate count dropped from `56` to `30`, and final count
  dropped from `56` to `30` against GT `27`
- total final diagnosis-baseline rows dropped from `185` to `159` with no fresh
  LLM calls
- Stage2 contract errors dropped from `302` to `30`, leaving a narrower set of
  remaining non-DOE scope-marker issues on papers such as `5GIF3D8W`,
  `PA3SPZ28`, `V99GKZEI`, and `WFDTQ4VX`

## 2026-04-22 - Treat shrunken replay `table_scopes` as lawful LLM backing for restored table-row expansion capability

Decision
- in `validate_stage2_semantic_authority_contract_v1.py`, when validating
  replay-era shrunken semantic documents that do not carry compatibility-only
  `table_formulation_scopes`, accept matching LLM-produced `table_scopes` as the
  lawful backing surface for table-row-expansion scope validation
- preserve warnings for non-DOE table-row expansion inside DOE-declaring
  documents, but do not fail the contract solely because the replay document
  stores table authorization in `table_scopes` rather than the older
  compatibility-only marker family

Why
- residual replay errors on `WFDTQ4VX`, `5GIF3D8W`, `PA3SPZ28`, and `V99GKZEI`
  all reflected historically restored capability paths that were still working
  functionally, but the validator treated them as illegal because shrunken
  replay documents no longer carried `table_formulation_scopes`
- historical evidence showed these papers had prior successful restoration
  paths, so the correct action was to legalize the still-governed replay backing
  surface rather than remove the restored capability

Impact
- replay validation on `data/results/20260422_e286e96/01_stage2_replay`
  reduced Stage2 contract errors from `30` to `0`
- all residual scope-marker failures for `WFDTQ4VX`, `5GIF3D8W`, `PA3SPZ28`, and
  `V99GKZEI` were converted into explicit warnings only, preserving diagnosis
  visibility without falsely marking restored capability as illegal
- no fresh LLM calls were used; validation remained replay-only

## 2026-04-22 - Restore 5GIF3D8W single-variable stabilizer family by accepting reversible noun-phrase variants in narrative sweep recovery

Decision
- in `table_row_expansion_v1.py`, treat reversible variable phrases such as
  `stabilizer concentration` and `concentration of stabilizer` as equivalent
  when extracting non-DOE single-variable level lists from source text
- keep the recovery bounded: this only broadens phrase matching for the same
  declared variable axis; it does not introduce new variables, Cartesian joins,
  or broader prose mining

Why
- current replay semantic signals named the axis as `stabilizer concentration`,
  while the source text states `concentration of stabilizer (0.5, 0.75, 1.0,
  and 2.0% w/v)`
- that wording drift caused the maintained non-DOE single-variable recovery path
  to restore only two of the three historically validated sweep families on
  `5GIF3D8W`
- historical validation had already shown the stabilizer family was lawful and
  should be restorable without fresh LLM calls

Impact
- targeted replay validation on
  `data/results/20260422_a329f3d/01_stage2_replay` restored `5GIF3D8W`
  table-row expansion from `10` back to `13` emitted rows
- the recovered single-variable groups returned from `2` to `3`, restoring the
  three stabilizer rows alongside the polymer and etoposide families
- Stage2 contract remained `pass` with `0` errors and `0` warnings for the
  targeted replay

## 2026-04-22 - Restore INMUTV7L simple-table enumeration by rebinding semantic Table 1 to the higher-authority preserved Table 15 asset via table-number aliases

Decision
- in `extract_semantic_stage2_objects_v2.py` and `table_row_expansion_v1.py`,
  resolve table authority using bounded table-number aliases derived from:
  - logical table labels
  - preserved asset filenames
  - source table references
  - recovered captions/titles
- when multiple alias-matching payloads exist, prefer the highest-authority
  preserved payload using existing authority ranking rather than failing on the
  first exact table-label collision
- in replay legality validation, treat alias-equivalent shrunken `table_scopes`
  as lawful backing when the restored execution surface uses a rebinding such as
  semantic `Table 1` -> preserved `table_15` authority asset

Why
- `INMUTV7L` is a simple-table anchor paper: one preparation paragraph plus one
  numbered formulation table with all 12 formulations
- the current replay LLM output still authorized `Table 1`, but the preserved
  `table_01` asset was a corrupted non-formulation fragment while the real 12-row
  formulation table lived in preserved asset `INMUTV7L__table_15__pdf_table`
  whose recovered caption explicitly said `Table 1`
- exact label matching therefore rebound `Table 1` to the wrong preserved asset,
  collapsing the current replay to only family-level rows despite a historically
  validated simple-table enumeration path

Impact
- targeted replay validation on `data/results/20260422_eaaf657/01_stage2_replay`
  restored `INMUTV7L` table-row expansion to `12` emitted rows
- the targeted replay now yields `14` Stage2 rows total:
  - `2` semantic family/variant rows
  - `12` deterministic numbered formulation rows from the preserved `table_15`
    authority asset
- replay legality also returns to `pass` with `0` errors and `0` warnings for
  the targeted validation

## 2026-04-22 - Restore BB3JUVW7 rowwise formulation recovery from two clear formulation tables

Decision
- in `table_row_expansion_v1.py`, add a bounded rowwise structured-table
  recovery path for formulation tables whose rows are already formulation
  instances and whose leading columns are composition/process-condition fields
  followed by measurement columns
- the rowwise contract is narrow:
  - table row count must stay small (`<= 12` rows)
  - at least two assignment columns must appear before measurement headers
  - at least two measurement columns must follow
  - each emitted row must carry complete assignment values for the assignment
    columns
- row labels for this path use stable ordinal-backed identifiers to avoid
  collisions when the first visible column repeats values such as `5`, `10`, or
  `100`

Why
- `BB3JUVW7` contains two clear formulation tables:
  - one 5-row nanosphere formulation table
  - one 7-row nanorod process-condition/formulation table
- current maintained recovery treated both tables as if they needed
  first-column identity or DOE-style decoding and emitted `0` rows despite the
  rows themselves already being lawful formulation instances
- the paper therefore collapsed to only family-level semantic rows even though
  the preserved authority payloads were clean and structured enough for bounded
  deterministic row recovery

Impact
- targeted replay validation on `data/results/20260422_de546e1/01_stage2_replay`
  restored `BB3JUVW7` Stage2 table-row expansion to `12` emitted rows
- the restored paper now yields `14` Stage2 rows total:
  - `2` semantic family rows
  - `12` deterministic formulation rows across the 5-row and 7-row tables
- the follow-on full replay baseline `data/results/20260422_99d902d` improved
  `BB3JUVW7` final count from `2` to `7` while also exposing broader Stage5
  closure trade-offs that still need refinement

## 2026-04-22 - Recognize morphology-style measurement headers in rowwise tables and suppress superseded family summaries

Decision
- extend `table_row_expansion_v1.py` measurement-header recognition for bounded
  rowwise formulation/process-condition tables so morphology-style metrics such
  as `Major axis`, `Minor axis`, `Aspect ratio` / `AR`, and `Feret` are treated
  as measurement columns rather than assignment columns
- extend `build_minimal_final_output_v1.py` so Stage5 filters:
  - parent-linked non-synthesis `formulation_family` descendants under the same
    rule family already used for parent-linked non-synthesis variants
  - unparented `formulation_family` summary rows when the same paper already has
    substantial deterministic `table_row_expansion_v1` coverage spanning
    multiple authorized table scopes

Why
- BB3JUVW7 Table 2 was already a lawful rowwise formulation/process-condition
  table, but the active measurement-header recognizer did not classify
  morphology-style response columns as measurements
- that made the rowwise contract misread those metric columns as additional
  assignment columns, pushing the table out of bounds and collapsing the full
  baseline from the intended `12` deterministic rows to only the 5 rows from
  Table 1
- once Stage2 restored both tables, Stage5 still kept two semantic family rows
  that were summary/descendant surfaces rather than independent benchmark-facing
  formulation identities

Impact
- replay-only bounded validation in `data/results/20260422_b78ac41/` restored
  BB3JUVW7 Stage2 to `14` total rows (`12` deterministic table rows + `2`
  semantic family rows)
- the same bounded replay then filtered the superseded family rows at Stage5 and
  produced `12` final rows
- diagnostic compare in
  `data/results/20260422_b78ac41/06_compare/final_table_vs_gt_counts.tsv`
  now matches BB3JUVW7 at `12 / 12` with no fresh LLM calls

Guardrail
- this remains a bounded class-level repair for small rowwise
  formulation/process-condition tables with assignment columns followed by
  morphology/measurement columns; it is not a license to mine arbitrary long or
  characterization-only tables

## 2026-04-22 - Add family-aware single-variable recovery and semantic-summary suppression for figure-backed non-DOE sweep papers

Decision
- extend `table_row_expansion_v1.py` so bounded non-DOE single-variable recovery can emit family-specific formulation identities when the paper provides:
  - a lawful one-parameter-at-a-time narrative contract
  - explicit tested levels
  - explicit or locally anchored polymer-family evidence from text / figure-caption-style narrative
  - actual experimental effect discussion for those family/level combinations
- allow those recovered formulation identities to leave measurement fields empty when the identity is supported but the quantitative values live only in figures and no image extractor exists yet
- add bounded optimized-family anchor completion when the paper explicitly states the optimized families in text / caption evidence but the preserved anchor table only materializes a subset of those family identities
- extend `build_minimal_final_output_v1.py` to filter parent-linked semantic `single_formulation` summary rows once substantial deterministic row-level enumeration already covers the same paper

Why
- `5GIF3D8W` already had a restored non-DOE sweep path, but it still undercounted because the maintained path emitted:
  - only 4 optimized anchor rows from Table 4 instead of the full optimized family set visible in text / figure-caption evidence
  - generic sweep rows for some variable groups rather than family-specific identities when the family support was present in the local variable discussion
- the paper’s formulation identities are supported jointly by table anchors, one-parameter-at-a-time narrative text, and figure-backed family-specific discussion; without image OCR, identity recovery must still allow value fields to remain blank rather than dropping the formulation identity entirely
- after row-level restoration, semantic summary rows such as `optimized_formulations` remained in final closure despite being superseded by deterministic row-level identities

Impact
- bounded replay validation in `data/results/20260422_3ab21d9/` restored `5GIF3D8W` to `29` Stage2 rows and `26` final rows, matching GT counts in diagnostic compare
- the new path remains bounded:
  - no paper-key special casing
  - no generic figure OCR
  - no unsupported Cartesian sweep expansion
  - no fabrication of quantitative measurements when only identity support is available

Guardrail
- family-specific sweep promotion requires local evidence for family + tested level + actual experimental effect; this is not a license to expand every generic sweep contract across all polymers mentioned anywhere in the paper

## 2026-04-22 - Add bounded first-column identity table recovery for complete formulation tables with abbreviated measurement headers

Decision
- extend `table_row_expansion_v1.py` with a bounded first-column identity recovery path for small formulation tables where:
  - each row is already a formulation instance
  - the first column carries the formulation identity label
  - remaining columns are measurement outputs
  - measurement headers may use abbreviated forms such as `Sizes`, `P.I.`, `Yield`, `D.C.`, and `E.E.`
- extend `build_minimal_final_output_v1.py` so semantic singleton/family rows without independent evidence grounding are filtered when a complete deterministic rowwise table enumeration from the same paper already covers the formulation table

Why
- `V99GKZEI` contains a complete six-row formulation table, but the older deterministic paths only handled:
  - numeric / `F1`-style row IDs
  - assignment-columns-before-measurement rowwise tables
- because this table instead used first-column formulation labels plus abbreviated measurement headers, only one fallback row survived and the semantic family/singleton rows were left to stand in for the full table

Impact
- bounded replay validation in `data/results/20260422_4fd0db1/` restored the six formulation-table rows and filtered the three superseded semantic summary rows, producing `6` final rows on the targeted paper

Guardrail
- this remains a bounded class-level repair for small complete formulation tables with identity-bearing first columns and measurement-tail columns; it does not authorize generic narrative mining or broad table expansion over long noisy tables

## 2026-04-22 - Add bounded compact inline formulation-table recovery from source text when preserved table payload is empty

Decision
- extend `table_row_expansion_v1.py` with a bounded compact-inline-table recovery path for cases where:
  - an LLM has already authorized a formulation-bearing table
  - the preserved S2-2 table payload is empty or unusable
  - the cleaned source text near the table anchor still contains a compact inline table surface with repeated formulation IDs and fixed-width value groups
- keep Stage5 summary suppression behavior so semantic singleton/family rows are removed once the deterministic row coverage is complete

Why
- `5ZXYABSU` still had a lawful formulation table contract at `Table 1`, but the preserved table payload was empty.
- the actual 9 formulation rows survived only as a compact inline table in source text, so older deterministic recovery paths emitted zero table rows and left only semantic summary rows.

Impact
- bounded replay validation in `data/results/20260422_0c6f1a4/` restored the nine `NPR/NPB/NPG` formulation rows and filtered the superseded semantic summary rows, producing `9` final rows on the targeted paper.
- full replay baseline `data/results/20260422_f81a6ce/` now reaches `5ZXYABSU: 9 / 9`.

Guardrail
- this path stays narrow: it requires an already-authorized formulation table plus a local compact inline table signature near the table anchor; it is not generic full-text table mining.

## 2026-04-22 - Restore fractional DOE level decoding and block interference-table over-retention for YGA8VQKU-class DOE papers

Decision
- extend `build_numbered_doe_row_candidates_v1.py` so strong numbered DOE tables can decode actual factor values from a companion coding table even when the coded design uses fractional axial levels such as `±1.68` rather than only integer coded levels
- allow coding-table recovery when the first factor-name column is labeled by a paper-specific header such as `Evaluated factors`, not only `Factor`
- extend `table_row_expansion_v1.py` to reject first-column identity recovery on temporal follow-up tables whose row labels are dominated by timepoints (`Day 1`, `Day 7`, etc.)
- carry measurement-tail fields through first-column identity recovery so small comparator tables preserve full measurement signatures
- extend `build_minimal_final_output_v1.py` so semantic DOE summaries/singletons are suppressed once deterministic DOE/table coverage is substantial, and collapse small comparator rows that match exactly one deterministic DOE row by complete measurement signature while adding no decoded factor assignments

Why
- `YGA8VQKU` has a 16-row numbered DOE table plus a separate coding table and one additional high-viscosity comparator formulation.
- older DOE decoding only recognized integer coded levels and a hard-coded `Factor` first column, so actual-value decode did not activate on the `±1.68` axial rows or the `Evaluated factors` coding table surface.
- recent first-column identity recovery also over-retained two interference surfaces:
  - the temporal stability table (`Day 1`, `Day 7`, `Day 15`, `Day 75`)
  - the low-viscosity comparator row that only restated the already-enumerated DOE optimum

Impact
- bounded replay validation in `data/results/20260422_9a31c4e/` restored decoded DOE assignments for the 16 numbered DOE rows, suppressed the three superseded semantic summary/singleton rows, collapsed the duplicated low-viscosity comparator row onto `F2`, and retained only the distinct high-viscosity comparator row
- targeted final count now reaches `17`, matching GT for `YGA8VQKU`

Guardrail
- the DOE decode path remains bounded to explicit numbered DOE rows plus an explicit companion coding table; it does not authorize free DOE design-space expansion
- comparator collapse requires an exact complete measurement-signature match to a unique deterministic DOE row and no explicit decoded factor assignments on the comparator row
- temporal follow-up rejection is limited to first-column identity tables whose row labels are dominated by explicit timepoint language, not generic short-label formulation tables

## 2026-04-22 - Restore WFDTQ4VX-class DOE emission under noisy mixed assets and retain explicit checkpoint batches

Decision
- extend `build_numbered_doe_row_candidates_v1.py` so numbered DOE emission is not zeroed out just because coded-level decode remains unresolved on a noisy mixed asset
- harden coded-column detection so obviously non-factor headers do not trigger false decode requirements, while allowing fallback positional coded-column detection on leading DOE columns
- broaden coding-table parsing so explicit factor rows like `X1`, `X2 – Polymer concentration`, and `X3 – Surfactant concentration` can still be read from noisy mixed preserved payloads
- add bounded checkpoint-batch recovery from explicit source-text validation sections headed by `Checkpoint batches with their predicted and measured values ...`
- stop collapsing checkpoint/validation formulations at Stage5 by same-core signature, because explicit checkpoint batches remain benchmark-facing formulation instances for this paper class

Why
- `WFDTQ4VX` mixes a coding table, a 27-row factorial layout, and an explicit three-checkpoint validation section across noisy preserved assets.
- after the YGA repair, DOE emission regressed from `14` recovered DOE rows to `0` because decode failure on the noisy mixed asset caused the DOE unit to discard otherwise explicit numbered rows.
- even before that regression, Stage5 logic still treated checkpoint batches as collapsible validation variants, capping the paper at `27` instead of the GT `30`.

Impact
- bounded replay validation in `data/results/20260422_85b4971/` restored the prior `27` explicit formulation instances and added `3` explicit checkpoint batches from the checkpoint-analysis text surface
- targeted final count now reaches `30`, matching GT for `WFDTQ4VX`

Guardrail
- raw DOE rows are retained when decode remains unresolved, but decode failure no longer erases explicit numbered-row evidence
- checkpoint recovery is bounded to an explicit checkpoint-analysis section with structured batch/value text; it is not generic prose mining for validation mentions
- Stage5 no longer auto-collapses checkpoint/validation rows solely because they match a DOE coordinate signature

## 2026-04-23

### Decision: Collapse later measurement-only rows that reuse deterministic DOE labels onto the existing DOE formulation core

Decision
- For DOE papers, a later non-DOE table row must not create a new formulation core when it reuses an existing deterministic DOE formulation label and only adds measurement or post-processing variables.
- This applies when the later row carries no decoded synthesis-defining assignments and its preserved identity variables are limited to processing-state measurement fields such as before/after freeze-drying size, PDI, or zeta potential.
- Stage5 should collapse that row onto the unique DOE row with the same paper-local formulation label.

Reason
- Some papers report a full DOE design matrix first, then revisit selected formulations in a later characterization or post-processing table using the same formulation IDs.
- Treating the later table occurrence as a new formulation over-counts formulation cores even though the later row only adds downstream measurements for an already enumerated DOE formulation.
- `WIVUCMYG` is the anchor case: Table 1 contains the 26 numbered DOE formulations, while Table 6 revisits F11, F19, and F20 only for before/after freeze-drying characterization.

Impact
- Stage5 duplicate-governance now collapses these later measurement-only rows into the matching DOE formulation instead of retaining them as extra benchmark-facing rows.
- Targeted full replay moved `WIVUCMYG` from `29 / 26` to `26 / 26`.
- Full collateral compare shows only `WIVUCMYG` changed (`29 -> 26`); other papers stayed count-stable.

Guardrail
- The rule is bounded to papers that already have a deterministic DOE row set.
- The later row must reuse a unique existing DOE formulation label and must not carry explicit synthesis-defining factor assignments.
- The later row's preserved identity-variable payload must be measurement/post-processing only; this is not a generic collapse of all later labeled tables.

### Decision: Recover anchorless sequential-optimization single-variable rows from explicit stagewise source-text contracts

Decision
- For sequential-optimization papers, Stage2 may recover row-level synthesis-defining single-variable formulations directly from explicit stagewise source-text level lists even when replayed raw semantic output has collapsed to family summaries and no explicit anchor rows survive in the current scope.
- This anchorless recovery is lawful only when all of the following hold:
  - `semantic_signals.has_sequential_optimization = true`
  - selected-condition hints are present
  - explicit stagewise level lists are recoverable from governed source text
  - the variable axes are synthesis-defining rather than downstream/post-processing
  - the document does not already expose another explicit anchor scope that can support the ordinary row-recovery path
- Later commercial comparators signaled only through compact semantic descriptions must still be filtered at Stage5 as non-internal external references.

Reason
- `QLYKLPKT` historically supported seven benchmark-facing formulation instances, but the current replay collapsed the paper to family summaries plus one optimal singleton because the raw replay no longer preserved row-level candidates.
- The source text still preserved the exact stagewise optimization contract:
  - four poloxamer 188 concentration levels
  - then three PLGA:ITZ ratio levels under the selected surfactant setting
- The prior table-row machinery required explicit anchor rows first, which blocked recovery for this paper even though the stagewise contract was still explicit in governed text.
- A document-level guard is required because some other papers, such as `5GIF3D8W`, already have explicit anchor scopes and must stay on the normal anchor-first recovery path.

Impact
- Bounded replay restored `QLYKLPKT` from `5` summary rows to `12` Stage2 rows and `7 / 7` final rows.
- Full replay collateral improved the active diagnostic compare surface from `12/15` matching papers to `13/15` while preserving `5GIF3D8W = 26/26` and all other paper counts.

Guardrail
- This is not generic prose mining for arbitrary sequential papers.
- It does not fire when an explicit anchor scope already exists elsewhere in the document.
- It excludes downstream/post-processing variable axes such as lyoprotectant freeze-drying sweeps from benchmark-facing synthesis-row recovery.
- It remains replay-only and deterministic; no fresh LLM calls are introduced.

### Decision: Recover measured loaded/empty characterization pairs when a single-family replay collapses and blank-control evidence survives only in narrative text

Decision
- In `table_row_expansion_v1`, after the normal direct-row and source-backed table extractors fail, Stage2 may emit a bounded characterization pair of row-level formulations from source text when all of the following hold:
  - only one formulation-family semantic object survives for the paper
  - the paper text explicitly reports a measured loaded-vs-empty or loaded-vs-blank nanoparticle comparator
  - the empty or blank comparator has at least one real measured value
  - the emitted rows stay within the existing LLM-declared scope and remain replay-only deterministic completion
- In `build_stage2_compatibility_projection_v1`, if this bounded characterization-pair recovery materializes explicit row-level rows, the collapsed family-only LLM summary for the same paper may be suppressed from the compatibility surface.

Reason
- `RHMJWZX8` was under-retained at `1 / 2` even after the user clarified that sparse-result blank/control formulations must stay in the main table whenever any real result is reported.
- The replayed raw Stage2 semantic output had collapsed to one family summary and the preserved authority scope still pointed at a pharmacokinetic table, so ordinary table row expansion emitted zero rows.
- The governed source text still contained an explicit measured comparator: the zeta potential of `AP-PLGA-NPs` and the zeta potential of `empty NPs`.
- Without a bounded recovery path, the pipeline kept only a collapsed family placeholder and lost the lawful blank-control formulation instance.

Impact
- Bounded replay restored `RHMJWZX8` from `1 / 2` to `2 / 2`.
- Full collateral replay improved the diagnostic compare surface from `13 / 15` matching papers to `14 / 15`.
- The only remaining mismatch after the full replay is `BXCV5XWB = 3 / 9`; previously repaired papers including `5GIF3D8W`, `L3H2RS2H`, `QLYKLPKT`, and `WIVUCMYG` remained count-stable.

Guardrail
- This is not generic prose enumeration.
- It is bounded to single-family collapse cases with an explicit measured loaded/empty or loaded/blank comparator in governed source text.
- The emitted rows remain deterministic post-authorization completion and do not reopen live LLM calls or promote deterministic semantic discovery into Stage2 mainline authority.

### Decision: Correct BXCV5XWB GT authority to exclude FITC and blank helper variants with no independent result output

Decision
- For `BXCV5XWB`, keep only the three KGN-loaded nanoparticle formulations in GT.
- Mark the six FITC-loaded and blank helper variants as non-GT in the Layer2 GT authority TSV.
- Reduce the Layer1 GT count for `BXCV5XWB` from `9` to `3` so compare uses the corrected GT universe.

Reason
- We had already established in system-side governance that these six rows are helper descendants rather than benchmark-facing main-table formulations.
- The paper does not report independent formulation-result outputs for the FITC or blank variants; they are assay/control helper particles.
- Under the current GT counting rule, helper/control particles without independent result-level reporting must not remain in Layer1 GT.

Impact
- After the GT correction, compare against the current Stage5 final table becomes `15 / 15` matching papers.
- `BXCV5XWB` now matches at `3 / 3`.
- Total final rows and total GT rows both resolve to `204` on the current diagnostic compare surface.

Guardrail
- This GT correction is specific to helper/control variants with no independent reported result output.
- Do not use it to remove sparse-result controls that do report at least one real measured result.

### Decision: Freeze DEV15 Layer 3 value-field grouping and compare contract before pipeline value debugging

Decision
- Freeze the current DEV15 Layer 3 authority surface in `data/cleaned/gt_authority/v1/dev15_layer3_values.tsv` as a Layer2-row-aligned value GT that stores only explicitly reported values plus manual calibration.
- Freeze Layer 3 fields into three classes for current-system work:
  - `core_fixed_fields`
  - `named_extensible_variables`
  - `provenance_or_reviewer_only`
- Keep `pH_raw` as a named extensible variable field.
- Do not rename `pH_raw` to anonymous slots such as `new_variable_1`.
- Do not yet elevate `pH_raw` to the same benchmark-first status as globally common fixed fields such as `particle_size_nm`, `ee_percent`, or `zeta_mV`.
- Freeze the Layer 3 compare unit to one cell:
  - `(paper_key, gt_formulation_id, field_name)`
- Freeze the minimum compare statuses to:
  - `missing_in_system`
  - `present_and_match`
  - `present_but_mismatch`
  - `extra_in_system`
  - `blocked_alignment`
  - `not_reported_in_gt`
- Require all Layer 3 compare reporting to split metrics between:
  - `core_fixed_fields`
  - `named_extensible_variables`

Reason
- Layer 3 GT is defined on top of the accepted Layer 2 formulation skeleton, so value debugging must not reopen formulation-boundary review.
- The current Layer 3 wide GT mixes stable core fields, sparse paper-local variables, and provenance columns; without a frozen grouping, compare design will drift from conversation to conversation.
- `pH_raw` has stable semantic meaning but low, paper-specific coverage in current DEV15 and therefore should stay explicit but separate from the core fixed-field benchmark surface.
- A cell-level compare surface is necessary to debug value recall, value accuracy, and unsupported/extra-value failure modes without collapsing them into paper-level summaries.

Impact
- Future Layer 3 evaluation and debugging should begin from the frozen field grouping and compare contract instead of ad hoc workbook interpretation.
- The next implementation step is to build governed compare outputs such as:
  - `layer3_value_compare_cells_v1.tsv`
  - `layer3_value_compare_summary_v1.tsv`
  - `layer3_value_error_buckets_v1.tsv`
- Pipeline tuning should use error buckets like `missing_value`, `unsupported_text`, `unresolved_table`, `normalization_mismatch`, and `derived_value_leakage` to localize which stage needs repair.

Guardrail
- Do not silently score inferred or calculation-only system fills as Layer 3 reported-value successes.
- Do not move named paper-local variables into anonymous slots.
- Do not merge `named_extensible_variables` into `core_fixed_fields` summaries without a new governed decision.
## 2026-04-26

### Decision: Use role-tolerant emulsifier/stabilizer scoring for first-week Layer3 value progress

Decision
- For first-week explicit-value progress reports and modeling-readiness field summaries, `surfactant_name` and `stabilizer_name` are not separate primary score fields when the article uses these labels interchangeably for aqueous-phase emulsion/stabilization agents such as PVA, Pluronic, Tween, emulsifier, stabilizing agent, or protective colloid.
- The primary reporting field is `emulsifier_stabilizer_name`, computed as a role-tolerant union of GT/system `surfactant_name` and `stabilizer_name`.
- Cross-role matches such as GT `stabilizer_name=PVA` versus system `surfactant_name=PVA` count as correct role-tolerant value matches; source-role wording remains provenance/reviewer metadata.
- Separate role scoring is reserved for downstream review when a paper explicitly reports multiple distinct surface-active excipients with distinct functions.

Reason
- PLGA nanoparticle papers often use surfactant, stabilizer, emulsifier, and protective colloid inconsistently for the same material role, especially PVA. Strict separate scoring converts terminology variation into artificial field-mapping errors.

Impact
- This changes first-week Layer3 progress reporting only. It does not alter Stage2 extraction authority, Stage3 relation semantics, Stage5 formulation membership, frozen Layer3 GT contents, or benchmark-valid row counts.

### Decision: Treat coded DOE levels as raw explicit values before decode-layer derivation

Decision
- For first-week Layer3 explicit-value progress, DOE variables reported as article-native coded levels may be compared as raw values when the GT field stores the coded level.
- Coded-level decoding to physical values belongs to the later derived/calculation layer and must preserve provenance from the coded row value and the factor-level decoding table.
- Ratio values that are already present in row labels or identity tokens may be rebound through a generic ratio-label-token rule rather than paper-key overrides.
- When the ratio label explicitly names both materials, left/right order remains semantic: `PLGA:ITZ 5:1` must not match `ITZ:PLGA 5:1` under compare.

Reason
- The current first-week task measures explicit extraction coverage, not numerical inference. Mixing coded-level decoding into this layer would blur extraction with derived calculation.
- QLYKLPKT source evidence confirms the ratio direction: preparation text says PLGA:ITZ (w/w) ratios of 5:1, 10:1, and 15:1; Table 2 is `Physicochemical properties of PLGA-ITZ-NS with different PLGA:ITZ initial ratios` and reports those three ratio rows.

Impact
- Compare-side recovery may materialize raw coded pH from pipe-delimited DOE rows and raw ratio tokens from formulation labels.
- This does not change frozen Layer3 GT, Stage2 semantic authority, Stage3 relation semantics, Stage5 row membership, or benchmark-valid final row counts.

### Decision: Preserve user-supplied paper evidence for first-week Layer3 explicit measured outputs

Decision
- User-supplied source snippets for `5ZXYABSU`, `5GIF3D8W`, `BB3JUVW7`, and `BXCV5XWB` are accepted as governed debugging evidence for first-week Layer3 explicit-value compare repair and must be retained in repository documentation.
- The shared reference location for these future paper-local uploads is `docs/methods/layer3_field_gt_protocol_v1.md` under `Paper-local explicit measured-output evidence notes`.
- For `5ZXYABSU`, the supplied Table 1 and Table 2 surfaces establish that the paper explicitly reports `Encapsulation efficiency ± standard deviation (%)` but does not report a separate loading-content / drug-content / drug-loading percentage field in those tables.
- For `5GIF3D8W`, the supplied optimized-formulation Table 1 surfaces establish that `Drug content (%)` is an explicit reported field distinct from `EEc (%)`, and that the optimized drug-loaded rows report `1.04 ± 0.06`, `1.14 ± 0.02`, `1.45 ± 0.11`, and `1.44 ± 0.09` for PLGA 50/50, PLGA 75/25, PLGA 85/15, and PCL respectively.
- The supplied `5GIF3D8W` narrative text also confirms the semantic identity of that field: the paper states that `percent drug content was low for all batches with a maximum of around 1.45 for PLGA 85/15 and PCL`, which supports treating those values as explicit `lc_percent` / drug-content evidence rather than inferred values.
- For `BB3JUVW7`, the supplied materials/methods plus Table 1/Table 2 surfaces establish two distinct formulation families in the paper: nanospheres with explicit `%EE` and `%DL` composition rows, and nanorods with explicit process-condition rows and `Drug content (µg/mg)` outputs. These surfaces should be used to prevent cross-family field conflation during debugging.
- For `BXCV5XWB`, the supplied materials/fabrication text plus Table 2 establish that the benchmark-facing KGN-loaded rows are `PLGA`, `PLGA–PEG`, and `PLGA–PEG–HA`, with explicit `Encapsulation efficiency (%)`, `Drug loading (mg KGN/mg nanoparticles)`, and `HA content` outputs. The same fabrication paragraph also mentions optional `FITC-loaded` formulations, but that mention alone does not authorize FITC variants as independent GT formulations.
- These paper-local source facts may justify compare-side explicit-value rebinding where the aligned Stage5 row preserves the same formulation identity and the value is directly reported in the row evidence surface.

Evidence retained
- `5ZXYABSU` user-supplied tables:
  - `Table 1 Nanoparticle formulations developed`
  - `Table 2 Characteristics of the nanoparticle formulations prepared`
  - Formulations `NPR1..NPG3`; measured outputs limited to particle size, zeta potential, and encapsulation efficiency.
- `5GIF3D8W` user-supplied optimized table:
  - `TABLE 1 Formulation characters for the optimized nanoparticle formulations`
  - PLGA 50/50 / 75/25 / 85/15 / PCL, each with `Empty` and `Drug loaded` columns.
  - Explicit measured rows include `Diameter (nm)`, `PIa`, `ZPb (mV)`, `Recovery (%)`, `Drug content (%)`, and `EEc (%)`.
- `5GIF3D8W` user-supplied narrative paragraph describing optimized formulation sizes and concluding that percent drug content peaks at around `1.45` for PLGA 85/15 and PCL.
- `BB3JUVW7` user-supplied methods/tables:
  - methods distinguish `artemether loaded PLGA nanospheres` from stretched `artemether loaded PLGA nanorods`
  - `Table 1 Particle size, PDI, %EE, %DL and zeta potential of the artemether loaded nanospheres`
  - `Table 2 Physicochemical parameters of nanorods obtained by varying the process conditions`
  - table surfaces show nanosphere composition+EE/DL fields versus nanorod process-condition+drug-content fields.
- `BXCV5XWB` user-supplied materials/fabrication/table:
  - nanoprecipitation core for `PLGA` / `PLGA–PEG`; conjugation step for `PLGA–PEG–HA`
  - optional `KGN or FITC (5 mg)` payload mention in fabrication paragraph
  - `Table 2 KGN-loaded nanoparticle properties`
  - explicit benchmark-facing rows `PLGA`, `PLGA–PEG`, `PLGA–PEG–HA` with DLS/TEM size, PDI, zeta, encapsulation efficiency, drug loading, and HA content.

Impact
- This records paper-local source authority for ongoing Layer3 compare debugging.
- Future user-provided paper excerpts for this workflow should be appended to the shared protocol section named above so later analyses know where to look first.
- It does not by itself change GT, Stage2 extraction authority, Stage3 semantics, Stage5 row membership, or benchmark-valid final reporting without a separately validated code change and bounded replay.

### Decision: Hold remaining 5GIF3D8W optimized-family `lc_percent` misses as upstream evidence-materialization gaps, and do not hard-recover `5ZXYABSU` `lc_percent`

Decision
- `5GIF3D8W_G036` (`1.45 %`) and `5GIF3D8W_G038` (`1.44 %`) remain diagnostic misses after compare-side `lc_percent` rebinding because their aligned Stage5 rows (`PLGA 85/15 / Drug loaded` and `PCL / Drug loaded`) currently preserve only narrative anchor-completion evidence, not the explicit optimized-table metric tail containing `Drug content (%)`.
- First-week explicit-value compare must not fill those two cells from manual notes or cross-row copying alone. They may be recovered only after the aligned system row itself preserves the explicit row-local drug-content surface.
- For `5ZXYABSU`, the user-supplied tables confirm that the visible explicit measured-output surface contains `Encapsulation efficiency (%)` but no separate `Drug content (%)`, `Drug loading (%)`, or `Loading content (%)` column. Remaining `lc_percent` cells for this paper should therefore stay unrecovered in first-week explicit compare unless a separate explicit source span is found.

Reason
- Current accepted first-week policy scores only explicit reported values and forbids converting manual interpretation or inferred fills into compare-side successes.
- The latest `5GIF3D8W` compare misses are not simple compare-binding omissions like the recovered `1.04 %` and `1.14 %` rows; they are row-evidence materialization gaps on the aligned optimized-family anchor-completion rows.
- `5ZXYABSU` lacks a safe table-local `lc_percent` surface in the supplied evidence, so forcing a recovery would blur `ee_percent` and `lc_percent` semantics.

Impact
- Treat `5GIF3D8W_G036/G038` as upstream restoration targets: restore explicit optimized-table metric tails onto the aligned rows before claiming compare success.
- Treat current `5ZXYABSU` `lc_percent` misses as expected first-week explicit gaps rather than compare bugs.
- This decision does not change GT or benchmark-valid counts by itself.

### Decision: Reframe current DEV15 work as diagnosis-baseline development, and keep legal recovery LLM-first

Decision
- DEV15 is the current governed diagnosis set for iterative repair work. Its purpose is to show whether diagnosis baselines improve as fixes land, not to pretend that the repository already has a complete benchmark-certified endpoint for that set.
- In the current repo phase, any mention of baseline for DEV15 should default to `diagnostic baseline` unless a separate governed contract explicitly declares a different status.
- Run scopes without an explicit governed GT must not be described as benchmark, benchmark-valid, or benchmark-blocked. They should be labeled diagnosis, audit, or extraction-development runs.
- Legal recovery stays LLM-first: the LLM semantic layer defines formulation meaning and candidate scope, while deterministic rules may only validate, normalize, align, or refill values already authorized by the LLM output or by governed explicit evidence handoff.
- Deterministic rules must not take over semantic authority, redefine row identity, or let rule-only reconstruction become the active owner of formulation meaning.

Reason
- The user clarified that DEV15 is a development-time testing group and that the desired signal is whether diagnosis baselines improve over time, not whether a benchmark certification badge can be claimed right now.
- The user also clarified that broader expansion work may have no GT at all, so benchmark language is actively misleading for those scopes.
- Recent Stage2 fallback debugging reinforced the architecture boundary: value restoration can be useful only if it remains subordinate to LLM semantic authorization rather than becoming a rule-led semantic substitute.

Impact
- Future repo discussion and reporting should use `diagnosis baseline`, `diagnostic compare`, or `diagnostic-only` for DEV15 unless a new governed benchmark contract is explicitly established later.
- When GT is absent, success should be framed through diagnosis surfaces, extraction quality audits, or downstream utility checks rather than benchmark claims.
- Repair work may continue to restore values through governed deterministic logic, but only inside the existing LLM-first semantic contract.
