# PLGA EE Literature-Mined ML Paper Minimum Publishable Plan

> For Hermes: this is the working execution plan for the near-term materials/PLGA EE prediction paper. Keep the extraction-method details intentionally brief in the paper-facing narrative; preserve implementation details in repository artifacts and a future methods paper.

Goal: Build the minimum publishable materials-science article from this repository by using LLM-assisted large-scale PLGA formulation extraction plus human audit to create an EE-focused dataset, then use interpretable machine learning to identify experimentally testable PLGA encapsulation-efficiency design rules and validate them with 3-5 targeted experiments.

Architecture: Use the existing governed Stage0-Stage5 pipeline as the data-production backbone. Do not redesign the extraction method for this paper; instead, finish the minimum DEV15 numeric-field backfill and risk-audit surfaces needed to support EE modeling, then scale the same narrow schema to 200+ papers. Separate the high-confidence human-audited modeling dataset from the broader LLM-extracted corpus.

Tech stack: existing PLGA pipeline, Stage2 composite extraction, Stage3 relation materialization, Stage5 final formulation closure, TSV/CSV modeling exports, RDKit/PubChem-style drug descriptors, scikit-learn/XGBoost/LightGBM, SHAP/permutation importance, grouped validation by DOI and drug.

---

## 1. Paper positioning

### Working title

LLM-assisted literature mining and interpretable machine learning reveal experimentally testable design rules for encapsulation efficiency in PLGA nanoparticles

### Main claim

A human-audited LLM-assisted formulation-level dataset can support interpretable ML discovery of PLGA nanoparticle encapsulation-efficiency design rules, and a small number of prospective experiments can validate the most actionable model-derived trends.

### What this paper is not

- Not a methods paper about the extraction pipeline.
- Not a benchmark paper claiming perfect automatic extraction.
- Not a universal predictor for every PLGA system.
- Not a full DOE optimization article.
- Not a formal benchmark-valid claim unless the repository governance gate later supports it.

### Intended novelty relative to uploaded/related literature

Detailed reference evidence is recorded in:

- `analysis/ee_modeling/reference_literature_field_model_matrix_v1.tsv`
- `analysis/ee_modeling/reference_literature_implications_for_minimum_publishable_scope_v1.md`

High-level synthesis:

1. Goren et al. 2025 Scientific Data provides a manually curated nanoprecipitation-only PLGA nanoparticle dataset: 433 formulations, 59 articles, 65 small molecules, 18 complete features plus size/EE/LC. Use it as a schema anchor and external reference dataset, not as a competitor.
2. Rezvantalab et al. 2024 Scientific Reports supports a minimum coarse PLGA ML field set: synthesis method, solvent 1/2, PLGA Mw, LA/GA ratio, PEG, PEG Mw, and PVA, with targets size/EE/DL and LASSO/SVR/RF-style modeling.
3. Hanari et al. 2025 microfluidic PLGA ML supports richer method-specific features such as flow rate, flow ratio, chip/channel fields, PLGA/drug/surfactant/PVA concentrations, and tree/boosting models for EE/DL, but those fields should stay microfluidic-specific rather than mandatory globally.
4. Seegobin et al. 2024 shows DOE + ML can guide PLGA nanoprecipitation experiments with a small controlled factor set, supporting our planned 3-5 model-guided validation experiments.
5. Noorain et al. 2023 supports early PLGA literature-extracted size/PDI/DL/EE modeling, but also illustrates why output-only sparse GP surfaces are insufficient for a design-rule paper.
6. Kim/Olivetti-style materials literature mining supports large-scale text-mined synthesis/formulation datasets and reinforces the need to audit route/formulation boundaries.

Our differentiator: broader PLGA formulation space plus LLM-assisted extraction plus human audit plus interpretable EE-focused ML plus 3-5 validation experiments.

---

## 2. Scope control for minimum publishable version

### Include in primary modeling scope

- PLGA or PLGA-dominant nanoparticles.
- Small-molecule-loaded systems.
- Formulation rows with unambiguous EE/entrapment/encapsulation efficiency target.
- Formulation rows whose EE can be aligned to a specific formulation instance.
- Preparation methods initially allowed:
  - nanoprecipitation / solvent displacement / interfacial deposition / solvent injection
  - single emulsion / emulsion solvent evaporation
  - double emulsion W1/O/W2
  - microfluidic PLGA NP preparation
  - DOE / Box-Behnken explicit formulation matrices when row identities and EE are explicit

### Exclude from primary modeling scope for minimum version

- Biologics unless deliberately added later.
- Active-targeting variants where targeting ligand is the main formulation variable.
- Lipid-polymer hybrids where PLGA is not the dominant carrier identity.
- Commercial comparator particles without internal preparation identity.
- Formulations with no EE target.
- Figure-only EE values unless manually extracted and audited.
- Release-only, cell-assay-only, or in vivo-only variants without formulation-level EE.

### Dataset tiers

#### Tier A: high-confidence modeling dataset

Use for main ML and paper figures. Requirements:
- formulation identity clear;
- EE target clear;
- formulation-to-EE alignment clear;
- at least Tier 0 predictors populated;
- human audit passed or risk score low enough for inclusion;
- provenance preserved.

#### Tier B: broad LLM-extracted dataset

Use for coverage reporting, sensitivity analyses, and future methods paper. Requirements are looser:
- LLM extraction may be incomplete;
- risk flags retained;
- ambiguous or high-risk rows excluded from main model unless manually resolved.

---

## 3. Minimum field schema

This field scope is literature-backed by `analysis/ee_modeling/reference_literature_field_model_matrix_v1.tsv` and `analysis/ee_modeling/reference_literature_implications_for_minimum_publishable_scope_v1.md`. In short: Goren 2025 anchors the nanoprecipitation small-molecule PLGA schema; Rezvantalab 2024 anchors the minimum coarse PLGA ML feature set; Hanari 2025 adds method-specific microfluidic features; Seegobin 2024 anchors experimental factor selection; Noorain 2023 supports output coverage but warns against output-only sparse models.

### Tier 0 fields: must exist for training rows

Identity/provenance:
- paper_key
- DOI
- final_formulation_id or stable formulation_id
- raw_formulation_label
- formulation_source_table_or_text_locator
- evidence_span_or_table_reference
- preparation_method_class

Target:
- ee_percent
- ee_value_text
- ee_source_type
- ee_alignment_status

Core design predictors:
- drug_name
- polymer_identity
- polymer_type / PLGA indicator
- preparation_method_class

### Tier 1 fields: highest-value predictors, prioritize extraction/backfill

Drug:
- drug_mass_mg or drug_feed_amount_text
- drug_polymer_ratio_raw or numeric drug/polymer ratio
- drug molecular descriptors: MW, logP/XLogP, TPSA, HBD, HBA, heteroatoms, rotatable bonds, formal charge if available

Polymer:
- polymer_mw_kDa
- LA/GA ratio
- polymer_mass_mg or polymer_concentration
- PEGylation / PEG presence
- PEG MW if present
- end group if reported, optional

Excipient/solvent/process:
- surfactant/stabilizer name
- surfactant/stabilizer concentration
- surfactant HLB when available
- organic solvent
- co-solvent if present
- solvent polarity index when available
- aqueous/organic phase ratio or W/O/W volumes when available
- pH if reported

Auxiliary outputs:
- LC/DL percent
- particle_size_nm
- PDI
- zeta_mV

### Tier 2 fields: useful but not blocking for minimum paper

- sonication time/power
- homogenization speed/time
- stirring speed/time
- evaporation time
- centrifugation speed/time
- temperature
- purification method
- release profile
- morphology descriptors
- cell assay variables

### Modeling input rule

Build two model variants:

1. Design-only EE model:
   - excludes particle_size_nm, PDI, zeta_mV, LC/DL as input features;
   - represents pre-experimental formulation design.

2. Characterization-assisted EE model:
   - may include particle_size_nm/PDI/zeta/LC;
   - clearly labeled as post-characterization or auxiliary model.

The main publishable design-rules claim should come from the design-only model.

---

## 4. Minimum publishable execution order

### Phase 0: record this plan and freeze scope

Output:
- this plan file in `docs/plans/`.

Acceptance:
- file exists;
- scope is explicitly EE-centered;
- extraction-method details are intentionally brief for paper-facing narrative.

### Phase 1: DEV15 numeric-readiness audit

Objective: determine exactly what DEV15 already supports and which numeric/value fields need narrow backfill before scaling.

Inputs:
- `data/results/ACTIVE_RUN.json`
- current active Stage5 final table
- current active Stage3 resolved fields
- current GT authority Layer3 values
- uploaded reference datasets/papers under analysis surfaces

Tasks:
1. Resolve active run only through `data/results/ACTIVE_RUN.json`.
2. Build a DEV15 EE modeling-readiness audit table with one row per final formulation.
3. Report coverage for Tier 0/Tier 1 fields.
4. Mark each row:
   - train_usable_now
   - needs_numeric_backfill
   - needs_identity_audit
   - target_missing
   - target_ambiguous
5. Separate true not-reported missingness from extraction failure where possible.

Minimum outputs:
- `analysis/ee_modeling/dev15_ee_modeling_readiness_v1.tsv`
- `analysis/ee_modeling/dev15_ee_modeling_readiness_summary_v1.md`

Acceptance:
- every DEV15 final formulation row is classified;
- EE target coverage is known;
- field gaps are ranked by model value, not by full Layer3 completeness.

### Phase 2: DEV15 minimum numeric backfill rules

Objective: add only the smallest deterministic backfill rules needed for an EE modeling-ready DEV15 surface.

Backfill priority:
1. EE / encapsulation efficiency target value and provenance.
2. LC/DL if directly reported or safely calculable.
3. drug/polymer ratio from explicit masses or table columns.
4. polymer_mw_kDa and LA/GA normalization from explicit text/table values.
5. surfactant concentration and surfactant name.
6. organic solvent and solvent class.
7. particle size only as auxiliary output.

Hard boundaries:
- no donor-fill from unrelated formulations;
- no assumption-based inference;
- no paper-specific fixes;
- no generic prose mining without scoped evidence;
- no treating Stage5 as semantic rediscovery;
- all new rules must be source-faithful and provenance-carrying.

Implementation placement:
- Stage2/Stage3 if the value is an extraction/projection/resolution issue.
- Stage5 only for materializing already-supported fields into final output or modeling export.
- Modeling export layer for derived ML-only descriptors and missingness indicators.

Minimum tests:
- unit tests for each numeric parser/backfill rule;
- sentinel regression checks for existing restored papers where count must not drift;
- DEV15 Stage5 count remains 204 final rows unless a governed GT decision changes.

Acceptance:
- DEV15 count stability preserved;
- numeric-field coverage improves for EE-relevant fields;
- every backfilled value has evidence or an explicit derivation formula.

### Phase 3: DEV15 modeling-ready export

Objective: produce a narrow dataset that can already run ML sanity checks.

Output schema:
- identity columns
- target columns
- Tier 0/Tier 1 predictors
- missingness indicators
- provenance fields
- risk flags
- train/test eligibility flags

Suggested output paths:
- `data/exports/modeling/ee/dev15_ee_modeling_dataset_v1.tsv`
- `analysis/ee_modeling/dev15_ee_modeling_export_summary_v1.md`

Acceptance:
- no high-risk target ambiguity rows in primary training subset;
- design-only and characterization-assisted feature sets can be generated from the same export;
- export is reproducible from explicit active-run artifacts.

### Phase 4: baseline ML sanity check on DEV15 and Goren 2025 reference data

Objective: validate modeling scripts and expected feature behavior before scaling to 200+ papers.

Inputs:
- DEV15 modeling export.
- Goren 2025 Mendeley dataset downloaded under `analysis/goren_2025_dataset/`.

Models:
- mean baseline
- ridge/elastic net
- random forest
- XGBoost or LightGBM

Validation modes:
- random row split for upper-bound sanity only;
- grouped split by DOI for real generalization estimate;
- grouped split by drug if enough data.

Interpretability:
- permutation importance
- SHAP where available
- partial dependence or accumulated local effects for top variables

Acceptance:
- ML code runs end-to-end;
- no claim of final model performance from DEV15 alone;
- Goren dataset used as schema/modeling reference, not as proof of our extraction quality.

### Phase 5: scale extraction to 200+ papers

Objective: run the narrow EE-focused extraction scope on the larger corpus.

Selection priority:
1. papers with formulation tables and explicit EE/entrapment efficiency columns;
2. DOE/Box-Behnken/factorial papers with row-level EE;
3. papers reporting drug/polymer and surfactant/process variables;
4. multiple formulation rows per paper;
5. small-molecule PLGA systems.

Operational rules:
- keep Stage2 authority LLM-centered;
- prefer replay when deterministic changes do not alter prompt/live-call behavior;
- preserve complete formulation tables into S2-2 authority surface;
- keep broad extraction output and high-confidence audited subset separate.

Expected outputs:
- `data/results/<lineage>/...` governed extraction runs;
- `data/exports/modeling/ee/plga_ee_broad_llm_extracted_dataset_v1.tsv`;
- `data/exports/modeling/ee/plga_ee_human_audit_queue_v1.tsv`.

Acceptance:
- target corpus count and paper list explicit;
- row-level extraction succeeds for enough papers to yield a useful high-confidence training set;
- every row has risk flags and provenance.

### Phase 6: risk-marked human audit

Objective: turn broad LLM extraction into a defensible high-confidence modeling dataset without manually re-curating everything.

Audit priority order:
1. EE target rows with high model leverage or high row count papers.
2. Rows with ambiguous formulation-to-EE alignment.
3. Rows with missing or conflicting drug/polymer ratio.
4. Rows with method ambiguity.
5. Rows with high-risk duplicate/variant/control classification.
6. Rows selected for experimental validation.

Risk flags:
- target_missing
- target_ambiguous
- formulation_identity_ambiguous
- multiple_ee_values_for_one_row
- table_row_alignment_uncertain
- possible_duplicate_variant
- method_uncertain
- drug_polymer_ratio_missing
- source_figure_only
- extraction_low_confidence

Audit outputs:
- reviewed decision: include_primary / include_secondary / exclude / needs_paper_check
- reviewer note
- corrected field values when needed
- evidence locator

Acceptance:
- high-confidence dataset has explicit inclusion decisions;
- broad dataset remains available but is not silently mixed into primary training.

### Phase 7: main ML modeling and design-rule extraction

Objective: train interpretable EE models and extract robust design rules.

Primary target:
- EE_percent regression.

Secondary target:
- high_EE classification, e.g. EE >= 70 percent or top-tertile threshold.

Required evaluation:
- random split shown only as optimistic upper bound;
- DOI-grouped split as main test;
- drug-grouped split if data supports it;
- paper-balanced or grouped analysis to avoid large DOE papers dominating.

Required model outputs:
- performance table: MAE, RMSE, R2, Spearman, classification AUC/F1 where applicable;
- feature-importance table;
- SHAP summary;
- top interaction candidates;
- model-derived experiment candidates.

Interpretation should emphasize directional design rules, e.g.:
- hydrophobicity/logP effects;
- drug/polymer ratio effects;
- PLGA MW and LA/GA effects;
- surfactant concentration/HLB effects;
- solvent polarity effects;
- preparation-method stratification.

Acceptance:
- conclusions are robust under DOI-grouped validation or explicitly labeled exploratory;
- no claim depends only on random row split;
- selected experimental hypotheses are feasible in the lab.

### Phase 8: 3-5 prospective validation experiments

Objective: validate one or two model-derived, experimentally actionable design rules.

Recommended design:
- fix one drug and one preparation method;
- change only 1-2 high-importance variables;
- choose points from model predictions:
  1. predicted low EE
  2. predicted medium EE
  3. predicted high EE
  4. high-uncertainty point
  5. predicted local optimum or replicate confirmation

Candidate variable pairs:
- drug/polymer ratio and surfactant concentration;
- PLGA MW and LA/GA ratio;
- surfactant concentration and solvent/aqueous ratio;
- polymer concentration and drug feed.

Required measurements:
- EE;
- LC/DL if feasible;
- particle size;
- PDI;
- zeta optional.

Acceptance:
- experiments test directional hypotheses, not exhaustive optimization;
- results can be plotted against predicted EE rank/order;
- at least one model-derived design rule is supported or clearly falsified.

---

## 5. Paper outline

### Abstract

- Problem: PLGA EE optimization is expensive and literature data are underused.
- Approach: LLM-assisted extraction + human audit + interpretable ML.
- Dataset: number of papers/formulations after final audit.
- Results: model performance and top design rules.
- Validation: 3-5 prospective experiments.
- Conclusion: literature-mined ML can guide PLGA EE formulation design.

### Introduction

1. PLGA nanoparticles are widely used but formulation optimization remains trial-and-error.
2. EE is a key quality attribute for dose, payload, process efficiency, and translational feasibility.
3. Prior datasets/ML studies show promise but are limited by scope, manual scale, narrow methods, or lack of prospective validation.
4. This work uses LLM-assisted extraction to scale formulation-level data construction, then human audit and interpretable ML to derive testable design rules.

### Methods, intentionally concise

- Literature corpus selection.
- LLM-assisted formulation-level extraction with provenance.
- Human audit and risk flags.
- Feature engineering and drug descriptors.
- Model training and grouped validation.
- Prospective experimental validation.

Keep extraction internals brief; cite future methods availability as repository/protocol where appropriate.

### Results

1. Dataset construction and audit flow.
2. Dataset composition and field coverage.
3. EE distribution and formulation-space coverage.
4. Model performance under random and grouped splits.
5. Interpretable design rules.
6. Prospective validation experiments.

### Discussion

- Agreement/disagreement with Goren 2025, Rezvantalab 2024, Hanari 2025, Seegobin 2024.
- Why drug descriptors and formulation-level identity matter.
- Limits of literature-mined data.
- Need for future active-learning/DOE loops.
- Extraction method reserved for separate methods paper.

---

## 6. Immediate next tasks

### Task 1: build DEV15 EE modeling readiness audit

Files:
- Create: `analysis/ee_modeling/dev15_ee_modeling_readiness_v1.tsv`
- Create: `analysis/ee_modeling/dev15_ee_modeling_readiness_summary_v1.md`

Use active source resolution only:
- read `data/results/ACTIVE_RUN.json`
- resolve Stage5 final table, Stage3 resolved fields, GT Layer3 values

Verify:
- rows equal active Stage5 final row count unless deliberately filtered in audit summary;
- fields include train_usable_now and risk flags.

### Task 2: list DEV15 minimum numeric backfill gaps

Files:
- Create: `analysis/ee_modeling/dev15_minimum_numeric_backfill_gap_list_v1.tsv`
- Create: `analysis/ee_modeling/dev15_minimum_numeric_backfill_plan_v1.md`

Prioritize gaps that block EE modeling, not all Layer3 fields.

### Task 3: implement first narrow numeric backfill rule

Choose only one high-value rule after Task 2, likely EE/LC target derivation or drug/polymer ratio parsing.

Rules:
- write failing test first;
- patch the appropriate Stage2/Stage3/export layer;
- rerun targeted tests;
- rerun DEV15 Stage5/export as diagnostic-only if needed.

### Task 4: create DEV15 modeling export

Files:
- Create: `data/exports/modeling/ee/dev15_ee_modeling_dataset_v1.tsv`
- Create: `analysis/ee_modeling/dev15_ee_modeling_export_summary_v1.md`

### Task 5: run Goren 2025 reference ML sanity check

Inputs:
- `analysis/goren_2025_dataset/A formulation dataset of poly(lactide-co-glycolide/NP_dataset.csv`
- DEV15 modeling export

Outputs:
- `analysis/ee_modeling/goren_reference_ml_sanity_v1.md`
- `analysis/ee_modeling/goren_reference_ml_metrics_v1.tsv`

### Task 6: define 200+ paper extraction scope

Files:
- Create: `analysis/ee_modeling/plga_ee_200paper_scope_selection_v1.tsv`
- Create: `analysis/ee_modeling/plga_ee_200paper_scope_selection_summary_v1.md`

Selection should prioritize EE-rich formulation-table papers.

---

## 7. Guardrails

- Do not broaden field scope until DEV15 minimum EE modeling export exists.
- Do not block scaling on PDI/zeta/process-minor-field completeness.
- Do not use random row split as the main performance claim.
- Do not mix broad LLM-extracted rows into high-confidence training without risk/audit status.
- Do not present diagnostic extraction outputs as benchmark-valid pipeline outputs.
- Do not make paper-specific fixes; generalize by error class.
- Do not turn the materials paper into an extraction-method paper.

---

## 8. Definition of minimum publishable completion

The minimum publishable package is complete when all are true:

1. A high-confidence audited PLGA EE dataset exists with enough rows for grouped ML.
2. The dataset includes formulation identity, EE target, drug/polymer/method core predictors, and provenance.
3. ML performance is reported under DOI-grouped validation.
4. Interpretable analysis identifies 2-4 plausible design rules.
5. 3-5 prospective experiments test at least one model-derived design rule.
6. Extraction method is described briefly and honestly as LLM-assisted plus human-audited, with detailed method reserved for a future paper.
