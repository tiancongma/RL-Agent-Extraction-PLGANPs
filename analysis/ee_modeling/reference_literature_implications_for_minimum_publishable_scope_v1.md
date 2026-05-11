# Reference Literature Implications For Minimum Publishable PLGA EE Scope

This note records how the uploaded/reference papers constrain the minimum publishable PLGA EE dataset and ML plan. It supports the working plan in `docs/plans/2026-04-25-plga-ee-literature-ml-paper-minimum-publishable-plan.md`.

## Purpose

The materials paper should not expand into a full extraction-method paper. The extraction method should be described briefly as LLM-assisted formulation-level extraction plus human audit. The literature below is used to justify a narrow field scope, model choices, validation design, and experimental strategy.

## Reference matrix

Tabular details are recorded in:

`analysis/ee_modeling/reference_literature_field_model_matrix_v1.tsv`

The matrix covers:

1. Noorain et al. 2023 Pharmaceutics, PLGA antiviral GP exploration.
2. Rezvantalab et al. 2024 Scientific Reports, PLGA literature-derived ML over size/EE/DL.
3. Hanari et al. 2025 Scientific Reports, microfluidic PLGA EE/DL prediction.
4. Seegobin et al. 2024 International Journal of Pharmaceutics, DOE + ML PLGA nanoprecipitation optimization.
5. Kim/Olivetti et al. 2017 Chemistry of Materials, materials literature text mining and ML synthesis insight.
6. Goren et al. 2025 Scientific Data, manually curated PLGA NP small-molecule nanoprecipitation dataset.

## What each paper contributes to our minimum scope

### Goren et al. 2025 Scientific Data

Key facts:

- 812 Web of Science hits screened.
- 59 articles included.
- 433 PLGA nanoparticle formulations.
- 65 small molecules.
- Nanoprecipitation / solvent displacement / interfacial deposition / solvent injection only.
- Small molecule systems only.
- No active targeting.
- Final dataset has 18 complete ML-ready columns plus source/reference support.

Fields in `NP_dataset.csv`:

- `polymer_MW`
- `LA/GA`
- `mol_MW`
- `mol_logP`
- `mol_TPSA`
- `mol_melting_point`
- `mol_Hacceptors`
- `mol_Hdonors`
- `mol_heteroatoms`
- `drug/polymer`
- `surfactant_concentration`
- `surfactant_HLB`
- `aqueous/organic`
- `pH`
- `solvent_polarity_index`
- `particle_size`
- `EE`
- `LC`

Implication for us:

- Use Goren 2025 as the schema anchor for the minimum PLGA EE modeling dataset.
- Add `preparation_method_class` because our intended scope is broader than nanoprecipitation.
- Treat Goren as external reference/sanity data, not proof of our extraction quality.
- Do not position our article as another dataset descriptor; position it as broader LLM-assisted extraction + interpretable ML + experimental validation.

### Rezvantalab et al. 2024 Scientific Reports

Key facts from extracted paper text:

- Dataset from over 100 PLGA NP research articles.
- Targets: size, EE%, DL%.
- Eight influential features:
  - synthesis method
  - solvent 1
  - solvent 2
  - PLGA Mw
  - LA/GA ratio
  - PEG presence
  - PEG Mw
  - PVA presence
- ML/feature analysis includes LASSO, SVR, RF, LR, MLP; tenfold validation; error/R-square comparisons.
- Conclusions emphasize PLGA Mw for EE and LA/GA for DL, with method/solvent/PVA/PEG also important.

Implication for us:

- These eight fields form the lowest defensible coarse predictor baseline.
- Our Tier 1 should refine binary PVA/PEG fields into surfactant identity/concentration and PEG details when possible.
- Our article should improve on this by adding drug descriptors and prospective validation.

### Hanari et al. 2025 Scientific Reports

Key facts from extracted paper text:

- Over 300 microfluidic PLGA nanoparticle formulations.
- 25 key features for microfluidic preparation.
- Targets: EE and DL.
- RF reported best in abstract, with high R2 values for EE and DL.
- Feature families include:
  - surfactant type
  - method
  - solvent 1/2
  - total/aqueous/organic flow
  - flow ratio
  - PLGA concentration
  - drug concentration
  - surfactant/PVA concentration
  - PLGA Mw
  - PEG percentage
  - chip type
  - channel diameter
  - PVA Mw
  - PEG Mw
  - LA/GA
  - size

Implication for us:

- Method-specific features matter, especially for microfluidics.
- Flow/channel fields should not become mandatory for all 200+ papers.
- They should be optional/method-specific Tier 2 fields.
- Tree/boosting models and feature reduction are appropriate for tabular formulation data.
- We must avoid comparing a broad-method DOI-grouped model directly against high R2 from a narrower microfluidic-only dataset.

### Seegobin et al. 2024 International Journal of Pharmaceutics

Key facts from extracted paper text:

- Combines DOE and ML for PLGA nanoprecipitation optimization.
- Targets are particle size and zeta potential, not EE.
- Experimental feature space:
  - PLGA type
  - anti-solvent type
  - PLGA concentration
  - anti-solvent solution concentration
- Uses XGBoost, RF, KNN, SVM, MLP, random search, 5-fold CV/LOOCV, R2/RMSE, DOE/RSM/ANOVA.
- Shows ML + DOE can identify influential parameters and guide limited experimental validation.

Implication for us:

- Use as justification for a small model-guided validation experiment rather than a full DOE.
- PLGA concentration and surfactant/anti-solvent concentration are high-value experimental variables.
- Do not let size/zeta optimization replace EE as our main target.

### Noorain et al. 2023 Pharmaceutics

Key facts from extracted paper text:

- Literature-extracted PLGA antiviral nanoparticle data.
- Uses size, PDI, DL, EE with Gaussian Process prediction surfaces.
- PLGA 50:50 table includes about 62 data points; other PLGA ratios include 9-point sets from one paper.
- Authors explicitly note sparse regions and extrapolation uncertainty.

Implication for us:

- Use as early precedent for PLGA literature-extracted ML.
- Treat size/PDI/DL/EE-only modeling as insufficient for our design goal.
- Particle size/PDI/DL should be auxiliary outputs or characterization-assisted inputs, not the main design-only feature set.

### Kim/Olivetti et al. 2017 Chemistry of Materials

Key facts from extracted paper text:

- Demonstrates literature-scale materials synthesis extraction and ML insight.
- Uses CrossRef/publisher retrieval, PDF/plain text conversion, paragraph relevance classification, synthesis-parameter extraction, and ML.
- Titania nanotube example uses 22,065 journal articles and 27 synthesis variables.
- Notes route-boundary ambiguity when multiple synthesis routes occur in a single paper.

Implication for us:

- Use as methods-context citation for literature-mined materials design rules.
- It supports our insistence on formulation-level row identity and target alignment.
- Human audit should prioritize formulation boundary and EE-target alignment, not every minor process field.

## Minimum field implications

### Tier 0: must be present for primary training rows

Justification: every relevant ML/dataset paper depends on row identity and target alignment, even when not stated as such.

- DOI / paper id
- stable formulation id
- raw formulation label
- formulation evidence locator
- preparation method class
- EE percent
- EE source/provenance
- drug name
- polymer identity / PLGA indicator

### Tier 1: prioritize for DEV15 backfill and 200+ paper extraction

Justification: supported by Goren 2025 schema, Rezvantalab 2024 influential features, Hanari 2025 microfluidic features, and Seegobin 2024 experimental factors.

- drug/polymer ratio
- drug amount/feed if available
- drug descriptors: MW, logP/XLogP, TPSA, HBD, HBA, heteroatoms, melting point when available
- PLGA MW
- LA/GA ratio
- polymer amount or concentration
- surfactant/stabilizer name
- surfactant/stabilizer concentration
- surfactant HLB when available
- solvent name
- solvent polarity index when available
- aqueous/organic or phase ratio
- pH
- LC/DL as secondary target
- particle size as auxiliary output

### Tier 2: useful but not blocking for minimum paper

Justification: appears in method-specific or process-focused papers but is not consistently necessary for a minimum cross-paper EE dataset.

- microfluidic total/aqueous/organic flow
- flow ratio
- chip type
- channel diameter
- sonication power/time
- homogenization speed/time
- stirring speed/time
- evaporation time
- centrifugation speed/time
- temperature
- zeta potential
- PDI
- release profile
- cell assay variables

## Model implications

Minimum model set:

- mean baseline
- ridge or elastic net
- random forest
- XGBoost or LightGBM
- optional SVR for comparison with Rezvantalab 2024

Interpretability:

- permutation importance
- SHAP for tree models
- partial dependence or accumulated local effects for top variables

Validation:

- random row split only as optimistic upper bound
- DOI-grouped split as main test
- drug-grouped split if enough data

Reason:

Uploaded/reference papers often use row-level cross validation or narrower method scopes. Our broader-method claim must be protected by grouped validation to avoid paper/family leakage.

## Experimental validation implications

Minimum experimental validation should test one or two model-derived design rules, not perform exhaustive optimization.

Preferred variables:

- drug/polymer ratio
- surfactant concentration
- PLGA MW or LA/GA ratio
- polymer concentration

Experimental design:

- fixed drug
- fixed method
- fixed solvent/surfactant family when possible
- 3-5 model-selected points: low, medium, high predicted EE, high uncertainty, and predicted local optimum or replicate

## What to update in the main plan

The main plan should explicitly reference this note and matrix in the paper-positioning and minimum-field sections, so future implementation decisions are literature-backed rather than preference-based.
