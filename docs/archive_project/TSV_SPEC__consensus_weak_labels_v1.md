# TSV Specification: Conservative Multi-Model Consensus Weak Labels (v1)

This spec defines the **outputs** of `multi_model_consensus_vote.py`.
It is designed for a publishable workflow:

primary extraction (multi-model) → field-level agreement → conservative consensus weak labels → conflict-only GT.

## Inputs

Input TSV must be **multi-model** (one row per key/formulation_id/model) and contain:

Required columns:
- `key`
- `formulation_id`
- `model`

Extraction fields (recommended; missing columns are created as empty):
- `emul_type`, `emul_method`, `la_ga_ratio`, `plga_mw_kDa`, `plga_mass_mg`, `pva_conc_percent`
- `organic_solvent`, `drug_name`, `drug_feed_amount_text`
- `size_nm`, `pdi`, `zeta_mV`, `encapsulation_efficiency_percent`, `loading_content_percent`
- `notes`

Evidence fields (strongly recommended; missing columns are created as empty):
- `evidence_section`
- `evidence_span_text`
- `evidence_span_start`
- `evidence_span_end`
- `evidence_method`
- `evidence_quality`

## Outputs

The script writes three TSV files.

### 1) `formulations_consensus_weak.tsv` (consensus weak labels)

One row per `(key, formulation_id)`.

Core triage columns:
- `has_any_conflict` (0/1): 1 if any tracked field is a conflict (both non-empty and disagree beyond tolerance).
- `risk_level` (LOW/HIGH):
  - HIGH if `has_any_conflict=1` OR any **key field** is missing consensus.
  - LOW otherwise.
- `conflict_fields`: semicolon-separated list of fields in conflict.
- `n_conflict_fields`: number of conflict fields.
- `key_fields_missing`: semicolon-separated list of key fields missing consensus.
- `particle_scale`: derived from `size_nm_main`:
  - `<1000` -> nano
  - `1000–9999` -> submicro/micro
  - `>=10000` -> micro
  - empty if missing/invalid

Model identity:
- `model1`, `model2`: the two model names being compared.
- `preferred_model`: used only when values are judged to agree (numeric close or text normalized equal).

Evidence blocks:
- For each evidence column X in:
  `evidence_section, evidence_span_text, evidence_span_start, evidence_span_end, evidence_method, evidence_quality`
  the consensus TSV includes:
  - `X_main`: chosen evidence block for quick human GT view.
    - Policy: pick the evidence from the model that contributes consensus for the first resolvable **key field** in order:
      `emul_type, emul_method, plga_mass_mg, organic_solvent, drug_name, size_nm`.
    - If a key field is `agree`, `preferred_model` decides which evidence becomes `*_main`.
    - If no key field has consensus, `*_main` is empty.
  - `X_model1`, `X_model2`: both models’ evidence (always kept for audit).

Field blocks (repeated for each field F):
- `F_main`: consensus value, conservatively defined:
  - text fields: normalized match -> consensus; conflict -> empty
  - numeric fields: relative difference ≤ numeric_rel_tol (default 0.2) -> consensus;
    otherwise conflict -> empty
  - single-sided value -> consensus from the side that provided it
- `F_status`: one of
  - `both_empty`, `only_model1`, `only_model2`, `agree`, `conflict`
- `F_conflict`: 0/1 (1 iff status is conflict)
- `F_model1`, `F_model2`: raw per-model values (strings, not coerced)

This layout is intentionally redundant to support:
- audit
- later arbitration (third-model judge or human)
- publishable tables
- robust downstream scripts

### 2) `formulations_conflict_queue.tsv` (conflict-only queue)

Subset of consensus TSV rows where `has_any_conflict=1`.
It contains the same columns as the consensus TSV.
This is the recommended input for your **minimal manual GT** step.

### 3) `field_level_qc_summary.tsv` (field-level consistency stats)

One row per field.
Columns:
- `field`
- `both_empty`, `only_model1`, `only_model2`, `agree`, `conflict`
- `total_formulations`
- `conflict_rate` = conflict / total_formulations
- `agree_rate` = agree / total_formulations

This table is meant to be directly referenced in Methods/Results to quantify model agreement.

## Recommended next step

1) Generate consensus + conflict queue.
2) Run manual GT only on `formulations_conflict_queue.tsv`.
3) Keep the consensus TSV as your weak-label baseline for evaluation and reporting.

