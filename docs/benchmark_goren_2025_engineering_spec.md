# Goren 2025 Benchmark Engineering Spec (Derivation + Projection + Alignment)

## Purpose
This spec defines a stable, auditable post-extraction benchmark pipeline for Goren 2025 comparability:

`extraction_full -> parsing -> derivation -> projection_to_curated -> alignment_eval`

Scope for this v1 spec:
- Do not change Stage2 extraction prompts or extraction logic.
- Build benchmark metrics only from projected curated-schema outputs.
- Preserve full value provenance for every derived/projected value.

## Inputs
- Frozen extraction baseline run:
  - `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv`
- Canonical dev18 manifest:
  - `data/cleaned/samples/sample_goren18.tsv`
- Curated benchmark source:
  - `data/benchmark/goren_2025/NP_dataset_formulations.csv`
- Optional curated overlap gold for evaluation:
  - `data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv`
- Immutable rules:
  - `data/benchmark/goren_2025/rules/derivation_rule_registry.v1.json`
  - `data/benchmark/goren_2025/rules/projection_ruleset.v1.json`

## Outputs
All outputs are written under:
- `data/results/<run_id>/benchmark_goren_2025/`

Core outputs:
- Derivation:
  - `derived_values.tsv`
  - `derivation_summary.json`
- Projection:
  - `projected_to_curated.tsv`
  - `projection_trace.tsv`
  - `projection_summary.json`
- Alignment evaluation:
  - `alignment_rows.tsv`
  - `metrics_summary.json`
  - `failure_types.tsv`

## Directory Layout
- Scripts:
  - `src/stage5_benchmark/run_derivation_v1.py`
  - `src/stage5_benchmark/run_projection_to_curated_v1.py`
  - `src/stage5_benchmark/run_alignment_eval_v1.py`
- Rules:
  - `data/benchmark/goren_2025/rules/*.v1.json`
- Run artifacts:
  - `data/results/<run_id>/benchmark_goren_2025/*`

## Naming Conventions
- Grouping key:
  - `group_key = key + "::" + formulation_id`
- File naming:
  - Stage artifacts are explicit and stable (`derived_values.tsv`, `projected_to_curated.tsv`, `alignment_rows.tsv`).
- Rule naming:
  - Derivation rules: `R_*`
  - Projection rules: `P_*`

## run_id Policy
- Baseline frozen extraction run is fixed for this branch:
  - `run_20260219_1623_780eb83_goren18_weaklabels_v1`
- Post-extraction stages write to:
  - `data/results/<run_id>/benchmark_goren_2025/`
- Re-runs for derivation/projection/eval do not mutate extraction outputs.

## Step-by-Step Workflow
1. Derivation (parsing + rule-based derivation)
   - Input: `weak_labels__gemini.tsv`
   - Parse:
     - `la_ga_ratio` -> `la_fraction`, `ga_fraction`, `LA/GA`
     - `plga_mw_kDa` -> lower/upper bounds (no midpoint derivation)
     - `plga_mass_mg`, `drug_feed_amount_text` -> numeric masses in mg
   - Derive:
     - `drug/polymer` if required inputs exist
     - Aqueous/organic only from explicit `evidence_span_text` with extracted anchors
   - Output: `derived_values.tsv` (long-form, auditable)

2. Projection to curated schema
   - Input: `derived_values.tsv`, curated template CSV, sample manifest DOI mapping
   - Project into exact curated columns set.
   - Keep nulls when not derivable.
   - Output:
     - `projected_to_curated.tsv` (strict curated columns only)
     - `projection_trace.tsv` (audit metadata)

3. Alignment evaluation
   - Input: `projected_to_curated.tsv`, curated gold TSV
   - Modes:
     - `strict`
     - `relaxed`
     - `canonicalized`
   - Output:
     - `alignment_rows.tsv` (per-row decision log)
     - `metrics_summary.json` (recall, precision, failure counts per mode)
     - `failure_types.tsv`

## Audit Requirements
For every derived/projected value, audit metadata must include:
- `rule_id`
- `derived_from`
- `value_source`
- `trace_pointer` to extracted evidence row context

Audit storage policy:
- `projected_to_curated.tsv` must contain only curated columns.
- All audit/provenance fields must be stored in trace artifacts (`derived_values.tsv`, `projection_trace.tsv`), not mixed into projected output.

## v1 Field Policies
- Group key: `key + formulation_id`
- Drug mass:
  - Parse from `drug_feed_amount_text` if no numeric column exists.
- Aqueous/organic:
  - Only derive from explicit evidence spans and extracted anchors.
  - Do not infer from notes or unrelated text.
  - Keep W1/O and (W1+W2)/O definitions separate; do not mix.

## Two-Table Schema
To reduce post-processing inflation, `schema_v1` adds a split model:

- `formulation_core.tsv`
  - One row per synthesis formulation definition.
  - Identity: DOI + deterministic `core_signature`.
  - Core signature uses normalized composition/synthesis anchors (drug, LA/GA, polymer MW bounds, solvent, drug/polymer ratio) with stable `missing` tokens for absent fields.

- `measurements.tsv`
  - Many rows per core formulation.
  - Stores characterization outcomes (`size_nm`, `ee_percent`, `lc_percent`, etc.) plus `condition_tag`.
  - `condition_tag` is heuristic:
    - `fresh` for no post-process indicators.
    - `postprocess` when evidence text contains keywords such as lyophilization, freeze-drying, cryoprotectants, storage, or reconstitution.

- `core_assignment_trace.tsv`
  - Auditable assignment of each extracted `group_key` to one `formulation_core_id`.
  - Captures assignment rule, condition tag, and matched condition keywords.

Why this solves inflation:
- Post-processing variants often preserve synthesis composition but alter measured outcomes.
- Treating these as measurement-condition rows under one core formulation prevents artificial formulation-count growth while retaining outcome variability.

## Two-Table Schema v2
`schema_v2` extends the core identity signature to reduce over-collapsing in DOE/orthogonal-design papers while still excluding post-processing from core identity.

- Ruleset:
  - `data/benchmark/goren_2025/rules/formulation_core_signature_ruleset.v2.json`
- Builder:
  - `src/stage5_benchmark/build_two_table_schema_v2.py`
- Outputs:
  - `data/results/<run_id>/benchmark_goren_2025/schema_v2/formulation_core.tsv`
  - `data/results/<run_id>/benchmark_goren_2025/schema_v2/measurements.tsv`
  - `data/results/<run_id>/benchmark_goren_2025/schema_v2/core_assignment_trace.tsv`

v2 identity behavior:
- Signature uses ordered inclusion of DOI plus composition and DOE-relevant variables (for example surfactant concentration / PLGA loading fields when available).
- Missing values are represented explicitly as deterministic `MISSING_<field>` tokens in signature composition.
- Post-processing/stability keywords do not change core identity and are recorded only as `condition_tag`/keyword context on measurement and assignment trace rows.

## CLI Usage
Derivation:
```bash
python src/stage5_benchmark/run_derivation_v1.py \
  --run-id run_20260219_1623_780eb83_goren18_weaklabels_v1
```

Projection:
```bash
python src/stage5_benchmark/run_projection_to_curated_v1.py \
  --run-id run_20260219_1623_780eb83_goren18_weaklabels_v1 \
  --curated-template data/benchmark/goren_2025/NP_dataset_formulations.csv \
  --sample-manifest data/cleaned/samples/sample_goren18.tsv
```

Alignment eval:
```bash
python src/stage5_benchmark/run_alignment_eval_v1.py \
  --run-id run_20260219_1623_780eb83_goren18_weaklabels_v1 \
  --curated-tsv data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv \
  --modes strict,relaxed,canonicalized
```

## core_eval_v1 Outputs
Purpose:
- Evaluate benchmark alignment at collapsed formulation-core granularity (one projected row per core), and quantify DOI-level multiplicity differences versus curated overlap rows.

Location:
- `data/results/<run_id>/benchmark_goren_2025/core_eval_v1/`

Key outputs:
- `core_projected_to_curated.tsv`
- `core_projection_trace.tsv`
- `core_alignment_rows.tsv`
- `core_metrics_summary.json`
- `core_failure_types.tsv`
- `core_row_membership_rows.tsv`
- `core_row_membership_summary.json`
- `core_row_membership_analysis.xlsx`

Reference command:
```bash
python src/stage5_benchmark/run_core_eval_pipeline_v1.py --run-id run_20260219_1623_780eb83_goren18_weaklabels_v1
```
## Tool Governance and Script Creation Policy

Before creating any new benchmark or evaluation scripts, developers must read docs/tool_index.md and verify whether equivalent functionality already exists. New scripts must not duplicate existing tools or introduce parallel implementations of the same logic.

If new functionality is required, it must either extend an existing script or clearly justify why a new script is necessary. Versioned scripts (e.g., _v2, _v3) must only be created when the previous version remains immutable for reproducibility purposes.

All new benchmark-related tools must:

- Respect the extraction -> parsing -> derivation -> projection -> alignment boundary.
- Avoid modifying Stage2 extraction logic or prompts.
- Write outputs only under data/results/<run_id>/benchmark_goren_2025/.
- Be registered in docs/tool_index.md immediately after creation.

No script may compute benchmark metrics directly from extraction outputs. All metrics must be computed exclusively from projected_to_curated.tsv.

## Full Database Export
For downstream analysis, a stable database snapshot interface is provided separately from benchmark/eval run artifacts.

- Export tool:
  - `src/stage5_benchmark/export_full_database_v1.py`
- Output interface:
  - `data/db/db_v1/formulation_core.tsv`
  - `data/db/db_v1/measurements.tsv`
  - `data/db/db_v1/factors.tsv`
  - `data/db/db_v1/schema_manifest.json`

Boundary:
- Benchmark/eval intermediates remain under `data/results/<run_id>/benchmark_goren_2025/`.
- Database exports are materialized under `data/db/<db_version>/` and can be consumed without coupling to run-folder internals.

## DOE coded factors: current table/evidence capability (recon)
Recon target run:
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/`

Stage2 extraction outputs found under run folder:
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.jsonl`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv`

DOE pain-point DOI inspected:
- DOI `10.1002/jps.24101` maps to key `WIVUCMYG` in `data/cleaned/samples/sample_goren18.tsv`.
- In `weak_labels__gemini.tsv`, this key has 29 extracted formulation rows.

Table/evidence capability findings:
- Tables present (structured table objects/anchors): **N**
- Structured table-like fields observed in Stage2 outputs:
  - None found for fields such as `table`, `table_id`, `table_text`, `row`, `col`, `cell`, `caption`.
- Section/evidence fields available in TSV:
  - `evidence_section`
  - `evidence_span_text`
  - `evidence_span_start`
  - `evidence_span_end`
  - `evidence_method`
  - `evidence_quality`
- For DOI `10.1002/jps.24101` specifically:
  - `evidence_section=fulltext` for all rows
  - `evidence_method=pattern_window` for all rows
  - Non-empty start/end offsets for all rows
  - No explicit table-caption/table-anchor metadata detected

Evidence granularity classification:
- **Section + character offsets** (plus extracted span text)
- Not limited to char offsets only.
- No table-caption anchors or structured table-cell provenance in current Stage2 extraction outputs.

## DOE coded factors support (v1)
Scope boundary:
- No Stage2 prompt or extraction-logic changes.
- DOE decoding support is implemented in parsing/derivation and downstream projection/export layers.

Input expectation from extraction outputs:
- Required minimum:
  - formulation-level extracted rows (for example `weak_labels__gemini.tsv`).
  - evidence metadata per row: `evidence_section`, `evidence_span_text`, `evidence_span_start`, `evidence_span_end`, `evidence_method`, `evidence_quality`.
- Optional richer input (if available in future runs):
  - raw table text blocks and/or structured table objects.
  - table-level anchors (table id/caption/row/column/cell provenance).
- v1 must work with current run reality (section + span evidence, no structured table objects).

Intermediate objects in parsing/derivation:
1. `doe_codebook`
- Purpose: normalize coded-factor legend mappings.
- Shape:
  - `factor_name -> level_code -> (real_value_num, unit, value_raw, evidence_anchor)`
- Notes:
  - `level_code` examples: `A1`, `B2`, `X3`.
  - `evidence_anchor` points to extraction evidence context (at minimum section + span offsets; include table anchor if available).
  - Keep both numeric and raw value text to preserve auditability.

2. `doe_runs`
- Purpose: represent each DOE run/row as factor assignments + outcomes.
- Shape per run:
  - `run_id_or_row_id`
  - `assignments`: `factor_name -> level_code`
  - `outcomes`: measured outputs (for example `size_nm`, `zeta_mV`, `encapsulation_efficiency_percent`, `loading_content_percent`)
  - `evidence_anchor`
- Notes:
  - Outcomes may be directly extracted or parsed from evidence spans.
  - Anchors must preserve trace to original extracted context.

Output policy:
- Always keep coded-factor rows in `factors.tsv` (coded values are first-class data, not discarded).
- Add decoded factor rows when `doe_codebook` mapping is available.
- Decoded values may be copied into formulation core fields only when field is in a small core-copy allowlist, and copied rows must carry provenance `decoded_from_codebook`.
- Decoding failures must be logged in diagnostics with reason categories.

Decoding diagnostics (minimum reason categories):
- `missing_codebook_for_factor`
- `missing_level_code_mapping`
- `ambiguous_level_code_mapping`
- `non_numeric_decoded_value_when_numeric_required`
- `unit_conflict_or_unknown_unit`
- `insufficient_evidence_anchor`

## Re-eval after float-coded DOE support (2026-02-23)
Run command:
- `python src/stage5_benchmark/run_alignment_eval_schema_v3_v1.py --run-id run_20260219_1623_780eb83_goren18_weaklabels_v1`

Decoder capability update:
- DOE coded levels now support discrete float codes in addition to integer codes.
- Example captured levels include `-1.68` and `1.68` (for `10.1016/j.colsurfb.2009.03.028`).

`schema_v3` overall metrics after float-coded support:
- `strict`: precision `0.532258`, recall `0.519685`, f1 `0.525896`
- `relaxed`: precision `0.669355`, recall `0.653543`, f1 `0.661355`
- `canonicalized`: precision `0.532258`, recall `0.519685`, f1 `0.525896`

Delta vs pre-rerun snapshot:
- `relaxed` improved (`+0.008198` precision, `+0.023622` recall, `+0.016193` f1).
- `strict` and `canonicalized` had small precision/f1 decreases (`-0.013196` precision, `-0.006362` f1) with unchanged recall; this is consistent with additional projected rows after DOE float-level expansion increasing strict/canonicalized unmatched projected rows.

DOE-enabled DOI recalls (`schema_v3`):
- `10.1002/jps.24101`: strict recall `0.961538`, relaxed recall `1.000000`, canonicalized recall `0.961538`.
- `10.1016/j.colsurfb.2009.03.028`: strict recall `0.000000`, relaxed recall `0.941176`, canonicalized recall `0.000000`.

DOE decoder CLI knobs (defaults):
- `--max-code-abs` (default `3.0`)
- `--max-code-unique` (default `9`)
- `--code-match-tol` (default `1e-6`)
