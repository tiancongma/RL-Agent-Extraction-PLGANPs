# Preparation Method Field Gap Diagnosis

## Scope inspected

- Active Stage 2 extractor: `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- Active Stage 5 closure/export path: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Downstream database export allowlist: `src/stage5_benchmark/export_full_database_v1.py`
- Latest deterministic DEV15 benchmark-valid lineage:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/RUN_CONTEXT.md`
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/final_formulation_table_v1.tsv`

## Direct answer

1. Does the current active schema have a general preparation-method field?

No. In the active Stage 2 extractor, `CORE_FIELDS` includes `emul_type` and `emul_method`, but no neutral `preparation_method` or equivalent field. Stage 5 then carries forward the original Stage 2 columns verbatim into `final_formulation_table_v1.tsv`, so the final table also has no general preparation-method column. The downstream export allowlist in `export_full_database_v1.py` also exposes `emul_method` and `emul_type`, not a broader method field.

2. Are non-emulsion methods at risk of being lost because only emulsion-specific fields are available?

Yes. Repo evidence shows that non-emulsion methods are currently forced into emulsion-specific columns when they survive at all. This creates two failure modes:

- partial preservation by semantic misfit:
  - `5ZXYABSU`: `emul_type_value = nanoprecipitation`, `emul_method_value = acetone-water system`
  - `INMUTV7L`: `emul_method_value = solvent displacement method`, `emul_type_value` blank
  - `PA3SPZ28`: `emul_method_value = nanoprecipitation`, `emul_type_value` blank
- full drop when extraction does not place the method into one of those emulsion-only slots:
  - `UFXX9WXE`: source text explicitly says nanoprecipitation, but both `emul_method_value` and `emul_type_value` are blank in active Stage 2 and active Stage 5

3. Is there repo evidence that nanoprecipitation-like methods are being dropped or weakly preserved?

Yes.

- Weakly preserved:
  - `5ZXYABSU` (`10.2147/IJN.S130908`): source says “prepared by nanoprecipitation”; active Stage 2 and Stage 5 preserve this only through `emul_*` columns.
  - `INMUTV7L` (`10.3390/nano10040720`): source says “prepared by using the solvent displacement method”; active Stage 2 and Stage 5 keep that only in `emul_method_value`.
  - `PA3SPZ28` (`10.1038/s41598-017-00696-6`): source says nanoprecipitation; active outputs keep it only in `emul_method_value`.
- Confirmed drop:
  - `UFXX9WXE` (`10.1155/2014/156010`): source says “successfully developed by nanoprecipitation method”, but both active Stage 2 and active Stage 5 leave the method fields blank.

4. What is the most likely loss mechanism?

Most likely: schema gap first, extraction inconsistency second.

- Schema gap:
  - the active schema gives non-emulsion methods no neutral destination;
  - surviving values are being packed into `emul_type` / `emul_method` by semantic misfit.
- Extraction inconsistency:
  - `UFXX9WXE` shows that even explicit nanoprecipitation evidence is not always landing in those emulsion-named slots.
- Final export omission:
  - not the primary problem in the active Stage 2 -> Stage 5 path;
  - Stage 5 largely preserves Stage 2 columns as-is, so whatever survives in Stage 2 usually survives into `final_formulation_table_v1.tsv`.

5. What schema change is recommended?

Recommended active schema change:

- add a top-level `preparation_method` field for the general formulation-preparation approach
- keep or rename emulsion-only structure detail as a subordinate field, for example `emulsion_structure`

Likely mapping:

- `preparation_method`
  - examples: `nanoprecipitation`, `solvent displacement`, `solvent diffusion`, `single emulsion solvent evaporation`, `double emulsion solvent evaporation`, `microfluidic`
- `emulsion_structure`
  - examples: `O/W`, `W/O/W`, `single emulsion`, `double emulsion`

This would let non-emulsion methods survive without being squeezed into emulsion-specific semantics while still preserving emulsion-only subtype detail when relevant.

## Evidence path

### Active schema and export behavior

- `auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  - active `CORE_FIELDS` includes `emul_type` and `emul_method`
  - no neutral preparation-method field is defined
- `build_minimal_final_output_v1.py`
  - Stage 5 reads Stage 2 candidate rows
  - `original_fieldnames = list(rows[0].keys())`
  - final-table writing appends `*original_fieldnames`, so Stage 2 columns survive directly into Stage 5
- `export_full_database_v1.py`
  - factor allowlist includes `emul_method` and `emul_type`
  - no `preparation_method` alias appears

### Focused diagnostic sample

See:

- `analysis/preparation_method_field_gap_diagnosis.tsv`

Interpretation of the sample:

- `5ZXYABSU`, `INMUTV7L`, and `PA3SPZ28` show the weak-preservation pattern
- `UFXX9WXE` shows a confirmed method drop in the active deterministic DEV15 lineage
- `RHMJWZX8` is a control showing that emulsion-shaped representations fit the current schema better

## Quantified current population

Active deterministic DEV15 run:

- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`

Counts:

- Stage 2 rows with nonblank `emul_method_value`: `103 / 269`
- Stage 2 rows with nonblank `emul_type_value`: `22 / 269`
- Stage 5 rows with nonblank `emul_method_value`: `91 / 223`
- Stage 5 rows with nonblank `emul_type_value`: `16 / 223`

Confirmed rows with explicit non-emulsion method evidence but blank method fields:

- `UFXX9WXE`
  - Stage 2 rows: `28`
  - Stage 5 rows: `27`
  - source text contains explicit nanoprecipitation evidence
  - both `emul_method_value` and `emul_type_value` are blank throughout the active outputs for this paper

Additional papers with automated term hits plus blank method fields do exist (`WIVUCMYG`, `BB3JUVW7`, `YGA8VQKU`), but the current local hits I inspected are dominated by references/recommended-article text rather than clean method statements from the paper body, so I am not counting those as confirmed drops.

## Conclusion

The current active schema likely does cause non-emulsion preparation-method loss.

- When extraction succeeds, non-emulsion methods are often only weakly preserved by being stored inside emulsion-specific columns.
- When extraction does not force the method into those columns, there is no neutral schema destination, and the method can disappear from both Stage 2 and Stage 5.
- `UFXX9WXE` is the clearest active DEV15 example of that loss mode.
