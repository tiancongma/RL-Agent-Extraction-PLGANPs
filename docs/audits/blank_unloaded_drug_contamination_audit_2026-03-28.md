# Blank / Unloaded Drug Contamination Audit

Date: `2026-03-28`

Primary workbook inspected:
`C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\value_gt_annotation_workbook_representation_repaired_v4.xlsx`

Authoritative run resolved:
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`
- Source authority file:
  `data/results/ACTIVE_RUN.json`

## 1. Workbook inspection

Workbook sheet names:
- `value_gt_annotation`
- `metadata`

Target sheet:
- `value_gt_annotation`

Drug-related / identity-related headers detected on the target sheet:
- `paper_key`
- `gt_formulation_id`
- `seed_pred_representative_source_formulation_id`
- `formulation_label`
- `drug_name`
- `drug_mass_mg`
- `drug_concentration_value`
- `drug_concentration_unit`
- `drug_to_polymer_ratio_raw`
- `candidate_notes`

Candidate blank / empty / unloaded rows found in the current v4 workbook: `13`

All candidate rows from the current v4 workbook:

| paper_key | gt_formulation_id | formulation_label | source formulation id | current workbook drug_name |
|---|---|---|---|---|
| 5GIF3D8W | 5GIF3D8W_G031 | PLGA 50/50 (Empty) | F1 | Etoposide |
| 5GIF3D8W | 5GIF3D8W_G033 | PLGA 75/25 (Empty) | F3 | Etoposide |
| 5GIF3D8W | 5GIF3D8W_G035 | PLGA 85/15 (Empty) | F5 | Etoposide |
| 5GIF3D8W | 5GIF3D8W_G037 | PCL (Empty) | F7 | Etoposide |
| BXCV5XWB | BXCV5XWB_G002 | Blank PLGA nanoparticles | PLGA-NP-Blank-01 | FITC |
| BXCV5XWB | BXCV5XWB_G006 | Blank PLGA–PEG–HA nanoparticles | PLGA-PEG-HA-NP-Blank-01 | FITC |
| BXCV5XWB | BXCV5XWB_G009 | Blank PLGA–PEG nanoparticles | PLGA-PEG-NP-Blank-01 | FITC |
| L3H2RS2H | L3H2RS2H_G006 | Empty nanocapsules | Nanocapsule-Empty-General | XAN |
| L3H2RS2H | L3H2RS2H_G007 | Empty nanocapsules (0.6 mL Myritol 318 and without xanthones) | Nanocapsule-Empty-Table5 | XAN |
| L3H2RS2H | L3H2RS2H_G017 | Empty nanospheres | Nanosphere-Empty | XAN |
| PA3SPZ28 | PA3SPZ28_G001 | Drug free nanoparticles | Blank-NP | Etoposide |
| QLYKLPKT | QLYKLPKT_G002 | 3 mg/mL Poloxamer 188 (Optimal Unloaded) | F1.2 | blank |
| RHMJWZX8 | RHMJWZX8_G002 | empty NPs | Formulation_2 | blank |

Current-workbook candidate rows whose `drug_name` cell is non-empty: `11`

Rows that are not current-workbook errors:
- `QLYKLPKT_G002`
  - `drug_name` is blank in the current `...representation_repaired_v4.xlsx`.
  - It is still an upstream error because `GAR` appears in `...representation_repaired_v2.xlsx` and `...representation_repaired_v3.xlsx`.
- `RHMJWZX8_G002`
  - `drug_name` is blank in the current `...representation_repaired_v4.xlsx`.
  - It is still an upstream error because `GAR` appears in `...representation_repaired_v2.xlsx` and `...representation_repaired_v3.xlsx`.

## 2. Confirmed erroneous rows

Confirmed erroneous current-workbook rows: `11`

These rows are confirmed erroneous because:
- the labels explicitly say `blank`, `empty`, `drug free`, or `unloaded`
- the benchmark-valid Stage 5 final table leaves `drug_name_value` blank
- the current v7 value-alignment rows leave `sys_drug_name` blank
- the raw weak-label responses explicitly mark the formulation as unloaded / drug-free / no drug

Confirmed erroneous current-workbook rows:

| paper_key | gt_formulation_id | formulation_label | wrong current workbook drug_name |
|---|---|---|---|
| 5GIF3D8W | 5GIF3D8W_G031 | PLGA 50/50 (Empty) | Etoposide |
| 5GIF3D8W | 5GIF3D8W_G033 | PLGA 75/25 (Empty) | Etoposide |
| 5GIF3D8W | 5GIF3D8W_G035 | PLGA 85/15 (Empty) | Etoposide |
| 5GIF3D8W | 5GIF3D8W_G037 | PCL (Empty) | Etoposide |
| BXCV5XWB | BXCV5XWB_G002 | Blank PLGA nanoparticles | FITC |
| BXCV5XWB | BXCV5XWB_G006 | Blank PLGA–PEG–HA nanoparticles | FITC |
| BXCV5XWB | BXCV5XWB_G009 | Blank PLGA–PEG nanoparticles | FITC |
| L3H2RS2H | L3H2RS2H_G006 | Empty nanocapsules | XAN |
| L3H2RS2H | L3H2RS2H_G007 | Empty nanocapsules (0.6 mL Myritol 318 and without xanthones) | XAN |
| L3H2RS2H | L3H2RS2H_G017 | Empty nanospheres | XAN |
| PA3SPZ28 | PA3SPZ28_G001 | Drug free nanoparticles | Etoposide |

Confirmed upstream-only rows with wrong non-empty drug in the representation-repair lineage, but blank in the current v4 workbook:

| paper_key | gt_formulation_id | formulation_label | upstream wrong value | first upstream file showing it |
|---|---|---|---|---|
| QLYKLPKT | QLYKLPKT_G002 | 3 mg/mL Poloxamer 188 (Optimal Unloaded) | GAR | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` |
| RHMJWZX8 | RHMJWZX8_G002 | empty NPs | GAR | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` |

## 3. Per-row provenance summary

Per-row chain summary:

| gt_formulation_id | workbook v4 drug_name | v7 sys_drug_name | Stage5 final drug_name_value | weak-label drug_name_value | first wrong artifact | root-cause class |
|---|---|---|---|---|---|---|
| 5GIF3D8W_G031 | Etoposide | blank | blank | blank | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=Etoposide`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| 5GIF3D8W_G033 | Etoposide | blank | blank | blank | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` | workbook materialization bug |
| 5GIF3D8W_G035 | Etoposide | blank | blank | blank | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` | workbook materialization bug |
| 5GIF3D8W_G037 | Etoposide | blank | blank | blank | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=Etoposide`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| BXCV5XWB_G002 | FITC | blank | blank | blank | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| BXCV5XWB_G006 | FITC | blank | blank | blank | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| BXCV5XWB_G009 | FITC | blank | blank | blank | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| L3H2RS2H_G006 | XAN | blank | blank | blank | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=3-Methoxyxanthone (3-MeOXAN)`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| L3H2RS2H_G007 | XAN | blank | blank | blank | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=Xanthone (XAN)`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| L3H2RS2H_G017 | XAN | blank | blank | blank | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=3-Methoxyxanthone (3-MeOXAN)`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` | loaded-to-blank variant contamination + workbook materialization bug |
| PA3SPZ28_G001 | Etoposide | blank | blank | blank | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=Garcinol`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` | workbook materialization bug |
| QLYKLPKT_G002 | blank in v4 | blank | blank | blank | `value_gt_annotation_workbook_representation_repaired_v2.xlsx` shows `GAR` | workbook materialization bug |
| RHMJWZX8_G002 | blank in v4 | blank | blank | blank with `drug_name_missing_reason=Not applicable (empty nanoparticles)` | `value_gt_annotation_workbook_v4.xlsx` already carries `l2_gt_formulation_label=Acetylpuerarin`; displayed `drug_name` first appears in `...representation_repaired_v2.xlsx` as `GAR` | workbook materialization bug |

## 4. Root cause

Main finding:
- The contamination does **not** start in the benchmark-valid Stage 5 final table.
- The contamination does **not** start in the active v7 value-alignment rows.
- The displayed non-empty `drug_name` values are introduced in the downstream workbook-representation-repair layer, especially:
  - `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py`
  - output workbook: `value_gt_annotation_workbook_representation_repaired_v2.xlsx`

Direct code-level cause:
- `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py:705-709` writes:
  - `output["drug_name"] = chosen.get("drug_name_candidates", "") or detect_drug_name(...)`
- `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py:446-451` defines `detect_drug_name`
- `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py:422-427` defines `detect_name`
- `detect_name` is a simple case-insensitive substring test over arbitrary text, with no row-local evidence requirement and no word-boundary requirement.

That creates two failure modes:
- workbook materialization assigns a paper-level or section-level drug token to a blank/unloaded row when `drug_name_candidates` is blank
- short codes such as `GAR` can be matched inside unrelated strings
  - confirmed example:
    - `QLYKLPKT.pdf.txt` first `gar` hit is in author name `Garcia ML`
  - likely example:
    - `RHMJWZX8.pdf.txt` first `gar` hit is not a real garcinol mention and appears inside unrelated text

Secondary earlier issue:
- Some rows already carried an incorrect `l2_gt_formulation_label` in the older compact workbook lineage:
  - `5GIF3D8W_G031`, `5GIF3D8W_G037`
  - `L3H2RS2H_G006`, `L3H2RS2H_G007`, `L3H2RS2H_G017`
  - `PA3SPZ28_G001`
  - `RHMJWZX8_G002`
- This earlier label leakage is an alignment / row-materialization problem inside the older workbook lineage.
- The representation-repair script then turns the problem into a visible `drug_name` field.

## 5. Evidence from source artifacts

Evidence that the formulations are truly blank / unloaded:
- `5GIF3D8W` raw response:
  - `PLGA 50/50 (Empty)` etc
  - `value_text: Drug free`
  - `missing_reason: Not applicable for empty formulation`
- `BXCV5XWB` raw response:
  - `Blank PLGA nanoparticles`
  - `No drug loaded into the nanoparticles.`
  - `missing_reason: No drug included`
- `L3H2RS2H` raw response:
  - `Empty nanospheres`
  - `Empty nanocapsules`
  - `omitting the xanthones`
- `PA3SPZ28` raw response:
  - `Drug free nanoparticles`
  - `Drug free nanoparticles were prepared according to the same procedure.`
- `QLYKLPKT` raw response:
  - `3 mg/mL Poloxamer 188 (Optimal Unloaded)`
  - `missing_reason: Unloaded formulation`
- `RHMJWZX8` raw response:
  - `empty NPs`
  - `No drug (Acetylpuerarin) loaded.`
  - `missing_reason: Not applicable (empty nanoparticles)`

Evidence that benchmark-valid mainline artifacts are clean for these rows:
- `final_formulation_table_v1.tsv`
  - all traced affected rows have blank `drug_name_value`
  - blank-control rows also show `loaded_state_final=empty` or blank-control payload state
- `value_gt_s1_core_alignment_contract_v1/value_gt_annotation_rows_v7.tsv`
  - all traced affected rows have blank `sys_drug_name`

## 6. Grouped counts

Current workbook non-empty wrong drug rows by paper:
- `5GIF3D8W`: `4`
- `BXCV5XWB`: `3`
- `L3H2RS2H`: `3`
- `PA3SPZ28`: `1`

Current workbook non-empty wrong drug rows by displayed drug:
- `Etoposide`: `5`
- `FITC`: `3`
- `XAN`: `3`

All confirmed affected rows across current workbook + upstream representation-repair lineage by paper:
- `5GIF3D8W`: `4`
- `BXCV5XWB`: `3`
- `L3H2RS2H`: `3`
- `PA3SPZ28`: `1`
- `QLYKLPKT`: `1`
- `RHMJWZX8`: `1`

All confirmed affected rows across current workbook + upstream representation-repair lineage by root-cause class:
- `loaded-to-blank variant contamination + workbook materialization bug`: `8`
- `workbook materialization bug`: `5`

## 7. GAR-specific or systematic?

Not GAR-specific.

What is true on `2026-03-28` for the exact workbook path provided:
- the current `...representation_repaired_v4.xlsx` workbook does **not** visibly show `GAR` in its `drug_name` cells for the blank/unloaded candidate rows
- the current v4 workbook still shows other wrong drugs:
  - `Etoposide`
  - `FITC`
  - `XAN`

What is true upstream:
- `GAR` is present in upstream workbook lineage artifacts:
  - `value_gt_annotation_workbook_representation_repaired_v2.xlsx`
  - `value_gt_annotation_workbook_representation_repaired_v3.xlsx`
- the `GAR` rows are:
  - `QLYKLPKT_G002`
  - `RHMJWZX8_G002`

So the issue is systematic reviewer-surface contamination, not a GAR-only phenomenon.

## 8. Recommended next action

Recommended next action:
1. Treat this as a downstream workbook-generation / workbook-repair defect, not a Stage 2 / Stage 3 / Stage 5 benchmark defect.
2. Audit and patch `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py` in a separate task:
   - remove `detect_drug_name(...)` fallback for blank / unloaded / control rows
   - require explicit row-local `drug_name_candidates` support before populating `drug_name`
   - add word-boundary-safe matching if any code-based fallback remains
   - do not scan full paper text or bibliography for blank-control drug names
3. Audit the older compact workbook alignment path that wrote incorrect `l2_gt_formulation_label` values for several blank rows.
4. Document the undocumented change that blanked `QLYKLPKT_G002` and `RHMJWZX8_G002` between `representation_repaired_v3.xlsx` and `representation_repaired_v4.xlsx`, because the v4 audit note only documents WIVUCMYG concentration repairs.

## 9. Exact files inspected

- `data/results/ACTIVE_RUN.json`
- `project/ACTIVE_DATA_SOURCE_CONTRACT.md`
- `project/0_PROJECT_CHARTER.md`
- `project/1_REQUIREMENTS.md`
- `project/2_ARCHITECTURE.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `project/memory_maintenance_audit_2026-03-27.md`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.audit.md`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v3.xlsx`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v3.audit.md`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v2.xlsx`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_v4.xlsx`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_v6_merged_alignment_correction_v1.xlsx`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_with_phase_values_v2.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_with_phase_and_polymer_values_v1.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_s1_core_alignment_contract_v1/value_gt_annotation_rows_v7.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/44_stage5_descendant_fix_v1fixed_recompare/run_20260321_1454_5fa3ed0_dev15_stage5_descendant_fix_v1fixed_recompare_v1/final_formulation_table_v1.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/44_stage5_descendant_fix_v1fixed_recompare/run_20260321_1454_5fa3ed0_dev15_stage5_descendant_fix_v1fixed_recompare_v1/final_output_decision_trace_v1.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/51_value_gt_annotation_representation_repair/run_20260325_163435_9d4c2ab_dev15_value_gt_annotation_representation_repair_v2/RUN_CONTEXT.md`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/51_value_gt_annotation_representation_repair/run_20260325_163435_9d4c2ab_dev15_value_gt_annotation_representation_repair_v2/value_gt_annotation_cell_preservation_summary_v2.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/02_L3H2RS2H_10.1016_j.ejpb.2004.09.002.txt`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/04_5GIF3D8W_10.1080_10717540802174662.txt`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/07_BXCV5XWB_10.1007_s10439-019-02430-x.txt`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/09_PA3SPZ28_10.1038_s41598-017-00696-6.txt`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/10_QLYKLPKT_10.2147_ijn.s54040.txt`
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/11_RHMJWZX8_10.1111_jphp.12481.txt`
- `data/cleaned/content_goren_2025/text/5GIF3D8W.pdf.txt`
- `data/cleaned/content_goren_2025/text/BXCV5XWB.html.txt`
- `data/cleaned/content_goren_2025/text/L3H2RS2H.pdf.txt`
- `data/cleaned/content_goren_2025/text/PA3SPZ28.pdf.txt`
- `data/cleaned/content_goren_2025/text/QLYKLPKT.pdf.txt`
- `data/cleaned/content_goren_2025/text/RHMJWZX8.pdf.txt`
- `data/cleaned/goren_2025/tables/5GIF3D8W/5GIF3D8W__table_04__pdf_table.csv`
- `data/cleaned/goren_2025/tables/L3H2RS2H/L3H2RS2H__table_03__pdf_table.csv`
- `data/cleaned/goren_2025/tables/L3H2RS2H/L3H2RS2H__table_04__pdf_table.csv`
- `data/cleaned/goren_2025/tables/QLYKLPKT/QLYKLPKT__table_08__pdf_table.csv`
- `data/cleaned/goren_2025/tables/PA3SPZ28/PA3SPZ28__table_07__pdf_table.csv`
- `data/cleaned/goren_2025/tables/RHMJWZX8/RHMJWZX8__table_19__pdf_table.csv`
- `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py`
- `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py`

## 10. Unresolved ambiguities

- The exact current workbook you requested, `...representation_repaired_v4.xlsx`, no longer shows `GAR` in the visible `drug_name` cells for `QLYKLPKT_G002` and `RHMJWZX8_G002`.
- The immediately earlier upstream files `...representation_repaired_v2.xlsx` and `...representation_repaired_v3.xlsx` do show `GAR` for both rows.
- The `...representation_repaired_v4.audit.md` file documents only WIVUCMYG concentration repairs and does not explain why those two GAR cells are now blank.
- That undocumented transition is unresolved and should be traced separately if the exact edit history of the workbook lineage matters.
