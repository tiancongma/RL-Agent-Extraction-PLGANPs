# 1. Executive conclusion

FACT: The most defensible lineage for `value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx` is a two-part chain: a maintained canonical chain that freezes row identity and value-candidate surfaces from Stage1 clean text through Stage2, Stage3, Stage5, boundary review, field-review export, and GT-skeleton value-workbook generation; then an advisory/manual root-side repair chain that mutates reviewer workbooks after the maintained child runs. FACT: The exact target workbook was not emitted by any recorded maintained run script or `RUN_CONTEXT.md`. INFERENCE: The immediate parent is `value_gt_annotation_workbook_representation_repaired_v4.xlsx`, and the target was created by an untracked root-side pH backfill step that added only `pH_raw` values (42 fills) to that workbook. UNCERTAINTY: the repo does not preserve the exact script, command, or operator action that created `..._with_pH.xlsx`, nor the exact producer steps for several earlier root-side helper/repair workbooks.

# 2. Artifact identity

- FACT: Target workbook absolute path: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`
- FACT: Producing lineage root: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`
- FACT: Artifact role: downstream Layer3 field/value annotation workbook surface, not the Layer2 boundary mother workbook.
- FACT: Boundary class: inherits the accepted formulation universe from the reviewed `include_gt` subset of `boundary_gt_review_workbook_v1.xlsx` (`project/ACTIVE_PIPELINE_RUNBOOK.md:811-814`, `project/4_DECISIONS_LOG.md:2945-2946`).
- FACT: Canonical vs review-support only: the target workbook is a review-support / repaired / advisory workbook, not a benchmark-valid canonical production artifact.

# 3. Direct producer

- FACT: No exact producer script or exact command was found for the exact filename `value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`.
- INFERENCE: Best reconstruction is:
  - immediate parent artifact: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\value_gt_annotation_workbook_representation_repaired_v4.xlsx`
  - mutation step: root-side pH backfill adding a new `pH_raw` column and 42 populated cells
  - evidence:
    - `value_gt_annotation_pH_backfill_audit.tsv` records 42 `FILL` actions and 168 `SKIP` actions for the workbook row set.
    - Workbook diff shows `v4 -> with_pH` changes only in new column `pH_raw`; no other preexisting cells differ.
    - Workbook diff shows `pdi_zeta_backfilled_from_semantic_stage2.xlsx` is a sibling branch, not the parent, because `with_pH` does not contain the 33 `pdi`/`zeta_mV` backfills present in that sibling.
- FACT: Embedded workbook metadata in the target still reports the March 25 representation-repair metadata block from the earlier v2 repair family, not a fresh producer record:
  - `current_workbook_xlsx = ...value_gt_annotation_workbook_with_phase_and_polymer_values_v1.xlsx`
  - `machine_baseline_tsv = ...value_gt_annotation_workbook_with_phase_and_polymer_values_v1.tsv`
  - `boundary_workbook_xlsx = ...boundary_gt_review_v1\\boundary_gt_review_workbook_v1.xlsx`
  - `gt_skeleton_tsv = ...dev15_formulation_skeleton_gt_v2_variantaware.tsv`
  - `generated_at_utc = 2026-03-25T20:35:46.230021+00:00`
- DECISION RULE: Because governance forbids naming-based inference and no exact producer run is recorded, the direct-producer claim is limited to the evidence-backed immediate-parent mutation, not an invented script invocation.

# 4. Recursive provenance tree

- FACT: `...\value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`
  - producer: unresolved root-side pH backfill step
  - role: repaired reviewer workbook
  - upstream inputs:
    - `...\value_gt_annotation_workbook_representation_repaired_v4.xlsx`
    - `...\value_gt_annotation_pH_backfill_audit.tsv`
    - unresolved pH source artifact(s) for 42 fills
    - evidence: workbook diff + audit TSV
  - FACT: `...\value_gt_annotation_workbook_representation_repaired_v4.xlsx`
    - producer: unresolved root-side v4 repair step
    - role: repaired reviewer workbook
    - upstream inputs:
      - `...\value_gt_annotation_workbook_representation_repaired_v3.xlsx`
      - Stage1 cleaned tables for WIVUCMYG:
        - `data\cleaned\goren_2025\tables\WIVUCMYG\WIVUCMYG__table_13__pdf_table.csv`
        - `data\cleaned\goren_2025\tables\WIVUCMYG\WIVUCMYG__table_05__html_table.csv`
        - `data\cleaned\goren_2025\tables\WIVUCMYG\WIVUCMYG__table_01__html_table.csv`
        - `data\cleaned\goren_2025\tables\WIVUCMYG\WIVUCMYG__table_14__pdf_table.csv`
      - advisory cross-checks:
        - parent Stage2 raw response `...\weak_labels_v7pilot_r3_fixparse\raw_responses\03_WIVUCMYG_10.1002_jps.24101.txt`
        - parent Stage2 TSV `...\weak_labels__v7pilot_r3_fixparse.tsv`
        - child 37 field seed `...\fgt_v5_dev15_v2\field_gt_review_seed_rows_v5.tsv`
        - `...\value_gt_s1_core_alignment_contract_v1\value_gt_annotation_rows_v7.tsv`
      - evidence: `value_gt_annotation_workbook_representation_repaired_v4.audit.md`
    - FACT: `...\value_gt_annotation_workbook_representation_repaired_v3.xlsx`
      - producer: unresolved root-side v3 repair step
      - role: repaired reviewer workbook
      - upstream inputs:
        - root copy of `...\value_gt_annotation_workbook_representation_repaired_v2.xlsx`
        - Stage1 cleaned tables for WFDTQ4VX:
          - `data\cleaned\goren_2025\tables\WFDTQ4VX\WFDTQ4VX__table_12__pdf_table.csv`
          - `data\cleaned\goren_2025\tables\WFDTQ4VX\WFDTQ4VX__table_14__pdf_table.csv`
        - advisory reconciliation TSV:
          - `data\results\doe_coordinate_reconciliation_v1\WFDTQ4VX_reconciled_instances.tsv`
        - evidence: `value_gt_annotation_workbook_representation_repaired_v3.audit.md`
      - FACT: `...\value_gt_annotation_workbook_representation_repaired_v2.xlsx`
        - producer: copied/derived root-side counterpart of child 51 representation repair output; copy step itself unresolved
        - role: representation-aware repair workbook
        - upstream inputs:
          - `...\value_gt_annotation_workbook_with_phase_and_polymer_values_v1.xlsx`
          - `...\value_gt_annotation_workbook_with_phase_and_polymer_values_v1.tsv`
          - `...\boundary_gt_review_v1\boundary_gt_review_workbook_v1.xlsx`
          - `data\cleaned\labels\manual\dev15_formulation_skeleton\dev15_formulation_skeleton_gt_v2_variantaware.tsv`
        - FACT: child 51 recorded producer script is `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py`
        - FACT: child 51 recorded outputs are only `...lineage\children\51_...\value_gt_annotation_workbook_representation_repaired_v2.xlsx/.tsv`
        - FACT: `...\value_gt_annotation_workbook_with_phase_and_polymer_values_v1.xlsx`
          - producer: unresolved root-side helper-fill step
          - role: helper-enriched reviewer workbook
          - upstream inputs:
            - `...\value_gt_annotation_workbook_with_phase_values_v2.xlsx`
            - Stage1 clean text / local evidence sources
          - evidence: `value_gt_annotation_workbook_with_phase_and_polymer_values_v1.summary.json`
        - FACT: `...\value_gt_annotation_workbook_with_phase_values_v2.xlsx`
          - producer: unresolved root-side helper-fill step
          - role: phase-helper reviewer workbook
          - upstream inputs:
            - `...\value_gt_annotation_workbook_with_phase_values_v1.xlsx`
            - Stage1 clean text files for affected papers
            - `analysis\value_gt_annotation_gt_skeleton_compact_v1\value_gt_annotation_workbook_gt_skeleton_compact_rows_v1.tsv`
          - evidence: `value_gt_annotation_workbook_with_phase_values_v2.summary.json`
        - FACT: `...\value_gt_annotation_workbook_with_phase_values_v1.xlsx`
          - producer: unresolved root-side helper-fill step
          - role: phase-helper reviewer workbook
          - upstream inputs: unresolved, but later summary names it as the direct predecessor of v2
        - FACT: `analysis\value_gt_annotation_gt_skeleton_compact_v1\value_gt_annotation_workbook_gt_skeleton_compact_rows_v1.tsv`
          - producer: unresolved root-side compact workbook step
          - role: advisory compact GT-skeleton surface
          - upstream inputs:
            - child 46 `final_formulation_table_audit_ready_v1.tsv`
            - child 46 `field_gt_review_seed_rows_v7.tsv`
            - child 44 `final_formulation_table_v1.tsv`
            - child 47 `value_gt_annotation_rows_v7.tsv`
            - `data\cleaned\labels\manual\dev15_formulation_skeleton\dev15_variant_alignment_scaffold_v1.tsv`
          - evidence: compact workbook metadata JSON
        - FACT: `analysis\value_gt_annotation_from_boundary_gt_keep_v1\value_gt_annotation_workbook_from_boundary_gt_keep_v1.xlsx`
          - producer: unresolved root-side workbook seed step
          - role: reviewed-boundary keep-only seed surface
          - upstream inputs:
            - `...\boundary_gt_review_v1\boundary_gt_review_workbook_v1.xlsx`
            - child 46 `final_formulation_table_audit_ready_v1.tsv`
            - child 46 `field_gt_review_seed_rows_v7.tsv`
            - child 44 `final_formulation_table_v1.tsv`
          - evidence: validation report says all output rows came exactly from human `include_gt` rows
- FACT: `...\boundary_gt_review_v1\boundary_gt_review_workbook_v1.xlsx`
  - producer: `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
  - role: practical mother boundary surface / authoritative reviewed inclusion universe
  - upstream inputs:
    - `...\final_formulation_table_v1.tsv`
    - `...\weak_labels_v7pilot_r3_fixparse\weak_labels__v7pilot_r3_fixparse.tsv`
    - `...\dev15_scope.tsv`
  - then reviewed by humans into `gt_row_decision = include_gt / exclude_*`
  - FACT: target value-workbook lineage later uses the reviewed `include_gt` subset, not the raw predicted row set
- FACT: `...\final_formulation_table_v1.tsv` (canonical Stage5 parent table)
  - producer: `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - role: canonical Stage5 final formulation table
  - upstream inputs:
    - `...\weak_labels_v7pilot_r3_fixparse\weak_labels__v7pilot_r3_fixparse.tsv`
    - Stage3 relation artifacts
- FACT: child 34 Stage3 relation artifacts
  - producer: `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
  - role: canonical Stage3 relation layer
  - upstream inputs:
    - `...\weak_labels_v7pilot_r3_fixparse\weak_labels__v7pilot_r3_fixparse.tsv`
    - `...\weak_labels_v7pilot_r3_fixparse\weak_labels__v7pilot_r3_fixparse.jsonl`
    - `...\dev15_scope.tsv`
- FACT: `...\weak_labels_v7pilot_r3_fixparse\weak_labels__v7pilot_r3_fixparse.tsv`
  - producer: `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  - role: canonical frozen Stage2 weak-label candidate surface reused by downstream runs
  - upstream inputs:
    - targeted-5 manifest `...\run_20260312_1321_455ac37_targeted5_stage2_regression_v1\targeted_manifest.tsv`
    - remaining-10 manifest `...\run_20260313_0950_f4912f3_dev15_remaining10_current_stage2_extraction_v1\remaining10_scope.tsv`
    - Stage1 clean text files named in those manifests
    - Stage1 cleaned table assets for certain deterministic helper paths / row recovery
- FACT: Stage1 clean-text/table roots
  - targeted 5 manifest: `5GIF3D8W`, `5ZXYABSU`, `L3H2RS2H`, `WFDTQ4VX`, `WIVUCMYG` -> `data\cleaned\content_goren_2025\text\*.txt`
  - remaining 10 manifest: `7ZS858NS`, `BB3JUVW7`, `BXCV5XWB`, `INMUTV7L`, `PA3SPZ28`, `QLYKLPKT`, `RHMJWZX8`, `UFXX9WXE`, `V99GKZEI`, `YGA8VQKU` -> `data\cleaned\content_goren_2025\text\*.txt`
  - repair-side direct Stage1 tables:
    - WFDTQ4VX tables 12/14
    - WIVUCMYG tables 01/05/13/14
    - YGA8VQKU pH source not explicitly recorded

# 5. Detailed hop-by-hop table

| hop_id | output_artifact | producer_script | function_or_entrypoint | run_dir | direct_inputs | config_or_cli | artifact_role | canonical_vs_advisory | evidence_source |
|---|---|---|---|---|---|---|---|---|---|
| H01 | `...\value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx` | unresolved | root-side pH backfill mutation | root run dir | `...\representation_repaired_v4.xlsx`; unresolved pH source artifact(s); `value_gt_annotation_pH_backfill_audit.tsv` | no recorded CLI | final repaired reviewer workbook | advisory | diff audit + `value_gt_annotation_pH_backfill_audit.tsv` |
| H02 | `...\value_gt_annotation_workbook_representation_repaired_v4.xlsx` | unresolved | root-side v4 repair | root run dir | `...\representation_repaired_v3.xlsx`; WIV tables 01/05/13/14; advisory Stage2 artifacts | no recorded CLI | repaired reviewer workbook | advisory | `value_gt_annotation_workbook_representation_repaired_v4.audit.md` |
| H03 | `...\value_gt_annotation_workbook_representation_repaired_v3.xlsx` | unresolved | root-side v3 repair | root run dir | root `...\representation_repaired_v2.xlsx`; `WFDTQ4VX_reconciled_instances.tsv`; WFDT tables 12/14 | no recorded CLI | repaired reviewer workbook | advisory | `value_gt_annotation_workbook_representation_repaired_v3.audit.md` |
| H04 | child 51 `...\value_gt_annotation_workbook_representation_repaired_v2.xlsx` | `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py` | `main` | child 51 run | `...\with_phase_and_polymer_values_v1.xlsx`; `...\with_phase_and_polymer_values_v1.tsv`; boundary workbook; GT skeleton TSV | child 51 recorded run order; script outputs fixed v2 names | representation-aware repair workbook | advisory / diagnostic | child 51 `RUN_CONTEXT.md`; script lines 891-922 |
| H05 | `...\value_gt_annotation_workbook_with_phase_and_polymer_values_v1.xlsx` | unresolved | root-side helper fill | root run dir | `...\with_phase_values_v2.xlsx`; Stage1 local evidence sources | no recorded CLI | polymer helper workbook | advisory | `value_gt_annotation_workbook_with_phase_and_polymer_values_v1.summary.json` |
| H06 | `...\value_gt_annotation_workbook_with_phase_values_v2.xlsx` | unresolved | root-side helper fill | root run dir | `...\with_phase_values_v1.xlsx`; Stage1 local evidence; compact rows TSV | no recorded CLI | phase helper workbook | advisory | `value_gt_annotation_workbook_with_phase_values_v2.summary.json` |
| H07 | `analysis\value_gt_annotation_from_boundary_gt_keep_v1\value_gt_annotation_workbook_from_boundary_gt_keep_v1.xlsx` | unresolved | boundary keep-only seed | root run dir | boundary workbook reviewed `review_gt_rows`; child 46 audit-ready + seed rows; child 44 final table | no recorded CLI | keep-only seed workbook | advisory | validation report JSON |
| H08 | `...\boundary_gt_review_v1\boundary_gt_review_workbook_v1.xlsx` | `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py` | `main` | parent root | parent Stage5 final table; parent Stage2 weak labels; scope manifest | script build; later human review decisions | practical mother boundary workbook | canonical review authority | script lines 937-940, 962-1021; workbook contents |
| H09 | child 37 `fgt_v5_dev15_v2/final_formulation_table_audit_ready_v1.tsv` and `field_gt_review_seed_rows_v5.tsv` | `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`; `src/stage5_benchmark/build_field_gt_review_workbook_v1.py` | `main` | child 37 run | child 34 final table + decision trace; parent Stage2 TSV; child 34 scope manifest; paper risk TSV; resolved relation TSV | exact commands recorded in child 37 | field-review export / seed rows | advisory review-support | child 37 `RUN_CONTEXT.md:30-69` |
| H10 | child 42 `value_gt_annotation_workbook_v4_repaired.xlsx` | `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py` | `main` | child 42 run | GT skeleton TSV; alignment scaffold; prior workbook v3; trusted prior alignment bridge child 40 rows v4; child 37 audit-ready + seed rows v5; child 34 final table | child 42 recorded run | repaired GT-skeleton workbook | advisory review-support | child 42 `RUN_CONTEXT.md:12-15,30` |
| H11 | child 43 `value_gt_annotation_workbook_v4_repaired2.xlsx` | `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py` | `main` | child 43 run | same core inputs as child 42 with stricter bridge inheritance | child 43 recorded run | repaired GT-skeleton workbook | advisory review-support | child 43 `RUN_CONTEXT.md:12-15,35` |
| H12 | child 47 `value_gt_annotation_rows_v7.tsv` / `value_gt_annotation_workbook_v6_merged_alignmentfix_generated.xlsx` | `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py` | `main` | child 47 run | child 46 audit-ready + seed rows v7; child 44 final table; GT skeleton; alignment scaffold; prior workbook `v6_merged.xlsx`; trusted alignment rows v6 | exact command recorded in child 47 | alignment-fixed GT-skeleton workbook | advisory but later reused | child 47 `RUN_CONTEXT.md:25-55` |
| H13 | child 34 `stage5_relation_first/final_formulation_table_v1.tsv` | `src/stage5_benchmark/build_minimal_final_output_v1.py` | `main` | child 34 run | parent Stage2 TSV; child 34 relation records; child 34 resolved relation fields | exact command recorded in child 34 | canonical Stage5 final table | canonical production | child 34 `RUN_CONTEXT.md:26-28,42-43,65-68,83` |
| H14 | child 34 `formulation_relation_v1/*` | `src/stage3_relation/build_formulation_relation_artifacts_v1.py` | `main` | child 34 run | parent Stage2 TSV + JSONL + copied scope manifest | exact command recorded in child 34 | canonical Stage3 relation layer | canonical production | child 34 `RUN_CONTEXT.md:26-28,42,65` |
| H15 | parent Stage2 TSV/JSONL `weak_labels__v7pilot_r3_fixparse.*` | `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` | `main` | targeted5 + remaining10 runs, then merged/replayed | targeted manifest / remaining scope manifest -> Stage1 clean text paths | `--manifest-tsv ... --model gemini-2.5-flash ... --out-dir ...` | canonical Stage2 candidate layer | canonical production input | Stage2 run contexts + script lines 3214-3273 |
| H16 | manifests `targeted_manifest.tsv` / `remaining10_scope.tsv` | runner / deterministic selection | manifest rows | targeted5 / remaining10 runs | split TSVs under `data\cleaned\goren_2025\index\splits\` | recorded in run contexts | Stage1 asset declarations | canonical source declarations | Stage2 run contexts + manifest files |
| H17 | `data\cleaned\content_goren_2025\text\*.txt` and repaired-branch `data\cleaned\goren_2025\tables\*.csv` | Stage1 cleaned asset layer | N/A | `data/cleaned` | earliest traced origin | N/A | clean text / cleaned table assets | canonical earliest origin | manifest rows + repair audits |

# 6. Repair / rewrite / backfill analysis

- FACT: `representation_repaired`
  - child 51 script `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py` remapped the compact workbook into a wider representation-aware schema and preserved human-edited cells against the machine baseline (`src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py:891-922`; child 51 `RUN_CONTEXT.md:24-44`).
  - output immediately after the maintained step: child 51 `...lineage\children\51_...\value_gt_annotation_workbook_representation_repaired_v2.xlsx`.
  - identity effect: none; it is an annotation-surface schema repair only.
- FACT: `representation_repaired_v3`
  - `value_gt_annotation_workbook_representation_repaired_v3.audit.md` documents WFDTQ4VX concentration repairs from v2 to v3 using `WFDTQ4VX_reconciled_instances.tsv` plus WFDT Stage1 table CSVs.
  - change type: value fills only, focused on WFDTQ4VX `drug_concentration_value` and `polymer_concentration_value`; row identity unchanged.
- FACT: `representation_repaired_v4`
  - `value_gt_annotation_workbook_representation_repaired_v4.audit.md` documents WIVUCMYG concentration/unit repairs from v3 to v4 using WIV Stage1 tables 01/05/13/14.
  - change type: value fills only; row identity unchanged.
  - advisory note: the same audit also says pH disappeared first in Stage2 weak-label extraction for WIVUCMYG, but that statement is only true for WIVUCMYG within the March 14 lineage, not necessarily for every later branch or paper.
- FACT: `pdi_zeta_backfilled_from_semantic_stage2`
  - `value_gt_annotation_workbook_representation_repaired_v4_pdi_zeta_backfill_audit.tsv` shows 33 fills relative to v4: 28 `pdi` fills and 5 `zeta_mV` fills.
  - change type: value fills only; row identity unchanged.
  - FACT: This workbook is a sibling advisory branch, not an ancestor of the target, because `with_pH` lacks those 33 filled cells.
- FACT: `with_pH`
  - `value_gt_annotation_pH_backfill_audit.tsv` shows 42 `FILL` actions and 168 `SKIP` actions.
  - workbook diff shows `v4 -> with_pH` adds only one new column, `pH_raw`, with 42 non-empty values; all other prior cells match v4 exactly.
  - filled papers are `WIVUCMYG` and `YGA8VQKU`; all fills use `exact_key_plus_formulation_label`.
  - change type: adds reviewer helper values in a new column; row identity unchanged.
  - UNCERTAINTY: the exact producer script and explicit source artifact for the YGA8VQKU pH values are not recorded in the March 14 lineage.
- FACT: carry-forward / bridge artifacts
  - child 42 and 43 explicitly use a trusted prior alignment bridge from child 40 `value_gt_annotation_rows_v4.tsv`.
  - child 47 explicitly uses prior workbook `value_gt_annotation_workbook_v6_merged.xlsx` plus trusted alignment rows v6 from child 46.
  - `build_value_gt_annotation_workbook_v1.py` marks these as advisory alignment inputs; the script itself says historical alignment scaffolds and prior workbook bridges may help map GT rows, but canonical current-system presence comes from the latest Stage5 final table plus audit-ready export (`src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py:1243-1278, 1732-1736`).

# 7. Mother workbook and inclusion-universe analysis

- FACT: The practical mother boundary surface is `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\boundary_gt_review_v1\boundary_gt_review_workbook_v1.xlsx`.
- FACT: The target value workbook inherits the accepted formulation universe from the reviewed `include_gt` subset of that boundary workbook.
- FACT: Mechanism:
  - `build_boundary_gt_review_workbook_v1.py` seeds `review_gt_rows`, instructs reviewers to choose `gt_row_decision`, and says `Use include_gt when the seeded row should become a GT formulation row` (`src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py:937-940`).
  - `analysis\value_gt_annotation_from_boundary_gt_keep_v1\value_gt_annotation_from_boundary_gt_keep_validation_report_v1.json` says `human_keep_row_count = 210`, `all_output_rows_come_exactly_from_human_keep_gt_rows = true`, `rows_added_not_in_human_keep_gt_set_count = 0`.
  - `value_gt_annotation_workbook_with_phase_and_polymer_values_v1.xlsx` metadata repeats `authoritative_workbook_path = ...boundary_gt_review_workbook_v1.xlsx`, `authoritative_sheet_used = review_gt_rows`, and `human_keep_row_count = 210`.
- FACT: Subset used: reviewed `review_gt_rows` where `gt_row_decision = include_gt`; excluded rows stayed out of the downstream value workbook row set.
- INFERENCE: The root-side keep-only seed workbook is the concrete bridge where the reviewed mother workbook’s `include_gt` subset became the downstream 210-row value-annotation universe.

# 8. From final table back to clean text

- FACT: Final workbook
  - `...\value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`
- FACT: prior workbook / seed
  - immediate parent: `...\value_gt_annotation_workbook_representation_repaired_v4.xlsx`
  - earlier root-side chain: `representation_repaired_v3.xlsx` <- `representation_repaired_v2.xlsx` <- `with_phase_and_polymer_values_v1.xlsx` <- `with_phase_values_v2.xlsx` <- unresolved root keep-only / compact seed surfaces
- FACT: audit-ready TSV / final table / field seed
  - child 37 `fgt_v5_dev15_v2/final_formulation_table_audit_ready_v1.tsv`
  - child 37 `fgt_v5_dev15_v2/field_gt_review_seed_rows_v5.tsv`
  - later compact/helper lineage also references child 46 `fgt_v7_dev15/*` and child 44 final table
- FACT: Stage5 outputs
  - child 34 `stage5_relation_first/final_formulation_table_v1.tsv`
  - child 44 later Stage5 descendant-fix final table
- FACT: Stage3 outputs
  - child 34 `formulation_relation_v1/formulation_relation_records_v1.tsv`
  - child 34 `formulation_relation_v1/resolved_relation_fields_v1.tsv`
- FACT: Stage2 completed artifact
  - parent `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
  - parent JSONL companion
- FACT: Stage2 evidence blocks / candidate blocks if used
  - extractor reads Stage1 clean text via manifest `text_path`
  - extractor also has deterministic helper paths that inspect Stage1 cleaned table assets for specific table/DOE recoveries
- FACT: Stage1 clean text and table assets
  - manifests point to `data\cleaned\content_goren_2025\text\*.txt`
  - repair branches additionally cite Stage1 cleaned tables for WFDTQ4VX and WIVUCMYG directly
- FACT: identity frozen vs values attached
  - identity is effectively frozen at the reviewed Layer2 boundary workbook `include_gt` subset and then preserved through keep-only / GT-skeleton row sets.
  - values are attached in layers:
    - system-side candidate values via Stage2 -> Stage3 -> Stage5 -> audit-ready / field seed / value workbook generation
    - later reviewer-support fills via root-side helper, repair, pH, and pdi/zeta backfill steps.

# 9. Ambiguities and missing evidence

- UNCERTAINTY 1
  - known: no script, `RUN_CONTEXT.md`, or recorded command emits `value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`.
  - inferred: it was created by mutating `representation_repaired_v4.xlsx` using the logic reflected in `value_gt_annotation_pH_backfill_audit.tsv`.
  - why unresolved: only the audit TSV and workbook diff remain.
  - most likely resolution: an untracked local Python/OpenPyXL backfill script or notebook added `pH_raw`.
- UNCERTAINTY 2
  - known: no recorded producer exists for root `representation_repaired_v3.xlsx` and `representation_repaired_v4.xlsx`.
  - inferred: they were manual/ad hoc workbook mutations backed by `.audit.md` files.
  - why unresolved: audits describe effects and source files but not the exact producing command.
  - most likely resolution: manual workbook-edit automation or one-off local scripts were used and not checked in.
- UNCERTAINTY 3
  - known: child 51 formally outputs only the child-run `representation_repaired_v2.xlsx`; the root copy step to `...\value_gt_annotation_workbook_representation_repaired_v2.xlsx` is not recorded.
  - inferred: the root v2 workbook is a copied/promoted descendant of the child 51 output.
  - why unresolved: no copy log or metadata refresh exists.
  - most likely resolution: the child 51 output was manually copied into the run root for reviewer use.
- UNCERTAINTY 4
  - known: no exact producer script/command was found for `value_gt_annotation_workbook_with_phase_values_v1.xlsx`, `...with_phase_values_v2.xlsx`, or `...with_phase_and_polymer_values_v1.xlsx`.
  - inferred: they are root-side helper enrichments built from the boundary keep-only seed plus local Stage1 evidence extraction.
  - why unresolved: only summary JSON/MD artifacts remain.
  - most likely resolution: one-off helper scripts or notebooks generated them without `RUN_CONTEXT.md`.
- UNCERTAINTY 5
  - known: `value_gt_annotation_pH_backfill_audit.tsv` records 42 pH fills, but it does not cite the source file paths used for YGA8VQKU.
  - inferred: the YGA8VQKU pH values likely came from an identity-variable source or table decode aligned by exact formulation label.
  - why unresolved: no sidecar metadata names the source artifact.
  - most likely resolution: a local lookup used either later Stage2-style identity-variable strings or direct table decode.
- UNCERTAINTY 6
  - known: `value_gt_annotation_workbook_with_phase_values_v1.xlsx` is the direct predecessor named by the v2 phase-helper summary, but its own provenance is not recorded.
  - inferred: it likely descends from `analysis\value_gt_annotation_from_boundary_gt_keep_v1\value_gt_annotation_workbook_from_boundary_gt_keep_v1.xlsx`.
  - why unresolved: no explicit sidecar connects those two files.
  - most likely resolution: the keep-only workbook was copied to the root and then extended into `with_phase_values_v1`.

# 10. Appendix: exact evidence excerpts

- FACT: Governed workbook identity
  - `project/ACTIVE_PIPELINE_RUNBOOK.md:811-814`
  - excerpt: `value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx` and `include_gt subset of the boundary workbook above`
- FACT: Governed downstream workbook note
  - `project/4_DECISIONS_LOG.md:2945-2946`
  - excerpt: `The downstream field-level workbook:` and `...value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`
- DECISION RULE: authoritative source resolution
  - `project/ACTIVE_DATA_SOURCE_CONTRACT.md:24-25,80,107,116-120,145`
  - excerpt: use the exact authoritative run directory / exact source artifact paths; do not assume newest child or timestamp order
- FACT: contracted workbook paths
  - `src/utils/paths.py:58,71,77`
  - excerpt: `DEV15_LAYER2_IDENTITY_TSV`, `DEV15_LAYER2_SOURCE_WORKBOOK_XLSX`, `DEV15_LAYER3_SOURCE_WORKBOOK_XLSX`
- FACT: child 34 Stage3 + Stage5 replay
  - `...child 34...\RUN_CONTEXT.md:65-68`
  - excerpt: exact commands for `build_formulation_relation_artifacts_v1.py` and `build_minimal_final_output_v1.py`
- FACT: child 37 field-review export
  - `...child 37...\RUN_CONTEXT.md:62-63`
  - excerpt: exact commands for `export_final_formulation_audit_ready_v1.py` and `build_field_gt_review_workbook_v1.py`
- FACT: child 47 alignment-fix export
  - `...child 47...\RUN_CONTEXT.md:49`
  - excerpt: exact command using `--audit-ready-tsv ...fgt_v7_dev15/final_formulation_table_audit_ready_v1.tsv --seed-rows-tsv ...field_gt_review_seed_rows_v7.tsv --final-table-tsv ...final_formulation_table_v1.tsv --gt-skeleton-tsv ...dev15_formulation_skeleton_gt_v2_variantaware.tsv --alignment-scaffold-tsv ...dev15_variant_alignment_scaffold_v1.tsv --prior-workbook-xlsx ...value_gt_annotation_workbook_v6_merged.xlsx --trusted-alignment-tsv ...value_gt_annotation_rows_v6.tsv`
- FACT: representation-repair script contract
  - `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py:891-922`
  - excerpt: required args `--current-workbook-xlsx`, `--baseline-tsv`, `--boundary-workbook-xlsx`, `--gt-skeleton-tsv`; outputs fixed `value_gt_annotation_workbook_representation_repaired_v2.xlsx/.tsv`
- FACT: value-workbook builder contract
  - `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py:1753-1769, 1851-1867, 1909-1916`
  - excerpt: args for `--audit-ready-tsv`, `--seed-rows-tsv`, `--final-table-tsv`, `--gt-skeleton-tsv`, `--alignment-scaffold-tsv`, `--prior-workbook-xlsx`, `--trusted-alignment-tsv`; output naming by artifact version
- FACT: canonical-vs-advisory alignment rule inside the builder
  - `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py:1243-1278, 1732-1736`
  - excerpt: latest Stage5 final table + audit-ready export are canonical; historical alignment scaffolds and prior workbook bridges are advisory only
- FACT: boundary mother-workbook rule
  - `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py:937-940`
  - excerpt: `On review_gt_rows, choose gt_row_decision... Use include_gt when the seeded row should become a GT formulation row.`
- FACT: root helper summaries
  - `value_gt_annotation_workbook_with_phase_values_v2.summary.json:3-5,19-21,236-242`
  - `value_gt_annotation_workbook_with_phase_and_polymer_values_v1.summary.json:2,4-7,190-195`
  - excerpt: direct predecessor workbook/TSV, local evidence sources, output paths, helper columns filled
- FACT: include-gt keep-only validation
  - `analysis\value_gt_annotation_from_boundary_gt_keep_v1\value_gt_annotation_from_boundary_gt_keep_validation_report_v1.json:2-3,12,14-15,213`
  - excerpt: authoritative workbook is boundary workbook `review_gt_rows`; `all_output_rows_come_exactly_from_human_keep_gt_rows = true`
- FACT: pH sibling repair evidence
  - `value_gt_annotation_pH_backfill_audit.tsv`
  - excerpt: `ACTIONS Counter({'SKIP': 168, 'FILL': 42})`; fills on `WIVUCMYG` and `YGA8VQKU`
- FACT: v4 repair evidence
  - `value_gt_annotation_workbook_representation_repaired_v4.audit.md`
  - excerpt: WIVUCMYG concentration/unit repairs from v3; pH present in paper tables but absent from March 14 Stage2 weak labels for WIVUCMYG
