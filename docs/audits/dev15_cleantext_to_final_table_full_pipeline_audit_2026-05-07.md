# DEV15 clean-text-to-final-table full-pipeline audit — old active baseline vs 20260507 diagnostic lineage

- generated_at_utc: `2026-05-07T14:28:24Z`
- repo: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs`
- audit mode: read-only audit plus this report write; no runtime code changes
- benchmark status: diagnostic-only; not benchmark-valid final performance evidence
- user request: forget the earlier S5 priority framing and audit the whole flow from clean text to final table against old baseline, new baseline, GT values, and key source excerpts/tables

## Correction / source-anchor authority note

User-provided original source excerpts are **not** primarily the short files under `data/results/`. The authoritative preserved original excerpts/tables are embedded verbatim in:

- `docs/methods/layer3_field_gt_protocol_v1.md`
- section: `## User-Provided Original Source Excerpts For Field-GT Debugging`
- overall line range: `1098-1864`

The `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/raw_anchor_snippets/*.md` files are derived diagnostic snippets/inventories, not the primary user-preserved source document. This report should use the `docs/methods/layer3_field_gt_protocol_v1.md` section as the primary source-anchor authority whenever referring to the user's uploaded/key original paragraphs and tables.

## Executive conclusion

The new `20260507_c1ad6ca` lineage is useful as a full-pipeline integration diagnostic, but it is not a clean improvement over the old active diagnostic baseline.

Main reasons:

1. The new lineage did run the maintained Stage2→Stage3→Stage5→Layer3 path, but it was a fresh live Stage2 run with a changed row universe, not a controlled replay of the old successful Stage2 authority surface plus newer downstream repairs.
2. The planned Stage1/Marker table-cell sidecar enhancement did **not** enter the actual downstream-used Stage2 run. The run metadata has `stage1_table_cell_sidecar_root: ""`.
3. Stage2 candidate/evidence/normalized-table surfaces were produced and consumed into Stage2 compatibility output, but the run had one live LLM failure (`UFXX9WXE`) and the feature activation gate was not promotion-clean.
4. Most checked source excerpts/tables already contain the GT values. The main failures are therefore not absence of source evidence; they are authority-preservation, row-universe, row/header geometry, table-cell binding/projection, live-call failure, and downstream alignment/materialization failures.
5. Compared with old active baseline, `present_and_match` dropped `1913 -> 1393`, while `blocked_alignment` rose `588 -> 1554`. The largest single regression is `UFXX9WXE`, where a live Stage2 request failure removed all final rows for that paper.

---

## 1. Baseline times, lineage, and run contents

### 1.1 Old active diagnostic baseline

Authority pointer:

- `data/results/ACTIVE_RUN.json`
- active run id: `20260423_9c4a03f`
- ACTIVE_RUN update time: `2026-05-03T21:11:48Z`
- benchmark_valid: `no`
- compare_mode: `diagnostic`

Important point: the old active baseline is a stitched governed diagnostic lineage under one parent bucket, not a single fresh clean-text-to-final-table rerun.

Exact active terminal lineage:

| layer | exact artifact |
|---|---|
| Stage2 compatibility TSV | `data/results/20260423_9c4a03f/116_stage2_mass_carrythrough_derived_provenance_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv` |
| Stage3 relation records | `data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/formulation_relation_records_v1.tsv` |
| Stage3 resolved fields | `data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/resolved_relation_fields_v1.tsv` |
| Stage5 final table | `data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic/final_formulation_table_v1.tsv` |
| Stage5 decision trace | `data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic/final_output_decision_trace_v1.tsv` |
| Layer3 compare cells | `data/results/20260423_9c4a03f/174_layer3_compare_source_completed_gt_authority_v1_diagnostic/layer3_value_compare_cells_v1.tsv` |

Old run content counts:

| artifact | rows |
|---|---:|
| Stage2 compatibility TSV | 255 |
| Stage3 relation records | 1802 |
| Stage3 resolved fields | 788 |
| Stage5 final table | 204 |
| Stage5 decision trace | 255 |
| Layer3 compare cells | 7224 |

Old Layer3 status counts:

| compare_status | count |
|---|---:|
| not_reported_in_gt | 3677 |
| present_and_match | 1913 |
| missing_in_system | 813 |
| blocked_alignment | 588 |
| present_but_mismatch | 203 |
| extra_in_system | 30 |

Old Stage2 row-source counts:

| candidate_source | rows |
|---|---:|
| table_row_expansion_v1 | 116 |
| doe_numbered_table_row_recovery | 85 |
| saved_raw_live_v2_replay_to_stage2_v2 | 51 |
| paper_driven_deterministic_semantic_emitter_v1 | 3 |

Interpretation: old baseline preserved more deterministic table-row expansion / DOE-derived row authority, but it did not have the newer clean Stage1/selector/feature-governance surfaces represented as a single fresh run-root lineage.

### 1.2 New diagnostic lineage

Root context:

- `data/results/20260507_c1ad6ca/RUN_CONTEXT.md`
- created_at_utc: `2026-05-07T12:01:12.615901Z`
- git_head_short: `c1ad6ca`
- benchmark_valid: `no`
- compare_mode: `diagnostic`
- ACTIVE_RUN_update: `no`

Refreshed summary:

- `data/results/20260507_c1ad6ca/diagnostic_baseline_summary_v1.md`
- generated_at_utc: `2026-05-07T12:28:40Z`
- refreshed_at_utc: `2026-05-07T12:35:05Z`

Important lineage correction: the root RUN_CONTEXT contains stale earlier paths (`01_stage2_live_llm_dev15 -> 02_stage3_relation -> 03_stage5_final -> 04_layer3_compare`). The refreshed summary and superseding RUN_CONTEXT section identify the actual downstream-used lineage:

`01_stage2_live_llm -> 03_stage3_relation_from_01_stage2 -> 04_stage5_final_from_01_stage2_03_stage3 -> 05_layer3_compare_source_completed_gt_from_04_stage5`

Exact actual terminal lineage:

| layer | exact artifact |
|---|---|
| Stage2 run dir | `data/results/20260507_c1ad6ca/01_stage2_live_llm` |
| Stage2 compatibility TSV | `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv` |
| Stage2 request summary | `data/results/20260507_c1ad6ca/01_stage2_live_llm/analysis/request_summary.tsv` |
| Stage2 run metadata | `data/results/20260507_c1ad6ca/01_stage2_live_llm/stage2_run_metadata_v1.json` |
| Stage3 relation records | `data/results/20260507_c1ad6ca/03_stage3_relation_from_01_stage2/formulation_relation_records_v1.tsv` |
| Stage3 resolved fields | `data/results/20260507_c1ad6ca/03_stage3_relation_from_01_stage2/resolved_relation_fields_v1.tsv` |
| Stage5 final table | `data/results/20260507_c1ad6ca/04_stage5_final_from_01_stage2_03_stage3/final_formulation_table_v1.tsv` |
| Stage5 decision trace | `data/results/20260507_c1ad6ca/04_stage5_final_from_01_stage2_03_stage3/final_output_decision_trace_v1.tsv` |
| Layer3 compare cells | `data/results/20260507_c1ad6ca/05_layer3_compare_source_completed_gt_from_04_stage5/layer3_value_compare_cells_v1.tsv` |

New run content counts:

| artifact | rows |
|---|---:|
| Stage2 compatibility TSV | 258 |
| Stage3 relation records | 2600 |
| Stage3 resolved fields | 1284 |
| Stage5 final table | 158 |
| Stage5 decision trace | 258 |
| Layer3 compare cells | 7224 |

New Layer3 status counts:

| compare_status | count |
|---|---:|
| not_reported_in_gt | 3159 |
| blocked_alignment | 1554 |
| present_and_match | 1393 |
| missing_in_system | 694 |
| present_but_mismatch | 386 |
| extra_in_system | 38 |

New Stage2 live-call status:

- request summary: `data/results/20260507_c1ad6ca/01_stage2_live_llm/analysis/request_summary.tsv`
- success_count: `14`
- failure_count: `1`
- failed key: `UFXX9WXE`
- failure: `DeadlineExceeded / 504 Deadline Exceeded`
- raw responses persisted for 14 papers; request metadata preserved for 15 papers
- retry child `data/results/20260507_c1ad6ca/02_stage2_live_llm_retry_complete` was incomplete/killed and was not used downstream

New Stage2 row-source counts:

| candidate_source | rows |
|---|---:|
| live_llm_stage2_v2 | 130 |
| doe_numbered_table_row_recovery | 84 |
| table_row_expansion_v1 | 44 |

---

## 2. What the new line attempted to repair or enhance

The new line tried to integrate several recent repair themes:

| enhancement / repair theme | intended design role | evidence of intent |
|---|---|---|
| Stage1 Marker/current dual surface | preserve PDF structured tables/cells as fixed Stage1 sidecars while keeping old clean text compatibility | `docs/plans/2026-05-07-dev15-full-diagnostic-baseline-from-cleantext-to-final-table-plan.md`; `docs/plans/2026-05-07-marker-table-authority-b-phase-progress.tsv`; modified Stage1 code/tests in current workspace |
| Stage1 section/noise structure | additive structure metadata; not semantic authority | `src/stage1_cleaning/pdf2clean.py`; `src/stage1_cleaning/run_stage1_parser_bakeoff_v1.py`; `tests/test_stage1_parser_bakeoff_v1.py` |
| Optional Stage1 sidecar consumption in Stage2 | attach Stage1 structure/table-cell metadata without defining candidate universe | `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`; `tests/test_stage2_stage1_sidecar_consumption_v1.py` |
| S2-2 candidate/evidence sidecars | freeze candidate segmentation and evidence blocks before prompt assembly | `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/candidate_blocks/`; `.../evidence_blocks/` |
| S2-2 normalized table payloads / table authority | preserve full table authority separate from prompt summary | `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/normalized_table_payloads/`; `analysis/table_authority_validation_v1.tsv` |
| Selector/evidence-priority work | rank/pack evidence without becoming irreversible table authority | `analysis/stage2_prompt_preview_v1.tsv`; `analysis/feature_activation_report_v1.tsv` |
| DOE / table-row recovery | deterministic post-LLM completion inside Stage2 after LLM semantic authorization | `semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`; Stage2 candidate_source counts |
| Shared carrythrough / deterministic direct materialization | Stage3/S5 relation and source-backed carrythrough into final table | Stage3 `resolved_relation_fields_v1.tsv`; Stage5 final table / compare `system_value_source_type=shared_carrythrough` |

---

## 3. Did the enhancements actually reach the maintained path and get consumed downstream?

### 3.1 Stage1 / Marker sidecar

Status: **not actually connected to the downstream-used new baseline**.

Evidence:

- `data/results/20260507_c1ad6ca/diagnostic_baseline_summary_v1.md` lines 51-53: no verified frozen Stage1 Marker table-cell sidecar root was resolved; `--stage1-table-cell-sidecar-root` was omitted.
- `data/results/20260507_c1ad6ca/01_stage2_live_llm/stage2_run_metadata_v1.json` line 35: `"stage1_table_cell_sidecar_root": ""`.

Implication: Marker/Stage1 table-cell/caption/continuation repairs did not influence this new full-lineage result. They remain current-workspace/bounded-diagnostic capability, not proven full-mainline consumption in this run.

### 3.2 S2-2 candidate/evidence/selector surfaces

Status: **connected and produced run-local artifacts**.

New run inventory:

| artifact family | count / path |
|---|---|
| candidate blocks | 15 JSON under `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/candidate_blocks/` |
| evidence blocks | 15 JSON under `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/evidence_blocks/` |
| normalized table payloads | 15 JSON and 97 CSV under `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/normalized_table_payloads/` |
| prompt preview | `data/results/20260507_c1ad6ca/01_stage2_live_llm/analysis/stage2_prompt_preview_v1.tsv` |
| feature activation | `data/results/20260507_c1ad6ca/01_stage2_live_llm/analysis/feature_activation_report_v1.tsv` |

Caveat: feature activation gate is not clean. Reported missing/not-proven items include `s2_2_duplicate_table_suppression`, `variant_aware_gt_authority_switch`, and unclear `s2_2a_primary_table_guardrail` observer status.

### 3.3 Table authority and downstream consumption

Status: **partly connected, but not sufficient for final table improvement**.

Evidence:

- The new run created normalized table payloads and `analysis/table_authority_validation_v1.tsv` with 97 rows.
- But the mechanical table-derived system-value source decreased in compare:
  - `structured_table_rebinding`: `520 -> 156` (`-364`)
  - `missing_system_field_surface`: `2472 -> 2997` (`+525`)
- Stage2 `table_row_expansion_v1` rows dropped from `116 -> 44` despite total Stage2 rows being similar (`255 -> 258`).

Implication: table authority surfaces exist, but the new live lineage did not preserve/activate the same row-local table expansion and table-cell projection behavior that made the old baseline successful.

### 3.4 Live LLM boundary

Status: **connected but failed for one critical paper**.

Evidence:

- `UFXX9WXE` live call failed with `DeadlineExceeded / 504`.
- New Stage3 has `0` `UFXX9WXE` relation/resolved rows.
- New Stage5 has `0` `UFXX9WXE` final rows.
- New compare has `UFXX9WXE blocked_alignment = 1092`.

Implication: `UFXX9WXE` is not a downstream materialization bug in this run; it is a live Stage2 failure that removed the row universe before Stage3/Stage5.

---

## 4. Authoritative step definitions: where the S2-4a / stage input-output contract lives

The current authoritative file that lists the fine-grained pipeline steps, including `S2-4a`, their function, inputs/outputs, and principles is:

- `project/ACTIVE_PIPELINE_FLOW.md`
  - fine-grained mapping: lines 90-123
  - canonical path and exact stage-by-stage flow: lines 160-310+
- `project/2_ARCHITECTURE.md`
  - Stage2 substeps and contracts: lines 88-150
  - Stage3/Stage5 contracts: lines 151-218

A concrete historical S2-4a run-context file that explicitly records S2-4a purpose, inputs, outputs, and stop rule is:

- `data/results/20260418_63bf985/03_s2_4/01_s2_4a/RUN_CONTEXT.md`
  - purpose: lines 20-23
  - boundary and owner script: lines 25-32
  - inputs: lines 34-51
  - outputs: lines 57-69
  - stop rule: lines 71-80

The audit below uses those contracts as the design requirements.

---

## 5. Step-by-step input/output audit against system design, GT, and source excerpts

### 5.1 Stage1 / clean text / table assets

Design requirement:

- Stage1 preserves source text, structure, table/cell geometry, parser provenance, and optional sidecars.
- Stage1 must not infer formulation semantics.
- Marker artifacts should be frozen once per raw PDF/parser contract and reused deterministically, not rerun as scratch.

New-line status:

- Maintained clean text/current table assets were used through Stage2.
- Frozen Marker table-cell sidecar root was not resolved and was omitted.

Failure/gap:

- The new baseline cannot be used as proof that Marker/Stage1 table-cell repairs improve final table, because those repairs were not connected to Stage2 input.
- Key source snippets show the original source values exist, so the gap is not source absence.

### 5.2 S2-1 / S2-2 / S2-2a / S2-2b evidence construction

Design requirement:

- `S2-1`: resolve declared manifest scope, clean assets, table assets.
- `S2-2`: freeze evidence package and full table authority.
- `S2-2a`: preserve execution-grade full-table authority and candidate segmentation.
- `S2-2b`: selector prioritizes evidence but must not become irreversible table authority.

New-line status:

- Candidate blocks, evidence blocks, and normalized table payloads were produced for 15 DEV15 papers.
- Prompt preview and feature activation artifacts exist.

Failure/gap:

- Table authority exists, but the downstream comparison shows reduced table-derived rebinding and increased missing system field surfaces.
- Some source-anchor audit rows are still classified as `payload_exists_but_text_normalization_mismatch` or `payload_exists_but_row_header_geometry_degraded`.

Key audit artifacts:

- `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/candidate_blocks/`
- `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/evidence_blocks/`
- `data/results/20260507_c1ad6ca/01_stage2_live_llm/semantic_stage2_objects/normalized_table_payloads/`
- `data/results/20260507_c1ad6ca/01_stage2_live_llm/analysis/table_authority_validation_v1.tsv`
- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/source_anchor_table_authority_visibility_v1.tsv`

### 5.3 S2-3 / S2-4a prompt assembly/freeze

Design requirement:

- Prompt assembly consumes `evidence_blocks_v1.json` only.
- Prompt table view is summary-only; full tables must remain outside prompt as execution authority.
- S2-4a writes frozen prompt artifacts and stops before live call.

New-line status:

- New Stage2 produced prompt preview from evidence artifacts.
- `stage2_prompt_preview_v1.tsv` reports summary-style prompt/table surfaces.

Failure/gap:

- Prompt summary being compact/lossy is legal, but it cannot replace full table execution authority.
- The current failure is not that every number must be in prompt; rather, full-table authority and row-local binding must survive into compatibility projection and Stage5. That did not hold consistently.

### 5.4 S2-4b live LLM call

Design requirement:

- Only nondeterministic Stage2 substep.
- Raw responses/request metadata must be run-scoped and traceable.

New-line status:

- Success: 14 papers.
- Failure: `UFXX9WXE`, `DeadlineExceeded / 504 Deadline Exceeded`.

Failure/gap:

- The failed paper has source and GT evidence, but no new Stage3/Stage5 rows. This is a Stage2 live-call failure before downstream materialization.

### 5.5 S2-5 / S2-6 / S2-7 semantic parse, contract validation, compatibility projection

Design requirement:

- S2-5 parses raw LLM responses.
- S2-6 validates semantic/provenance contracts.
- S2-7 projects to the completed compatibility surface consumed by Stage3, without replacing LLM semantic discovery as candidate-universe authority.

New-line status:

- Stage2 compatibility projection succeeded and emitted 258 rows.
- But row source mix changed sharply:
  - old `table_row_expansion_v1`: 116
  - new `table_row_expansion_v1`: 44
  - new `live_llm_stage2_v2`: 130

Failure/gap:

- New Stage2 preserved row count at a superficial level but changed the row universe: many old row-local table identities became family-level or differently aligned rows.
- This caused final-table row loss and compare blocked/mismatch growth.

### 5.6 Stage3 relation materialization/resolution

Design requirement:

- Stage3 consumes completed Stage2 TSV and builds relation records/resolved fields without live LLM.
- Stage3 cannot recover rows absent from Stage2 compatibility surface.

New-line status:

- Relation rows increased `1802 -> 2600`.
- Resolved fields increased `788 -> 1284`.

Failure/gap:

- More Stage3 relation rows did not mean better final output because the critical row universe changed.
- `UFXX9WXE` has 0 new Stage3 rows due to Stage2 failure.
- Papers such as `L3H2RS2H` kept relation activity but lost final row-local table identities.

### 5.7 Stage5 final table materialization

Design requirement:

- Stage5 consumes fixed Stage2 row universe plus Stage3 relations.
- S5-2 performs source-faithful deterministic direct materialization only.
- Stage5 must not create or rediscover formulation rows.

New-line status:

- Stage5 final rows dropped `204 -> 158`.
- Main final-row losses:
  - `UFXX9WXE`: `27 -> 0`
  - `L3H2RS2H`: `21 -> 4`
  - `PA3SPZ28`: `5 -> 3`
  - `BXCV5XWB`: `3 -> 2`

Failure/gap:

- Stage5 mostly reflects upstream row-universe loss/change.
- Some deterministic materialization issues remain, especially sign preservation and row/header geometry dependent values, but the largest failures begin upstream.

### 5.8 B-1 Layer3 compare against GT

GT authority file:

- `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`

Old compare:

- `data/results/20260423_9c4a03f/174_layer3_compare_source_completed_gt_authority_v1_diagnostic/layer3_value_compare_cells_v1.tsv`

New compare:

- `data/results/20260507_c1ad6ca/05_layer3_compare_source_completed_gt_from_04_stage5/layer3_value_compare_cells_v1.tsv`

Status transition summary:

| transition | count |
|---|---:|
| present_and_match -> present_and_match | 1128 |
| present_and_match -> blocked_alignment | 491 |
| present_and_match -> missing_in_system | 178 |
| present_and_match -> present_but_mismatch | 116 |
| missing_in_system -> present_and_match | 127 |
| blocked_alignment -> present_and_match | 137 |

Net result:

- old match lost to non-match: 785 cells
- old non-match improved to match: 265 cells
- net match delta: -520

---

## 6. Re-find map for user-uploaded/key source paragraphs and tables

Primary governed source-anchor authority:

- `docs/methods/layer3_field_gt_protocol_v1.md`
- section: `## User-Provided Original Source Excerpts For Field-GT Debugging`
- overall exact section range: lines `1098-1864`

This `docs/methods/layer3_field_gt_protocol_v1.md` section is the repository-preserved verbatim record of the user's key original paragraphs and tables. It must not be replaced by summaries and must not be treated as a transient diagnostic artifact.

Per-paper line ranges in the primary source-anchor document:

| paper_key | primary source address | important source/table content |
|---|---|---|
| INMUTV7L | `docs/methods/layer3_field_gt_protocol_v1.md:1100-1129` | Preparation paragraph and Table 1 characterization rows 1-12. |
| BB3JUVW7 | `docs/methods/layer3_field_gt_protocol_v1.md:1130-1167` | Materials/preparation methods; Table 1 nanospheres; Table 2 nanorods. |
| BXCV5XWB | `docs/methods/layer3_field_gt_protocol_v1.md:1168-1257` | Materials/fabrication paragraphs; Table 2 PLGA / PLGA-PEG / PLGA-PEG-HA KGN-loaded nanoparticle properties. |
| L3H2RS2H | `docs/methods/layer3_field_gt_protocol_v1.md:1258-1325` | Materials/preparation paragraphs; Tables 1-5 for XAN/3-MeOXAN nanospheres/nanocapsules. |
| PA3SPZ28 | `docs/methods/layer3_field_gt_protocol_v1.md:1326-1417` | GAR-NP preparation paragraph; Table 1 characterization values. |
| QLYKLPKT | `docs/methods/layer3_field_gt_protocol_v1.md:1418-1454` | PLGA-ITZ-NS preparation/optimization paragraphs; Tables 1-2. |
| RHMJWZX8 | `docs/methods/layer3_field_gt_protocol_v1.md:1455-1471` | AP-PLGA-NP materials/preparation/characterization paragraphs. |
| UFXX9WXE | `docs/methods/layer3_field_gt_protocol_v1.md:1472-1547` | Lorazepam PLGA-NP materials/preparation; Table 1 BBD variables; Table 2 DOE rows 1-26. |
| V99GKZEI | `docs/methods/layer3_field_gt_protocol_v1.md:1548-1571` | MB loaded PLGA/SC6OH preparation paragraphs; Table 1 NP properties. |
| WFDTQ4VX | `docs/methods/layer3_field_gt_protocol_v1.md:1572-1639` | Lopinavir PLGA NP formulation/optimization paragraphs; Tables 1, 2, 7. |
| WIVUCMYG | `docs/methods/layer3_field_gt_protocol_v1.md:1640-1691` | Pranoprofen PLGA NP methods; Tables 1-2 central composite design. |
| YGA8VQKU | `docs/methods/layer3_field_gt_protocol_v1.md:1692-1747` | Flurbiprofen PLGA nanospheres methods/design/results; Tables 1, 2, 6. |
| 7ZS858NS | `docs/methods/layer3_field_gt_protocol_v1.md:1748-1764` | Mometasone furoate NP preparation; Table 1 properties. |
| 5ZXYABSU | `docs/methods/layer3_field_gt_protocol_v1.md:1765-1813` | Rh/Gatifloxacin NP preparation; Tables 1-2. |
| 5GIF3D8W | `docs/methods/layer3_field_gt_protocol_v1.md:1814-1864` | Etoposide NP preparation/results paragraphs; optimized formulation Table 1. Line 1865 begins a later diagnostic note and is not part of this raw user excerpt. |

Derived diagnostic inventories/snippets, not primary authority:

- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/source_anchor_table_authority_visibility_summary_v1.tsv`
- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/source_anchor_table_authority_visibility_v1.tsv`
- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/raw_anchor_snippets/`

Original Zotero/source inventory for these papers:

- `analysis/layer3_field_repairs/dev15_zotero_original_file_inventory_v1.tsv`

---

## 7. Source/GT/final-output representative traces

### 7.1 UFXX9WXE — Stage2 live failure removes an otherwise source/GT-supported DOE table

Primary source address:

- `docs/methods/layer3_field_gt_protocol_v1.md:1472-1547`

Derived diagnostic snippet:

- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/raw_anchor_snippets/UFXX9WXE_raw_excerpt.md`

GT address:

- `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`
- `UFXX9WXE_G001` starts around line 100 and DOE rows continue through line 125.

Source vs GT examples:

| formulation | field | source value | GT value | old status | new status |
|---|---|---|---|---|---|
| UFXX9WXE_G001 | drug_concentration_value | Table 2 row 1: 1 | 1 | present_and_match | blocked_alignment |
| UFXX9WXE_G001 | polymer_concentration_value | Table 2 row 1: 35 | 35 | present_and_match | blocked_alignment |
| UFXX9WXE_G001 | emulsifier_stabilizer_concentration_value | Table 2 row 1: 2 | 2 | present_and_match | blocked_alignment |
| UFXX9WXE_G001 | particle_size_nm | Table 2 row 1: 211 | 211 | present_and_match | blocked_alignment |
| UFXX9WXE_G001 | ee_percent | Table 2 row 1: 70 | 70 | present_and_match | blocked_alignment |

Failure localization:

- Source has values.
- GT has values.
- Old Stage2/Stage5 carried 27 final rows.
- New Stage2 live request failed for the whole paper.
- New Stage3 and Stage5 have 0 rows for UFXX9WXE.
- First failed step in new line: S2-4b live LLM request boundary.

### 7.2 L3H2RS2H — source table values exist, but new row-local table materialization collapses

Primary source address:

- `docs/methods/layer3_field_gt_protocol_v1.md:1258-1325`

Derived diagnostic snippet:

- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/raw_anchor_snippets/L3H2RS2H_raw_excerpt.md`

GT address:

- `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`
- `L3H2RS2H` rows around lines 65-85.

Examples:

| formulation | field | source/GT fact | old/new behavior |
|---|---|---|---|
| L3H2RS2H_G003 | drug_concentration_value | 3-MeOXAN nanocapsule theoretical concentration 1400 μg/mL; GT line around 67 | old matched; new often missing/misaligned |
| L3H2RS2H_G005 | particle_size_nm / PDI / zeta | source exact nanocapsule case: size 271, PDI 0.43, zeta -41.8; GT line around 69 | final output may preserve size/PDI but sign/value scope issues remain |

Stage counts:

- old Stage5 final rows: 21
- new Stage5 final rows: 4

Failure localization:

- Source values and GT values exist.
- New full run did not activate the old amount of table-row expansion and row-local table materialization.
- First major failed boundary: S2-7 compatibility projection / S5-1 fixed-row intake from changed Stage2 row universe; downstream S5-2 cannot recover absent rows.

### 7.3 BB3JUVW7 / YGA8VQKU / INMUTV7L / BXCV5XWB — table geometry and sign preservation examples

Source anchor classifications:

| paper | source-anchor visibility status | first failure class |
|---|---|---|
| BB3JUVW7 | absent | payload_exists_but_text_normalization_mismatch |
| YGA8VQKU | absent | payload_exists_but_text_normalization_mismatch |
| INMUTV7L | absent | payload_exists_but_text_normalization_mismatch |
| BXCV5XWB | absent | payload_exists_but_row_header_geometry_degraded |

Representative source/GT agreement:

- `docs/methods/layer3_field_gt_protocol_v1.md:1130-1167` contains BB3JUVW7 Table 1 zeta `−8.0`; GT `BB3JUVW7_G001` has zeta `−8.0`.
- `docs/methods/layer3_field_gt_protocol_v1.md:1692-1747` contains YGA8VQKU Table 2 F1 zeta `−22.43`; GT `YGA8VQKU_G001` has zeta `−22.43`.
- `docs/methods/layer3_field_gt_protocol_v1.md:1100-1129` contains INMUTV7L Table 1 formulation 1 zeta `−12.2`; GT `INMUTV7L_G001` has zeta `−12.2`.
- `docs/methods/layer3_field_gt_protocol_v1.md:1168-1257` contains BXCV5XWB Table 2 PLGA/PLGA-PEG/PLGA-PEG-HA rows align with GT rows `BXCV5XWB_G001`, `G004`, `G007`.

Failure localization:

- For BB3JUVW7/YGA8VQKU/INMUTV7L: source and GT agree, but table authority/text normalization and value sign/materialization are unstable.
- For BXCV5XWB: the first classified failure is row/header geometry degradation, which is a Stage2 table-authority preservation issue before Stage5.

---

## 8. Quantitative regression localization

### 8.1 Compare status delta

| status | old | new | delta |
|---|---:|---:|---:|
| present_and_match | 1913 | 1393 | -520 |
| present_but_mismatch | 203 | 386 | +183 |
| missing_in_system | 813 | 694 | -119 |
| blocked_alignment | 588 | 1554 | +966 |
| not_reported_in_gt | 3677 | 3159 | -518 |
| extra_in_system | 30 | 38 | +8 |

### 8.2 Top old-match regressions by paper

| paper_key | present_and_match -> non-match |
|---|---:|
| UFXX9WXE | 416 |
| YGA8VQKU | 81 |
| BB3JUVW7 | 78 |
| INMUTV7L | 60 |
| L3H2RS2H | 48 |
| WFDTQ4VX | 40 |
| BXCV5XWB | 36 |
| QLYKLPKT | 12 |
| 5ZXYABSU | 10 |

### 8.3 Top old-match regressions by field

| field | present_and_match -> non-match |
|---|---:|
| drug_name | 76 |
| drug_concentration_value | 70 |
| emulsifier_stabilizer_concentration_value | 57 |
| polymer_name | 53 |
| emulsifier_stabilizer_concentration_unit | 50 |
| emulsifier_stabilizer_name | 46 |
| drug_concentration_unit | 42 |
| la_ga_ratio_normalized | 39 |
| la_ga_ratio_raw | 39 |
| solvent_name | 38 |
| ee_percent | 34 |
| particle_size_nm | 34 |
| method_type | 31 |

### 8.4 Compare source-type evidence

| system_value_source_type | old | new | delta |
|---|---:|---:|---:|
| missing_system_field_surface | 2472 | 2997 | +525 |
| structured_table_rebinding | 520 | 156 | -364 |
| direct_extracted | 3157 | 3021 | -136 |
| shared_carrythrough | 69 | 193 | +124 |
| ordinal_grid_semantics | 120 | 45 | -75 |
| relation_or_direct | 126 | 71 | -55 |
| stage2_table_cell_binding_authority | 120 | 167 | +47 |

Interpretation: new run increased some table-cell-binding authority and shared carrythrough, but lost much more old structured-table rebinding and increased missing field surfaces. The net effect is regression.

---

## 9. Which steps failed system design requirements?

### Primary failure boundary A: Stage1 Marker sidecar integration

Requirement: fixed Stage1 table-cell sidecars should be available as additive evidence authority and deterministically consumed when explicitly provided.

Observed: no verified frozen Stage1 Marker table-cell sidecar root was passed into actual Stage2. Therefore this enhancement was not tested end-to-end.

Classification: not connected to mainline in this run.

### Primary failure boundary B: S2-4b live LLM boundary for UFXX9WXE

Requirement: live LLM raw response should exist for every declared paper or the run should clearly be partial/diagnostic.

Observed: `UFXX9WXE` failed with 504. This removed all downstream rows for the paper.

Classification: live-call failure; not a Stage5 repair target.

### Primary failure boundary C: S2-2/S2-7 table authority to compatibility projection

Requirement: full table authority and LLM-authorized row scopes should preserve row-local table identity into the completed Stage2 compatibility surface.

Observed: `table_row_expansion_v1` rows dropped `116 -> 44`, while compare `structured_table_rebinding` dropped `520 -> 156`.

Classification: table authority exists but is not being sufficiently projected/materialized into the row-local downstream surface.

### Primary failure boundary D: S2 table normalization / row-header geometry

Requirement: coordinate-preserving full-table payload/grid must preserve header/value geometry, blank placeholders, row headers, multi-row headers, and negative signs.

Observed source-anchor audit classes include:

- `payload_exists_but_text_normalization_mismatch`
- `payload_exists_but_row_header_geometry_degraded`

Classification: Stage2 table-authority preservation / normalization defect; Stage5 cannot fully repair once coordinates/signs are degraded.

### Secondary failure boundary E: S5-2 deterministic direct materialization

Requirement: apply source-faithful deterministic direct materialization over a fixed row universe.

Observed: some final rows carry values but lose sign/scope or mismatch GT; however the largest row losses are upstream.

Classification: secondary downstream materialization issue after upstream row/table authority defects.

---

## 10. Audit actions performed and specialist profiles used

This report consolidates main-orchestrator checks plus three read-only specialist audits:

1. lineage/features specialist: audited old/new lineage, run contents, feature integration, Marker/sidecar connection status.
2. quantitative transition specialist: audited status counts, transitions, per-paper/per-field regressions, stage row deltas, source-type deltas.
3. source/GT specialist: found key source snippet addresses, source-anchor TSVs, GT authority rows, and representative source-vs-GT traces.

No specialist modified files. This markdown report is the only repository write from this audit task.

---

## 11. Durable re-use notes

If this audit is referenced later, use this file first:

- `docs/audits/dev15_cleantext_to_final_table_full_pipeline_audit_2026-05-07.md`

If the user says “之前上传的关键原文段落和表格”, use the primary governed verbatim source section first:

- `docs/methods/layer3_field_gt_protocol_v1.md:1098-1864`
- section header: `## User-Provided Original Source Excerpts For Field-GT Debugging`

Only then use derived diagnostic inventories/snippets if needed for visibility audits:

- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/raw_anchor_snippets/`
- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/source_anchor_table_authority_visibility_v1.tsv`
- `data/results/20260506_end_to_end_boundary_repair/03_table_authority_failure_buckets/source_anchor_table_authority_visibility_summary_v1.tsv`

If the user asks where S2-4a and other step input/output/function/principle contracts are recorded, use:

- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/2_ARCHITECTURE.md`
- concrete S2-4a example: `data/results/20260418_63bf985/03_s2_4/01_s2_4a/RUN_CONTEXT.md`
