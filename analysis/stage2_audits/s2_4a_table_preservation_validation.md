# S2-4a Table Preservation Validation

## Executive conclusion

The current maintained implementation has **not** fully caught up to the confirmed-noise-only `S2-2b` governance contract.

Bounded validation result on six papers:

- preserved correctly through `S2-4a`:
  - `INMUTV7L`
  - `WIVUCMYG`
  - `UFXX9WXE`
  - `5GIF3D8W`
- still lost at `S2-2b` with explicit `hard_drop_table_noise`:
  - `5ZXYABSU`
  - `WFDTQ4VX`

So governance is still ahead of code. The current implementation can preserve key formulation-bearing / DOE-bearing tables for some papers, but it still irreversibly drops critical non-noise tables for at least two anchor cases.

## Validation run context

Maintained entrypoints used:

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - executed with `--stop-before-live-call`
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`

Exact manifest path:

- [dev15_scope.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv)

Bounded paper set:

- `5ZXYABSU`
- `WFDTQ4VX`
- `INMUTV7L`
- `WIVUCMYG`
- `UFXX9WXE`
- `5GIF3D8W`

Fresh governed lineage:

- pre-LLM build:
  - [20260421_9c4a03f/01_s2_4a_table_preservation_validation](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/01_s2_4a_table_preservation_validation)
- prompt freeze:
  - [20260421_9c4a03f/02_s2_4a_prompt_construction_v2](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/02_s2_4a_prompt_construction_v2)

No live LLM calls were made.

Primary fresh artifacts used:

- [candidate_segmentation_debug_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/01_s2_4a_table_preservation_validation/analysis/candidate_segmentation_debug_v1.tsv)
- [table_selection_debug_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/01_s2_4a_table_preservation_validation/analysis/table_selection_debug_v1.json)
- `semantic_stage2_objects/candidate_blocks/<paper>/candidate_blocks_v1.json`
- `semantic_stage2_objects/evidence_blocks/<paper>/evidence_blocks_v1.json`
- `semantic_stage2_objects/normalized_table_payloads/<paper>/normalized_table_payloads_v1.json`
- [s2_4a_prompts_v1.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/02_s2_4a_prompt_construction_v2/analysis/s2_4a_prompts_v1.jsonl)
- [s2_4a_prompt_audit_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/02_s2_4a_prompt_construction_v2/analysis/s2_4a_prompt_audit_v1.tsv)

Interpretation rule for `S2-4a` prompt presence:

- `yes` means the expected table survived into `evidence_blocks_v1.json`, and the prompt audit confirms `all_selected_blocks_included=yes`
- `no` means the expected table never reached the selected evidence pack, so it could not appear as a frozen prompt table block

## Per-paper table preservation results

| paper_key | expected key table | S2-2a | S2-2b | normalized payload | S2-4a prompt | hard_drop_table_noise | first failed preservation boundary |
|---|---|---:|---:|---:|---:|---:|---|
| `5ZXYABSU` | `Table 1` and `Table 2` formulation-bearing pair | yes | no | no | no | yes | `S2-2B` |
| `WFDTQ4VX` | recovered `Table 1 / Table 2` DOE design surface | yes | no | no | no | yes | `S2-2B` |
| `INMUTV7L` | simple formulation-bearing `Table 1` | yes | yes | yes | yes | no | none in bounded slice |
| `WIVUCMYG` | DOE-bearing `Table 1` | yes | yes | yes | yes | no | none in bounded slice |
| `UFXX9WXE` | key preserved formulation/DOE-bearing tables (`Table 1` and `Table 2`) | yes | yes | yes | yes | no | none in bounded slice |
| `5GIF3D8W` | explicit formulation-bearing `Table 4` | yes | yes | yes | yes | no | none in bounded slice |

## Key tables expected vs actually preserved

### 5ZXYABSU

Expected:

- `Table 1`
- `Table 2`

Observed:

- both still survive `S2-2a` as candidate tables
- neither survives `S2-2b`
- only `Table 14` is preserved into `evidence_blocks`, normalized payloads, and the `S2-4a` prompt

Fresh artifact-backed evidence:

- candidate presence:
  - `5ZXYABSU__candidate_table__07` from `5ZXYABSU__table_01__pdf_table.csv`
  - `5ZXYABSU__candidate_table__08` from `5ZXYABSU__table_02__pdf_table.csv`
- selector result:
  - both are suppressed in `selector_debug.suppression_events` with `reason=hard_drop_table_noise`
- preserved table:
  - `5ZXYABSU__table__01` -> `Table 14`

### WFDTQ4VX

Expected:

- the real DOE design surface, concretely represented in the fresh run by DOE-like candidates such as:
  - `WFDTQ4VX__candidate_table__08` from `WFDTQ4VX__table_12__pdf_table.csv`
  - `WFDTQ4VX__candidate_table__02`
  - `WFDTQ4VX__candidate_table__06`

Observed:

- DOE-like tables survive `S2-2a`
- all of them are still suppressed at `S2-2b` as `hard_drop_table_noise`
- only downstream `Table 8` survives into `evidence_blocks`, normalized payloads, and the frozen prompt

Fresh artifact-backed evidence:

- `WFDTQ4VX__candidate_table__08` is present in candidate segmentation with `table_role_hint=design matrix`
- `selector_debug.suppression_events` includes:
  - `WFDTQ4VX__candidate_table__02` -> `hard_drop_table_noise`
  - `WFDTQ4VX__candidate_table__06` -> `hard_drop_table_noise`
  - `WFDTQ4VX__candidate_table__08` -> `hard_drop_table_noise`
- preserved table:
  - `WFDTQ4VX__table__01` -> `Table 8`

### INMUTV7L

Expected:

- simple formulation-bearing `Table 1`

Observed:

- preserved cleanly through all checked boundaries
- `Table 1`, `Table 2`, `Table 3`, and `Table 4` all reach the frozen prompt pack in this bounded run

### WIVUCMYG

Expected:

- DOE-bearing `Table 1`

Observed:

- preserved cleanly through all checked boundaries
- `Table 1`, `Table 3`, `Table 5`, and `Table 6` all survive into the frozen prompt pack

### UFXX9WXE

Expected:

- key preserved formulation/DOE-bearing tables, especially `Table 1` and `Table 2`

Observed:

- preserved cleanly through all checked boundaries
- `Table 1`, `Table 2`, and `Table 15` survive into the frozen prompt pack

### 5GIF3D8W

Expected:

- explicit formulation-bearing `Table 4`

Observed:

- preserved cleanly through all checked boundaries
- `Table 4` survives into evidence, normalized payloads, and the prompt

## Whether hard_drop_table_noise still affects key tables

Yes.

It still directly affects critical key tables in the fresh current-code run:

- `5ZXYABSU`
  - `candidate_table__07`
  - `candidate_table__08`
- `WFDTQ4VX`
  - `candidate_table__02`
  - `candidate_table__06`
  - `candidate_table__08`

For the four guard/passing papers in this bounded set, the expected key tables were not hard-dropped.

## Whether current implementation matches the confirmed-noise-only contract

No.

Why:

- The fresh bounded run still irreversibly removes critical non-noise formulation-bearing / DOE-bearing tables for `5ZXYABSU` and `WFDTQ4VX`.
- Those removals happen specifically through `hard_drop_table_noise`.
- The preserved fallback/minimal-evidence floor did not rescue those formulation surfaces:
  - `minimal_evidence_floor_applied = no`
  - `floor_added_formulation_surface = no`

So the current implementation remains only partially aligned:

- aligned for some papers:
  - `INMUTV7L`
  - `WIVUCMYG`
  - `UFXX9WXE`
  - `5GIF3D8W`
- misaligned for key anchor failures:
  - `5ZXYABSU`
  - `WFDTQ4VX`

## FACTS

- A fresh maintained bounded run was executed with the current codebase and stopped before `S2-4b`.
- The fresh run wrote new candidate blocks, evidence blocks, normalized payloads, and frozen `S2-4a` prompt artifacts under `data/results/20260421_9c4a03f/`.
- `5ZXYABSU` still preserves only `Table 14` in `evidence_blocks` and normalized payloads.
- `WFDTQ4VX` still preserves only `Table 8` in `evidence_blocks` and normalized payloads.
- `INMUTV7L`, `WIVUCMYG`, `UFXX9WXE`, and `5GIF3D8W` preserve their expected key tables through the frozen prompt boundary in this bounded set.
- `selector_debug.suppression_events` still records `hard_drop_table_noise` for critical `5ZXYABSU` and `WFDTQ4VX` table candidates.

## INFERENCES

- The confirmed-noise-only governance language has not yet been fully implemented in `S2-2b`.
- The current implementation is paper-sensitive: some table families now survive correctly, but the anchored preservation failures remain active in code.

## UNCERTAINTIES

- This validation is intentionally bounded to six papers, so it does not prove full-repo compliance or non-compliance beyond this slice.
- For `UFXX9WXE`, this validation confirms table preservation only; it does not relocalize later DOE-emission behavior.
