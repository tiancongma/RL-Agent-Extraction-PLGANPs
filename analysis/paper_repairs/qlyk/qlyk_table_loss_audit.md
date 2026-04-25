# QLYKLPKT Table Loss Audit

## Scope

Audit the maintained Stage1 -> Stage2 -> Stage5 path for `QLYKLPKT`
(`10.2147/IJN.S54040`) to determine where the paper's second optimization
table disappears, whether the loss is segmentation-, selector-, or prompt-side,
and whether historical runs ever preserved both optimization tables.

## Fixed facts

- The paper contains a two-stage sequential optimization structure.
- Paper Table 1 is a single-variable surfactant optimization over poloxamer 188
  concentration with a selected optimum of `3 mg/mL`.
- Paper Table 2 is a second single-variable optimization over `PLGA:ITZ` ratio,
  explicitly conducted after selecting the `3 mg/mL` condition from Table 1.
- Expected formulation universe across those two tables is `7` formulations:
  `4` unloaded rows from Table 1 and `3` loaded rows from Table 2.

## Evidence reviewed

- Stage1 cleaned text:
  - `data/cleaned/content/text/QLYKLPKT.pdf.txt`
- Stage1 table assets:
  - `data/cleaned/goren_2025/tables/QLYKLPKT/QLYKLPKT__table_08__pdf_table.csv`
  - `data/cleaned/goren_2025/tables/QLYKLPKT/QLYKLPKT__table_09__pdf_table.csv`
- Maintained current-stage artifact chain:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/semantic_stage2_objects/candidate_blocks/QLYKLPKT/candidate_blocks_v1.json`
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/semantic_stage2_objects/evidence_blocks/QLYKLPKT/evidence_blocks_v1.json`
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/semantic_stage2_objects/raw_responses/QLYKLPKT__stage2_v2_raw_response.json`
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/final_formulation_table_v1.tsv`
- Historical comparison runs:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/final_formulation_table_v1.tsv`
  - `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/final_formulation_table_v1.tsv`

## Table presence audit

### Stage1

- table1 present? `yes`
- table2 present? `yes`

Evidence:

- The cleaned text contains both paper tables contiguously:
  - `Table 1 Physicochemical properties of unloaded PLGA nanoparticles...`
  - `Table 2 Physicochemical properties of PLGA-ITZ-NS with different PLGA:ITZ initial ratios`
- Stage1 table asset mapping for the paper's optimization tables is:
  - paper Table 1 -> `QLYKLPKT__table_08__pdf_table.csv`
  - paper Table 2 -> `QLYKLPKT__table_09__pdf_table.csv`
- Stage1 table 08 contains the Table 1 rows `2.5 / 3 / 4 / 10 mg/mL` plus the
  note selecting `3 mg/mL`.
- Stage1 table 09 contains the Table 2 rows `5:1 / 10:1 / 15:1` plus the note
  selecting `10:1`.

### S2-2a

- number of table candidates: `16`
- table1 present? `yes`
- table2 present? `yes`

Evidence:

- `candidate_blocks_v1.json` ranks the relevant optimization tables as the top
  two table candidates:
  - `QLYKLPKT__candidate_table__01` ->
    `.../QLYKLPKT__table_08__pdf_table.csv`
    - score `324`
    - section label `Experimental design variable table.`
  - `QLYKLPKT__candidate_table__02` ->
    `.../QLYKLPKT__table_09__pdf_table.csv`
    - score `179`
    - section label beginning `Table 2 Physicochemical properties...`
- `coverage_summary.table_candidates = 16`

### S2-2b

- table1 preserved? `no`
- table2 preserved? `no`
- if missing: why
  - `evidence_blocks_v1.json` uses
    `input_contract.selector_strategy = sorted_csv_first_4`
  - `role_aware_evidence_selection = false`
  - `table_selection_scoring = false`
  - the selected table blocks are lexicographic file picks
    `QLYKLPKT__table_01__pdf_table.csv` through
    `QLYKLPKT__table_04__pdf_table.csv`, not the actual optimization tables
    `table_08` and `table_09`

Important distinction:

- The paper's real optimization tables are lost as explicit selected table
  blocks at S2-2b.
- They are not absent from the prompt entirely, because the raw-text prefix
  still contains both paper tables inside the first `30000` characters.

### S2-3 / S2-4

- which tables in prompt / raw response
  - Prompt assembly input contract is:
    - `metadata`
    - `raw_prefix`
    - `table_01`
    - `table_02`
    - `table_03`
    - `table_04`
  - Therefore the structured prompt-side table blocks do **not** include paper
    Table 1 (`table_08`) or paper Table 2 (`table_09`).
  - The raw response still references `Table 8`, `Table 9`, and `Table 15`,
    which means the model recovered those signals from the long raw-text prefix,
    not from the selector-preserved table excerpts.

Observed raw-response behavior:

- References `Table 8` for the selected `3 mg/mL` poloxamer 188 condition
- References `Table 9` for the `5:1 / 10:1 / 15:1` ratio rows and selected
  `10:1`
- Does not materialize Table 8 row variants as formulation candidates

### S2-5 / S2-7

- which tables actually contribute rows
  - `semantic_stage2_v2_objects.jsonl` emits formulation candidates only for
    `PLGA:ITZ 5:1`, `10:1`, and `15:1`
  - `poloxamer 188 concentration = 3 mg/mL` is retained only as a shared
    context / selected condition
  - `weak_labels__v7pilot_r3_fixparse.tsv` contains only `3` QLYK rows, all
    sourced from `Table 9`
  - `final_formulation_table_v1.tsv` contains only `3` QLYK final rows, all
    from `Table 9`

Current maintained run outcome:

- Table 1 contributes a selected carry-forward condition only.
- Table 2 contributes the entire row universe.
- The current maintained run therefore materializes only one table's row set.

## Earliest loss boundary

`S2-2b selector evidence prioritization: paper Table 1 (table_08) and paper Table 2 (table_09) are dropped from explicit evidence blocks by the hardwired selector strategy sorted_csv_first_4.`

## Root cause classification

`selector_loss`

## Minimal root cause statement

This is not a Stage1 visibility failure and not a candidate-segmentation loss.
Both optimization tables are present in cleaned text and are the top two table
candidates in `S2-2a`. The earliest real loss occurs in `S2-2b`, where the
selector ignores those ranked candidates and instead emits the lexicographically
first four Stage1 CSV assets (`table_01` to `table_04`). That removes the
paper's actual optimization tables from the explicit structured evidence
package. The prompt still partially recovers them through the long raw-text
prefix, which is why the raw response can mention `Table 8` and `Table 9`.
However, downstream semantic parsing and projection only materialize Table 2's
ratio variants as rows; Table 1 survives only as a selected inherited condition.

## Minimal restoration target

- Replace `sorted_csv_first_4` with selector logic that preserves the highest-
  scoring relevant table candidates from `candidate_blocks_v1.json`, including
  `table_08` and `table_09` for QLYKLPKT.
- Keep the structured selected-table surface distinct from the raw-prefix
  fallback so prompt success does not depend on accidental recovery from prose.
- Add a regression guard that asserts sequential-optimization papers retain both
  stage tables in `evidence_blocks_v1.json` before S2-4 execution.

## Historical comparison

- Historical evidence **does** show both tables were previously preserved.
- In
  `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`,
  `QLYKLPKT` has exactly `7` rows:
  - `4` unloaded rows from Table 1:
    - `QLYKLPKT_Unloaded_01` `2.5 mg/mL`
    - `QLYKLPKT_Unloaded_02` `3 mg/mL`
    - `QLYKLPKT_Unloaded_03` `4 mg/mL`
    - `QLYKLPKT_Unloaded_04` `10 mg/mL`
  - `3` loaded rows from Table 2:
    - `QLYKLPKT_Loaded_01` `5:1`
    - `QLYKLPKT_Loaded_02` `10:1`
    - `QLYKLPKT_Loaded_03` `15:1`
- The same run's `final_formulation_table_v1.tsv` also contains the expected
  `7` QLYK rows spanning both tables.
- A later comparison run,
  `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/final_formulation_table_v1.tsv`,
  again shows both table families contributing rows, although with extra noisy
  rows.
