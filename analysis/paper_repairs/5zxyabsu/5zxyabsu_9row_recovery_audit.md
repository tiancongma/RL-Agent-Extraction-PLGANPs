# 5ZXYABSU 9-Row Recovery Audit

## Executive Conclusion

- The main blocker is **not** the row-numbering parser.
- The article text clearly shows 9 explicit formulation entries in `Table 1` (`NPR1`–`NPR3`, `NPB1`–`NPB3`, `NPG1`–`NPG3`) and `Table 2` is clearly the corresponding results table for the same labels.
- But in the accepted pre-LLM authority lineage used by the replay, `Table 1` and `Table 2` were not preserved as normalized table payloads at all; only `Table 14` survived into the authority payload family.
- As a result, the last boundary where the 9 rows are still recoverable in principle is the article text / original table surfaces, and the first place recovery stops in practice is the Stage2 authority-surface path before row emission: `S2-7 table_row_expansion_v1` fails on `Table 1` with `missing_table_authority_payload`.

## Table 1 / Table 2 Evidence Check

- Clean-text evidence from `data/cleaned/content/text/5ZXYABSU.pdf.txt` shows:
  - `Table 1` lists 9 formulations: `NPR1`, `NPR2`, `NPR3`, `NPB1`, `NPB2`, `NPB3`, `NPG1`, `NPG2`, `NPG3`.
  - The identifier family is **alphanumeric formulation-family labels**, not bare digits and not `F<number>`.
  - `Table 2` is the characterization/results table for those same labels, with rows `NPR1`–`NPG3` and measurements such as mean particle size and encapsulation efficiency.
- The raw extracted CSVs for `5ZXYABSU__table_01__pdf_table.csv` and `5ZXYABSU__table_02__pdf_table.csv` are effectively empty/noisy in the current cleaned table assets, so the row-bearing structure is visible in article text but not recoverable from the selected Stage1 CSV surfaces.

## S2-5 Semantic Authorization Check

- `S2-5` survives with one formulation table scope for `Table 1` and one unclear scope for `Table 14`.
- `Table 1` is authorized as formulation-bearing: `scope_kind=formulation_table`, `is_formulation_bearing=true`, `is_doe=false`, and `table_formulation_authorization_scope` references `Table 1`.
- `Table 2` does **not** survive as an authorized table scope in the replayed semantic document.
- The paper is routed as **non-DOE** in the surviving table scope (`Table 1 is_doe=false`).
- The semantic document still contains only one LLM formulation candidate: `NPR1`.

## S2-7 Row-Emission Check

- Explicit authority reopen succeeds for the replay lineage in general, but not for the relevant `5ZXYABSU` formulation table.
- Execution ledger entries for `5ZXYABSU`:
  - `table_id=Table 1` `was_unit_authorized=yes` `was_unit_called=no` `rows_emitted=0` `skip_reason=missing_table_authority_payload`
  - `table_id=Table 14` `was_unit_authorized=no` `was_unit_called=no` `rows_emitted=0` `skip_reason=not_formulation_table`
- For the relevant scope (`Table 1`), the exact first failing runtime condition is `missing_table_authority_payload` in `src/stage2_sampling_labels/table_row_expansion_v1.py:1595-1604`.
- The replayed completed Stage2 artifact therefore still contains only the original LLM row `NPR1`, and Stage5 likewise retains only one final formulation row for this paper.

## Is Numbering The Blocker?

- **Answer: partially, but not mainly.**
- Why not mainly: the current replay never reaches a row-label parsing step for `Table 1`, because no reopenable normalized payload exists for that table in the accepted authority surface.
- Why partially: if `Table 1` or `Table 2` were preserved as payloads later, the current numbered-DOE parser still would not recognize `NPR1` / `NPB1` / `NPG1`, because `build_numbered_doe_row_candidates_v1.py:170-186` only accepts bare digits or `F<number>` labels.
- But that parser limitation is secondary in the current failure chain, because the system stops earlier at payload availability.

## Exact First Blocking Boundary

- **Chosen classification:** `other`
- Precise one-sentence blocker: the accepted pre-LLM authority surface omitted `Table 1` and `Table 2`, so replay-time `S2-7` cannot reopen a row-bearing payload for the only authorized formulation table and stops at `missing_table_authority_payload` before any numbering-family parser could matter.
- Last boundary where the 9 formulation rows are still recoverable in principle: the article-native `Table 1` / `Table 2` evidence preserved in clean text, plus the original PDF table surfaces referenced in `tables_manifest.json`.
- First exact place where recovery stops in practice: accepted pre-LLM authority preservation and downstream reopen binding, observed at `S2-7 execution_ledger_v2.tsv` for `Table 1` and implemented in `table_row_expansion_v1.py:1595-1604`.

## FACTS

- `data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/normalized_table_payloads/5ZXYABSU/normalized_table_payloads_v1.json` contains only `Table 14`.
- `data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/evidence_blocks/5ZXYABSU/evidence_blocks_v1.json` shows `candidate_table__01` and `candidate_table__02` were suppressed as `hard_drop_table_noise`, and only `Table 14` was added by the minimal evidence floor.
- `data/results/20260421_c8f4b61/01_s2_5/...semantic_stage2_v2_objects.jsonl` authorizes `Table 1` as formulation-bearing and non-DOE, but not `Table 2`.
- `data/results/20260421_c8f4b61/03_s2_7/analysis/execution_ledger_v2.tsv` records `Table 1` with `skip_reason=missing_table_authority_payload`.
- `data/results/20260421_c8f4b61/05_stage5/final_formulation_table_v1.tsv` contains one `5ZXYABSU` row, and compare remains `1 vs 9`, status `under`.

## INFERENCES

- The replay baseline did not fail on `5ZXYABSU` because of Stage5 filtering after successful recovery; the missing 8 formulations were never materialized at Stage2.
- The paper belongs to a broader failure family where article-native table content is semantically understood enough to authorize a table, but the actual row-bearing authority payloads for the relevant tables are absent.

## UNCERTAINTIES

- The current cleaned Stage1 CSV extraction for `Table 1` and `Table 2` is too degraded to tell whether a repaired direct table payload would be recoverable from those exact CSV assets without revisiting upstream extraction/repair surfaces.
- `Table 2` is clearly the results table in article text, but because it is not authorized in `S2-5`, this audit cannot observe how a linked `Table 1` + `Table 2` recovery would behave downstream.
