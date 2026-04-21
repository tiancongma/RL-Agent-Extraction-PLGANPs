# 5ZXYABSU Table Loss Boundary Audit

## Executive conclusion

`Table 1` and `Table 2` are **not** first lost at `S2-2a` candidate segmentation, and they are **not** mainly blocked by numbering style. They are first lost at **`S2-2b selector / evidence preservation`**, where both table candidates are hard-dropped as `hard_drop_table_noise`. After that, only `Table 14` is preserved into `evidence_blocks_v1.json` and then into `normalized_table_payloads`, so replay-time `S2-7` has no row-bearing authority payload to reopen for `Table 1` or `Table 2`.

**Final classification:** `LOST_AT_S2_2B`

One-sentence bottom line:
`Table 1` and `Table 2` first disappear in practice at `S2-2b`, when `5ZXYABSU__candidate_table__07` and `5ZXYABSU__candidate_table__08` are hard-dropped as table noise, leaving only `Table 14` to be preserved downstream.

## Upstream table presence check

The upstream extracted table assets do contain the relevant tables:

- `data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_01__pdf_table.csv`
- `data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_02__pdf_table.csv`
- `data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_14__pdf_table.csv`

`tables_manifest.json` confirms:

- `table_01__pdf_table.csv`: `page_number=7`, `extraction_method=camelot_lattice`, `n_rows=2`, `n_cols=12`
- `table_02__pdf_table.csv`: `page_number=8`, `extraction_method=camelot_lattice`, `n_rows=2`, `n_cols=8`
- `table_14__pdf_table.csv`: `page_number=8`, `extraction_method=camelot_stream`, `n_rows=13`, `n_cols=6`

The cleaned paper text also explicitly describes the formulation family for `Table 1`:

- `NPR1`, `NPR2`, `NPR3`
- `NPB1`, `NPB2`, `NPB3`
- `NPG1`, `NPG2`, `NPG3`

That means the expected formulation identifiers are present upstream as an alphanumeric family, not as bare digits.

## S2-2a candidate segmentation check

Authority root inspected:

- `data/results/20260421_3579206/09_selector_contract_dev15_prellm`

Candidate artifact:

- `semantic_stage2_objects/candidate_blocks/5ZXYABSU/candidate_blocks_v1.json`

Relevant candidates do exist in `S2-2a`:

- `5ZXYABSU__candidate_table__07`
  - `origin_locator = data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_01__pdf_table.csv`
  - `selector_readiness_label = unresolved`
  - `representation_status = unrepaired_corrupted`
  - `quality_flags = ["corrupted_or_sparse_table"]`
- `5ZXYABSU__candidate_table__08`
  - `origin_locator = data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_02__pdf_table.csv`
  - `selector_readiness_label = unresolved`
  - `representation_status = unrepaired_corrupted`
  - `quality_flags = ["corrupted_or_sparse_table"]`
- `5ZXYABSU__candidate_table__04`
  - `origin_locator = data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_14__pdf_table.csv`
  - `selector_readiness_label = weak`
  - `representation_status = repair_insufficient`

Conclusion:

- `Table 1` is present at `S2-2a`
- `Table 2` is present at `S2-2a`
- therefore `S2-2a` is **not** the first failing boundary

## S2-2b selector/evidence check

Evidence artifact:

- `semantic_stage2_objects/evidence_blocks/5ZXYABSU/evidence_blocks_v1.json`

Only one table evidence block survives:

- `5ZXYABSU__table__01`
  - `table_id = Table 14`
  - `candidate_id = 5ZXYABSU__candidate_table__04`
  - `selection_reason = selected_high_signal_table`

Selector debug shows the actual first loss:

- `5ZXYABSU__candidate_table__07` -> `reason = hard_drop_table_noise`
- `5ZXYABSU__candidate_table__08` -> `reason = hard_drop_table_noise`
- `5ZXYABSU__candidate_table__04` -> `reason = minimal_evidence_floor_added_formulation_surface`

`table_selection_debug_v1.json` confirms:

- `selected_tables = ['5ZXYABSU__table_14__pdf_table.csv']`
- `5ZXYABSU__table_01__pdf_table.csv`
  - `selected = false`
  - `authority_rank = 7`
  - `authority_score = -26.25`
  - `quality_flags = ['corrupted_or_sparse_table']`
- `5ZXYABSU__table_02__pdf_table.csv`
  - `selected = false`
  - `authority_rank = 8`
  - `authority_score = -26.3`
  - `quality_flags = ['corrupted_or_sparse_table']`
- `5ZXYABSU__table_14__pdf_table.csv`
  - `selected = true`

Conclusion:

- `Table 1` and `Table 2` are first lost at `S2-2b`
- the loss mechanism is selector/evidence suppression, not replay-time reopen binding

## Normalized payload preservation check

Normalized payload artifact:

- `semantic_stage2_objects/normalized_table_payloads/5ZXYABSU/normalized_table_payloads_v1.json`

Only one normalized payload exists:

- `Table 14`
  - `source_filename = 5ZXYABSU__table_14__pdf_table.csv`
  - `table_inclusion_class = must_include`

There is **no** normalized payload for:

- `Table 1`
- `Table 2`

Conclusion:

- normalized payload preservation is incomplete for this paper
- but it is not the first loss boundary
- the payload family is incomplete because the relevant tables were already dropped at `S2-2b`

## Replay reopen availability check

Replay lineage inspected:

- `data/results/20260421_c8f4b61`

Replay `S2-5` semantic doc shows:

- `authority_run_dir = data/results/20260421_3579206/09_selector_contract_dev15_prellm`
- `authority_payload_root = data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/normalized_table_payloads`
- surviving semantic scopes:
  - `Table 1` as `formulation_table`, `is_formulation_bearing=true`
  - `Table 14` as `unclear`, `is_formulation_bearing=false`

Replay `S2-7` execution ledger shows:

- `Table 1`
  - `was_unit_authorized = yes`
  - `was_unit_called = no`
  - `rows_emitted = 0`
  - `skip_reason = missing_table_authority_payload`
- `Table 14`
  - `was_unit_authorized = no`
  - `skip_reason = not_formulation_table`

Conclusion:

- replay is binding to the **correct** authority root
- the failure is not `SURVIVES_UPSTREAM_BUT_REPLAY_BINDS_WRONG_ROOT`
- replay cannot reopen `Table 1` because the accepted upstream normalized payload family never preserved `Table 1`

## Why Table 14 survived instead

`Table 14` survived because the selector floor added it as the minimal preserved formulation surface after other candidate tables were suppressed. The evidence artifact records:

- `minimal_evidence_floor_applied = yes`
- `floor_added_formulation_surface = yes`
- `floor_rationale = added_authoritative_formulation_surface`

This points to a selector/evidence preservation outcome driven by degraded extracted-table quality:

- `Table 1` and `Table 2` were labeled `unresolved` + `unrepaired_corrupted`
- both received `quality_flags = ["corrupted_or_sparse_table"]`
- both were hard-dropped as noise
- `Table 14` was weak but still salvageable enough to satisfy the minimal evidence floor

This is more consistent with **role/quality misclassification and selector suppression** than with numbering style. Numbering never gets a chance to matter because the relevant tables are gone before normalized-payload preservation and replay reopen.

## Final classification

`LOST_AT_S2_2B`

## FACTS

- `Table 1` and `Table 2` are present in upstream extracted table assets and in cleaned paper text.
- `5ZXYABSU__candidate_table__07` and `5ZXYABSU__candidate_table__08` exist in `candidate_blocks_v1.json`.
- Both of those candidates are marked `unresolved` / `unrepaired_corrupted` with `corrupted_or_sparse_table`.
- `evidence_blocks_v1.json` preserves only `Table 14`.
- `selector_debug.suppression_events` explicitly records `hard_drop_table_noise` for candidate tables `07` and `08`.
- `normalized_table_payloads_v1.json` contains only `Table 14`.
- Replay `S2-5` points to the correct pre-LLM authority root.
- Replay `S2-7` fails `Table 1` with `missing_table_authority_payload`.

## INFERENCES

- The first practical loss boundary is selector/evidence preservation, not replay binding.
- `Table 14` survived because it was the only table left available for the minimal evidence floor after `Table 1` and `Table 2` were suppressed.
- Numbering is not the main blocker for this paper, because the relevant tables are removed before any row-family parser is invoked.

## UNCERTAINTIES

- The exact subcause of the `corrupted_or_sparse_table` assessment for `Table 1` and `Table 2` would require a narrower extraction-quality audit, which is outside this boundary-localization task.
- `Table 2` does not appear as a surviving semantic scope in replay, but that absence is downstream of the already-established `S2-2b` loss.
