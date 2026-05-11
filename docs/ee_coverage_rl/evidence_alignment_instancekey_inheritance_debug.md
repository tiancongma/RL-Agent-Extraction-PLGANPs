# Instance Key + Inheritance Debug Summary

Run: run_20260219_1623_780eb83_goren18_weaklabels_v1

## Coverage
- total formulation rows: 217
- rows without explicit formulation ID: 118
- rows using table-row condition fingerprint keys: 118
- rows using figure-group keys: 0
- rows using enumerated-text keys: 0

## Inheritance Transparency
- rows with inherited-base fields: 32
- rows with inheritance but no explicit constant-parameter claim: 32
- rows with local EE support: 7
- average fingerprint completeness: 0.6119

## Confidence Tier Distribution
- A: 1
- B: 1
- C: 215

## Notes
- Confidence no longer depends on same-paragraph anchor cohesion.
- Tiering now uses fingerprint completeness + local EE support + inheritance transparency.
- Records with inheritance and no explicit constant-parameter claim are downgraded.

## Polymer Donor Inheritance Update
- run_id: `run_20260219_1623_780eb83_goren18_weaklabels_v1`
- donor_conflicts_la_ga_ratio_docs: 3
- donor_conflicts_plga_mw_kDa_docs: 1
- donor_nonconflict_la_ga_ratio_docs: 12
- donor_nonconflict_plga_mw_kDa_docs: 3
- inherited_base_count_la_ga_ratio_rows: 4
- inherited_base_count_plga_mw_kDa_rows: 3
- N_EE_and_polymer: 2
- N_EE_and_loading_and_polymer: 0

## Merged-Instance Modeling Readiness (Conservative)
- input_run: `run_20260219_1623_780eb83_goren18_weaklabels_v1`
- merge criteria: exact only (no fuzzy matching)
  - explicit-ID rows: `doc_key + ::merge::id:: + normalized formulation_id`
  - table-fingerprint rows: `doc_key + ::merge::fp:: + existing condition fingerprint hash token`
  - fallback: exact `condition_instance_key` bucketed deterministically
- N_merged_EE_and_loading: 3
- N_merged_EE_and_polymer: 2
- N_merged_EE_and_loading_and_polymer: 0
- A_rows: 4, B_rows: 2, A?B on group_key: 0
- root cause note: current A/B rows are mostly from different doc keys, so row-level intersection is naturally zero.

## Schema V2 Core-Collapse Fix (UFXX9WXE)
- root cause: in `src/stage5_benchmark/build_two_table_schema_v2.py`, `core_signature` did not include explicit formulation IDs, so many `UFXX9WXE` rows (`F-1..F-26`) shared the same signature and collapsed to one `formulation_core_id`.
- fix: when a row has an explicit formulation identifier, a normalized token is now injected into `core_signature` before the dedup/groupby step.
  - normalization: trim -> uppercase -> non `[A-Z0-9_-]` to `_` -> collapse repeated separators -> strip edge separators.
  - explicit-ID gate: mixed alpha+numeric token (for example `F-1`, `NPG2`) to avoid treating plain numeric indices as explicit IDs.
- regression check output:
  - `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/schema_v2_explicit_id_regression__UFXX9WXE.tsv`
- expected behavior:
  - `UFXX9WXE` `formulation_core` count must be `> 1`.
  - current check result: before `1`, after `26`, `acceptance_pass=1`.
