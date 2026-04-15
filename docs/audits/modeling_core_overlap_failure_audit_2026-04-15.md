# Modeling Core Overlap Failure Audit 2026-04-15

## Executive conclusion
- Target EE-positive but Core-A-failing rows: `53`.
- Fact: all 53 target rows gained EE after row binding, but every one of them still lacks `drug_name`, `polymer_mw_kDa`, and `la_ga_ratio` on the same `final_formulation_id` in the combined-binding run.
- Fact: the failure set is concentrated in four papers: `WIVUCMYG`, `YGA8VQKU`, `V99GKZEI`, and `BB3JUVW7`.
- Inference: the overlap failure is not mainly caused by Step 1 membership drift at this point; it is caused by field co-location failure on the frozen rows.
- Inference: the dominant blocker mix is `ownership_unresolved` for `drug_name` plus `candidate_absent_upstream` for `polymer_mw_kDa` and `la_ga_ratio`.
- Recommendation category: `mixed`.

## Exact target set size
- Rows with EE absent in baseline v2, EE present after binding, and still outside Core Set A after combined binding: `53`.
- Core Set B subset size: `53`. It is identical because every target row already fails Core Set A.
- Core Set C subset size: `53`. It is identical because every target row already fails Core Set A.

## Top blocker fields
- `drug_name`: `53` of `53` target rows
- `polymer_mw_kDa`: `53` of `53` target rows
- `la_ga_ratio`: `53` of `53` target rows

## Top blocker reason classes
- `candidate_absent_upstream`: `106` Core A missing-field instances
- `ownership_unresolved`: `53` Core A missing-field instances

## Papers dominating the failure set
- `BB3JUVW7`: `5` target rows
- `V99GKZEI`: `6` target rows
- `WIVUCMYG`: `26` target rows
- `YGA8VQKU`: `16` target rows

## Facts vs inference
- Fact: none of the four failure-set papers contribute any Core Set A rows after combined binding.
- Fact: the only papers that contribute Core Set A rows are `L3H2RS2H` and `BXCV5XWB`.
- Fact: there are no papers that simultaneously contain Core Set A successes and newly EE-positive blocked rows, so the requested same-paper successful-vs-blocked contrast is empty in the current DEV15 run.
- Inference: same-paper identity/family misalignment is therefore not the dominant current failure mode for this 53-row set.

## Why the 53 rows stay outside the core
- `drug_name`: every target row still shows `unresolved_table`. The final frozen row often already contains a table-derived drug value, but current Step 2 does not treat it as lawfully row-owned for these rows.
- `polymer_mw_kDa`: every target row remains unsupported. For these papers, the formulation-parameter binding unit resolved no lawful target, and the frozen final rows usually carry `not_projectable_from_current_replacement_objects` for this field.
- `la_ga_ratio`: same pattern as `polymer_mw_kDa`.
- Result: EE recovery improved measurement density, but the recovered EE rows mostly belong to papers that still lack co-located polymer lineage fields on the same frozen row.

## Paper-level interpretation
- `BB3JUVW7`: `5` newly EE-positive rows, dominant Core A blockers `drug_name, polymer_mw_kDa, la_ga_ratio`, dominant reasons `candidate_absent_upstream, ownership_unresolved`, likely direction `mixed`.
- `V99GKZEI`: `6` newly EE-positive rows, dominant Core A blockers `drug_name, polymer_mw_kDa, la_ga_ratio`, dominant reasons `candidate_absent_upstream, ownership_unresolved`, likely direction `mixed`.
- `WIVUCMYG`: `26` newly EE-positive rows, dominant Core A blockers `drug_name, polymer_mw_kDa, la_ga_ratio`, dominant reasons `candidate_absent_upstream, ownership_unresolved`, likely direction `mixed`.
- `YGA8VQKU`: `16` newly EE-positive rows, dominant Core A blockers `drug_name, polymer_mw_kDa, la_ga_ratio`, dominant reasons `candidate_absent_upstream, ownership_unresolved`, likely direction `mixed`.

## Cross-row same-paper contrast
- None. The blocked EE-positive rows are disjoint from the papers that already satisfy Core Set A.

## Core-schema pressure audit
- `Core Set A`: `20 -> 20 -> 20`
- `Core Set B`: `20 -> 20 -> 20`
- `Core Set C`: `20 -> 20 -> 20`
- `Core Set A1`: `20 -> 20 -> 20`
- `Core Set A2`: `20 -> 20 -> 20`
- `Core Set A3`: `20 -> 20 -> 20`
- Fact: even the exploratory A1/A2/A3 sets stay flat at `20 -> 20 -> 20`.
- Inference: pilot-core relaxation that still requires `drug_name` does not unlock these rows, because `drug_name` remains unresolved on the whole 53-row failure set.

## Dominant next-step direction
- `parameter alignment` alone is insufficient, because the added EE rows still lack `polymer_mw_kDa` and `la_ga_ratio` candidates on the same rows.
- `identity alignment` is not the main issue in this failure slice, because no target paper already has successful core rows to align against.
- `paper-specific branch resolution` may help for some composition fields, but it does not explain the entire failure set.
- `pilot-core relaxation` alone does not help with the current A1/A2/A3 definitions, because unresolved `drug_name` blocks every target row.
- Best classification: `mixed`.

## Exact recommended next implementation target
- Add a narrow lawful row-local binding path for `drug_name` on table-derived rows, then separately audit whether `polymer_mw_kDa` and `la_ga_ratio` are truly absent in `WIVUCMYG`, `YGA8VQKU`, `BB3JUVW7`, and `V99GKZEI` or merely missing from current deterministic projection surfaces.
- If those polymer lineage fields are genuinely absent from the papers, the current official core will not expand for this slice without changing the modeling objective; if they are present, the next implementation target should be paper-specific deterministic extraction rather than further broad alignment changes.
