# weak_labels_v7pilot3 Evaluation (2026-03-06)

## Selected 3 papers
- `10.2147/IJN.S130908` | key `5ZXYABSU` | shared/global baseline condition risk (global inheritance seen in diagnostics)
- `10.1016/j.ejpb.2004.09.002` | key `L3H2RS2H` | segmentation/alignment difficulty (top EE mismatch and contamination risk)
- `10.1002/jps.24101` | key `WIVUCMYG` | table-heavy multi-formulation (high row multiplicity and under-enumeration risk)

## Number of extracted formulations per DOI
- `10.1002/jps.24101`: 26
- `10.1016/j.ejpb.2004.09.002`: 8
- `10.2147/ijn.s130908`: 9

## Distribution of field-level scope values
- `instance_specific`: 336
- `unknown`: 243
- `global_shared`: 109

## Distribution of membership_confidence values
- `high`: 431
- `low`: 243
- `medium`: 14

## Unknown/uncertain rate by field
- `plga_mw_kDa`: 34/43 (79.1%)
- `la_ga_ratio`: 28/43 (65.1%)
- `organic_solvent`: 26/43 (60.5%)
- `pdi`: 11/43 (25.6%)
- `encapsulation_efficiency_percent`: 7/43 (16.3%)
- `drug_name`: 5/43 (11.6%)
- `drug_feed_amount_text`: 5/43 (11.6%)
- `surfactant_concentration_text`: 3/43 (7.0%)
- `size_nm`: 2/43 (4.7%)
- `surfactant_name`: 0/43 (0.0%)

## v6 comparison (same 3 DOIs)
- `10.1002/jps.24101`: v6=29, v7=26, delta=-3
- `10.1016/j.ejpb.2004.09.002`: v6=8, v7=8, delta=0
- `10.2147/ijn.s130908`: v6=9, v7=9, delta=0
- Shared-vs-instance ambiguity appears reduced where `scope=global_shared` is explicit, but unknown scope remains in sparse fields.

## Should we scale from 3 papers to all 15 DEV papers?
Recommendation: **yes with prompt tweaks**
- Pilot is promising, but unknown/uncertain rates should be reduced before 15-paper rollout.
