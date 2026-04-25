# Replay Baseline From S2-5 Report

## Exact Replay Start Boundary Used

- replay start boundary: `S2-5`
- source frozen `S2-4b` lineage: [20260421_43ed145/02_s2_4b](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/02_s2_4b)
- source raw responses dir: [raw_responses](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/02_s2_4b/raw_responses)
- source manifest: [dev15_scope.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv)
- new replay lineage: [20260421_7a5c2d1](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1)

## Child Layout / Execution Order

1. [01_s2_5](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/01_s2_5)
2. [02_s2_6](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/02_s2_6)
3. [03_s2_7](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/03_s2_7)
4. [04_stage3](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/04_stage3)
5. [05_stage5](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/05_stage5)
6. [06_compare](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/06_compare)

## Compare Result Summary

- previous diagnostic baseline: [20260421_43ed145/08_compare](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/08_compare)
- new replay compare: [20260421_7a5c2d1/06_compare](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/06_compare)
- previous totals:
  - final rows: `35`
  - GT rows: `210`
  - status mix: `{'under': 13, 'match': 1, 'over': 1}`
- new totals:
  - final rows: `35`
  - GT rows: `210`
  - status mix: `{'under': 13, 'match': 1, 'over': 1}`
- material improvement relative to the previous baseline: `no`

## Identity Freeze Result

- identity freeze summary: [identity_freeze_summary_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/05_stage5/audit/identity_freeze_guardrail_v1/identity_freeze_summary_v1.tsv)
- identity freeze failed: `yes`
- compare mode: `diagnostic`
- benchmark_valid: `no`
- replay classification: `diagnostic-only, not benchmark-valid final output`

## Why The Replay Did Not Improve

The repaired authority-reopen and row-emission code was present in this replay, but the replayed `S2-5` semantic objects still carried blank authority handles for the target papers.

Evidence:
- [semantic_stage2_v2_objects.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/01_s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl)
  - `5GIF3D8W`, `UFXX9WXE`, and `WIVUCMYG` have empty `authority_run_dir` and empty `authority_payload_root`
- [execution_ledger_v2.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/03_s2_7/analysis/execution_ledger_v2.tsv)
  - repaired table emitters are skipped with `missing_table_authority_payload`

So the replay did not fail because fresh Stage3 or Stage5 filtering removed newly recovered rows. It failed earlier because the replayed `S2-5 -> S2-7` handoff did not supply the explicit authority handles required by the new reopen contract.

## Dedicated 5GIF3D8W Section

- Stage2 completed row count:
  - previous: `2`
  - new: `2`
- Stage5 final row count:
  - previous: `1`
  - new: `1`
- compare status:
  - previous: `under`
  - new: `under`
  - GT rows: `26`
- whether the 8 explicit anchor formulations are present: `no`
- Stage2 rows actually present in the replay:
- Drug-free nanoparticles (variant_formulation)
- Etoposide-loaded nanoparticles (formulation_family)
- Stage5 rows actually present in the replay:
- Etoposide-loaded nanoparticles | payload_state=blank_control | field_source_type=unresolved_blank
- blank-control audit:
  - No EE-based exclusion is evidenced in this replay. The anchor rows never materialize at S2-7, so Stage5 never gets a chance to filter them by EE.
- if excluded, where and under which rule:
  - boundary: `S2-7`
  - rule: `table_row_expansion_v1 skip_reason = missing_table_authority_payload`
  - consequence: the explicit anchor rows never reach Stage3 or Stage5
- Stage5 decision-trace audit:
  - there are no `5GIF3D8W` row-level drop entries in [final_output_decision_trace_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_7a5c2d1/05_stage5/final_output_decision_trace_v1.tsv)
  - that supports the narrower conclusion that the anchor rows were absent before Stage5, not later filtered because they lacked EE

## Dedicated UFXX9WXE Quick Check

- Stage2 completed row count:
  - previous: `2`
  - new: `2`
- Stage5 final row count:
  - previous: `2`
  - new: `2`
- compare status:
  - previous: `under`
  - new: `under`
- did the previously recovered rows survive through Stage5 in this replay: `no`
- first replay blocker evidenced here:
  - `S2-7 table_row_expansion_v1 skip_reason = missing_table_authority_payload, not_formulation_table`

## Dedicated WIVUCMYG Quick Check

- Stage2 completed row count:
  - previous: `3`
  - new: `3`
- Stage5 final row count:
  - previous: `3`
  - new: `3`
- compare status:
  - previous: `under`
  - new: `under`
- did the previously recovered rows survive through Stage5 in this replay: `no`
- first replay blocker evidenced here:
  - `S2-7 table_row_expansion_v1 skip_reason = missing_table_authority_payload, not_formulation_table`

## FACTS

- Replay start boundary used: S2-5 only, reusing frozen raw responses from data/results/20260421_43ed145/02_s2_4b/raw_responses.
- Maintained downstream chain executed: S2-5 -> S2-6 -> S2-7 -> Stage3 -> Stage5 -> compare.
- New replay lineage: data/results/20260421_7a5c2d1.
- New compare totals: final rows 35, GT rows 210, matched papers 1, mismatched papers 14.
- Identity freeze remained failed and compare stayed diagnostic-only with benchmark_valid = no.
- Paper-level final counts for 5GIF3D8W, UFXX9WXE, and WIVUCMYG were unchanged relative to the previous diagnostic baseline.
- In the replayed S2-5 semantic documents for 5GIF3D8W, UFXX9WXE, and WIVUCMYG, authority_run_dir and authority_payload_root are blank.
- In the replayed S2-7 execution ledger, the repaired table emitters skip with missing_table_authority_payload for the target formulation tables.
- 5GIF3D8W reaches Stage5 as a single family row only; no explicit anchor rows appear in the final table or decision trace.

## INFERENCES

- This replay does not measure the full downstream effect of the explicit authority-reopen repair, because the replayed S2-5 documents do not carry the authority handles that S2-7 now expects.
- The unchanged final counts are primarily caused by the S2-5-to-S2-7 handoff in this replay lineage, not by Stage3 or Stage5 newly filtering recovered rows.
- For 5GIF3D8W, the dedicated blank-control / missing-EE concern is not the active blocker in this replay. The blocker occurs earlier, when the anchor rows fail to materialize at S2-7.

## UNCERTAINTIES

- This replay does not prove whether a different lawful replay start boundary that preserves authority handles would carry the recovered rows through Stage3 and Stage5.
- Because the target rows never materialize in this replay, the audit cannot test whether later Stage5 rules would retain all eight explicit anchor formulations for 5GIF3D8W.
