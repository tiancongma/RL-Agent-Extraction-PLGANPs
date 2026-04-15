# WFDTQ4VX Dual Lineage Stage Trace

## Scope

This file compares the WFDTQ4VX row universe stage by stage between:

- Lineage A:
  - older frozen raw-response replay lineage used by the `2026-04-15`
    operational replay
- Lineage B:
  - newer live/current Stage2 recovery lineage

## Side-by-side trace

| stage metric | Lineage A | Lineage B | difference | interpretation |
| --- | --- | --- | --- | --- |
| raw response present | yes | yes | no difference | both lineages contain a WFDTQ4VX raw-response artifact |
| raw-response formulation-candidate count | 3 | 34 | `+31` in B | the row universe already diverges at raw response |
| S2-5 semantic formulation count | 3 | 34 | `+31` in B | parsing preserves the divergence rather than causing it later |
| S2-6 validation status | pass | pass | no difference | S2-6 does not reject WFDTQ4VX in either lineage |
| S2-6 WFDTQ4VX-specific errors / warnings | none | none | no difference | no contract-validation evidence of suppression |
| S2-7 projected weak-label rows | 3 | 34 | `+31` in B | projection keeps the upstream row-universe split |
| Stage3 relation records | 28 | 218 | `+190` in B | relation graph size scales with upstream row supply |
| Stage3 resolved relation fields | 7 | 7 | no difference | Stage3 resolves similar descriptive fields in both lineages |
| Stage5 final rows | 2 | 33 | `+31` in B | Stage5 result difference is downstream of the Stage2 split |
| Stage5 decision-trace rows | 3 | 34 | `+31` in B | Stage5 receives many more source candidates in B because they already exist upstream |

## Main finding

The strongest divergence happens before Stage3 and before Stage5.

The decisive split is:

- raw response:
  - `3` formulation candidates in Lineage A
  - `34` formulation candidates in Lineage B
- S2-5:
  - still `3` vs `34`
- S2-7:
  - still `3` vs `34`

That means the WFDTQ4VX row universe does **not** primarily diverge in:

- `S2-6`
- `Stage3`
- or `Stage5`

It diverges earlier, and the later stages mostly preserve that difference.

## Supporting artifact anchors

- Lineage A raw response:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
- Lineage A S2-5 summary:
  - `data/results/20260415_4f1c2ab/01_s2_5_semantic_parsing_replay/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- Lineage A final table:
  - `data/results/20260415_4f1c2ab/04_operational_baseline_final/final_formulation_table_v1.tsv`
- Lineage B raw response:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
- Lineage B S2-5 summary:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- Lineage B final table:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/final_formulation_table_v1.tsv`
