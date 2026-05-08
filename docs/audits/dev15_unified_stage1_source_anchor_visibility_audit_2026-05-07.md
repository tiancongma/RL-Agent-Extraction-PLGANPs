# DEV15 unified Stage1 source-anchor visibility audit — 2026-05-07

Diagnostic-only visibility audit. Not benchmark-valid; not row binding; not value authority; no Stage2/Stage3/Stage5/compare execution.

## Authority and inputs

- Primary user-provided source-anchor authority: `docs/methods/layer3_field_gt_protocol_v1.md:1098-1864`.
- Audited downstream Stage1 surface: `data/cleaned/index/stage1_unified_current_marker_manifest_v1.tsv`.
- Compared surfaces: current clean text, unified clean text, unified structure JSON text fields, table-cell sidecar when bound, and frozen Marker surface for diagnostics.

## Summary

- `complete_or_near_complete`: 7
- `partial_review_needed`: 2
- `major_gap_or_missing`: 1
- `mostly_covered_minor_gaps`: 4
- `complete_with_minor_formatting_drift`: 1

First-failure boundary counts:
- `none_or_minor_formatting_drift`: 12
- `stage1_unified_visibility_partial`: 2
- `html_table_cell_sidecar_absent_or_incomplete`: 1

## Per-paper result

| paper_key | source | status | exact/partial/numeric/missing | boundary | note |
|---|---:|---|---:|---|---|
| INMUTV7L | PDF | complete_or_near_complete | 18/0/0/0 | none_or_minor_formatting_drift |  |
| BB3JUVW7 | HTML | partial_review_needed | 23/0/1/4 | stage1_unified_visibility_partial | first missing: Artemether (>98.0%) was purchased from Tokyo Chemical Industry Co. Ltd. (Tokyo, Japan). Poly(lactic-co-glycolic) acid (P |
| BXCV5XWB | HTML | major_gap_or_missing | 10/1/0/18 | html_table_cell_sidecar_absent_or_incomplete | first missing: 166.63 ± 4.48 |
| L3H2RS2H | PDF | mostly_covered_minor_gaps | 37/7/0/2 | none_or_minor_formatting_drift | first missing: ζ (mV)	−36.2±5.2	−38.9±1.3	−36.0±3.0 |
| PA3SPZ28 | PDF | complete_or_near_complete | 32/0/0/0 | none_or_minor_formatting_drift |  |
| QLYKLPKT | PDF | complete_or_near_complete | 18/0/0/0 | none_or_minor_formatting_drift |  |
| RHMJWZX8 | PDF | mostly_covered_minor_gaps | 4/0/0/1 | none_or_minor_formatting_drift | first missing: Zeta potential can influence physical stability of a colloidal dispersion. High negative charge of zeta potential indica |
| UFXX9WXE | PDF | mostly_covered_minor_gaps | 48/0/0/1 | none_or_minor_formatting_drift | first missing: −1	0	1 |
| V99GKZEI | HTML | mostly_covered_minor_gaps | 12/2/0/1 | none_or_minor_formatting_drift | first missing: MB loaded-PLGAb	220 ± 4	0.19 ± 0.02	43.21 ± 2.69	0.52 ± 0.19	3.12 ± 1.12 |
| WFDTQ4VX | PDF | complete_with_minor_formatting_drift | 27/21/0/0 | none_or_minor_formatting_drift |  |
| WIVUCMYG | HTML | complete_or_near_complete | 38/0/0/0 | none_or_minor_formatting_drift |  |
| YGA8VQKU | HTML | partial_review_needed | 31/0/1/5 | stage1_unified_visibility_partial | first missing: The poly(lactic-co-glycolic) acid (PLGA) polymers Resomer® RG756S and Resomer® RG753S composed of lactide:glycolide 75:2 |
| 7ZS858NS | PDF | complete_or_near_complete | 8/0/0/0 | none_or_minor_formatting_drift |  |
| 5ZXYABSU | PDF | complete_or_near_complete | 27/1/0/0 | none_or_minor_formatting_drift |  |
| 5GIF3D8W | PDF | complete_or_near_complete | 33/2/0/0 | none_or_minor_formatting_drift |  |

## Focused review of flagged/near-flagged cases

A focused token check over the combined unified downstream surface confirmed the high-level pattern:

- `BB3JUVW7`: important table values are visible (`190.2`, `510.7`, `Drug content`), but preparation/material paragraphs such as `Artemether loaded PLGA nanospheres were prepared` and `PLGA (75 mg) and artemether` were not found. This is a source-visibility gap for methods text, not a Stage5 issue.
- `BXCV5XWB`: the key KGN-loaded table values are largely not visible (`166.63`, `297.32`, `507.01` absent) and no HTML table-cell sidecar is bound. This is the strongest remaining Stage1/source extraction gap.
- `YGA8VQKU`: table values are visible (`240.00`, `97.75`, `Table 6`, `232.80`), but important materials/preparation text such as `Resomer RG756S` and `Nanospheres composed of PLGA 75:25` was not found. This needs source review before claiming full preservation.
- `L3H2RS2H`: most tables are visible, including later incorporation values (`1173`, `2780`), but selected zeta-potential rows from Tables 2/4 were not found in normalized search (`-36.2`, `-38.9`, `-40.9`). Treat as minor table-row formatting/visibility review unless manual inspection proves the row is present in another encoding.
- `V99GKZEI`: focused values are visible (`MB loaded-PLGA`, `220`, `3.12`, `57.89`); the single missing row in the fragment audit is likely formatting/row-label drift rather than source disappearance.

## Interpretation

Conclusion: the current unified Stage1 surface is **much improved and usable for controlled pre-live gating**, but it is **not yet safe to claim that all user-uploaded key paragraphs and tables are fully extracted without important loss**.

- 12/15 papers are complete/mostly covered with at most minor formatting/table-fragment review.
- 3/15 papers require review before broad extraction is treated as safe: `BB3JUVW7`, `BXCV5XWB`, `YGA8VQKU`.
- The main blocker is `BXCV5XWB` HTML table extraction/sidecar absence; secondary blockers are missing/partial methods paragraphs in `BB3JUVW7` and `YGA8VQKU`.

Recommended next step before expanding all original-text extraction: repair or manually rehydrate the three flagged source-visibility cases into the unified Stage1 surface, then rerun this visibility audit. Do not spend live Stage2 LLM calls on full expansion until these three are resolved or explicitly accepted as manual exceptions.

## Output files

- TSV: `docs/audits/dev15_unified_stage1_source_anchor_visibility_audit_2026-05-07.tsv`
- Details JSON: `docs/audits/dev15_unified_stage1_source_anchor_visibility_audit_2026-05-07.details.json`
