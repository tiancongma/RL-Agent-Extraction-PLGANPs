# DEV15 Stage1 Current PDF/HTML vs Marker vs Source Anchor Visibility Audit — 2026-05-07

Scope: **Stage1-only** clean-text/structure visibility audit. No Stage2, Stage5, live LLM, ACTIVE_RUN update, or benchmark claim was run/made.

Primary source-anchor authority: `docs/methods/layer3_field_gt_protocol_v1.md:1098-1864`. `data/results/.../raw_anchor_snippets/` are derived only and were not used as primary authority.

Compared surfaces:
- Current Stage1 clean text/structure from `key2txt.tsv` and `key2structure.tsv`.
- Frozen Marker PDF clean text/structure from `data/cleaned/content/marker_pdf/<paper_key>/` and `key2marker_pdf_v1.tsv`.
- Additive current+Marker fusion from `data/cleaned/content/pdf_current_marker_fusion/<paper_key>/` and `key2stage1_pdf_fusion_v1.tsv`.

## High-level result

- DEV15 papers: 15. PDF Marker-target papers: 10. HTML papers where Marker PDF is not applicable: 5.
- DEV15 PDF governed Marker coverage: 10/10.
- DEV15 PDF current+Marker fusion coverage: 10/10.
- Main finding: Marker is valuable as an **additive structure/table-cell/block surface**, not as a wholesale replacement for current clean text. Current PDF/HTML text often has stronger literal prose continuity; Marker provides much richer page/block/section segmentation and exposes explicit `Table`/`TableCell` blocks for PDFs.
- Fusion is therefore the correct Stage1 direction: preserve current clean text for compatibility and prose recall, attach Marker text/structure/table blocks additively for downstream no-live preparation after table/cell validation.

## Per-paper metrics

| paper | src | anchor table mentions | current numeric | marker numeric | current key phrases | marker key phrases | current blocks/T/TC/heads | marker blocks/T/TC/heads | assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| INMUTV7L | PDF | 2 | 75/75 (100.0%) | 43/75 (57.3%) | 17/18 (94.4%) | 6/18 (33.3%) | 25/1/0/0 | 704/7/434/32 | marker_partial_review |
| BB3JUVW7 | HTML | 2 | 87/101 (86.1%) | 0/101 (0.0%) | 21/24 (87.5%) | 0/24 (0.0%) | 168/2/0/58 | 0/0/0/0 | html_current_only_marker_not_applicable |
| BXCV5XWB | HTML | 1 | 26/45 (57.8%) | 0/45 (0.0%) | 7/15 (46.7%) | 0/15 (0.0%) | 364/0/0/47 | 0/0/0/0 | html_current_only_marker_not_applicable |
| L3H2RS2H | PDF | 5 | 57/107 (53.3%) | 14/107 (13.1%) | 15/38 (39.5%) | 0/38 (0.0%) | 11/1/0/0 | 22/0/0/4 | marker_partial_review |
| PA3SPZ28 | PDF | 1 | 36/36 (100.0%) | 36/36 (100.0%) | 8/9 (88.9%) | 8/9 (88.9%) | 15/1/0/0 | 296/2/110/11 | marker_good_additive |
| QLYKLPKT | PDF | 4 | 45/45 (100.0%) | 45/45 (100.0%) | 20/20 (100.0%) | 19/20 (95.0%) | 12/1/0/0 | 388/4/160/22 | marker_good_additive |
| RHMJWZX8 | PDF | 0 | 30/30 (100.0%) | 28/30 (93.3%) | 7/7 (100.0%) | 7/7 (100.0%) | 14/1/0/0 | 416/3/133/37 | marker_good_additive |
| UFXX9WXE | PDF | 4 | 127/127 (100.0%) | 52/127 (40.9%) | 38/46 (82.6%) | 12/46 (26.1%) | 15/1/0/0 | 551/4/303/9 | marker_partial_review |
| V99GKZEI | HTML | 1 | 64/64 (100.0%) | 0/64 (0.0%) | 13/13 (100.0%) | 0/13 (0.0%) | 185/2/0/35 | 0/0/0/0 | html_current_only_marker_not_applicable |
| WFDTQ4VX | PDF | 5 | 132/133 (99.2%) | 81/133 (60.9%) | 25/48 (52.1%) | 18/48 (37.5%) | 15/1/0/0 | 714/10/462/33 | marker_partial_review |
| WIVUCMYG | HTML | 3 | 162/162 (100.0%) | 0/162 (0.0%) | 39/39 (100.0%) | 0/39 (0.0%) | 695/6/0/0 | 0/0/0/0 | html_current_only_marker_not_applicable |
| YGA8VQKU | HTML | 6 | 102/107 (95.3%) | 0/107 (0.0%) | 29/33 (87.9%) | 0/33 (0.0%) | 157/7/0/44 | 0/0/0/0 | html_current_only_marker_not_applicable |
| 7ZS858NS | PDF | 1 | 22/22 (100.0%) | 17/22 (77.3%) | 7/8 (87.5%) | 5/8 (62.5%) | 9/1/0/0 | 294/3/89/13 | marker_good_additive |
| 5ZXYABSU | PDF | 6 | 53/53 (100.0%) | 53/53 (100.0%) | 21/22 (95.5%) | 21/22 (95.5%) | 11/1/0/0 | 323/3/118/21 | marker_good_additive |
| 5GIF3D8W | PDF | 3 | 95/95 (100.0%) | 53/95 (55.8%) | 25/25 (100.0%) | 12/25 (48.0%) | 17/7/0/0 | 266/5/117/18 | marker_partial_review |

## PDF paper notes

### INMUTV7L
- Source-anchor table mentions: 2; current table blocks/cells: 1/0; Marker table blocks/cells: 7/434.
- Numeric visibility: current 75/75; Marker 43/75; fusion 75/75.
- Key phrase visibility: current 17/18; Marker 6/18; fusion 18/18.
- Structure clarity: current blocks/Table/TableCell/section headers = 25/1/0/0; Marker = 704/7/434/32.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `1	PLGA 503 H	PVA	234.1 ± 0.5	0.081 ± 0.009	−12.2 ± 1.3	93.4`
  - `2	Tween80®	146.0 ± 0.6	0.054 ± 0.008	−25.2 ± 0.6	87.5`
  - `3	Lutrol	159.5 ± 0.8	0.058 ± 0.021	−26.0 ± 0.1	85.1`
- Current missing/reformatted key anchor examples:
  - `Table 1. Characterization of the different formulations developed.`

### L3H2RS2H
- Source-anchor table mentions: 5; current table blocks/cells: 1/0; Marker table blocks/cells: 0/0.
- Numeric visibility: current 57/107; Marker 14/107; fusion 57/107.
- Key phrase visibility: current 15/38; Marker 0/38; fusion 16/38.
- Structure clarity: current blocks/Table/TableCell/section headers = 11/1/0/0; Marker = 22/0/0/4.
- Table completeness risk: Marker exposes no explicit Table/TableCell blocks despite source-anchor table mentions; inspect raw Marker output and PDF pages before treating Marker as table authority.
- Marker missing/reformatted key anchor examples:
  - `2. Materials and methods`
  - `Xanthone (XAN), PLGA (50:50) MW 50 000–75 000, Pluronic F-68, phosphate buffered saline tablets and soybean lecithin (40% purity by thin-layer chromatography) were purchased from S`
  - `2.2. Preparation of nanospheres`
- Current missing/reformatted key anchor examples:
  - `Table 1. Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanospheres`
  - `50	13.0±1.1	26.1±2.1	19.0±0.6	38.1±1.1`
  - `60	20.0±2.4	33.0±4.1	24.9±4.6	41.5±7.6`

### PA3SPZ28
- Source-anchor table mentions: 1; current table blocks/cells: 1/0; Marker table blocks/cells: 2/110.
- Numeric visibility: current 36/36; Marker 36/36; fusion 36/36.
- Key phrase visibility: current 8/9; Marker 8/9; fusion 8/9.
- Structure clarity: current blocks/Table/TableCell/section headers = 15/1/0/0; Marker = 296/2/110/11.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `Table1 Characterizaton of GAR-NPs.`
- Current missing/reformatted key anchor examples:
  - `Table1 Characterizaton of GAR-NPs.`

### QLYKLPKT
- Source-anchor table mentions: 4; current table blocks/cells: 1/0; Marker table blocks/cells: 4/160.
- Numeric visibility: current 45/45; Marker 45/45; fusion 45/45.
- Key phrase visibility: current 20/20; Marker 19/20; fusion 20/20.
- Structure clarity: current blocks/Table/TableCell/section headers = 12/1/0/0; Marker = 388/4/160/22.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `Note: Ratio of 10:1 was then selected to prepare PLGA-ITZ-NS for the remaining studies.`

### RHMJWZX8
- Source-anchor table mentions: 0; current table blocks/cells: 1/0; Marker table blocks/cells: 3/133.
- Numeric visibility: current 30/30; Marker 28/30; fusion 30/30.
- Key phrase visibility: current 7/7; Marker 7/7; fusion 7/7.
- Structure clarity: current blocks/Table/TableCell/section headers = 14/1/0/0; Marker = 416/3/133/37.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.

### UFXX9WXE
- Source-anchor table mentions: 4; current table blocks/cells: 1/0; Marker table blocks/cells: 4/303.
- Numeric visibility: current 127/127; Marker 52/127; fusion 127/127.
- Key phrase visibility: current 38/46; Marker 12/46; fusion 38/46.
- Structure clarity: current blocks/Table/TableCell/section headers = 15/1/0/0; Marker = 551/4/303/9.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `Table 1. Independent and dependent variables levels in Box-Behnken design.`
  - `X1 = polymer concentration (w/v)	10	35	60`
  - `X2 = surfactant concentration (w/v)	2	8.50	15`
- Current missing/reformatted key anchor examples:
  - `Table 1. Independent and dependent variables levels in Box-Behnken design.`
  - `X1 = polymer concentration (w/v)	10	35	60`
  - `X2 = surfactant concentration (w/v)	2	8.50	15`

### WFDTQ4VX
- Source-anchor table mentions: 5; current table blocks/cells: 1/0; Marker table blocks/cells: 10/462.
- Numeric visibility: current 132/133; Marker 81/133; fusion 133/133.
- Key phrase visibility: current 25/48; Marker 18/48; fusion 28/48.
- Structure clarity: current blocks/Table/TableCell/section headers = 15/1/0/0; Marker = 714/10/462/33.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `Sr. No.	X1	X2	X3	Y1Table Footnotea (EE, %)	Y2Table Footnotea (PS, nm)`
  - `1	−1	−1	−1	36.5 ± 2.21	126.6 ± 4.16`
  - `2	−1	−1	0	29.4 ± 1.05	131.3 ± 6.13`
- Current missing/reformatted key anchor examples:
  - `Sr. No.	X1	X2	X3	Y1Table Footnotea (EE, %)	Y2Table Footnotea (PS, nm)`
  - `1	−1	−1	−1	36.5 ± 2.21	126.6 ± 4.16`
  - `2	−1	−1	0	29.4 ± 1.05	131.3 ± 6.13`

### 7ZS858NS
- Source-anchor table mentions: 1; current table blocks/cells: 1/0; Marker table blocks/cells: 3/89.
- Numeric visibility: current 22/22; Marker 17/22; fusion 22/22.
- Key phrase visibility: current 7/8; Marker 5/8; fusion 7/8.
- Structure clarity: current blocks/Table/TableCell/section headers = 9/1/0/0; Marker = 294/3/89/13.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `MF NPs	117 ± 13	0.26 ± 0.02	–32 ± 1.2	90 ± 2.1	22.4 ± 0.5`
  - `aEncapsulation efficiency (%) = (amount of drug in nanoparticles/amount of drug fed initially) × 100.`
  - `bDrug loading content (%) = [amount of drug/(amount of drug + amount of polymer)] × 100.`
- Current missing/reformatted key anchor examples:
  - `aEncapsulation efficiency (%) = (amount of drug in nanoparticles/amount of drug fed initially) × 100.`

### 5ZXYABSU
- Source-anchor table mentions: 6; current table blocks/cells: 1/0; Marker table blocks/cells: 3/118.
- Numeric visibility: current 53/53; Marker 53/53; fusion 53/53.
- Key phrase visibility: current 21/22; Marker 21/22; fusion 21/22.
- Structure clarity: current blocks/Table/TableCell/section headers = 11/1/0/0; Marker = 323/3/118/21.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `Download CSVDisplay Table`
- Current missing/reformatted key anchor examples:
  - `Download CSVDisplay Table`

### 5GIF3D8W
- Source-anchor table mentions: 3; current table blocks/cells: 7/0; Marker table blocks/cells: 5/117.
- Numeric visibility: current 95/95; Marker 53/95; fusion 95/95.
- Key phrase visibility: current 25/25; Marker 12/25; fusion 25/25.
- Structure clarity: current blocks/Table/TableCell/section headers = 17/7/0/0; Marker = 266/5/117/18.
- Table structure signal: Marker exposes explicit Table/TableCell structure; still needs row/cell validation before downstream use.
- Marker missing/reformatted key anchor examples:
  - `PLGA 50/50 (Mean ± SD)	PLGA 75/25 (Mean ± SD)`
  - `Diameter (nm)	87.2 ± 0.25	91.8 ± 2.74	96.9 ± 1.06	103.7 ± 2.98`
  - `PIa	0.14 ± 0.01	0.13 ± 0.01	0.12 ± 0.01	0.14 ± 0.01`

## HTML note

These DEV15 papers are HTML in the current manifest; Marker PDF extraction is not applicable, so they are current/HTML clean-text vs source-anchor audit targets only:

BB3JUVW7, BXCV5XWB, V99GKZEI, WIVUCMYG, YGA8VQKU

## Conclusion

For DEV15 PDFs, Marker processing and current+Marker fusion are complete as governed Stage1 surfaces. The audit supports using fusion rather than replacement: current clean text protects prose/key paragraph completeness, while Marker contributes stronger structural segmentation and explicit Table/TableCell blocks. Before Stage2 is rerun, the next Stage1 engineering gate should validate Marker/fusion table rows/cells against the uploaded source anchors and expose the fused structure through the maintained Stage2 no-live input-preparation path. This report remains Stage1-only and diagnostic for visibility; it is not benchmark-valid final output evidence.

Companion files:
- `docs/audits/dev15_stage1_current_marker_source_anchor_visibility_audit_2026-05-07.tsv`
- `docs/audits/dev15_stage1_current_marker_source_anchor_visibility_audit_2026-05-07.details.json`
