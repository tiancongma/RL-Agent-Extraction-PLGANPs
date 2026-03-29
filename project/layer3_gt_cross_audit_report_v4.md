# Layer 3 GT Cross-Audit Report v4

## Scope

- Surface: value GT annotation workbook v4
- Principle: retain only values explicitly supported by source text or tables
- Audit mode: report-only; no workbook edits and no value corrections

## Outputs

- Merged TSV: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\analysis\layer3_gt_cross_audit_report_v4.tsv`
- High-risk subset: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\analysis\layer3_gt_cross_audit_report_v4_high_priority.tsv`

## Counts

- Total flagged cells: `32`
- High-risk cells: `8`

## Execution

- Candidate cells after refined rule pass: `32`
- Cells actually sent to Gemini: `32`
- Cells actually sent to NVIDIA: `32`
- Cells returned by Gemini: `32`
- Cells returned by NVIDIA: `31`
- Actually executed: `gemini,nvidia`

### By Risk Type

- `direction_mismatch`: `24`
- `blank_should_be_null`: `8`

### By Source

- `nvidia+rule`: `31`
- `rule`: `1`

## Source Availability

- `rule` rows merged: `32`
- `gemini` rows merged: `0`
- `nvidia` rows merged: `31`

## High-Risk Sample

| paper_id | doi | formulation_id | field_name | current_value | risk_type | source_of_flag | reason |
|---|---|---|---|---|---|---|---|
| 10.1007/s10439-019-02430-x | 10.1007/s10439-019-02430-x | BXCV5XWB_G002 | drug_name | FITC | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |
| 10.1007/s10439-019-02430-x | 10.1007/s10439-019-02430-x | BXCV5XWB_G006 | drug_name | FITC | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |
| 10.1007/s10439-019-02430-x | 10.1007/s10439-019-02430-x | BXCV5XWB_G009 | drug_name | FITC | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |
| 10.1016/j.ejpb.2004.09.002 | 10.1016/j.ejpb.2004.09.002 | L3H2RS2H_G006 | drug_name | XAN | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |
| 10.1016/j.ejpb.2004.09.002 | 10.1016/j.ejpb.2004.09.002 | L3H2RS2H_G007 | drug_name | XAN | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |
| 10.1016/j.ejpb.2004.09.002 | 10.1016/j.ejpb.2004.09.002 | L3H2RS2H_G017 | drug_name | XAN | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |
| 10.1038/s41598-017-00696-6 | 10.1038/s41598-017-00696-6 | PA3SPZ28_G001 | drug_name | Etoposide | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. \| nvidia: Blank... |
| 10.1080/10717540802174662 | 10.1080/10717540802174662 | 5GIF3D8W_G031 | drug_name | Etoposide | blank_should_be_null | nvidia+rule | rule: Blank/empty/unloaded formulation carries a non-null drug value. Layer 3 blank rows must preserve a null drug field. Secondary signa... |

## Notes

- `supported` means explicit source support was available and the cell was still flagged for another risk such as contamination or ambiguity.
- `derived` means the workbook value appears computable or formatting-dependent rather than explicitly stated.
- `unsupported` means the value was not found explicitly in cleaned text/tables for the current paper.
- Model outputs are audit signals only and must not be treated as edits or truth labels.
- Use the TSV to locate the cell in the existing v4 workbook and manually verify the cited paper-local evidence.

