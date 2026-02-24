# Regression Diagnosis: Benchmark18 Table-Rich

Run: `run_20260219_1623_780eb83_goren18_weaklabels_v1`

## Doc Selection
Top-3 benchmark docs by expected formulation rows from `overlap_goren18_v1/dedup_ee_v1/extracted_ee_dedup_rows.tsv`.
- `WIVUCMYG` (26 raw extracted rows)
- `WFDTQ4VX` (25 raw extracted rows)
- `UFXX9WXE` (22 raw extracted rows)

## Per-Doc Counts by Layer
| key      |   expected_rows_raw_table |   n_formulation_core_v2 |   n_formulation_core_v3 |   n_measurements_total |   n_measurements_EE_type |   n_rows_projected_has_EE |   n_rows_ee_local_support_1 | drop_location         |
|:---------|--------------------------:|------------------------:|------------------------:|-----------------------:|-------------------------:|--------------------------:|----------------------------:|:----------------------|
| WIVUCMYG |                        26 |                       5 |                      28 |                    113 |                       26 |                        26 |                           0 | evidence gating layer |
| WFDTQ4VX |                        25 |                      10 |                      10 |                     55 |                       27 |                        27 |                           5 | evidence gating layer |
| UFXX9WXE |                        22 |                       1 |                       1 |                     79 |                       26 |                        26 |                           5 | splitting/dedup layer |

Interpretation key:
- `n_formulation_core_v2/v3`: full-database core rows after schema splitting/dedup
- `n_rows_projected_has_EE`: benchmark projected view rows with non-empty `EE`
- `n_rows_ee_local_support_1`: benchmark evidence-gated EE-supported rows

## Where Drop Happens
- `WIVUCMYG`: raw=26, core_v3=28, projected_has_EE=26, ee_supported=0 -> **evidence gating layer**
- `WFDTQ4VX`: raw=25, core_v3=10, projected_has_EE=27, ee_supported=5 -> **evidence gating layer**
- `UFXX9WXE`: raw=22, core_v3=1, projected_has_EE=26, ee_supported=5 -> **splitting/dedup layer**

## Code Path Audit (Benchmark Flags vs Full DB)
- Full DB core construction is in `src/stage5_benchmark/build_two_table_schema_v2.py` and `src/stage5_benchmark/build_two_table_schema_v3.py`.
- Core dedup/grouping happens in `build_two_table_schema_v2.py` at the groupby on `[reference_normalized_doi, core_signature]` (around lines 303-327), where multiple `group_key` rows collapse when their signature is identical.
- V3 DOE splitting is applied in `build_two_table_schema_v3.py` (`build_once`, around lines 191-373), then measurements are reassigned by `group_key`.
- Benchmark-only gating flags are computed in `src/stage5_benchmark/run_evidence_token_qc_v1.py` (`ee_local_support` around lines 957-1041) and written to `confidence_tiers__formulation_level.tsv`; these are downstream view/eval artifacts.
- Search check found no `has_EE`/`ee_local_support`/`modeling_ready` usage in upstream full-DB builders (`build_two_table_schema_*`, `export_full_database*`).

## Conclusion
- No evidence that benchmark-view flags (`has_EE`, `ee_local_support`, modeling-ready) are constraining full-database outputs upstream.
- For table-rich docs, low benchmark supported-row counts are mainly downstream evidence-gating loss (e.g., `WIVUCMYG`).
- The strongest true row-collapse signal is `UFXX9WXE`, where many extracted table rows map to one core signature already in schema v2; this indicates a core-signature granularity issue in `build_two_table_schema_v2.py` rather than a benchmark-view filter regression.
