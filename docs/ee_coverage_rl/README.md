# EE Coverage RL Docs

This directory collects EE coverage / RL branch diagnostics and related notes
for the `feature/ee-coverage-rl` workstream. It is a supporting documentation
area for experiments, regressions, and coverage debugging. It is not an
authoritative governance surface.

## Status

This directory is experimental and diagnostic. The files here help explain
specific regressions, run slices, and local engineering questions, but they do
not define active pipeline authority.

## Content Types

### Audits

- [Benchmark18 stability status](./ee_coverage_rl__benchmark18_status.md)
- [Benchmark18 table-rich regression diagnosis](./regression_diagnosis__benchmark18_table_rich.md)
- [Modeling-ready diagnosis](./ee_coverage_rl__modeling_ready_diagnosis__run_20260219_1623_780eb83.md)
- [Core modeling summary](./ee_coverage_rl__core_modeling_summary__run_20260219_1623_780eb83.md)

### Experiments

- [Run metrics summary](./ee_coverage_rl__run_20260219_1623_780eb83__metrics_summary.md)
- [Stage1 HTML-first tables manifest](./2026-02-26_stage1_html_first_tables_manifest.md)

### Diagnostics

- [Evidence alignment instance-key + inheritance debug](./evidence_alignment_instancekey_inheritance_debug.md)
- [Step2 EE table support regression](./step2_ee_table_support_regression.md)

### Notes

- [2026-02-24 engineering log](./2026-02-24_engineering_log.md)

## Suggested Reading Order

1. Start with [Benchmark18 stability status](./ee_coverage_rl__benchmark18_status.md)
   for the high-level branch state.
2. Read [Run metrics summary](./ee_coverage_rl__run_20260219_1623_780eb83__metrics_summary.md)
   to understand the referenced run surface.
3. Use [Modeling-ready diagnosis](./ee_coverage_rl__modeling_ready_diagnosis__run_20260219_1623_780eb83.md)
   and [Core modeling summary](./ee_coverage_rl__core_modeling_summary__run_20260219_1623_780eb83.md)
   for the main failure analysis.
4. Consult the regression and debug notes only as needed for narrower issues.

## Navigation Notes

- Most files are branch-specific and historically named.
- Run-specific files usually include the relevant run id in the filename.
- This directory is intentionally kept intact in place; this pass adds
  navigation only and does not split or rename the existing files.
