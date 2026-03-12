# RUN_SPEC Template

Use this template in every `data/results/run_*` directory.

The file may be named `RUN_CONTEXT.md` or `RUN_SPEC.md`, but the content must include all sections below.

## 1. Run ID

- Exact run directory name

## 2. Run Type

- One of:
  - `intermediate_diagnostic_run`
  - `component_regression_run`
  - `full_pipeline_benchmark_run`

## 3. Purpose

- Why the run exists
- What question it is intended to answer

## 4. Benchmark Status

- One of:
  - `benchmark-valid final output`
  - `diagnostic-only, not benchmark-valid final output`

## 5. Starting Inputs

- Exact input file paths
- Exact manifest paths
- Exact benchmark or GT paths if used
- Exact prior run artifacts if reused

## 6. Environment Assumptions

- Required model or API configuration
- Required branch or commit if relevant
- Any required environment variables

## 7. Step-by-Step Execution Plan

For each step include:

- step number
- script path
- exact command or command pattern
- upstream artifacts consumed
- intermediate artifacts produced

## 8. Script Paths Used

- Flat list of every script used in the run

## 9. Intermediate Artifacts

- Every important intermediate file or directory
- Why it exists

## 10. Final Outputs

- Every final file or directory the run is meant to produce

## 11. Comparison Contract

- State whether the run is comparing:
  - candidate-layer output
  - component-only output
  - full final-output layer
- If GT is involved, state why the comparison is diagnostic-only or benchmark-valid

## 12. Reproduction Checklist

- Can a human rerun the same scripts in the same order?
- Are all input paths recorded?
- Are all script paths recorded?
- Are run type and benchmark status explicit?
- Are final outputs named explicitly?
