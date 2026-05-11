# RUN_CONTEXT

## Purpose
- Full pipeline freeze snapshot for DEV15 (non-benchmark-valid)
- Preserve Stage2 through Stage5 artifacts plus identity-freeze audit for reproducibility
- This freeze is a structured promotion only; no stages were rerun

## Source Run
- `data/results/20260412_8517d36/18_full_pipeline_benchmark_dev15_v1/`

## Stage Coverage
- `S2-4b` -> `S2-5` -> `S2-6` -> `S2-7` -> `Stage3` -> `Stage5` -> `identity_freeze`

## Boundary Classification
- Stage2: `mainline_resume_boundary` (from S2-7 compatibility projection)
- Stage5: `diagnostic_boundary` (identity freeze failed)

## Identity Freeze Status
- `FAIL`
- failure classes:
  - row count drift
  - identity reassignment
  - unresolved scaffold binding

## Notes
- THIS IS NOT A BENCHMARK-VALID RUN
- GT comparison was not executed for this frozen lineage
