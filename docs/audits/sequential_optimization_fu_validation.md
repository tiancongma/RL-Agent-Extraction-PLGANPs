# Sequential Optimization Function Unit Validation

## Title
Sequential Optimization Function Unit (QLYKLPKT Pattern)

## Maintained path and source resolution
- Maintained entrypoint used: `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- Maintained internal completion path: `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- Maintained validator used: `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
- Validation run directory: `data/results/20260407_ab12cd3/03_seqopt_fu_validation_units`
- Governed replay source for existing Stage2 LLM outputs:
  - manifest: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
  - raw responses: `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses`

## Minimal integration changes
- Added `src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py`
- Hooked the unit into `build_stage2_compatibility_projection_v1.py` after semantic row construction and separately from DOE handling
- Added a narrow `sequential_optimization_resolved` validator branch in `validate_stage2_semantic_authority_contract_v1.py`
- Recorded the function unit in `run_stage2_composite_v1.py` run metadata and `docs/src_script_registry.tsv`

## QLYKLPKT behavior
- Expected:
  - recognize the stagewise optimization chain
  - materialize at most one final optimized formulation row
  - preserve only explicitly selected values plus parent-shared formulation fields
  - avoid any DOE-style expansion or cross-table Cartesian merge
- Actual:
  - `compatibility_projection_summary_v1.json` reports `sequential_optimization_resolved_rows = 1`
  - `QLYKLPKT` summary note is `resolved_final_formulation`
  - the original unresolved `FC_PLGA_ITZ_NS_Optimal` row is suppressed and replaced by one governed resolved row
  - the resolved row preserves:
    - parent-shared fields from `FC_PLGA_ITZ_NS_General`: `PLGA`, `1% (w/v)`, `poloxamer 188`, `acetone`, `Itraconazole`
    - optimal-row measurements: particle size, PDI, zeta potential, encapsulation efficiency
    - explicit selected values:
      - `poloxamer 188 concentration = 3 mg/mL`
      - `PLGA:ITZ (w/w) ratio = 10:1`
      - `lyoprotectant concentration = 2%`
      - `lyoprotectant type = sucrose`
  - trace evidence is recorded in `compatibility_projection_trace_v1.tsv` under `sequential_optimization_resolution`

## No illegal universe construction
- No DOE scope is used for `QLYKLPKT`
- No numbered-row or DOE row expansion occurs
- No cross-table Cartesian merge occurs
- The function unit emits one resolved row only
- The trace payload records:
  - `cross_table_cartesian_merge = no`
  - `doe_scope_present = no`
  - `materialization_basis = explicit_stagewise_selection_text_plus_parent_inheritance`

## Regression checks
- `5GIF3D8W`
  - expected: no trigger
  - actual: summary note `missing_stagewise_selected_values`
  - result: no sequential optimization row emitted
- `UFXX9WXE`
  - expected: no interference with DOE-governed paper
  - actual: summary note `blocked_by_doe_scope`
  - result: sequential unit does not run when DOE scope is present

## Contract result
- `stage2_semantic_authority_contract_report_v1.json` status: `pass`
- DOE behavior remained unchanged in this validation run
- `5GIF3D8W` remained excluded
- `QLYKLPKT` is no longer blocked by missing function-unit coverage under the validated pattern
