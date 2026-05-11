**Old vs New Execution Path**

This comparison focuses on DOE and table-row expansion responsibilities before and after Stage2 decomposition.

**Old System (Pre-Decomposition)**
Execution shape:
- Single composite entrypoint (`run_stage2_composite_v1.py`) executed the full Stage2 chain.
- LLM semantic discovery was immediately followed by deterministic completion.
- DOE enumeration and table-row expansion ran inside the deterministic completion step.

Where enumeration happened:
- `build_stage2_compatibility_projection_v1.py` invoked:
  - DOE enumerator (`doe_row_expansion_function_unit_v1`)
  - Table-row expansion (`table_row_expansion_v1`)
- Unit activation was guaranteed by the composite chain.

Execution authority:
- LLM discovered semantic scope.
- Deterministic units executed within the same Stage2 run.
- The completed Stage2 artifact was always produced when Stage2 ran.

**New System (S2-2 → S2-7 Decomposition)**
Execution shape:
- Stage2 is split into explicit boundaries: S2-2, S2-3, S2-4a, S2-4b, S2-5, S2-6, S2-7.
- S2-5 and S2-6 are legal stop points and do not execute deterministic completion.
- S2-7 alone runs the compatibility projection and functional units.

Where enumeration is supposed to occur:
- S2-7 (`run_stage2_s2_7_compatibility_projection_v1.py`) calls `build_stage2_compatibility_projection_v1.py`.
- DOE and table-row expansion are still in the same deterministic substep, but this substep is now optional and separable.

Execution authority movement:
- Authority for execution moved from the composite Stage2 wrapper into a later, explicitly invoked step.
- If the pipeline stops at S2-5 or S2-6, execution does not occur.
- If S2-7 runs without full inputs or authorized markers, execution is suppressed even if markers exist.

