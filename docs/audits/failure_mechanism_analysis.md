**Failure Mechanism Analysis**

This analysis explains why functional units stopped controlling execution after Stage2 was split into fine-grained steps, and where the execution path broke.

**Summary**
Stage2 decomposition created new stop boundaries (S2-5 and S2-6) that are valid for replay and validation but do not execute deterministic completion. The functional units that enumerate DOE rows and table rows live inside the deterministic completion step (`S2-7` via `build_stage2_compatibility_projection_v1.py`). When the split path is run or replayed without `S2-7`, or when `S2-7` runs without the full set of inputs or authorized markers, the functional units either never execute or are forced to skip. This manifests as downstream rows missing even though markers existed earlier.

**Where Execution Authority Moved**
Old composite execution:
- LLM semantic discovery and deterministic completion were run in one composite path.
- DOE and table-row expansion were executed immediately after semantic parsing within Stage2.
- Functional unit activation was implicit in the composite entrypoint and its completion substep.

New decomposed execution:
- Semantic parsing (S2-5) and contract validation (S2-6) are explicit stop boundaries.
- Deterministic completion (S2-7) is now optional and must be called explicitly.
- Execution authority moved from the composite wrapper into a separate, later step that can be bypassed.

**Primary Failure Mode**
Execution path broken (root cause category 1), compounded by marker-to-execution disconnect (category 2) and artifact boundary loss (category 4).

**Mechanism Details**
- The deterministic completion step is no longer guaranteed to run after parsing.
- S2-5 artifacts are valid intermediates but are not a mainline resume boundary.
- S2-6 validation is a gate, not an executor; it does not invoke functional units.
- S2-7 is the only step that invokes the deterministic compatibility projection and functional units.
- When the flow stops at S2-5 or S2-6 (common during replay, freeze, or audit), functional units never run.
- When inputs required by functional units are not carried forward (e.g., table assets, evidence bundle pointers, marker readiness context), the completion step runs but the units skip due to missing prerequisites.

**Why This Degrades Functional Units**
Functional units were designed as deterministic completion inside Stage2. The decomposition created multiple legal stopping points that are not aligned with functional-unit execution. The system now permits a valid Stage2 semantic intermediate without forcing or proving unit activation, so execution authority effectively shifts back toward the LLM outputs alone.

