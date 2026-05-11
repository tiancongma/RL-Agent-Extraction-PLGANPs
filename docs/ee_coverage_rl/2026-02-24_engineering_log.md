EE Coverage RL – Engineering Log

Date: 2026-02-24
Branch: feature/ee-coverage-rl
Run: run_20260219_1623_780eb83_goren18_weaklabels_v1

1️⃣ Step 1 – Fix schema_v2 core collapse (explicit formulation IDs)

Problem:
UFXX9WXE collapsed 22–26 explicit formulation IDs into 1 core due to signature missing formulation_id.

Fix:

Inject normalized explicit_formulation_id into schema_v2 core signature before dedup.

Added regression check script.

Result:

UFXX9WXE core count: 1 → 26

Full database structure restored.

This re-established the Full database invariant.

2️⃣ Documentation restructuring

Moved ee_coverage_rl reports into docs/ee_coverage_rl/

Moved benchmark engineering spec into docs/benchmarks/

Moved method-level documents into docs/methods/

Updated tool_index and added index README.

3️⃣ Step 2 – EE Table-Block Structured Support

Problem:
Table-rich docs (WIVUCMYG, WFDTQ4VX) had EE values but ee_supported=0.

Fix:

Added table_block structured support mode.

EE support allowed when numeric candidate present + table context header detected within same/nearby table block.

Added instrumentation columns:

ee_support_level

ee_fail_reason

ee_evidence_snippet

ee_evidence_block_id

Results:

Doc	EE non-empty	EE supported (before)	EE supported (after)
WIVUCMYG	26	0	26
WFDTQ4VX	27	5	6

Guardrails verified:

No upstream effect on Full DB.

UFXX9WXE core count unchanged (26).

No spurious support when EE candidate missing.

4️⃣ Current State

EE extraction and table support working for table-rich papers.

Loading and polymer support remain bottlenecks.

schema_v3 collapse still to verify (next step).

Modeling-ready still 0 under strict 3-field definition.

Core (EE + any key feature) = 5.

5️⃣ System Stability

The pipeline now respects:

Full DB invariant

View layer separation

Debug contract (DOI requirement)

Regression checks for structural changes

Engineering now stable and diagnosable.