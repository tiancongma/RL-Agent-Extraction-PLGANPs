**Enforcement Plan**

Concrete mechanisms to prevent functional-unit bypass.

**1. Assertion checks**
Add explicit assertions in S2-7 that must pass before writing the completed Stage2 artifact.
- Assert that DOE unit ran when DOE markers exist.
- Assert that table-row expansion ran when table authorization markers exist.
- Assert that required inputs (tables, manifest context, marker readiness) are present.

**2. Execution gates**
Introduce a gate between S2-6 and S2-7 that validates readiness and required input availability.
- If readiness is insufficient, fail with a structured error class.
- Do not allow fallback to “no-op” completion.

**3. No silent skip rules**
Any skip must be explicit and recorded.
- `unit_status = skipped` must include `skip_reason`.
- `skip_reason` must be a governed enum, not free text.

**4. Unit activation validator**
Add a validator that reads the S2-7 run directory and enforces:
- each required unit has an activation record
- required units attempted
- missing activation is a hard failure

**5. Pipeline integrity checks**
Before Stage3, enforce:
- completed Stage2 artifact exists
- S2-7 activation report exists
- semantic source mode and marker readiness are consistent

**6. Observability guarantees**
Write a `unit_activation_report_v1.json` in the S2-7 output directory.
Include:
- unit name
- required or optional
- attempt timestamp
- input paths used
- output counts
- skip reason if any

