**Safeguard Design**

This design adds system-level protections to prevent functional-unit bypass after decomposition.

**A. Execution guarantees**
Guarantee deterministic unit execution when authorized markers are present.
- If `is_doe == true` in semantic scope, DOE enumerator must run and record an attempt.
- If `table_formulation_scope` is declared for a non-DOE table, table-row expansion must run and record an attempt.
- If markers exist but the unit produces zero rows, the attempt is still required and must be logged with a reason.

**B. Hard failure rules**
Disallow silent skips when execution should occur.
- If a required unit did not run, fail the pipeline at S2-7.
- If markers are present but the unit is not invoked, fail fast.
- If a unit is invoked but critical inputs are missing, fail and record missing inputs.

**C. Execution ownership enforcement**
Preserve deterministic ownership of execution.
- LLM outputs may authorize execution but may not replace deterministic unit behavior.
- No LLM fallback may be substituted for DOE or table-row expansion.
- Any fallback path must be explicitly declared and tagged as non-authoritative.

**D. Boundary contracts**
Formalize required inputs at each S-step.
- S2-5 must preserve the semantic scope markers required by completion.
- S2-6 must validate marker readiness and required input availability for S2-7.
- S2-7 must require explicit availability of tables, scope manifests, and evidence pointers needed by units.

**E. Observability guarantees**
Make unit activation provable.
- Emit a run-scoped `unit_activation_report` at S2-7.
- Record per-unit status: `required`, `attempted`, `succeeded`, `skipped_with_reason`.
- Require this report before Stage3 consumption.

