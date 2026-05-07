# Drug Mass Residual Generic Capability Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Keep every runtime change generic. Concrete paper keys below are validation/audit samples only, not implementation conditions.

**Goal:** Close the remaining `drug_mass_mg` diagnostic failures by implementing the next generic material-value capabilities exposed by the residuals: variable-sweep scoped binding, method-paragraph mass-pair carrythrough, stronger negative-context filtering, mismatch rejection/audit, and alignment-gated handling.

**Architecture:** Extend the existing generic `src/stage5_benchmark/material_value_binding_v1.py` + Stage5 adapter path. Keep Stage2 as semantic authority for formulation rows; Stage3/S5-2 may only bind, validate, scope, carry through, reject, and audit direct source-backed values for already admitted rows. Do not use S5-3, GT lookup, paper-specific branches, DOI branches, or hard-coded drug-specific value maps.

**Tech Stack:** Python standard library, existing `unittest` suite, `src/stage5_benchmark/material_value_binding_v1.py`, `src/stage5_benchmark/build_minimal_final_output_v1.py`, explicit diagnostic artifacts under `data/results/20260504_ab9f61e/`, governed contracts in `project/`.

---

## Current Diagnostic Baseline

Use this baseline unless the user explicitly promotes a newer diagnosis lineage:

- Before generic material-value binding validation:
  - `data/results/20260504_ab9f61e/032_layer3_compare_current_s5_2_no_s5_3_baseline_doe_explicit_only_diagnostic/layer3_value_compare_cells_v1.tsv`
- Current after-line:
  - `data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_value_compare_cells_v1.tsv`
  - `data/results/20260504_ab9f61e/049_stage5_p9_polymer_mgml_mass_guard_bounded_diagnostic/final_formulation_table_v1.tsv`
- Locked Layer3 GT authority:
  - `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`
- Stage5 replay inputs used by current after-line:
  - Stage2: `data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
  - Stage3 relation records: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv`
  - Stage3 resolved fields: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/resolved_relation_fields_v1.tsv`

Current `drug_mass_mg` status after `050`:

```text
present_and_match:     26
present_but_mismatch:   7
missing_in_system:     17
blocked_alignment:      6
extra_in_system:        0
not_reported_in_gt:   116
```

Residual capability classes to close:

```text
variable-sweep / formulation-family scoped binding
method paragraph shared drug/polymer mass-pair carrythrough
negative-context / method-scope guard in multi-mass text
evidence-selection/alignment gated handling, not ordinary carrythrough
numeric misbinding / non-input mass rejection
blocked alignment gating: alignment first, no value materializer override
```

Concrete paper keys may be used only as offline validation/audit examples after the generic capability is implemented. They must not determine execution order, runtime conditions, hard-coded aliases, or success criteria.

---

## Non-Negotiable Rules

1. **No paper-specific branches.** No `if paper_key == ...`, DOI switches, paper-key allowlists, or DEV15 value maps.
2. **No drug-specific hard-coded amount rules.** The runtime must not encode that a concrete token like `MB`, `GAR`, `AP`, or a named drug has a fixed mass.
3. **No GT-driven filling.** GT may only validate diagnostics; it must not drive runtime materialization.
4. **No S5-3 compensation.** This plan is S3/S5-2 deterministic direct materialization. S5-3 remains residual source-evidenced extraction only after these generic capabilities are exhausted.
5. **No global donor fill.** Do not propagate a paper-level mass unless scope is unique, source-backed, and compatible with admitted row semantics.
6. **Row-local direct evidence wins.** Shared/method/footnote values must not overwrite row-local table cells or typed row-local assignments.
7. **Negative context blocks direct materialization.** Release, assay, dose, PK, LC/DL result, sample-prep, injection, DSC, cell uptake, and measurement contexts must be rejected or review-queued for direct input mass fields.
8. **Alignment failures are not value failures.** If no unique GT/system row alignment exists, stop at alignment review; do not fabricate values to bypass blocked alignment.
9. **All outputs remain diagnostic-only until promoted by governance.** Do not report benchmark-valid improvement from these runs.

---

## Desired Capability Decomposition

### Capability A: Variable-Sweep Scoped Binding

Detect when source text/table describes several formulation families or axes and bind masses only to rows whose semantic signature matches the value's local axis/scope.

Generic signals:

- multiple mass values for the same material role in one source region;
- row labels or table headers carrying axis tokens;
- polymer/drug/ratio/family labels that distinguish subgroups;
- source spans like `drug/polymer ratio`, `initial drug amount`, `loading study`, `formulation variable`.

Runtime behavior:

- if one drug mass is unique inside a row-local/table-row scope -> promote;
- if multiple drug masses exist in the same paper/method scope without row discriminator -> reject as `ambiguous_variable_sweep_mass`;
- if row discriminator matches a local source value -> promote only to that admitted row;
- otherwise audit, do not fill.

### Capability B: Method-Paragraph Mass-Pair Carrythrough

Recognize preparation sentences that bind at least one drug mass and one polymer mass as a shared formulation input pair, then carry each direct value only to admitted rows inside the same method/formulation family.

Generic signals:

- preparation-positive verbs: `prepared`, `dissolved`, `weighed`, `mixed`, `emulsified`, `added`, `organic phase`, etc.;
- entity-bound pair: `polymer role + mass` and `drug role + mass` in the same sentence or adjacent method sentence;
- paper-local alias graph can resolve drug full name/abbreviation from row hints, abbreviation definitions, or formulation labels.

Runtime behavior:

- promote `drug + mass` to `drug_feed_amount_text` only when row has compatible drug-loaded/admitted semantics;
- promote `polymer + mass` to `plga_mass_mg` only when polymer identity is compatible;
- reject if same scope has conflicting masses without row/family discriminator.

### Capability C: Stronger Negative-Context and Measurement Rejection

Prevent LC/DL, release, dose, PK, assay, sample-prep, and characterization masses from being materialized as input `drug_mass_mg`.

Generic signals:

- context words: `release`, `loading capacity`, `encapsulation`, `dose`, `mg/kg`, `AUC`, `Cmax`, `plasma`, `cell`, `uptake`, `cytotoxicity`, `sample preparation`, `HPLC injection`, `DSC sample`, `lyophilized sample`;
- value family is downstream measurement rather than preparation input;
- unit or nearby header suggests concentration/dose/result rather than input mass.

Runtime behavior:

- reject such candidates with reason `negative_context_non_input_mass` or a more specific subtype;
- route to audit sidecar/review queue, not final direct field.

### Capability D: Mismatch-Aware Direct Mass Authority Guard

Prevent already-populated `drug_feed_amount_text` values from retaining likely non-input numeric masses when evidence context indicates they are measurement/result masses.

Generic signals:

- field is populated from table or text but value source context is LC/DL/result/measurement rather than formulation input;
- row has result fields whose numeric values match the supposed drug input mass;
- source region type is characterization/result table rather than preparation method or formulation composition table.

Runtime behavior:

- do not replace with GT;
- either blank invalid direct mass with an auditable missing/rejected reason, or keep value but mark risk depending on existing Stage5 policy;
- prefer no value over wrong value in direct compare surfaces when evidence is clearly non-input.

### Capability E: Alignment-Gated Handling

Residual blocked alignment rows should remain outside value materialization until row identity/alignment is repaired.

Runtime behavior:

- add diagnostic audit classification only;
- do not implement value fill for `no_unique_alignment` cases;
- once alignment is repaired, generic value materializers can run normally.

---

## Progress Ledger

Create or append:

```text
docs/plans/2026-05-05-drug-mass-residual-generic-capability-progress.tsv
```

Columns:

```text
task_id	capability	status	tests_run	commit	diagnostic_run	notes	updated_at
```

Rules for unattended execution:

- execute exactly the next pending task;
- run targeted tests before and after implementation;
- commit only coherent passing changes;
- if tests fail after one focused generic fix attempt, write blocker and stop;
- never broaden scope to paper-specific repair to make a diagnostic row pass.

---

## Execution Sequencing Principle

This plan is intended to repair **all** residual generic capability defects, not to choose one article or one field case over another. “Priority” means only safe implementation order:

1. establish validators/rejection guards before adding broader carrythrough, so expansion cannot silently create wrong direct values;
2. implement scope primitives before using them for method/table/family inheritance;
3. implement carrythrough only after conflict/ambiguity handling exists;
4. run bounded diagnostic replays after each capability cluster to catch regressions early;
5. keep alignment-gated cases classified separately until an alignment repair exists.

Execution order must be based on dependency and regression risk, not on paper keys or the number of cells in any one paper.

---

## Task 1: Add Residual Audit Fixture Tests for Variable-Sweep Ambiguity

**Objective:** Prove that multiple drug masses in one preparation scope are rejected unless row/family discriminator is available.

**Files:**

- Modify: `tests/test_material_value_binding_v1.py`
- Later modify: `src/stage5_benchmark/material_value_binding_v1.py`

**Step 1: Add failing tests**

Add synthetic tests using generic labels, not DEV15 paper keys:

```python
def test_variable_sweep_multiple_drug_masses_without_row_discriminator_is_rejected(self):
    graph = build_material_alias_graph(
        "The drug candidate API (API) and PLGA were used to prepare nanoparticles."
    )
    candidates = extract_entity_bound_values(
        "Nanoparticles were prepared with API amounts of 2.5 mg, 5 mg, and 10 mg while PLGA was varied.",
        graph,
    )
    review = evaluate_canonical_promotions(candidates, [{"final_formulation_id": "row-1", "drug_mass_mg": ""}])
    self.assertEqual(review["proposals"], [])
    self.assertIn(
        "ambiguous_variable_sweep_mass",
        {row["rejection_reason"] for row in review["rejections"]},
    )
```

**Step 2: Run failing test**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1.MaterialValueBindingTests.test_variable_sweep_multiple_drug_masses_without_row_discriminator_is_rejected
```

Expected: FAIL until ambiguity detection exists.

---

## Task 2: Implement Generic Variable-Sweep Conflict Classification

**Objective:** Reject shared promotion when multiple values for the same role+field appear in the same scope without row/family discriminator.

**Files:**

- Modify: `src/stage5_benchmark/material_value_binding_v1.py`
- Test: `tests/test_material_value_binding_v1.py`

**Implementation guidance:**

- Extend `evaluate_canonical_promotions()` or its grouping helper.
- Group candidates by:

```text
canonical_field + scope_type + scope_key/family_key if present
```

- If a group has multiple normalized values and no row-local target / family discriminator:

```python
rejection_reason = "ambiguous_variable_sweep_mass"
```

- Do not promote any candidate from that ambiguous group.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: OK.

---

## Task 3: Add Row/Family Discriminator Tests for Scoped Variable Binding

**Objective:** Allow variable-sweep masses only when a row-specific or family-specific discriminator ties the value to one admitted row.

**Files:**

- Modify: `tests/test_material_value_binding_v1.py`
- Later modify: `src/stage5_benchmark/material_value_binding_v1.py`

**Test pattern:**

```python
def test_variable_sweep_mass_promotes_only_when_row_family_matches(self):
    candidates = [
        {
            "material_alias": "API",
            "entity_role": "drug",
            "value_type": "mass",
            "normalized_value": "5",
            "normalized_unit": "mg",
            "source_provenance": "direct_text",
            "scope_type": "row_family",
            "scope_key": "family-a",
        }
    ]
    admitted = [
        {"final_formulation_id": "row-a", "drug_mass_mg": "", "scope_key": "family-a"},
        {"final_formulation_id": "row-b", "drug_mass_mg": "", "scope_key": "family-b"},
    ]
    review = evaluate_canonical_promotions(candidates, admitted)
    self.assertEqual([p["final_formulation_id"] for p in review["proposals"]], ["row-a"])
```

**Verification:** targeted unittest expected FAIL before implementation.

---

## Task 4: Implement Generic Row/Family Scope Matching

**Objective:** Promote row/family-scoped candidates only to admitted rows with matching scope markers.

**Files:**

- Modify: `src/stage5_benchmark/material_value_binding_v1.py`
- Optional adapter modify: `src/stage5_benchmark/build_minimal_final_output_v1.py` if row scope hints need to be passed from final rows.

**Implementation guidance:**

- Add conservative `scope_key` handling in promotion evaluation.
- Do not infer broad paper-global scope from family labels.
- Accept only exact normalized match between candidate scope key and row scope key.
- If candidate has `scope_type=row_family` but admitted row lacks matching scope key, reject as:

```text
row_family_scope_not_matched
```

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1
```

Expected: OK.

---

## Task 5: Add Method-Paragraph Mass-Pair Carrythrough Tests

**Objective:** Validate generic extraction of drug/polymer mass pairs from preparation method text.

**Files:**

- Modify: `tests/test_material_value_binding_v1.py`

**Test pattern:**

```python
def test_method_paragraph_drug_polymer_mass_pair_promotes_to_admitted_loaded_rows(self):
    graph = build_material_alias_graph(
        "The drug candidate API (API) was encapsulated in PLGA nanoparticles.",
        row_hints=[{"drug_name_value": "API", "polymer_name_value": "PLGA"}],
    )
    candidates = extract_entity_bound_values(
        "For nanoparticle preparation, PLGA (90 mg) and API (5 mg) were dissolved in organic solvent.",
        graph,
    )
    for candidate in candidates:
        candidate.update(infer_value_scope(candidate, admitted_rows=[{"final_formulation_id": "row-1"}]))
    review = evaluate_canonical_promotions(candidates, [{"final_formulation_id": "row-1", "drug_mass_mg": "", "polymer_mass_mg": ""}])
    proposed = {(p["canonical_field"], p["normalized_value"]) for p in review["proposals"]}
    self.assertIn(("drug_mass_mg", "5"), proposed)
    self.assertIn(("polymer_mass_mg", "90"), proposed)
```

**Verification:** targeted unittest may already pass partially; keep it as regression guard.

---

## Task 6: Strengthen Method-Pair Extraction Without Static Drug-Name Dependence

**Objective:** Reduce dependence on `_DRUG_CUES` static named-drug list by using row hints and abbreviation definitions as primary authority.

**Files:**

- Modify: `src/stage5_benchmark/material_value_binding_v1.py`
- Test: `tests/test_material_value_binding_v1.py`

**Implementation requirements:**

- Ensure `build_material_alias_graph()` prioritizes:
  1. `drug_name_value` / `drug_name` row hints;
  2. full-name abbreviation definitions near drug role context;
  3. formulation labels only when loaded/drug context is present;
  4. static cues only as weak fallback.
- Static `_DRUG_CUES` must not be necessary for a row-hint drug to resolve as `drug`.

**Add test:**

```python
def test_row_hint_unknown_drug_name_resolves_without_static_drug_cue(self):
    graph = build_material_alias_graph("", row_hints=[{"drug_name_value": "NovelPayloadX"}])
    self.assertEqual(graph.resolve_role("NovelPayloadX"), "drug")
```

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

---

## Task 7: Add Negative-Context Tests for Non-Input Drug Masses

**Objective:** Prevent assay/release/dose/measurement values from becoming `drug_mass_mg`.

**Files:**

- Modify: `tests/test_material_value_binding_v1.py`

**Test pattern:**

```python
def test_release_assay_and_dose_masses_are_rejected_for_direct_drug_input(self):
    graph = build_material_alias_graph("NovelPayloadX (NPX) was the drug.", row_hints=[{"drug_name_value": "NPX"}])
    text = (
        "For nanoparticle preparation, NPX (5 mg) and PLGA (50 mg) were dissolved. "
        "For release assay, 10 mg of NPX-loaded nanoparticles were placed in buffer. "
        "Rats received NPX at 20 mg/kg. "
        "HPLC sample preparation used 2 mg standard."
    )
    candidates = extract_entity_bound_values(text, graph)
    spans = " ".join(c["source_span"] for c in candidates)
    self.assertIn("NPX (5 mg)", spans)
    self.assertNotIn("20 mg/kg", spans)
    self.assertNotIn("2 mg standard", spans)
    self.assertNotIn("10 mg of NPX-loaded nanoparticles", spans)
```

**Verification:** targeted unittest expected FAIL if any negative context leaks.

---

## Task 8: Implement Negative-Context Classifier with Audit Reasons

**Objective:** Reject non-formulation input mass contexts with explicit, testable reasons.

**Files:**

- Modify: `src/stage5_benchmark/material_value_binding_v1.py`

**Implementation guidance:**

- Extend `_NEGATIVE_CONTEXT_CUES` conservatively.
- Add structured rejection helper if needed:

```python
classify_value_context(sentence) -> {
    "context_class": "preparation_input" | "release_assay" | "dose" | "measurement" | "sample_prep" | "ambiguous",
    "materialization_allowed": bool,
    "reason": str,
}
```

- Do not simply add more strings without tests.
- Keep preparation-positive and negative-context checks independent; negative wins.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

---

## Task 9: Add Mismatch Guard Tests for Result/Measurement Table Values

**Objective:** Ensure direct mass fields reject populated values when their evidence/header/source context indicates result metrics, not input mass.

**Files:**

- Modify: `tests/test_material_value_binding_v1.py`
- Possibly modify: `tests/test_compare_layer3_values_v1.py` if the guard lives in Stage5 final-table adapter.

**Test pattern:**

```python
def test_result_table_numeric_mass_is_not_valid_direct_drug_input_mass(self):
    result = validate_direct_value("1.8 mg", "mass", context="loading capacity result table; released drug amount")
    self.assertEqual(result["status"], "invalid")
    self.assertEqual(result["reason"], "invalid_mass_non_input_context")
```

If `validate_direct_value()` should remain context-free, introduce a separate helper:

```python
validate_direct_value_context(expression, value_type, context)
```

**Verification:** targeted unittest expected FAIL before implementation.

---

## Task 10: Implement Context-Aware Direct Mass Validation

**Objective:** Add a generic guard that can blank/reject direct mass candidates from result or measurement contexts before they enter final direct fields.

**Files:**

- Modify: `src/stage5_benchmark/material_value_binding_v1.py`
- Modify adapter only if necessary: `src/stage5_benchmark/build_minimal_final_output_v1.py`

**Implementation guidance:**

- Keep existing context-free `validate_direct_value()` behavior if widely used.
- Add context-aware wrapper instead of changing all callers unexpectedly:

```python
def validate_direct_value_with_context(expression: str, value_type: str, context: str = "") -> dict[str, str]:
    base = validate_direct_value(expression, value_type)
    if base["status"] != "valid":
        return base
    context_class = classify_value_context(context)
    if value_type == "mass" and not context_class["materialization_allowed"]:
        return {"status": "invalid", "reason": "invalid_mass_non_input_context", ...}
    return base
```

- Stage5 adapter should use this only when source/header/context is available; otherwise preserve current behavior.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1
```

---

## Task 11: Add Stage5 Adapter Tests for “Reject Wrong Value Rather Than Fill” Policy

**Objective:** For direct compare fields, prove that suspicious non-input mass values do not remain as authoritative `drug_feed_amount_text` when context proves they are measurement outputs.

**Files:**

- Modify: `tests/test_compare_layer3_values_v1.py` or create a focused `tests/test_stage5_direct_mass_authority_v1.py`
- Modify later: `src/stage5_benchmark/build_minimal_final_output_v1.py`

**Test requirements:**

- Build a synthetic final-row/materialization input with:
  - `drug_feed_amount_text_value_text = "1.8 mg"`
  - evidence/header/context containing `loading capacity`, `release`, or result-table cues;
  - no valid row-local preparation input mass.
- Expected behavior:
  - either field blanked with `drug_feed_amount_text_missing_reason=invalid_mass_non_input_context`, or
  - field retained only with explicit risk flag and excluded from direct compare if that policy already exists.

Prefer blanking invalid direct fields for current direct compare unless governance says otherwise.

---

## Task 12: Implement Stage5 Direct Mass Authority Guard

**Objective:** Apply context-aware mass validation to Stage5 `drug_feed_amount_text` materialization paths.

**Files:**

- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py`

**Implementation guidance:**

- Identify all locations that set or preserve `drug_feed_amount_text_*`.
- Apply guard only to direct mass fields:

```text
drug_feed_amount_text
plga_mass_mg
surfactant_mass_mg
```

- Do not affect drug identity fields.
- Do not overwrite row-local valid direct masses.
- If invalid due to context:
  - blank the field or prevent promotion;
  - set missing/rejection reason if field schema supports it;
  - preserve source evidence in audit sidecar/decision trace if available.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1 tests.test_material_value_binding_v1
```

---

## Task 13: Add Alignment-Gated Residual Audit Helper

**Objective:** Classify remaining `drug_mass_mg` residuals without filling blocked-alignment cases.

**Files:**

- Create only if needed and user accepts diagnostic script creation: `src/stage5_benchmark/audit_drug_mass_residual_capabilities_v1.py`
- Otherwise implement as internal analysis in tests/scripts and record in `RUN_CONTEXT.md` of validation run.

**Inputs:**

- compare cells TSV from `050` or newer explicit compare;
- final table TSV;
- optional alignment resolution TSV.

**Output columns:**

```text
paper_key
gt_formulation_id
matched_system_formulation_id
compare_status
residual_capability_class
recommended_next_layer
must_not_fill_reason
```

Classes:

```text
variable_sweep_scope_needed
method_pair_carrythrough_needed
negative_context_or_measurement_rejection_needed
alignment_blocked_no_value_action
present_but_mismatch_numeric_misbinding
```

**Verification:** run helper on `050`; it must classify blocked cases as `alignment_blocked_no_value_action`, not materialization targets.

---

## Task 14: Bounded Diagnostic Replay After Each Capability Cluster

**Objective:** Validate improvements without conflating them with benchmark-valid performance.

**Files:**

- New run directories under `data/results/20260504_ab9f61e/<NN>_stage5_<cue>_diagnostic`
- New compare dirs under `data/results/20260504_ab9f61e/<NN>_layer3_compare_<cue>_diagnostic`

**Command pattern:**

```bash
PYTHONPATH=. python3 src/stage5_benchmark/build_minimal_final_output_v1.py \
  --input-tsv data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv \
  --relation-records-tsv data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv \
  --resolved-relation-fields-tsv data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/resolved_relation_fields_v1.tsv \
  --out-dir data/results/20260504_ab9f61e/<NN>_stage5_drug_mass_residual_capability_<cue>_diagnostic
```

Then run the maintained Layer3 compare entrypoint registered in `docs/maintained_script_surface.tsv` with explicit final-table and locked GT paths.

**Validation criteria:**

- `drug_mass_mg present_and_match` increases or `present_but_mismatch` decreases;
- `drug_mass_mg extra_in_system` remains `0`;
- total core `extra_in_system` does not increase;
- `present_but_mismatch` must not increase for drug/polymer mass fields;
- blocked alignment counts may remain unchanged and must not be described as value-materialization failure.

---

## Task 15: Update Decision Trace / Audit Provenance

**Objective:** Make every new carrythrough/rejection explainable.

**Files:**

- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Existing outputs: `final_output_decision_trace_v1.tsv`, `final_output_summary_v1.md`

**Requirements:**

Add trace values such as:

```text
material_value_binding_method_pair_direct_mass
material_value_binding_variable_sweep_rejected
material_value_binding_negative_context_rejected
material_value_binding_row_family_scope_matched
material_value_binding_alignment_gated_no_action
```

Each trace should include enough source/provenance detail to audit why a value was filled or rejected.

---

## Task 16: Regression Test Bundle and Commit Gate

**Objective:** Prevent future generic materialization regressions.

**Commands:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1
PYTHONPATH=. python3 -m unittest tests.test_table_structure_dictionary_v1
```

Expected: all pass.

If broader tests fail due to unrelated pre-existing issue, record exact failing test and run targeted suites plus a focused compare replay before committing.

---

## Task 17: Final Drug Mass Delta Audit

**Objective:** Quantify final effect after all generic residual capabilities.

**Compute and report:**

```text
drug_mass_mg 050 -> final:
  present_and_match
  present_but_mismatch
  missing_in_system
  blocked_alignment
  extra_in_system
  value_recall
  correct-value recall
  conditional accuracy relaxed/canonicalized
```

Also report transitions:

```text
missing_in_system -> present_and_match
present_but_mismatch -> present_and_match
present_but_mismatch -> missing/rejected_invalid_context
missing_in_system -> missing_in_system
blocked_alignment -> blocked_alignment
new extra_in_system
```

**Interpretation rules:**

- Improvements from missing/mismatch to match are valid diagnostic deltas.
- Mismatch to blank/rejected can be a correctness improvement if it removes a false direct value, but it should not be counted as recall improvement.
- Blocked alignment unchanged is acceptable if classified correctly.
- Any extra increase is a blocker unless explained and accepted.

---

## Task 18: Governance/Memory Update After Validation

**Objective:** Record only durable generic lessons after implementation, not paper-specific counts.

**Files:**

- Append/update only if implementation succeeds:
  - `project/4_DECISIONS_LOG.md`
  - `data/mem/v1/dec.tsv`
  - relevant Hermes skill if the workflow changed

**Do not store:**

- individual paper rows;
- run-specific counts in Hermes always-on memory;
- one-off excerpts.

**Do store:**

- generic capability contracts;
- validation/rejection principles;
- authority and scope rules.

Validate governed memory:

```bash
PYTHONPATH=. python3 src/utils/check_mem_v1.py
```

---

## Expected Outcome Bands

This plan should not promise all `drug_mass_mg` cells become matches, because some residuals are alignment-gated or may lack source-authorized direct values.

Expected diagnostic effects if capabilities work:

```text
Likely positive:
- reduce method-pair missing cells if source scope is unique;
- reduce variable-sweep missing cells only where row/family discriminator is explicit;
- reduce numeric mismatches when wrong measurement masses can be rejected or corrected from source-backed input mass.

Likely unchanged until separate work:
- blocked_alignment cells;
- evidence-selection cases where the matched row is not a formulation-input row;
- multi-mass source regions without a safe preparation-scope discriminator.
```

Success is not “fill every blank.” Success is:

```text
more correct direct masses, fewer wrong direct masses, zero new extras, and explicit audit reasons for cases that remain unfilled.
```

---

## Execution Handoff

Plan is ready for implementation with subagent-driven-development or unattended scheduled execution. Preferred execution mode:

1. create/update progress ledger;
2. execute one task per run;
3. run targeted tests;
4. commit passing coherent changes;
5. perform bounded diagnostic replay after each capability cluster;
6. stop on ambiguity rather than paper-patching.
