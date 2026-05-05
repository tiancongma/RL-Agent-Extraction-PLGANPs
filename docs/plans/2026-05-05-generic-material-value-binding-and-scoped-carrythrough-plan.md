# Generic Material-Value Binding and Scoped Carrythrough Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Keep all implementation generic; concrete papers are validation cases only.

**Goal:** Add a generic S3/S5-2 capability for source-backed material-value binding and scoped carrythrough so drug/polymer/surfactant preparation values propagate safely without paper-specific patches or S5-3 database completion.

**Architecture:** Build a paper-local material alias graph, bind mass/concentration/ratio expressions to material entities, infer value scope, and promote entity-bound direct values into canonical fields only for admitted rows inside the authorized scope. Stage2 remains semantic authority for formulation meaning and row membership; deterministic S3/S5-2 performs relation resolution, materialization, carrythrough, and validation only after lawful Stage2 handoff.

**Tech Stack:** Python standard library, existing Stage2/Stage3/Stage5 TSV/JSON artifacts, `unittest` project test suite, maintained entrypoints from `project/ACTIVE_PIPELINE_RUNBOOK.md`.

---

## Non-Negotiable Scope Rules

1. **No paper-specific runtime branches.** Do not implement `if paper_key == ...`, hard-code AP/MB/GAR/etoposide/dexibuprofen, or create one-off value maps for DEV15 papers.
2. **No field-only repair framing.** `drug_mass_mg` is a validation symptom, not the feature. The feature is generic material-value binding and scoped carrythrough.
3. **No semantic-authority drift.** Deterministic code may normalize, bind, validate, and carry values only after Stage2 has authorized the formulation row/candidate scope. It must not define candidate membership or row semantics.
4. **No blank-slot completion.** S5-3 remains residual source-evidenced gap fill. S5-2/S3 generic repairs should reduce simple/direct residuals before S5-3 is invoked.
5. **No latest-by-sort input selection.** All diagnostic replays must use explicit paths or `data/results/ACTIVE_RUN.json` per `project/ACTIVE_DATA_SOURCE_CONTRACT.md`.
6. **Direct vs derived separation.** This plan covers direct source-backed values only. Derived calculations stay in S5-5 sidecars and must not contaminate direct Layer3 compare.

---

## Current Diagnostic Anchors

Use these as fixed diagnostic validation artifacts until a newer explicit authority is selected:

- Latest explicit no-S5-3 diagnostic final table:
  - `data/results/20260504_ab9f61e/031_stage5_current_s5_2_no_s5_3_baseline_doe_explicit_only_diagnostic/final_formulation_table_v1.tsv`
- Latest explicit Layer3 compare:
  - `data/results/20260504_ab9f61e/032_layer3_compare_current_s5_2_no_s5_3_baseline_doe_explicit_only_diagnostic/layer3_value_compare_cells_v1.tsv`
- Completed Stage2 handoff used by that lineage:
  - `data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 relation artifacts used by that lineage:
  - `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv`
  - `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/resolved_relation_fields_v1.tsv`

These are diagnostic-only and not benchmark-valid final output.

---

## Capability Model

### New conceptual objects

The implementation may be file-local helpers first; create separate modules only if reuse becomes clear.

1. **MaterialAliasGraph**
   - Paper-local map from textual names, abbreviations, formulation-label tokens, and table labels to material entities.
   - Entity roles: `drug`, `polymer`, `stabilizer`, `surfactant`, `helper_material`, `solvent`, `unknown`.
   - Sources: Stage2 row fields, formulation labels, abbreviation definitions in source text, table headers/footnotes, relation artifacts.

2. **EntityBoundValue**
   - A direct source expression linked to a material entity.
   - Examples:
     - `5 mg of dexibuprofen` -> entity role `drug`, field family `mass`
     - `PLGA (90 mg)` -> role `polymer`, field family `mass`
     - `1% w/v PVA` -> role `stabilizer`, field family `concentration`
   - Must carry source span/context and scope evidence.

3. **ValueScope**
   - `row_local_table_cell`
   - `typed_row_local_assignment`
   - `table_footnote_shared`
   - `table_scoped_constant`
   - `method_paragraph_shared`
   - `paper_global_unique_constant`
   - `ambiguous_or_conflicted`

4. **PromotionTarget**
   - Map entity role + value type to canonical fields.
   - Examples:
     - `drug + mass` -> `drug_feed_amount_text_*` / compare field `drug_mass_mg`
     - `polymer + mass` -> `plga_mass_mg_*` / compare field `polymer_mass_mg`
     - `stabilizer/surfactant + concentration` -> stabilizer concentration value/unit fields

---

## Task 1: Add field-type validation tests for amount fields

**Objective:** Prevent identity text from entering numeric/mass fields before adding new carrythrough.

**Files:**
- Modify: `tests/test_compare_layer3_values_v1.py` or create a focused Stage5 test if an existing nearby test file is more appropriate.
- Modify later: `src/stage5_benchmark/build_minimal_final_output_v1.py` or the Stage3 relation builder if validation belongs earlier.

**Step 1: Write failing tests**

Add tests asserting:

```python
def test_drug_feed_amount_rejects_identity_only_text():
    assert not is_valid_direct_mass_text("acetylpuerarin")
    assert not is_valid_direct_mass_text("AP")
    assert is_valid_direct_mass_text("7 mg")
    assert is_valid_direct_mass_text("7 mg AP")
```

Use the actual helper name chosen during implementation; if no helper exists, create it in Task 2.

**Step 2: Run targeted test**

```bash
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1
```

Expected: fails because helper/guard does not exist or does not reject identity-only text.

---

## Task 2: Implement generic direct mass validator

**Objective:** Add a reusable validator that distinguishes mass expressions from material identities.

**Files:**
- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Optional if cleaner: create `src/stage5_benchmark/material_value_binding_v1.py`

**Implementation requirements:**

- Accept values with numeric + mass unit, e.g. `0.5 mg`, `5 mg`, `7 mg`, `100 µg`, `0.2 g`.
- Reject values without numeric mass units, e.g. drug names, abbreviations, loaded labels.
- Reject concentration-only values for mass fields unless a later derived sidecar explicitly calculates mass.
- Preserve source text in audit fields where available; do not silently coerce non-mass values.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1
```

Expected: new tests pass; existing compare tests continue passing.

---

## Task 3: Add paper-local material alias graph tests

**Objective:** Define generic abbreviation/full-name/entity-role behavior without paper-specific constants.

**Files:**
- Create or modify test file: `tests/test_material_value_binding_v1.py`
- Implementation later: `src/stage5_benchmark/material_value_binding_v1.py` or equivalent helper surface.

**Step 1: Test abbreviation definition extraction**

Use synthetic examples, not DEV15 paper-specific fixtures:

```python
def test_alias_graph_links_full_name_and_parenthetical_abbreviation():
    text = "Acetylpuerarin (AP)-loaded PLGA nanoparticles were prepared. 7 mg of AP was dissolved."
    graph = build_material_alias_graph(source_text=text, row_hints={"drug_name": "acetylpuerarin"})
    entity = graph.resolve("AP")
    assert entity.role == "drug"
    assert "acetylpuerarin" in entity.aliases
```

**Step 2: Test formulation-label alias extraction**

```python
def test_alias_graph_uses_loaded_formulation_label_tokens():
    row_hints = {"drug_name": "methylene blue", "formulation_identity_label": "MB-loaded PLGA NPs"}
    graph = build_material_alias_graph(source_text="", row_hints=row_hints)
    assert graph.resolve("MB").role == "drug"
```

**Step 3: Run tests and expect failure**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: fails until alias graph exists.

---

## Task 4: Implement minimal MaterialAliasGraph

**Objective:** Build paper-local alias/entity resolution from generic evidence cues.

**Files:**
- Create: `src/stage5_benchmark/material_value_binding_v1.py`
- Test: `tests/test_material_value_binding_v1.py`

**Implementation requirements:**

- Parse full-name abbreviation patterns:
  - `<full name> (<ABBR>)`
  - `<ABBR> (<full name>)` only when supported by row hints or repeated context.
- Use row hints from existing surfaces:
  - `drug_name_value`, `polymer_name_value`, `emulsifier_stabilizer_name_value`, `formulation_identity_label`, table row labels.
- Assign roles conservatively:
  - drug role requires drug-name hint, loaded-label evidence, or explicit semantic row field.
  - polymer role recognizes PLGA/PCL/PLA-like polymer hints from existing canonical fields, not arbitrary uppercase tokens.
  - unknown role must stay unknown if role evidence is weak.
- Return confidence/reason strings for auditability.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: alias tests pass.

---

## Task 5: Add entity-value binding tests

**Objective:** Bind mass/concentration expressions to material entities instead of treating any nearby number as a target value.

**Files:**
- Modify: `tests/test_material_value_binding_v1.py`

**Test cases:**

```python
def test_binds_mass_to_drug_alias_in_preparation_sentence():
    graph = build_material_alias_graph(
        source_text="Acetylpuerarin (AP)-loaded PLGA nanoparticles were prepared.",
        row_hints={"drug_name": "acetylpuerarin"},
    )
    values = extract_entity_bound_values("18 mg of PLGA and 7 mg of AP were dissolved in acetone.", graph)
    assert any(v.role == "drug" and v.value_text == "7 mg" for v in values)
    assert any(v.role == "polymer" and v.value_text == "18 mg" for v in values)
```

```python
def test_does_not_bind_animal_dose_as_formulation_mass():
    graph = build_material_alias_graph(source_text="Drug (DG) nanoparticles were prepared.", row_hints={"drug_name": "Drug"})
    values = extract_entity_bound_values("Mice received 25 mg/kg DG by injection.", graph)
    assert not any(v.role == "drug" and v.value_type == "mass" for v in values)
```

**Run:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: fails until binder exists.

---

## Task 6: Implement entity-bound value extraction with negative-context guards

**Objective:** Extract source-backed values only from preparation-relevant contexts.

**Files:**
- Modify: `src/stage5_benchmark/material_value_binding_v1.py`

**Implementation requirements:**

- Recognize direct mass forms:
  - `<mass> of <material>`
  - `<material> (<mass>)`
  - `<material> amount = <mass>`
  - `<material> amount (<mass-like column>)`
- Recognize preparation-positive contexts:
  - prepared, dissolved, weighed, organic phase, aqueous phase, formulation, nanoprecipitation, emulsion, solvent evaporation.
- Reject negative/downstream contexts:
  - release sample, animal dose, mg/kg, assay, calibration, DSC/TGA sample, pharmacokinetic dose, cell treatment, biodistribution, injection dose.
- Attach `value_scope_candidate` but do not yet inherit to rows.
- Preserve raw context snippet and match reason.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: all entity-binding tests pass.

---

## Task 7: Add scope inference tests

**Objective:** Classify whether an entity-bound value can be carried to rows.

**Files:**
- Modify: `tests/test_material_value_binding_v1.py`

**Test cases:**

1. `table_footnote_shared`: text says amounts were always maintained.
2. `method_paragraph_shared`: one preparation sentence contains drug/polymer mass pair.
3. `ambiguous_or_conflicted`: multiple candidate drug masses in different preparation contexts with no row axis signal.
4. `row_local_table_cell`: row-local direct table cell should outrank shared values.

**Run:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: fails until scope inference exists.

---

## Task 8: Implement conservative value scope inference

**Objective:** Decide when a value is safe to carry to admitted rows.

**Files:**
- Modify: `src/stage5_benchmark/material_value_binding_v1.py`

**Implementation requirements:**

- Authority ladder:
  1. row-local direct table cell
  2. typed row-local assignment / identity variable
  3. table-footnote or table-scoped constant
  4. method-paragraph shared constant
  5. paper-global unique direct constant
  6. ambiguous/conflicted -> no promotion
- If multiple candidate drug masses exist, require row-local axis evidence or same-scope disambiguation; otherwise mark conflicted.
- Footnote/shared constant must carry scope text and cannot apply to blank/control/helper rows.
- Method-paragraph mass-pair carrythrough should require co-occurring drug and polymer/preparation context.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: scope tests pass.

---

## Task 9: Add canonical promotion tests

**Objective:** Promote entity-bound values to canonical final-table fields only when role, type, and scope are valid.

**Files:**
- Modify: `tests/test_material_value_binding_v1.py`
- Modify later: `src/stage5_benchmark/build_minimal_final_output_v1.py`

**Test cases:**

- drug + mass -> `drug_feed_amount_text_value` and `_text`.
- polymer + mass -> existing polymer mass canonical fields.
- stabilizer + concentration -> stabilizer/emulsifier concentration fields.
- unknown + mass -> no canonical promotion.
- drug identity text -> no amount promotion.

**Run:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: fails until promotion helper exists.

---

## Task 10: Implement canonical promotion helper

**Objective:** Convert validated `EntityBoundValue` objects into canonical field updates.

**Files:**
- Modify: `src/stage5_benchmark/material_value_binding_v1.py`

**Implementation requirements:**

- Return proposed updates, not mutate rows directly:

```python
{
  "field": "drug_feed_amount_text",
  "value_text": "7 mg",
  "value_num": "7",
  "unit": "mg",
  "source": "method_paragraph_shared",
  "evidence": "18 mg of PLGA and 7 mg of AP were dissolved...",
  "confidence": "high",
}
```

- Include rejection reasons for invalid/conflicted candidates.
- Never overwrite a higher-authority row-local value.
- Never promote unknown-role material values into drug/polymer/stabilizer fields.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1
```

Expected: promotion tests pass.

---

## Task 11: Integrate typed row-local identity-variable promotion into S5-2

**Objective:** Promote existing row-local identity variables such as generic `<drug_alias> amount = 5 mg` without hard-coded drug names.

**Files:**
- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Tests: existing or new Stage5 final-output test file.

**Implementation requirements:**

- Read row-local `identity_variables_json` / `change_descriptions` if present.
- Use `MaterialAliasGraph` to decide whether the variable name is bound to a drug entity.
- Promote only mass values to `drug_feed_amount_text_*`.
- Reject polymer/surfactant/helper amount variables from drug mass promotion.
- Do not override an existing row-local table-cell value.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_table_structure_dictionary_v1 tests.test_compare_layer3_values_v1 tests.test_material_value_binding_v1
```

Expected: all pass.

---

## Task 12: Integrate table-footnote/shared constant carrythrough into S5-2

**Objective:** Carry source-backed shared constants to admitted rows within safe table/method scope.

**Files:**
- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Optional helper: `src/stage5_benchmark/material_value_binding_v1.py`

**Implementation requirements:**

- Consume Stage2 source/table evidence already available to Stage5, not raw CSV fallback unless an existing lawful authority surface is missing and the workflow is diagnostic.
- Build per-paper alias graph once, then evaluate shared values per row.
- Apply only when `row_allows_shared_preparation_mass_carrythrough(...)` or an equivalent generic eligibility check passes.
- Add decision-trace entries showing:
  - entity alias matched
  - value text
  - value scope
  - evidence snippet/path
  - rejection reason when not carried

**Verification:**

Run unit tests, then bounded diagnostic replay on explicit lineage paths only:

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_table_structure_dictionary_v1 tests.test_compare_layer3_values_v1
```

Expected: tests pass. Bounded replay should be diagnostic-only and compare exact before/after status deltas, not benchmark claims.

---

## Task 13: Add conflict/audit sidecar or decision trace rows

**Objective:** Make non-promotions auditable so future S5-3 gets true residuals, not silent S5-2 failures.

**Files:**
- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Output: existing `final_output_decision_trace_v1.tsv` preferred; create a new sidecar only if the trace schema cannot represent candidate/rejection details.

**Implementation requirements:**

- Record accepted and rejected material-value candidates.
- Rejection reasons:
  - ambiguous multiple masses
  - negative downstream context
  - role unknown
  - row ineligible blank/control/helper
  - lower authority than existing direct value
  - missing Stage2 authorization / illegal boundary
- Keep sidecar diagnostic/audit only; do not make it a benchmark terminal surface.

**Verification:**

Inspect a bounded run decision trace and verify accepted/rejected rows are explainable.

---

## Task 14: Bounded diagnostic validation on symptom set

**Objective:** Use known missing-value cases as validation samples without adding paper-specific code.

**Files:**
- No source changes unless failures reveal generic defects.
- Run outputs under a new governed diagnostic child directory with `RUN_CONTEXT.md`.

**Diagnostic sample categories:**

- abbreviation/full-name alias: full name in row, abbreviation in mass sentence.
- table-footnote constant: value maintained across rows.
- method paragraph shared mass pair: drug/polymer dissolved/weighed together.
- identity-variable promotion: `<drug alias> amount = <mass>` already exists in row-local variable surface.
- negative controls: downstream assay/release/animal-dose masses must not carry.

**Run requirements:**

- Use explicit input paths from current diagnostic lineage.
- Print final table path, Stage2 path, Stage3 path, GT authority path.
- Label outputs `diagnostic-only, benchmark_valid=no`.

**Verification metrics:**

- `drug_mass_mg missing_in_system` should decrease for valid generic cases.
- `present_but_mismatch` and `extra_in_system` must not increase outside expected reviewed cases.
- Non-target fields must remain unchanged except fields intentionally covered by the generic capability.

---

## Task 15: Regression guardrails

**Objective:** Prevent the generic capability from becoming an uncontrolled donor-fill engine.

**Files:**
- Tests: `tests/test_material_value_binding_v1.py`
- Existing compare/table tests.

**Add tests for:**

- two conflicting drug masses in separate method paragraphs -> no paper-global carrythrough.
- blank/control rows do not inherit drug mass.
- helper/model-drug substitution rows do not inherit active-drug mass unless Stage2 authorized the substitution as the benchmark drug entity.
- concentration values do not enter mass fields.
- row-local value beats shared value.
- unknown abbreviation does not become drug role without alias evidence.

**Run:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_table_structure_dictionary_v1 tests.test_compare_layer3_values_v1
```

Expected: all pass.

---

## Task 16: Documentation and governed memory update after implementation

**Objective:** Record the implemented generic capability only after tests and bounded diagnostics pass.

**Files:**
- Append/update: `project/4_DECISIONS_LOG.md`
- Append/update: `data/mem/v1/dec.tsv`
- Append/update: relevant skill/reference if a reusable workflow emerges.

**Requirements:**

- The decision entry must state the generic capability and authority guards.
- Do not record paper-specific values as project policy.
- If implementation creates a new maintained helper script, update `docs/maintained_script_surface.tsv` only after it has a stable contract and user approval.
- If only helper functions are added inside an existing active entrypoint, no maintained-script registry update is needed.

---

## Acceptance Criteria

A future implementation of this plan is acceptable only if all of the following hold:

1. No paper-specific runtime branches or hard-coded DEV15 drug dictionaries.
2. Paper-local alias graph can map full names, abbreviations, and formulation label tokens to material entities when source evidence supports the link.
3. Entity-bound values are extracted with preparation-positive and downstream-negative context guards.
4. Shared values carry only inside row/table/method/footnote scopes with admitted-row eligibility checks.
5. Canonical promotion writes direct values only when role, value type, and scope are valid.
6. Type validation prevents identities from entering mass/concentration fields.
7. Existing unit tests pass.
8. Bounded diagnostic replay records exact lineage and remains diagnostic-only.
9. S5-3 target scope shrinks only because S5-2/S3 generic direct-value capability improved, not because blank slots were filled indiscriminately.

---

## Suggested Implementation Order

1. Validator first: stop field contamination.
2. Alias graph second: establish paper-local material identities.
3. Entity-value binding third: attach values to entities.
4. Scope inference fourth: decide inheritance safety.
5. Canonical promotion fifth: write final fields through proposals.
6. S5-2 integration sixth: mutate final rows only after all guards pass.
7. Diagnostic replay last: compare exact lineage deltas and inspect regressions.

---

## What Not To Do

- Do not implement a DEV15 lookup table of drug abbreviations.
- Do not treat any uppercase token near `mg` as a drug.
- Do not use whole-paper unique mass carrythrough without scope and negative-context guards.
- Do not overwrite row-local direct table values with shared method values.
- Do not let S5-3 fill values that deterministic S5-2 should materialize from direct table/method/footnote evidence.
- Do not report diagnostic compare improvements as benchmark-valid final performance.
