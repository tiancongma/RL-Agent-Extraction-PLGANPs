# Decoded Structured-Table Rebinding v1 Implementation Plan

> For Hermes: implement this as a generalized compare repair, not a paper-specific fix.

Goal: Replace the current UFXX9WXE-style paper-local compare override with a schema-driven decoded structured-table rebinding mechanism that can be reused across papers with the same failure mode.

Architecture:
- Add a generalized compare-side rebinding layer that operates on already-materialized row evidence (`evidence_span_text` / `supporting_evidence_refs`) and governed schema definitions.
- Keep this bounded: no new semantic discovery, no formulation-universe expansion, no cross-row guessing.
- Preserve current global mechanisms (`ordinal bridge`, field-aware canonicalization) and migrate UFXX9WXE to the new rebinding engine as the first schema.

Tech stack: Python, unittest, existing `compare_layer3_values_to_gt_v1.py` pipeline.

---

## Scope and non-goals

In scope:
- Generalized parsing of structured row evidence with fixed column order.
- Schema-driven mapping from parsed columns to compare fields.
- Schema-driven safe suppressions for known misbound residual fields.
- Regression tests proving behavior on UFXX9WXE and proving guardrails.

Out of scope:
- New LLM calls.
- Stage2 or Stage5 semantic redesign.
- Generic prose mining.
- Cross-paper fuzzy inference.
- Per-paper if/else growth.

---

## Target files

Modify:
- `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- `tests/test_compare_layer3_values_v1.py`

Optional diagnostic note update after implementation:
- `analysis/ufxx9wxe_layer3_triage_table_20260424.tsv` (refresh only if useful; not required for code correctness)

Do not modify:
- `project/`
- GT authority files during this implementation step

---

## Design overview

Introduce three new generalized concepts inside `compare_layer3_values_to_gt_v1.py`:

1. `DecodedStructuredRowSchema`
- Describes a bounded table-row schema.
- Example fields:
  - `name`
  - `eligibility_fn(row) -> bool`
  - `column_extractors` or `column_positions`
  - `field_bindings`
  - `safe_suppressions`

2. `parse_structured_row_evidence(row) -> ParsedRow | None`
- Reads `evidence_span_text` first.
- Falls back to `supporting_evidence_refs[*].span_text`.
- Splits only pipe-delimited structured rows.
- Rejects rows with insufficient columns.
- Returns normalized column list.

3. `apply_decoded_structured_table_rebinding(field_name, row, paper_key) -> override tuple | None`
- Finds matching schema(s) by eligibility.
- Parses row once.
- Rebinds requested compare field if schema defines it.
- Applies only bounded field-level returns.
- Can also return safe blank suppressions for explicitly declared misbound residual surfaces.

Important: this engine is keyed by schema eligibility, not by paper id branching. UFXX9WXE will be the first consumer schema, but the code path must not be named or structured as a one-off paper hack.

---

## Guardrails

The generalized rebinding mechanism must only activate when all are true:
- Row already exists in current system output.
- Evidence source is a structured table row, not arbitrary prose.
- Column count matches or safely exceeds schema minimum.
- Schema only maps fixed columns to fixed compare fields.
- Returned values are row-local or paper-level shared constants already governed by source text.
- No new formulation rows are created.
- No field is filled by cross-row interpolation.

Hard prohibitions:
- No `if paper_key == ...` in the new rebinding engine.
- No direct expected-count logic.
- No schema that depends on the answer for one specific GT row.
- No using this layer to reinterpret ambiguous prose.

---

## Schema v1 to implement

Create first schema class/factory for:
- decoded structured value tables with fixed pipe-delimited row order

v1 column order (validated on UFXX9WXE):
1. row label
2. polymer concentration value
3. surfactant concentration value
4. phase ratio value
5. drug concentration value
6. particle size value
7. ee percent value
8. pdi value

Shared governed constants for this schema class may include, when already explicitly proven in source text for the paper class/run context:
- polymer molecular weight raw
- LA/GA ratio raw/normalized

But keep constants injected by schema config, not by hardcoded paper branch.

v1 safe suppressions supported by schema:
- suppress `drug_mass_mg` when the same row is known to carry drug concentration, not drug mass
- suppress `polymer_grade` when current surface is only a generic identity spillover and GT is blank

Do not add more suppressions in v1.

---

## Implementation tasks

### Task 1: Snapshot current paper-local logic to preserve behavior target

Objective: Record the current UFXX9WXE behavior before refactor.

Files:
- Read: `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- Read: `tests/test_compare_layer3_values_v1.py`

Step 1: Locate current UFXX9WXE-specific helper and branch.

Step 2: Note exact fields currently handled:
- `polymer_concentration_value`
- `polymer_concentration_unit`
- `surfactant_concentration_value`
- `surfactant_concentration_unit`
- `drug_concentration_value`
- `drug_concentration_unit`
- `phase_ratio_raw`
- `particle_size_nm`
- `ee_percent`
- `drug_mass_mg` suppression
- `polymer_grade` suppression
- `polymer_mw_raw`
- `la_ga_ratio_raw`
- `la_ga_ratio_normalized`

Step 3: Keep existing tests green before any refactor.

Run:
- `python3 -m unittest tests/test_compare_layer3_values_v1.py`

Expected:
- PASS

### Task 2: Introduce generic parsed-row helper

Objective: Add a non-paper-specific structured row parser.

Files:
- Modify: `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- Test: `tests/test_compare_layer3_values_v1.py`

Step 1: Add a helper with a generic name, e.g.:
- `_extract_structured_row_span_text(row)`
- `_parse_pipe_delimited_structured_row(row)`

Behavior:
- Prefer `evidence_span_text`
- Fallback to `supporting_evidence_refs[*].span_text`
- Split on `|`
- Normalize whitespace
- Drop empty edge fragments
- Return list of columns or `None`

Step 2: Add failing tests for:
- direct `evidence_span_text`
- fallback from `supporting_evidence_refs`
- reject too-short row

Step 3: Implement minimal parser and rerun tests.

### Task 3: Introduce schema container and UFXX-compatible schema config

Objective: Represent decoded structured tables as schemas rather than paper branches.

Files:
- Modify: `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- Test: `tests/test_compare_layer3_values_v1.py`

Step 1: Add a small schema representation.

Recommended simple shape for v1:
- plain dict or dataclass with:
  - `name`
  - `min_columns`
  - `eligibility_fn`
  - `field_to_column_index`
  - `field_to_formatter`
  - `suppressed_fields`
  - `shared_constant_fields`

Step 2: Implement first schema config for the current decoded structured table class.

Important:
- eligibility may temporarily include a bounded paper-class detector if absolutely needed for safe rollout, but do NOT branch inside the field override function by paper key.
- preferred eligibility signals:
  - pipe-delimited table row with 8 columns
  - row carries `preparation_method`
  - row carries concentration surfaces already known to be legacy fixed-slot remnants
  - row evidence section points to numbered formulation table

Step 3: Add tests proving schema object maps:
- col 2 -> `polymer_concentration_value`
- col 3 -> `surfactant_concentration_value`
- col 4 -> `phase_ratio_raw`
- col 5 -> `drug_concentration_value`
- col 6 -> `particle_size_nm`
- col 7 -> `ee_percent`
- col 8 -> `pdi`

### Task 4: Implement generalized rebinding engine

Objective: Replace direct UFXX-specific field returns with schema-driven rebinding.

Files:
- Modify: `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- Test: `tests/test_compare_layer3_values_v1.py`

Step 1: Add a function like:
- `_decoded_structured_table_override(field_name, row, paper_key="")`

Behavior:
- iterate over registered schemas
- if schema eligible:
  - parse row columns
  - if field is in `suppressed_fields`, return blank override
  - if field is in `shared_constant_fields`, return configured constant
  - if field is in `field_to_column_index`, extract and format value
  - otherwise no override

Step 2: Formatters needed in v1:
- unit fields -> `mg/mL`
- `phase_ratio_raw` -> `<value> w/o phase volume ratio`
- `particle_size_nm` -> uncertainty-stripped numeric
- `ee_percent` -> uncertainty-stripped percent text with `%`
- `pdi` -> uncertainty-stripped numeric or preserved decimal text according to existing compare conventions

Step 3: Add tests covering:
- concentration rebinding
- outcome rebinding
- safe suppressions
- shared constants

### Task 5: Route compare through the generalized engine

Objective: Make `get_system_value()` use the new generalized mechanism before legacy direct-field mapping.

Files:
- Modify: `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- Test: `tests/test_compare_layer3_values_v1.py`

Step 1: Insert the new generalized rebinding call near the top of `get_system_value()` or in a pre-mapping stage.

Order should be:
1. paper-independent generalized rebinding rules
2. remaining legacy paper-local overrides
3. direct field map
4. decision-trace fallback
5. preparation-method fallback

Step 2: Keep behavior unchanged for existing non-UFXX tests.

### Task 6: Migrate UFXX behavior to generalized schema and remove direct UFXX branch

Objective: Delete the explicit UFXX9WXE branch once generalized behavior is proven equivalent.

Files:
- Modify: `src/stage5_benchmark/compare_layer3_values_to_gt_v1.py`
- Test: `tests/test_compare_layer3_values_v1.py`

Step 1: Confirm all UFXX tests pass through generalized path.

Step 2: Remove:
- `_ufxx9wxe_row_values` if superseded
- direct `if resolved_paper_key == "UFXX9WXE"` field override block

Step 3: Rerun full tests.

### Task 7: Add guardrail tests against overreach

Objective: Prove the new mechanism does not fire on arbitrary prose or unrelated rows.

Files:
- Modify: `tests/test_compare_layer3_values_v1.py`

Add tests for:
- non-pipe evidence row -> no rebinding
- too-short row -> no rebinding
- unrelated paper row with ordinary evidence -> no rebinding
- no support for cross-row guessing
- schema does not create values for fields outside explicit mapping

### Task 8: Verify compare output on current baseline

Objective: Prove generalized mechanism preserves or improves current result without regressions.

Files:
- Output only under `data/results/...`

Run:
- `python3 -m unittest tests/test_compare_layer3_values_v1.py`
- `python3 -m py_compile src/stage5_benchmark/compare_layer3_values_to_gt_v1.py tests/test_compare_layer3_values_v1.py`
- rerun `compare_layer3_values_to_gt_v1.py` against the current maintained final table and GT

Expected:
- UFXX9WXE no longer relies on a paper-key branch
- UFXX9WXE residuals remain at or below current level
- no large regression in total error rows

### Task 9: Multi-paper safety check

Objective: Validate this mechanism remains bounded and conceptually reusable.

Files:
- output compare artifacts only

Check at least:
- UFXX9WXE
- one non-decoded-table paper where rule should NOT fire
- one paper from another failure class (e.g. INMUTV7L or WIVUCMYG) to confirm no interference

Expected:
- decoded-table rebinding only affects eligible rows
- no new spurious values on other papers

---

## Test additions required

Add or refactor tests so that they are no longer paper-branch tests but capability tests.

Must-have tests:
- structured row parser extracts columns from direct span text
- structured row parser extracts columns from supporting refs
- decoded schema maps concentrations correctly
- decoded schema maps particle size / ee / pdi correctly
- decoded schema suppresses misbound `drug_mass_mg`
- decoded schema suppresses extra `polymer_grade`
- decoded schema returns shared constants only when eligible
- ineligible row does not trigger override

---

## Success criteria

Functional:
- No explicit UFXX9WXE compare override branch remains.
- Generalized rebinding engine exists and is schema-driven.
- Current UFXX9WXE gains are preserved.
- No new broad regressions appear on compare rerun.

Architectural:
- Compare repair is organized by failure mode, not paper key.
- New schemas can be added without branching core logic by paper name.
- Guardrails prevent generic prose mining or uncontrolled value invention.

---

## Follow-up after v1

If v1 succeeds, next generalization targets should be:
1. `simple_numbered_row_label_semantics_v1`
   - absorbs INMUTV7L-style row-label identity repair
2. `shared_material_process_carrythrough_v1`
   - absorbs 5GIF3D8W / YGA8VQKU-style shared method/solvent carry-through
3. `safe_extra_surface_suppression_v1`
   - absorbs recurring misbound extra-surface suppression patterns

Do NOT implement these follow-ups in the same patch.
Keep decoded structured-table rebinding v1 isolated and verifiable.

---

## Verification commands

From repo root:

```bash
python3 -m unittest tests/test_compare_layer3_values_v1.py
python3 -m py_compile src/stage5_benchmark/compare_layer3_values_to_gt_v1.py tests/test_compare_layer3_values_v1.py
PYTHONPATH=. python3 src/stage5_benchmark/compare_layer3_values_to_gt_v1.py \
  --final-table-tsv data/results/20260423_9c4a03f/89_stage5_live_minplus_final_r3/final_formulation_table_v1.tsv \
  --decision-trace-tsv data/results/20260423_9c4a03f/89_stage5_live_minplus_final_r3/final_output_decision_trace_v1.tsv \
  --layer3-gt-tsv data/cleaned/gt_authority/v1/dev15_layer3_values.tsv \
  --scope-manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv \
  --alignment-scaffold-tsv data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv \
  --out-dir data/results/20260423_9c4a03f/<new_out_dir>
```

---

## Final note

This plan intentionally converts a validated paper-local fix into a generalized capability restoration path. The implementer should prefer a small, explicit schema system over clever heuristics. Bounded reuse beats paper-specific branching, but only if the guardrails remain strict.