# EE Numeric Covariate Mainline Repair Plan

**Status:** accepted execution plan, diagnostic-development mode.

**Scope:** this plan contains only the three mainline repairs below. It does
not implement S5-3 LLM numeric value backfill, derived-value calculation, GT
backfill, or a new pipeline.

**Active baseline:** `data/results/ACTIVE_RUN.json` currently pins DEV15
diagnostic lineage `20260423_9c4a03f`:

- S2-7: `675_s2_7_role_header_value_type_guard_source_identity_diagnostic`
- Stage3: `676_stage3_role_header_value_type_guard_source_identity_diagnostic`
- Stage5: `677_stage5_role_header_value_type_guard_source_identity_diagnostic`
- Layer1 compare: `678_compare_role_header_value_type_guard_source_identity_diagnostic`
- Layer3 compare: `679_layer3_compare_role_header_value_type_guard_source_identity_diagnostic`

All results under this plan are diagnostic-only unless a future governed
benchmark contract explicitly says otherwise.

---

## Step 1: S2 Prompt/Schema Typed Relation Cues

**Goal:** first repair the S2 prompt/schema handshake with Stage3 by adding
lightweight optional typed relation cues, while protecting the LLM's primary
task: formulation/table/formulation-row semantic identification.

**Rules:**

- Relation cues are optional and additive.
- The LLM must not be asked to fill the full Layer3 table.
- Existing formulation identification instructions stay primary.
- Empty relation cues are acceptable.
- Only execution-ready cues may be consumed by Stage3.
- Partial or review-only cues may be preserved for audit but must not authorize
  deterministic materialization.

**Minimal cue contract:**

```text
relation_type:
  formulation_variable | protocol_parameter | result_measurement |
  material_identity | row_result_binding | doe_factor_definition
target_scope:
  row_local | table_scope | method_group | paper_global | selected_condition
target_ref:
  formulation id, row id, table id, group label, or source label token
field_name:
  canonical field if known, otherwise blank
value_text:
  direct source text if present
value_kind:
  direct_text | coded_level | physical_value | unit_only | result_measurement
unit_text:
  direct unit text if present
evidence_anchor:
  short source cue or table-cell locator
execution_readiness:
  execution_ready | partial_semantic | review_only
confidence:
  high | medium | low
```

**Verification before promotion:**

1. Unit tests prove parsing/projection preserves relation cues without breaking
   existing markers.
2. S2-4a prompt freeze/audit passes with no prompt-legality failure and no
   material prompt inflation.
3. Bounded replay/live validation confirms formulation row identification is
   not degraded.
4. Full DEV15 replay through S2-7 -> Stage3 -> Stage5 -> Layer1 compare ->
   Layer3 compare keeps Layer1 at `202/202` and `15/15`.
5. If accepted, update repository records, memory, repair index, and
   `ACTIVE_RUN.json` before starting Step 2.

---

## Step 2: S5 Value/Unit Splitter And DOE Coded/Physical Guard

**Goal:** then repair S5 deterministic value/unit splitting and coded/physical
DOE handling, especially for surfactant and drug concentration fields.

**Target fields:**

```text
drug_concentration_value
drug_concentration_unit
emulsifier_stabilizer_concentration_value
emulsifier_stabilizer_concentration_unit
polymer_concentration_value
polymer_concentration_unit
phase_ratio_raw
pH_raw
drug_mass_mg
polymer_mass_mg
```

**Rules:**

- Split direct source expressions into value/unit only when source evidence
  supports that split.
- Decode DOE coded levels only when the source cell is typed as coded.
- Do not decode already-physical table values again.
- Generic role headers such as `surfactant concentration` must not create
  concrete material identities or mass/feed fields.
- Direct fields must not contain derived calculations.
- Compare-side normalization may explain equivalence but must not invent system
  extraction.

**Verification before promotion:**

1. Unit tests cover `%`, `% w/v`, `%w/v`, `mg/mL`, `ug/mL`, `μg/mL`, coded
   levels, already-physical values, and generic role headers.
2. Full DEV15 replay keeps Layer1 at `202/202` and `15/15`.
3. Layer3 surfactant/drug concentration mismatches or missing values improve
   without increasing `blocked_alignment`.
4. Any `missing_in_system` increase is accepted only when it suppresses a
   previously polluted value and is documented field-by-field.
5. If accepted, update repository records, memory, repair index, and
   `ACTIVE_RUN.json` before starting Step 3.

---

## Step 3: Measurement Binding For EE/Size Result Covariates

**Goal:** finally repair measurement binding because EE, particle size, PDI,
and zeta are the most important EE-modeling result covariates, and they depend
on correctly binding result-table rows to formulation rows.

**Target fields:**

```text
ee_percent
particle_size_nm
pdi
zeta_mV
dl_percent
lc_percent
```

**Rules:**

- Measurement binding must not create formulation rows.
- Bind result fields only to already-admitted rows.
- Prefer row-local table-cell/grid evidence.
- Use Stage3 relation records for explicit row/result binding.
- Reject spread from selected-case legends, characterization summaries, scale
  bars, filter/pore sizes, assay settings, release contexts, and other
  non-formulation measurement contexts.

**Verification before promotion:**

1. Unit tests cover same-row result binding, row-label binding, DOE-row binding,
   metric-tail extraction, and false-spread rejection.
2. Full DEV15 replay keeps Layer1 at `202/202` and `15/15`.
3. Layer3 `ee_percent` and/or `particle_size_nm` improves without increasing
   result-field `extra_in_system`.
4. False propagation into non-selected rows is zero in targeted audits.
5. If accepted, update repository records, memory, repair index, and
   `ACTIVE_RUN.json`.

---

## Serial Execution Gate

For each step:

```text
resolve active paths
read repair index and memory
patch the smallest existing mainline surface
run targeted tests
run bounded validation when applicable
run full DEV15 diagnostic replay
compare against the current active diagnostic baseline by field and paper
promote only if validation passes
record repository decision, mem rows, repair-index row, and ACTIVE_RUN update
validate active data-source contract
validate memory surface and note any pre-existing failures
then start the next step
```
