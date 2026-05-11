# Generic Characterization Measurement Metric Binding and Projection Repair Plan

> **For Hermes:** Execute with TDD, bounded diagnostic replays, and lineage/legal-boundary checks. Keep runtime repairs generic; concrete size/EE/zeta/PDI/DL/LC papers are validation anchors only.

**Goal:** Add a generic characterization/result measurement capability line for row-bound metrics such as `particle_size_nm`, `pdi`, `zeta_mV`, `ee_percent`, `lc_percent`, and `dl_percent`, without reusing preparation/material carrythrough semantics or adding paper-specific runtime rules.

**Architecture:** Clean text and Stage2 table authority remain the only Stage2 source surfaces. Stage2/S2-7 may project row-local measurement candidates and coordinate table cells after LLM/governed evidence handoff authorizes row/formulation scope. Stage5 may materialize only source-backed row-local measurement values into admitted rows, guarded by metric endpoint policy, typed validators, and alignment legality. Diagnostic replays remain `benchmark_valid=no` unless a complete governed benchmark-valid lineage is explicitly run.

**Tech Stack:** Python 3 standard library, Stage2 compatibility TSV/JSON table-cell binding artifacts, Stage5 final table materialization, Layer3 compare TSVs, `unittest`, diagnostic sidecars under `data/results/20260504_ab9f61e/`.

---

## 0. Non-Negotiable Scope Rules

1. **No paper-specific runtime branches.** Concrete examples may appear only in tests, diagnostics, and bounded validation reports.
2. **Do not model measurement fields as material fields.** Preparation/material values use material/entity binding; characterization metrics use row/endpoint/header binding.
3. **Clean text and governed Stage2 table payloads are authority.** Do not read raw PDFs/HTML/XML, GT notes, protocol excerpts, or user snippets as downstream replacement evidence.
4. **No method-shared carrythrough for measured outcomes by default.** Size/PDI/zeta/EE/DL/LC are row-local unless the source explicitly states a common measured value and row identity is lawful.
5. **Alignment-gated projection.** Blocked alignment, competing row universes, and non-unique formulation identities must not receive measurement fills.
6. **Typed validators must be field-specific.** Unit/context guards differ for nm, mV, %, and dimensionless PDI.
7. **Diagnostic-only reporting.** Bounded replay outputs are diagnostic baselines, not benchmark-valid final outputs.

---

## 1. Pre-Plan Baseline Summary

Before runtime modification, this plan writes a diagnostic field-level summary so after-plan results can be compared against the accepted material baseline.

**Output already created:**

```text
data/results/20260504_ab9f61e/051_characterization_metric_pre_plan_baseline_summary_diagnostic/
```

**Contains:**
- `pre_plan_field_level_summary.md`
- `historical_to_current_material_baseline.tsv`
- `accepted_pre_measurement_baseline.tsv`
- `measurement_fields_pre_plan_focus.tsv`
- `RUN_CONTEXT.md`

**Pre-plan accepted baseline:** `032 → 050`, with measurement fields unchanged by the material P9 fix:

```text
particle_size_nm recall 80.00%, correct-value recall 65.62%, extra 0
pdi recall 75.27%, correct-value recall 75.27%, extra 0
zeta_mV recall 75.29%, correct-value recall 75.29%, extra 0
ee_percent recall 78.26%, correct-value recall 78.26%, extra 0
lc_percent recall 70.00%, correct-value recall 70.00%, extra 0
dl_percent recall 0.00%, correct-value recall 0.00%, extra 0
```

---

## 2. Allowed Generic Capability Model

1. **Measurement endpoint/header dictionary**
   - Map source headers/phrases to canonical fields plus endpoint subtype.
   - Examples: `Particle size (nm)`, `PS (nm)`, `Diameter (nm)` → `particle_size_nm`; `PDI` → `pdi`; `ZP (mV)` → `zeta_mV`; `EE (%)` → `ee_percent`; `DL (%)` → `dl_percent`; `LC (%)` → `lc_percent`.

2. **Row-local characterization metric binding**
   - Bind values through table row identity, source row/column coordinates, raw header, metric endpoint, and formulation/admitted-row identity.
   - Authority ladder: coordinate table cell + metric header > typed row-local assignment > unique labeled result prose > diagnostic source-CSV rebinding.

3. **Measurement endpoint policy**
   - Preserve endpoint subtypes such as hydrodynamic diameter, Z-average, major axis, minor axis, Feret diameter, predicted/observed, before/after lyophilization, timepoint/stability.
   - Ambiguous morphology/process-state endpoints require review rather than silent primary-field projection.

4. **Metric typed validation**
   - `particle_size_nm`: numeric with nm/size context; reject UV wavelength, filter pore size, scale bars, and morphology endpoints unless policy allows.
   - `pdi`: numeric dimensionless with PDI/polydispersity context; reject arbitrary decimal columns.
   - `zeta_mV`: numeric with zeta/mV context; reject instrument voltage settings.
   - `ee_percent`, `dl_percent`, `lc_percent`: numeric percent with endpoint wording; reject release %, recovery %, viability %, yield %, and assay-only percentages.

5. **Alignment-gated materialization**
   - Materialize only into admitted final rows with lawful unique row identity and no blocked alignment/competing row-universe marker.

---

## 3. Execution Phases

### Phase 1: Diagnostic residual boundary audit

**Objective:** Add or run a diagnostic-only audit that classifies measurement residuals by first failure boundary.

**Outputs:** `052_characterization_metric_residual_boundary_audit_diagnostic/`

**Columns:**
```text
paper_key
field_name
compare_status
gt_formulation_id
matched_system_formulation_id
system_value_raw
gt_value_raw
system_value_source_type
evidence_status_detail
alignment_rule
first_failure_boundary
notes
```

**Status taxonomy:**
- `missing_system_field_surface`
- `measurement_table_header_binding_gap`
- `measurement_projection_gap`
- `present_but_mismatch_endpoint_or_value_policy`
- `alignment_blocked_before_metric_projection`
- `extra_metric_surface_requires_review`

### Phase 2: Measurement endpoint/header dictionary guards

**Objective:** Strengthen generic header aliases for size/PDI/zeta/EE/DL/LC and negative headers.

**Tests:** Extend `tests/test_compare_layer3_values_v1.py` and/or `tests/test_table_structure_dictionary_v1.py` with RED tests for:
- `Y2 (PS, nm)` → `particle_size_nm`
- `Z-average (nm)` → `particle_size_nm`
- `D.L. (%)` → `dl_percent`
- `L.C. (%)` → `lc_percent`
- `Recovery (%)`, `Release (%)`, `Cell viability (%)`, `HPLC method`, `LC-MS assay` → no canonical metric field

### Phase 3: Generic measurement metric binding helper

**Objective:** Add a small shared helper surface, preferably `src/stage5_benchmark/measurement_metric_binding_v1.py`, or keep a local helper if the first patch is minimal, to validate and materialize row-local measurement fields generically.

**Initial supported fields:**
`particle_size_nm`, `pdi`, `zeta_mV`, `ee_percent`, `lc_percent`, `dl_percent`.

### Phase 4: Stage5 row-local table-cell integration

**Objective:** Ensure Stage5 consumes Stage2 `table_cell_grid_v1_row_local_header_binding` for all supported measurement metrics, including DL/LC which were not covered by the prior material plan.

**Rules:**
- Apply row-local values only when metric header validates for the field.
- Preserve raw text in `*_value_text` fields when available.
- Do not fill blocked alignment rows or create rows.
- Do not use method-shared carrythrough.

### Phase 5: S2-7 projection review / minimal repair

**Objective:** Inspect whether measurement candidates authorized by Stage2 are lost before final rows. Only patch S2-7 if a generic projection gap is proven by diagnostic audit and RED tests.

**Likely files:**
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- `src/stage2_sampling_labels/table_row_expansion_v1.py`

### Phase 6: Morphology/process-state endpoint policy

**Objective:** Prevent overfill from nm-valued non-primary endpoints.

**Review-first examples:**
major axis, minor axis, Feret diameter, scale bar, UV wavelength, filter pore size, before/after lyophilization, stability timepoint.

### Phase 7: Bounded diagnostic replay and comparison

**Objective:** Rebuild Stage5 final table from the accepted DOE-explicit Stage2/Stage3 lineage, compare to GT, and compare against pre-plan summary.

**Accepted input lineage:**
- Stage2: `data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 relations: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv`
- Stage3 resolved fields: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/resolved_relation_fields_v1.tsv`
- Compare baseline: `data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/`

**Acceptance gates:**
- `benchmark_valid=no` unless separately promoted by governance.
- No new unexplained `extra_in_system` in core/measurement fields.
- No blocked-alignment forced fills.
- Measurement gains must come from source-backed row-local/header binding or audited projection, not paper-specific completion.

---

## 4. Progress Ledger

Use companion ledger:

```text
docs/plans/2026-05-05-characterization-measurement-metric-binding-progress-ledger.md
```

Each phase must record status, artifacts, tests, lineage, and diagnostic boundary.

---

## 5. Execution Log

2026-05-05 current-session execution:

- Wrote pre-plan baseline diagnostic `051_characterization_metric_pre_plan_baseline_summary_diagnostic/` for comparison against accepted material baseline `050`.
- Added Phase 1 residual audit script/test and ran `052_characterization_metric_residual_boundary_audit_diagnostic/`.
- Completed Phase 2 endpoint/header guards with TDD for PS/Z-average/P.I./ZP/D.L./L.C. aliases and negative recovery/release/viability/HPLC/LC-MS/morphology guards.
- Completed Phase 3/4 Stage5 row-local metric materialization for DL/LC plus fixed source-CSV row-index alignment to Stage2 physical CSV `source_row_index`.
- Completed Phase 6 minimal morphology policy: `major axis (nm)` can project to primary particle size; `minor axis` and `Feret` do not silently project.
- Completed Phase 7 bounded diagnostic replay:
  - Stage5: `053_stage5_characterization_metric_binding_bounded_diagnostic/`, final_rows=202, benchmark_valid=no.
  - Layer3 compare: `054_layer3_compare_characterization_metric_binding_bounded_diagnostic/`, benchmark_valid=no.
  - `050 -> 054`: missing 688->676 (-12), present_and_match 2039->2051 (+12), present_but_mismatch 339->339 (+0), blocked 252->252 (+0), extra 34->34 (+0).
  - Measurement gains: `particle_size_nm` system_on_GT 128->135 with correct-value recall 65.62%->70.00%; `dl_percent` system_on_GT 0->5; new-only extra rows=0.
