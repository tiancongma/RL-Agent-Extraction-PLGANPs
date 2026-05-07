# Characterization Measurement Full-Line Repair and Live LLM Rerun Plan

> **For Hermes:** Execute with TDD, governed repair-index intake, replay-first validation, then fresh live LLM only after deterministic line repairs are green. Use subagent-driven-development for implementation/review tasks if executing this plan.

**Goal:** Repair the full characterization/result measurement capability line for `particle_size_nm`, `pdi`, `zeta_mV`, `ee_percent`, `lc_percent`, and `dl_percent`, then run a fresh live-LLM baseline only after replay proves the deterministic pipeline is safe.

**Architecture:** Keep Stage2 LLM as semantic authority for formulation/table/result scope. Deterministic code may preserve, project, validate, and materialize measurement values only after LLM/governed scope authorization. Fix the line in upstream-to-downstream order: evidence preservation -> cell/endpoint projection -> compatibility surface -> Stage5 materialization -> compare visibility -> full replay -> live LLM. Do not use S5-3 as a shortcut for row-local table/result metrics.

**Tech Stack:** Python 3 stdlib, `unittest`, maintained Stage2/Stage3/Stage5/Layer3 entrypoints from `project/ACTIVE_PIPELINE_RUNBOOK.md`, governed diagnostic run directories under `data/results/20260504_ab9f61e/` or a new dated lineage, Layer3 GT compare TSVs.

---

## 0. Current State and Why a Full-Line Plan Is Needed

The completed bounded repair `053/054` fixed only one downstream segment:

```text
row-local table/grid metric evidence already present -> Stage5 field bundle materialization
```

Observed bounded diagnostic result:

```text
050 -> 054
missing_in_system: 688 -> 676 (-12)
present_and_match: 2039 -> 2051 (+12)
present_but_mismatch: 339 -> 339 (+0)
blocked_alignment: 252 -> 252 (+0)
extra_in_system: 34 -> 34 (+0)
```

The small delta is expected because the repair did not reopen:

- Stage2 evidence selection and preservation;
- Stage2 table-cell binding / metric-tail projection;
- S2-7 compatibility projection gaps;
- compare-surface visibility for row-local metric snippets;
- endpoint/process-state policy beyond a minimal `major axis` guard;
- blocked alignment / row universe issues.

Repair index confirms already-validated related patterns exist and must be reused rather than redesigned:

- `PAT_STAGE2_TABLE_ROW_MEASUREMENT_TAIL_SIZE_CARRYTHROUGH_V1`: explicit governed activation; full replay restored `particle_size_nm` missing 57->45 and match 81->93.
- `PAT_STAGE2_TABLE_ROW_ABBREVIATED_EE_HEADER_MAPPING_V1`: explicit governed activation; EE abbreviated headers restored `ee_percent` missing 56->50 and match 85->91.
- `PAT_LAYER3_EE_EVIDENCE_METRIC_REBINDING_V1`: row-local labeled EE evidence compare visibility; +6 changed cells, no collateral.
- `PAT_LAYER3_PARTICLE_SIZE_EVIDENCE_METRIC_REBINDING_V1`: row-local labeled size evidence compare visibility; +7 changed cells, no collateral.
- `PAT_LABELED_MEASUREMENT_TABLE_DUPLICATE_OF_DOE_ROW_V1`: later measurement-only DOE-label tables should collapse onto existing DOE cores, not create new rows.

---

## 1. Non-Negotiable Rules

1. Do not make paper-specific runtime branches. Paper keys are validation anchors only.
2. Do not fill measurement metrics from GT, protocol notes, user excerpts, raw PDFs, or raw HTML. Stage2 clean text and governed preserved table payloads are authority.
3. Do not use method-shared carrythrough for measured outcomes unless the source explicitly reports a common measured value for the row scope.
4. Do not materialize into blocked alignment or ambiguous row-universe surfaces.
5. Do not spend fresh live LLM tokens until deterministic repairs pass full frozen-response replay.
6. Do not treat a replay diagnostic as benchmark-valid.
7. Do not update `ACTIVE_RUN.json` until the selected diagnostic/live lineage has full Stage2->Stage3->Stage5->Layer3 artifacts and `RUN_CONTEXT.md`.
8. Do not claim benchmark improvement unless modified artifact lineage, evaluation artifact lineage, and GT authority source are explicitly aligned.

---

## 2. Repair Scope by Capability Layer

### Layer A0 — Clean-text inclusion and source conversion gate

**Question:** Are the measurement-bearing sentences/tables actually present in the cleaned text that Stage2 is allowed to use?

This is the first gate. If a value never enters clean text, neither LLM rerun nor downstream deterministic repair can lawfully recover it. Downstream stages must not search raw PDFs/HTML/XML, user snippets, protocol notes, or GT to compensate.

**Repair targets:**

- audit clean-text coverage for measurement-bearing result/characterization tables and nearby explanatory paragraphs;
- detect truncated body text, missing table bodies, caption-only table remnants, OCR/CSV-to-text drops, and selector-visible summaries whose underlying numeric values are absent from clean text;
- repair the text-cleaning / table-to-clean-text conversion path when clean text lacks source-present values;
- add sidecar diagnostics that distinguish:
  - `source_value_absent_from_clean_text`,
  - `clean_text_has_value_but_selector_omitted`,
  - `selector_included_summary_but_not_numeric_table`,
  - `llm_prompt_has_value_but_llm_did_not_authorize`,
  - `llm_authorized_but_downstream_projection_lost`.

**Likely files:**

```text
src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py
src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py
# plus the maintained cleaned-text/table-conversion utilities identified by ACTIVE_PIPELINE_RUNBOOK
```

**Acceptance:** for every sampled residual selected for repair, the plan records whether the original value is present in clean text before blaming selector, LLM, Stage2 projection, Stage5, or compare.

### Layer A1 — Evidence selector / prompt-pack inclusion

**Question:** If clean text contains the value, did S2-2/S2-4a select and present the measurement-bearing evidence to the LLM with enough local context?

Previously observed selector failures must be treated as first-class repair targets, not as optional live-LLM cleanup. A fresh LLM call cannot authorize values that were never placed in the prompt/evidence pack.

**Repair targets:**

- audit S2-2 selector inclusion for characterization/result tables that are row-local value evidence even when they are not formulation-universe creators;
- preserve measurement-bearing tables/windows as semantic-facing evidence summaries while separately marking whether they may create formulation rows;
- ensure compact summary rendering does not drop numeric measurement columns needed for endpoint authorization;
- ensure evidence budget / role-coverage logic does not exclude the only result table because a preparation/formulation table already satisfied table coverage;
- keep the distinction between `source_role=value_evidence_only` and `source_role=formulation_universe_source`.

**Likely files:**

```text
src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py
src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py
src/stage2_sampling_labels/table_structure_dictionary_v1.py
```

**Acceptance:** the S2-4a prompt/evidence audit proves whether each sampled residual value was visible to the LLM; if not, selector/prompt-pack repair comes before live LLM.

### Layer B — Stage2 row-local table cell binding and metric-tail projection

**Question:** Once a formulation row is admitted, are its row-local metric cells preserved with header/coordinate/endpoint metadata?

**Repair targets:**

- ensure `table_cell_grid_v1` / `table_cell_bindings_json` includes measurement-tail cells for size/PDI/zeta/EE/DL/LC;
- restore measurement-tail assignment projection from authorized explicit tables;
- map abbreviated assay headers: `E.E.`, `E.E.%`, `D.C.`, `D.L.`, `L.C.`, `P.I.`, `ZP`, `Sizes`, `Z-average`;
- preserve source row/column index and header path; no compare-time CSV rescue should be the primary path after this repair.

**Likely files:**

```text
src/stage2_sampling_labels/table_row_expansion_v1.py
src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py
src/stage2_sampling_labels/table_structure_dictionary_v1.py
```

**Acceptance:** Stage2 compatibility rows expose non-empty metric binding surfaces before Stage5 runs.

### Layer C — Endpoint taxonomy and typed validators

**Question:** Are fields mapped to the correct metric endpoint instead of any numeric nm/%/mV column?

**Repair targets:**

- endpoint subtype object: `primary_size`, `z_average`, `hydrodynamic_diameter`, `major_axis`, `minor_axis`, `feret`, `before_lyophilization`, `after_lyophilization`, `stability_timepoint`, `predicted`, `observed`, `release`, `recovery`, `viability`, `yield`, `assay_method`;
- positive mappings: canonical characterization endpoints;
- negative mappings: release/recovery/viability/yield/HPLC/LC-MS/UV wavelength/filter pore/scale bar;
- field-specific value validators for nm, mV, %, PDI.

**Likely files:**

```text
src/stage2_sampling_labels/table_structure_dictionary_v1.py
src/stage5_benchmark/build_minimal_final_output_v1.py
# Optional if extracted:
src/stage5_benchmark/measurement_metric_binding_v1.py
```

**Acceptance:** recall gains must not increase mismatch/extra; ambiguous endpoints remain sidecar/review, not primary-field fills.

### Layer D — Stage5 materialization from Stage2 metric surfaces

**Question:** Are admitted final rows consuming Stage2's metric surfaces consistently?

**Repair targets:**

- prefer Stage2 `table_cell_bindings_json` / grid bindings over raw source CSV fallback;
- support all metric bundles: `size_nm`, `pdi`, `zeta_mV`, `ee_percent`, `lc_percent`, `dl_percent`;
- expose value/raw text/source/header/provenance fields;
- collapse later measurement-only rows onto unique existing DOE rows when they reuse DOE labels and add no synthesis-defining assignments;
- block final-row creation from measurement-only evidence surfaces.

**Likely files:**

```text
src/stage5_benchmark/build_minimal_final_output_v1.py
```

**Acceptance:** final row count stays stable unless row-universe bug is explicitly fixed; metric values fill only admitted rows.

### Layer E — Layer3 compare visibility / audit sidecars

**Question:** Does compare see row-local metric evidence already present in final rows?

**Repair targets:**

- expose short row-local labeled metric snippets in compare only when final field bundle is blank and snippet is row-local/header-labeled;
- no long article-wide source-text rebinding;
- add delta audits by field/status/paper and changed-cell audit.

**Likely files:**

```text
src/stage5_benchmark/compare_layer3_values_to_gt_v1.py
tests/test_compare_layer3_values_v1.py
```

**Acceptance:** changed cells are target-field only, mostly `missing_in_system -> present_and_match`, with zero new unexplained extras/mismatches.

### Layer F — Live LLM rerun readiness

**Question:** Does a fresh LLM call add semantic scope/evidence quality now that deterministic consumers are ready?

**Prerequisites:**

- all deterministic repairs pass unit tests;
- full frozen replay produces no row-count collateral, no new extras, no new mismatches;
- prompt/evidence changes are intentional and documented;
- cost/lineage/run context are explicit.

**Acceptance:** live baseline is executed only after replay and is evaluated separately from replay.

---

## 3. Execution Order

### Phase 0 — Freeze and ledger the current working state

**Objective:** Avoid mixing the completed `054` bounded repair with the larger full-line repair.

**Actions:**

1. Run `git status --branch --short`.
2. Decide whether to commit/stash the current `054` repair before starting this plan.
3. Create/update a progress ledger:

```text
docs/plans/2026-05-05-characterization-measurement-full-line-progress-ledger.md
```

**Stop condition:** Do not start full-line code changes on top of ambiguous uncommitted work.

### Phase 1 — Clean-text and selector visibility audit on current `054`

**Objective:** Prove whether residual metric values are absent from clean text, present in clean text but omitted by selector/prompt-pack, visible to LLM but not authorized, or authorized then lost downstream.

**Inputs:**

```text
data/results/20260504_ab9f61e/054_layer3_compare_characterization_metric_binding_bounded_diagnostic/layer3_value_compare_cells_v1.tsv
accepted Stage2 clean-text/table authority artifacts from the active run contract
S2-4a prompt/evidence-pack artifacts for the accepted baseline or fresh replay
```

**Outputs:**

```text
data/results/20260504_ab9f61e/055_characterization_metric_cleantext_selector_visibility_audit_diagnostic/
```

**Required columns:**

```text
paper_key
field_name
gt_formulation_id
compare_status
gt_value_raw
clean_text_value_visibility
clean_text_locator
selector_visibility
selector_artifact_ref
prompt_visibility
prompt_artifact_ref
llm_authorization_status
downstream_boundary
first_failure_boundary
notes
```

**Failure classes:**

```text
source_value_absent_from_clean_text
clean_text_has_value_but_selector_omitted
selector_included_summary_but_not_numeric_table
llm_prompt_has_value_but_llm_did_not_authorize
llm_authorized_but_stage2_projection_lost
stage2_projected_but_stage5_materialization_lost
stage5_has_evidence_but_compare_surface_blank
endpoint_or_alignment_blocked
```

**Acceptance:** no live LLM rerun and no Stage5 patch until the sampled residuals prove where values disappeared. If `source_value_absent_from_clean_text` dominates, repair cleaning/conversion first. If `clean_text_has_value_but_selector_omitted` dominates, repair selector/prompt-pack first.

### Phase 2 — Residual stratification by deterministic boundary

**Objective:** After the clean-text/selector audit, partition remaining metric misses by downstream deterministic repair layer.

**Inputs:**

```text
data/results/20260504_ab9f61e/054_layer3_compare_characterization_metric_binding_bounded_diagnostic/layer3_value_compare_cells_v1.tsv
```

**Outputs:**

```text
data/results/20260504_ab9f61e/056_characterization_metric_full_line_residual_stratification_diagnostic/
```

**Buckets:**

```text
A_missing_evidence_or_table_not_preserved
B_stage2_cell_binding_absent
C_stage2_projection_field_blank
D_stage5_materialization_blank_despite_binding
E_compare_visibility_blank_despite_final_evidence
F_endpoint_mismatch_or_policy
G_alignment_blocked_or_row_universe
H_gt_not_reported_or_not_target
```

**Acceptance:** no code patch until top residual buckets are quantified.

### Phase 3 — Restore Stage2 metric-tail binding first

**Objective:** Move fixes upstream so Stage5 consumes authoritative cell/endpoint surfaces instead of source-CSV rescue.

**TDD anchors:**

- split header: `Average` + `Size (nm)` -> `particle_size_nm`;
- shifted header: `Number | Used | Size (nm) | P.I. | ZP (mV)` maps size under `Size`, not `Used`;
- abbreviated EE: `E.E.% ± S.D.` -> `encapsulation_efficiency_percent`;
- DL/LC: `D.L. (%)`, `Drug loading (%)`, `L.C. (%)`, `Loading content (%)`;
- negative `%`: recovery/release/viability/yield not EE/DL/LC;
- morphology: `major axis` review/allowed policy; `minor axis` and `Feret` not primary by default.

**Run:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1 tests.test_table_structure_dictionary_v1
PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/table_row_expansion_v1.py src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py src/stage2_sampling_labels/table_structure_dictionary_v1.py
```

**Acceptance:** Stage2 compatibility artifacts show explicit metric binding columns/surfaces.

### Phase 4 — Full frozen-response Stage2 replay

**Objective:** Validate upstream deterministic repair without spending LLM tokens.

**Run type:** diagnostic-only, no ACTIVE_RUN promotion.

**Entry:** maintained Stage2 composite/runbook command only; use frozen raw responses from the accepted baseline lineage.

**Required checks:**

- `analysis/request_summary.tsv` has replay backend for every paper;
- semantic authority contract passes or any validator issue is localized;
- Stage2 row counts do not over-admit measurement-only tables;
- `table_cell_bindings_json` / grid metric surfaces are present for target fields.

**Output suggestion:**

```text
data/results/20260504_ab9f61e/057_stage2_measurement_full_line_replay_diagnostic/
```

### Phase 5 — Stage3 + Stage5 replay with metric materialization

**Objective:** Validate that repaired Stage2 metric surfaces survive into final rows.

**Inputs:** exact Stage2 output from Phase 4.

**Outputs:**

```text
data/results/20260504_ab9f61e/058_stage3_measurement_full_line_replay_diagnostic/
data/results/20260504_ab9f61e/059_stage5_measurement_full_line_replay_diagnostic/
```

**Checks:**

- final row counts by paper vs `054` and accepted baseline;
- metric bundle non-empty deltas by field;
- no measurement-only row over-retention;
- decision trace shows source-backed materialization rule.

### Phase 6 — Layer3 compare and changed-cell audit

**Objective:** Measure replay effect and collateral.

**Output:**

```text
data/results/20260504_ab9f61e/060_layer3_compare_measurement_full_line_replay_diagnostic/
```

**Acceptance gates:**

- `extra_in_system` does not increase for measurement fields unless explicitly reviewed;
- `present_but_mismatch` does not increase;
- changed cells are explainable by `changed_cell_audit.tsv`;
- row-count deltas are explained before value claims;
- compare metadata and `RUN_CONTEXT.md` mark `benchmark_valid=no`.

### Phase 7 — Residual round 2: only fix remaining largest safe bucket

**Objective:** Do not jump to live LLM if replay still shows deterministic failures.

**Order of remaining fixes:**

1. Stage2 projection blanks with existing bindings.
2. Stage5 blanks with existing Stage2 metric surfaces.
3. Compare visibility blanks with final row-local evidence.
4. Endpoint mismatch policy.
5. Alignment/row-universe, only if the metric is blocked by identity rather than value extraction.
6. Selector/prompt omission after clean-text presence is proven.
7. Clean-text/source-conversion absence.

**Stop condition:** If top residual is absent from clean text, stop downstream repair and fix source conversion / cleaned-text generation. If values are in clean text but absent from prompt, fix selector/prompt-pack before live LLM. Otherwise continue deterministic replay repairs.

### Phase 8 — Live LLM prompt/evidence-pack readiness audit

**Objective:** Decide whether fresh LLM is warranted and what it should be asked to do.

**Live LLM is warranted only if replay proves one of:**

- Stage2 preserved/evidence pack does not expose result/characterization table summaries to the LLM;
- LLM semantic scope is too collapsed and deterministic repair cannot lawfully recover row-level metric scope;
- clean-text/evidence selection repair changed S2-4a prompt content;
- fresh semantic classification is needed for endpoint/process-state scope.

**Do not live-rerun merely because Stage5/compare had a deterministic mapping bug.**

### Phase 9 — Fresh live LLM baseline

**Objective:** Run a clean fresh LLM baseline after deterministic consumers are repaired.

**Run type:** full live diagnostic baseline first, not benchmark-valid promotion.

**Required run context fields:**

```text
llm_backend: live
model/provider/config
prompt/evidence-pack version
starting input corpus
script execution order
Stage2/Stage3/Stage5/Layer3 exact paths
GT authority source
benchmark_valid: no unless separately promoted
```

**Immediate post-live checks:**

- compare live Stage2 row counts vs frozen replay;
- identify semantic drift separately from deterministic repair effect;
- run full Stage3/Stage5/Layer3 compare;
- changed-cell audit split into replay-effect vs live-LLM-effect.

### Phase 10 — Promotion decision

**Objective:** Decide whether the new live lineage becomes the active diagnosis baseline or benchmark-valid run.

**Promotion requires:**

- no unresolved Section 11 boundary violation;
- exact source paths printed and in `RUN_CONTEXT.md`;
- no unexplained row-count collateral;
- no unexplained measurement extras/mismatches;
- user approval before changing governed active pointers if benchmark-valid semantics are claimed.

---

## 4. Subagent Review Pattern

Use read-only subagents in parallel for these audits before code patches:

1. **Clean-text coverage auditor:** verify whether sampled residual values exist in allowed clean text, with locators.
2. **Selector/prompt-pack auditor:** verify whether clean-text-present values were selected into S2-2/S2-4a evidence visible to LLM.
3. **Residual stratifier:** quantify buckets A-H on `054` after A0/A1 status is known.
4. **Stage2 surface auditor:** inspect whether missing cells already exist in Stage2 weak labels/grid bindings.
5. **Stage5 materialization auditor:** inspect final rows for blank fields despite evidence.
6. **Endpoint policy auditor:** list mismatch/extras from endpoint/process-state ambiguity.

Main/default agent serializes writes and verifies specialist claims by reading artifacts.

---

## 5. Expected Outcome Shape

A successful full-line repair should produce larger gains than `054`, but the expected shape is still conservative:

```text
missing_in_system decreases
present_and_match increases
present_but_mismatch flat or decreases
extra_in_system flat or decreases
blocked_alignment unchanged unless identity repair is explicitly included
final row count stable unless row-universe repair is explicitly included
```

If recall improves while mismatches/extras rise, the repair is not accepted; classify it as endpoint overfill or row-universe over-admission and stop interpretation.

---

## 6. Why Live LLM Comes After Replay

The current small `054` gain does not prove LLM failure. It proves only that one downstream deterministic consumer was incomplete. The governed workflow should therefore be:

```text
repair deterministic consumers -> full frozen replay -> collateral audit -> only then fresh live LLM
```

A live LLM rerun before deterministic repair risks paying for new semantic outputs that downstream still cannot consume, and it makes it harder to separate prompt/evidence gains from deterministic projection/materialization bugs.
