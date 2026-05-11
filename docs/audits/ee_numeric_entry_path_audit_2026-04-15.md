# EE Numeric Entry Path Audit

Date: 2026-04-15

## 1. Executive conclusion

The historical higher numeric density was `mixed`, not a single mechanism.

The repo evidence supports four distinct facts:

1. The `2026-03-14` "no-LLM" refresh did carry many numeric values into the Stage5 final table, but it did so by replaying saved Stage2 raw LLM responses rather than rebuilding semantics from cleaned assets. Evidence: [data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/RUN_CONTEXT.md](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\RUN_CONTEXT.md).
2. The `2026-04-11` rules-only comparator also carried many numeric values into the Stage5 final table without any replayed LLM raw responses. Evidence: [data/results/run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1/final_formulation_table_v1.tsv](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1\final_formulation_table_v1.tsv).
3. In both lineages, the dominant numeric path was not row-local table binding. Most numeric cells reached Stage5 as table-like values without a bound `table_row_id`. Evidence: both historical final tables show `table_like_with_row = 0` for EE, size, PDI, zeta, and LC, while `table_like_without_row` is large.
4. The new deterministic Step 2 helper is much stricter than those older Stage5 and workbook surfaces. It intentionally converts table-like values without `table_row_id` into `unresolved_table` rather than treating them as explicit-supported machine values. Evidence: [src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_deterministic_step2_value_backfill_v1.py): lines around the `direct_support()` unresolved-table gate.

So the current EE sparsity is not mainly "no numeric values were historically extracted." It is mainly:

- intentional contract tightening in Step 2,
- plus the absence of a lawful row-local table binding unit in the current deterministic two-step baseline,
- plus some historical dependence on replayed LLM-derived Stage2 surfaces in the March 14 lineage,
- plus some downstream workbook surfaces that displayed or preserved values more liberally than the new Step 2 machine output now allows.

## 2. Historical richer numeric paths

### 2.1 March 14 replay lineage

Run: [data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1)

Documented run contract:

- "Refresh the active DEV15 benchmark deterministically from existing saved Stage2 raw responses without any new LLM or API calls."
- replay mode = `reuse_existing_raw_llm_outputs`
- reused raw responses from `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`

Evidence: [RUN_CONTEXT.md](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\RUN_CONTEXT.md)

Observed Stage5 final-table numeric density:

- `encapsulation_efficiency_percent_value = 154`
- `size_nm_value = 168`
- `pdi_value = 108`
- `zeta_mV_value = 101`
- `loading_content_percent_value = 35`
- `plga_mass_mg_value = 135`
- `surfactant_concentration_text_value = 85`

Observed candidate-source split for numeric fields:

- EE: `26` from `doe_numbered_table_row`, `138` from `llm_extracted`
- size: `26` from `doe_numbered_table_row`, `142` from `llm_extracted`
- PDI: `26` from `doe_numbered_table_row`, `82` from `llm_extracted`
- zeta: `101` from `llm_extracted`
- LC: `57` from `llm_extracted`

Inference:

- March 14 numeric density was partly lawful deterministic downstream carry-through, but the underlying candidate universe was still largely inherited from replayed LLM Stage2 outputs.
- The main numeric density did not come from a row-local deterministic table binder.

### 2.2 April 11 rules-only comparator lineage

Run: [data/results/run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1)

Documented run contract:

- rules-only comparator
- no reuse of active Stage2 or Stage5 outputs as production inputs
- selected Stage2 producer = `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- downstream reused unchanged:
  - `build_stage2_compatibility_projection_v1.py`
  - `build_formulation_relation_artifacts_v1.py`
  - `build_minimal_final_output_v1.py`
  - review/export helpers

Evidence: [RUN_CONTEXT.md](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1\RUN_CONTEXT.md)

Observed Stage5 final-table numeric density:

- `encapsulation_efficiency_percent_value = 104`
- `size_nm_value = 96`
- `pdi_value = 77`
- `zeta_mV_value = 64`
- `loading_content_percent_value = 21`
- `plga_mass_mg_value = 62`
- `surfactant_concentration_text_value = 51`

All of those rows had `candidate_source = paper_driven_deterministic_semantic_emitter_v1`.

Inference:

- Richer numeric density was not exclusive to replay-backed lineages.
- The deterministic comparator could already populate many numeric Stage5 fields.
- But that density still did not rely on row-local `table_row_id` preservation.

### 2.3 Workbook and export surfaces

Relevant code:

- [src/stage5_benchmark/export_final_formulation_audit_ready_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\export_final_formulation_audit_ready_v1.py)
- [src/stage5_benchmark/build_field_gt_review_workbook_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_field_gt_review_workbook_v1.py)
- [src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_value_gt_annotation_workbook_v1.py)

Observed behavior:

- `export_final_formulation_audit_ready_v1.py` exposes Stage5 numeric values for review, but it is export-only, not a new extraction path.
- `build_field_gt_review_workbook_v1.py` explicitly labels many rows as `unresolved_table` or `unsupported_text` rather than suppressing them entirely.
- `build_value_gt_annotation_workbook_v1.py` builds value workbooks from the audit-ready export plus field-review seed rows and can preserve prior human annotation cells via `--prior-workbook-xlsx` and trusted alignment rows.

Evidence:

- [src/stage5_benchmark/build_field_gt_review_workbook_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_field_gt_review_workbook_v1.py): lines around `resolve_field_evidence()`, `classify_evidence_status_detail()`, and `choose_extracted_value()`
- [src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_value_gt_annotation_workbook_v1.py): `merge_preserved_cells`, `--prior-workbook-xlsx`, `--trusted-alignment-tsv`

Important result:

- Historical workbook density was richer than the current Step 2 machine output partly because the workbook path showed extracted values even when the evidence status was unresolved or inherited from prior human annotation.
- Example: in the March 14 lineage, [value_gt_annotation_workbook_with_phase_and_polymer_values_v1.tsv](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1\value_gt_annotation_workbook_with_phase_and_polymer_values_v1.tsv) shows `sys_EE = 137` and `sys_particle_size = 141`, while the downstream field-review seed surface for the same lineage had only `28` EE rows and `31` particle-size rows classified as `supported`; the rest were mostly `unresolved_table`.

## 3. New deterministic two-step numeric path

Current baseline chain:

Step 1:

- [src/analysis/run_deterministic_step1_baseline_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\analysis\run_deterministic_step1_baseline_v1.py)
- [src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage2_sampling_labels\emit_semantic_objects_from_cleaned_papers_v1.py)
- [src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage2_sampling_labels\build_stage2_compatibility_projection_v1.py)
- [src/stage3_relation/build_formulation_relation_artifacts_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage3_relation\build_formulation_relation_artifacts_v1.py)
- [src/stage5_benchmark/build_minimal_final_output_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_minimal_final_output_v1.py)

Step 2:

- [src/analysis/run_deterministic_step2_baseline_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\analysis\run_deterministic_step2_baseline_v1.py)
- [src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_deterministic_step2_value_backfill_v1.py)

Observed DEV15 diagnostic output:

- Step 1 final table still had many numeric values:
  - EE nonblank = `111`
  - size nonblank = `96`
  - PDI nonblank = `77`
  - zeta nonblank = `64`
- But Step 2 machine output only kept:
  - EE explicit-supported = `20`
  - particle_size explicit-supported = `5`
  - PDI explicit-supported = `2`
  - zeta explicit-supported = `5`

Evidence:

- [data/results/20260415_23c14f0/08_dev15_deterministic_two_step_diag_v2/final_formulation_table_v1.tsv](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260415_23c14f0\08_dev15_deterministic_two_step_diag_v2\final_formulation_table_v1.tsv)
- [data/results/20260415_23c14f0/08_dev15_deterministic_two_step_diag_v2/step2_outputs/step2_value_backfill_table_v1.tsv](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260415_23c14f0\08_dev15_deterministic_two_step_diag_v2\step2_outputs\step2_value_backfill_table_v1.tsv)
- [data/results/20260415_23c14f0/08_dev15_deterministic_two_step_diag_v2/analysis/layer3_compare_summary_v1.md](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260415_23c14f0\08_dev15_deterministic_two_step_diag_v2\analysis\layer3_compare_summary_v1.md)

Key structural fact:

- For the current Step 1 final table, numeric fields have `table_like_with_row = 0` across EE, LC, size, PDI, zeta, polymer amount, drug amount, and surfactant concentration.
- The same fields have large `table_like_without_row` counts.

So the current Step 2 helper is mostly not "missing candidates"; it is rejecting unbound table-like values on purpose.

## 4. Field-by-field mechanism comparison

| field_name | historical richer path | current path | likely reason for loss/sparsity | recoverable deterministically | recoverable without relaxing current strict contract | likely needs a new table-row binding unit | confidence |
|---|---|---|---|---|---|---|---|
| `encapsulation_efficiency_percent` | March 14: mostly replayed `llm_extracted` Stage2 rows plus some DOE rows; April 11: deterministic emitter measurement candidates projected into Stage5; workbook path surfaced unresolved values | Step 2 reads Stage5 value, then blanks table-like rows lacking `table_row_id` as `unresolved_table` | values survive into Stage5, but row-local proof is missing | yes | limited | yes | high |
| `loading_capacity_percent` | same pattern as EE, but lower overall density; workbook path also displayed unresolved values | same strict unresolved-table suppression | same as EE plus lower upstream field availability | yes | limited | yes | high |
| `particle_size_nm` | March 14 and April 11 Stage5 both dense; mainly table-like projected measurements; workbook path showed many system values | explicit-supported only when non-table or safely anchored; most table-like rows blanked | no row-local table binding | yes | limited | yes | high |
| `pdi` | same as particle size | same | same | yes | limited | yes | high |
| `zeta_potential_mV` | same as particle size | same | same | yes | limited | yes | high |
| `drug_polymer_ratio` | not usually direct extraction; review workbook derives from final-row mass values | Step 2 derives only when both drug and polymer amounts are explicit-supported or relation-carried explicit | current contract is stricter than workbook derivation path | yes | yes, if explicit inputs exist | no, if upstream masses bind; yes otherwise | high |
| `polymer_amount` | historically entered Stage5 from component amount expressions in compatibility projection; many rows still table-like | Step 2 keeps only explicit or relation-carried values; many table-like rows downgraded | same row-local binding gap | yes | limited | yes | high |
| `surfactant_concentration` | historically came from component amount expressions and some review-layer normalization hints; workbook kept many unresolved rows | Step 2 keeps some explicit values, but many table-like cases blank | some rows are text/component explicit; many remain unbound table values | yes | partly | yes | medium-high |
| `phase_ratio` | mainly representation-repair / workbook-level value surface, not a strong Stage5 final-table field in current baseline | Step 2 has no direct Step 1 source field and therefore stays blank | field did not have a maintained frozen-final machine source in current baseline | maybe | no | yes | medium |

## 5. Actual provenance path by mechanism

### 5.1 True row-local deterministic table extraction

Evidence found:

- The deterministic emitter can create table-derived measurement candidates and handoffs from explicit table rows for supported papers. Examples are in [emit_semantic_objects_from_cleaned_papers_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage2_sampling_labels\emit_semantic_objects_from_cleaned_papers_v1.py), where papers such as `UFXX9WXE`, `WIVUCMYG`, `YGA8VQKU`, `BB3JUVW7`, `INMUTV7L`, and `V99GKZEI` emit `measurement_candidates` from table rows.
- `table_row_expansion_v1.py` is the maintained unit that would emit real `table_row_id` values such as `table_id::row_XX`.

But practical result in the audited lineages:

- final wide-row and final Stage5 numeric values almost never carried `table_row_id`.
- The dense historical numeric paths therefore were not actually using this unit as the dominant entry path.

Conclusion:

- true row-local deterministic table extraction exists as an intended mechanism,
- but it was not the mechanism that historically produced most of the numeric density seen in Stage5 final tables.

### 5.2 Carried through from replayed LLM raw-response surfaces

Strong evidence:

- March 14 explicitly replayed saved raw LLM responses.
- March 14 final-table numeric density was dominated by `llm_extracted` rows for EE, size, PDI, zeta, and LC.

Conclusion:

- historical numeric density was partly dependent on replayed LLM-derived Stage2 surfaces.

### 5.3 Relation-based carry-through

Relevant code:

- [src/stage3_relation/build_formulation_relation_artifacts_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage3_relation\build_formulation_relation_artifacts_v1.py)
- [src/stage5_benchmark/build_minimal_final_output_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_minimal_final_output_v1.py)

Important limit:

- `build_minimal_final_output_v1.py` only governs resolved relation carry-through for:
  - `polymer_mw_kDa`
  - `surfactant_name`
  - `organic_solvent`
  - `preparation_method`

It does not relation-carry EE, LC, size, PDI, zeta, or phase ratio into the benchmark-final table.

Conclusion:

- relation carry-through explains some descriptive/shared fields,
- but not the historical density of the physicochemical numeric fields.

### 5.4 Stage5 closure / projection logic

Relevant code:

- [build_minimal_final_output_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage5_benchmark\build_minimal_final_output_v1.py)
- [build_stage2_compatibility_projection_v1.py](C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\src\stage2_sampling_labels\build_stage2_compatibility_projection_v1.py)

Key facts:

- Stage2 compatibility projection maps measurement candidates into legacy wide-row numeric fields by name matching.
- The projection is explicitly documented as:
  - `"size_nm/pdi/zeta_mV/encapsulation_efficiency_percent/loading_content_percent"` <- `measurement_candidate`
  - `"supporting_evidence_refs"` <- coarse evidence handoff only
  - `"instance_evidence_region_type/evidence_section/evidence_span_text"` <- coarse direct only, "Not audit-grade ownership binding."
- The compatibility projection writes the same coarse `region`, `locator`, and `snippet` into projected field bundles, not a row-local ownership binding.

Conclusion:

- one major historical numeric-entry mechanism was compatibility projection from candidate objects into legacy wide-row fields,
- but the projection preserved only coarse evidence, not a lawful row-local table binding.

### 5.5 Workbook/export-only heuristics

Strong evidence:

- `build_field_gt_review_workbook_v1.py` can still emit rows with extracted values marked `unresolved_table`, `unsupported_text`, or `derived_without_direct_text`.
- `build_value_gt_annotation_workbook_v1.py` populates `sys_*` value cells from the final table and field-review seed rows, then optionally preserves prior human annotation cells and trusted alignment rows.

Conclusion:

- workbook/export surfaces were richer than the current Step 2 machine table because they were designed for review and annotation, not strict machine-ready explicit support.
- Some historical impression of "the system had many numeric values" came from those review surfaces rather than from a strict explicit-only downstream value table.

## 6. Old path versus new path

### 6.1 Source of candidate universe

- Old richer path:
  - March 14: replayed prior LLM raw responses.
  - April 11: deterministic emitter semantic objects from cleaned assets.
- New path:
  - deterministic emitter + compatibility projection only.

Change assessment:

- changed, but not the whole story.
- replay removal explains part of the loss from March 14 to April 11.
- it does not explain the loss from April 11 Stage5 final table to current Step 2.

### 6.2 Evidence-binding strictness

- Old richer path:
  - Stage5 final table accepted coarse table-like field projection.
  - workbook paths kept unresolved values visible.
- New path:
  - Step 2 requires explicit support and downgrades unbound table-like values to `unresolved_table`.

Change assessment:

- yes, materially changed.
- this is the main direct cause of current sparsity.

### 6.3 Table-row locality requirement

- Old richer path:
  - effectively no hard requirement; dense numeric paths operated with coarse table provenance.
- New path:
  - hard practical requirement in Step 2 for table-like values.

Change assessment:

- yes, materially changed.
- this is the clearest mechanism-level explanation.

### 6.4 Relation carry-through policy

- Old and new:
  - relation carry-through remains narrow and mostly descriptive.

Change assessment:

- not the main cause of EE sparsity.

### 6.5 Stage5 closure semantics

- Old and new benchmark-final Stage5:
  - still willing to emit numeric values from compatibility-projected measurement fields.
- New two-step baseline:
  - Step 2 is downstream of Stage5 and is stricter than Stage5 itself.

Change assessment:

- the gap is mostly between Stage5 final-table permissiveness and Step 2 backfill strictness, not primarily a Stage5 regression.

### 6.6 Workbook/export helper behavior

- Old richer path:
  - review helpers surfaced unresolved values and preserved human edits.
- New path:
  - Step 2 requires machine-readable explicit support, not just review visibility.

Change assessment:

- yes, materially changed for what counts as "present."

### 6.7 Use of replayed LLM surfaces

- Old richer path:
  - March 14 yes.
- New deterministic two-step:
  - no.

Change assessment:

- yes, but only partly explanatory because April 11 still had substantial numeric density without replay.

### 6.8 Value preservation versus value suppression

- Old richer path:
  - preserve value text in Stage5 and review surfaces even when evidence was coarse or unresolved.
- New path:
  - preserve only if explicit-supported or safe relation-carried explicit.

Change assessment:

- yes, this is the most direct reason for the sparse machine table.

## 7. Why EE is sparse now

### Facts

1. The current Step 1 final table still contains many EE and physicochemical values.
2. Nearly all of those table-like numeric values lack `table_row_id`.
3. The current Step 2 helper explicitly blanks such rows as `unresolved_table`.
4. Historical workbook and annotation surfaces displayed many of these unresolved values instead of suppressing them.
5. The March 14 lineage also benefited from replayed LLM-origin Stage2 candidates.

### Inference

The current EE sparsity is mostly not a pure extraction-capability collapse. It is a contract mismatch:

- upstream compatibility projection can still place many numeric values into Stage5,
- but downstream Step 2 now requires a stronger evidence-binding standard than those projected values actually carry.

### Split by cause

Actual capability loss:

- some loss from March 14 to April 11 is real because replayed LLM candidates were richer than the deterministic emitter on some papers.

Intentional contract tightening:

- the major loss from April 11 Stage5 numeric density to current Step 2 machine density is intentional.

Hidden dependency on historical LLM-derived artifacts:

- yes for the March 14 lineage.
- not sufficient to explain the whole sparsity gap because April 11 remained much denser than Step 2.

Hidden dependency on workbook/review surfaces:

- yes.
- older "system has many numeric values" impressions were partly formed from review surfaces that tolerated unresolved evidence status.

## 8. Which losses are intentional versus accidental

Intentional:

- Step 2 unresolved-table blanking when `table_row_id` is absent.
- Step 2 refusal to reuse donor-fill, assumption-based inference, or workbook-only heuristics.
- Step 2 narrow derivation policy for `drug_polymer_ratio`.

Accidental or at least non-goal side effects:

- the deterministic Step 1 path still does not hand Step 2 a lawful row-local binding unit for most numeric table values.
- compatibility projection preserves coarse evidence but not ownership-grade row binding.
- this creates a large gap between "Stage5 visibly has values" and "Step 2 can lawfully re-emit them."

## 9. What can likely be recovered without violating the current baseline goal

High-probability recovery:

- lawful deterministic table-row binding for numeric table measurements, so that Step 2 can mark more cells `explicit_supported` instead of `unresolved_table`.

Moderate-probability recovery:

- selected deterministic carry-through from already explicit table-row bindings or execution-grade Stage2 table authority payloads where the formulation row and table row can be linked without ambiguity.

Low-probability recovery under the current strict contract:

- broad recovery of phase-ratio-like fields without a new row-binding unit or additional explicit upstream field materialization.

Not recoverable without changing the experiment’s contract:

- replay-based numeric density that only existed because prior LLM raw responses had already created richer Stage2 candidate surfaces.

## 10. Recommendation categories

Overall classification: `mixed conclusion`

1. recover numeric density by restoring lawful deterministic table-row binding
   - `yes`
   - this is the best evidence-backed next target.
2. recover numeric density by selectively relaxing current Step 2 evidence contract
   - `possible`, but lower priority than building a lawful binding unit.
   - the audit does not support doing this first.
3. recover numeric density only if historical LLM-derived surfaces are reused
   - `no`
   - reuse would recover some density, but April 11 proves it is not the only route.
4. no real loss, current sparsity is expected and historically denser paths were not comparable
   - `no`
   - there is a real gap between historical Stage5 density and current Step 2 machine density.
5. mixed conclusion
   - `yes`
   - some historical density was replay-backed, but the dominant current sparsity gap is stricter row-local evidence requirements without an upstream binding unit.

## 11. Exact recommended next implementation target

Recommended next target: `table-row binding unit`

Reason:

- It addresses the main audited mechanism gap without weakening the current explicit-only Step 2 contract.
- It is the only recommendation that aligns all observed evidence:
  - historical Stage5 already had many numeric values,
  - current Step 2 mostly suppresses them because row-local ownership is missing,
  - workbook paths show those values were treated as present-but-unresolved rather than absent,
  - relation carry-through is too narrow to solve EE and physicochemical fields,
  - replay reuse is not necessary to explain the gap and should not be the first repair target.

Concretely, the next implementation target should be:

- preserve or reconstruct a lawful per-formulation numeric table-row binding from execution-grade table authority into the frozen Step 1 output family,
- so that Step 2 can re-emit EE, LC, size, PDI, zeta, polymer amount, drug amount, and surfactant concentration as `explicit_supported` when the row-level link is truly deterministic.

