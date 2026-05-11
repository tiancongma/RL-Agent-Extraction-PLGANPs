# Preparation Evidence Selector and Typed Materialization Repair Progress Ledger

Created: 2026-05-05 15:18:16 EDT
Scope: governed unattended execution ledger for Phase 2-9 of `2026-05-05-preparation-evidence-selector-and-typed-materialization-repair-plan.md`.

## Baseline at ledger creation

- Working directory: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs`
- Branch: `feature/ee-coverage-rl`
- Baseline dirty state: heavy pre-existing WIP was present before this tick, including tracked modifications across governance/docs/src/tests and untracked plan/test/stage scripts plus root `.docx`/markdown files. This tick must not commit because `src/stage1_cleaning/pdf2clean.py`, `tests/test_pdf2clean_html_fallback_v1.py`, and the plan file were already dirty/untracked before this executor's edits, making commit scope ambiguous.
- Data-source boundary: no benchmark/audit/workbook comparison was run this tick. Any future workflow consuming `data/results/` must resolve authority via explicit CLI/`--run-dir` or `data/results/ACTIVE_RUN.json` and print exact source paths before interpretation.

## Task ledger

| Task ID | Status | Bounded task | Notes / next action |
|---|---|---|---|
| P2.1 | completed-before-ledger | Start generic BS4 fallback for content-bearing HTML blocks outside `<p>` tags | Recorded in plan execution log before this ledger existed. |
| P2.2 | completed-uncommitted | Add generic quality-gated BS4 supplement when trafilatura succeeds but cleaned HTML extraction is prose-insufficient | Completed 2026-05-05 15:18:16 EDT. Not committed due pre-existing dirty same-file WIP. |
| P3.1 | completed-uncommitted | Add preparation-core sufficiency floor to Stage2 selector candidate/evidence package selection | Completed 2026-05-05 15:33:13 EDT. Not committed due pre-existing dirty same-file WIP and broad repository WIP. |
| P4.1 | completed-uncommitted | Add table-role negative taxonomy and source-role separation | Completed 2026-05-05 16:07:40 EDT. Not committed due pre-existing dirty same-file WIP and broad repository WIP. |
| P5.1 | completed-uncommitted | Implement generic material-value binding helper surface | Completed 2026-05-05 16:21:31 EDT. Not committed due broad pre-existing dirty WIP and untracked same-scope files in workspace. |
| P6.1 | completed-uncommitted | Integrate generic S5-2 shared preparation / admitted-row carrythrough | Completed 2026-05-05 16:43:06 EDT. Not committed due broad pre-existing dirty WIP and same-scope Stage5/test modifications in workspace. |
| P7.1 | completed-uncommitted | Enforce type validators at Stage2 compatibility and Stage5 final boundaries | Completed in-session after cron removal. Not committed due broad pre-existing dirty WIP and same-scope Stage5/test modifications. |
| P8.1 | completed-uncommitted | Harden direct vs derived provenance separation | Completed in-session. Derived mass calculations now carry explicit no-direct-fill provenance/status in sidecar JSON only. |
| P9.1 | completed-uncommitted | Run bounded diagnostic replay validation with lineage/legal boundary checks | Completed in-session. Final accepted diagnostic artifacts: `049_stage5_p9_polymer_mgml_mass_guard_bounded_diagnostic` and `050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic`; benchmark_valid=no. |

## Execution entries

### 2026-05-05 15:18:16 EDT — P2.2 quality-gated BS4 supplement

- Task attempted: P2.2 — generic quality-gated BeautifulSoup supplement for successful-but-insufficient trafilatura HTML extraction.
- Files touched:
  - `src/stage1_cleaning/pdf2clean.py`
  - `tests/test_pdf2clean_html_fallback_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-repair-plan.md`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
- TDD RED:
  - Command: `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1.Pdf2CleanHtmlFallbackTests.test_trafilatura_success_with_sparse_heading_is_supplemented_by_bs4_body`
  - Expected failure observed: `AssertionError: 'PLGA dissolved in dichloromethane' not found in 'Preparation of PLGA nanoparticles'`.
- Implementation:
  - Added `html_blocks_need_bs4_supplement()` generic source-quality gate based on sparse paragraph/list body word count with broad method/preparation-like terms.
  - Added `merge_bs4_supplement_blocks()` exact normalized text deduplication.
  - Updated `extract_bs4_blocks()` to process explicit text tags and fallback `article`/`main`/`section`/`div` tags in DOM order and to suppress nested fallback container amplification.
  - Updated `extract_text_from_html()` to supplement from BS4 when trafilatura succeeds but fails the generic body-prose sufficiency gate; BS4 blocks become the base order-preserving surface and trafilatura-only extras are appended. Warning marker: `trafilatura_insufficient_prose_bs4_supplemented`; parser marker: `trafilatura_plus_beautifulsoup_supplement`.
  - No paper-specific branches, paper-key checks, expected-count logic, GT-derived snippets, raw PDF/XML supplementation, or S5-3 changes.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1.Pdf2CleanHtmlFallbackTests.test_trafilatura_success_with_sparse_heading_is_supplemented_by_bs4_body` -> RED failure before implementation, OK after implementation.
  - `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1.Pdf2CleanHtmlFallbackTests.test_bs4_fallback_preserves_dom_order_for_mixed_heading_and_div_body` -> RED failure before DOM-order fix, OK after implementation.
  - `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1.Pdf2CleanHtmlFallbackTests.test_trafilatura_supplement_uses_bs4_dom_order_when_trafilatura_keeps_later_heading` -> RED failure before supplement-order fix, OK after implementation.
  - `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1` -> 5 tests OK. Note: environment emitted existing urllib3 LibreSSL warning.
  - `PYTHONPATH=. python3 -m py_compile src/stage1_cleaning/pdf2clean.py tests/test_pdf2clean_html_fallback_v1.py` -> passed.
  - Read-only final quality re-review -> APPROVED.
- Status: PASS, but uncommitted.
- Commit: none. Reason: commit scope ambiguous because repo had heavy pre-existing dirty WIP and same task files were already modified/untracked before this tick.
- Next pending task: P3.1 — add preparation-core sufficiency floor to Stage2 selector candidate/evidence package selection.

### 2026-05-05 15:33:13 EDT — P3.1 preparation-core selector floor

- Task attempted: P3.1 — generic preparation-core sufficiency floor for Stage2 selector candidate/evidence package selection.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: heavy pre-existing tracked and untracked WIP remained present before/through this tick, including `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` already dirty with unrelated table-structure/coordinate-preservation changes. Commit scope remained ambiguous.
- Files touched this tick:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - `tests/test_stage2_preparation_core_selector_floor_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
- TDD RED:
  - Command: `PYTHONPATH=. python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1` -> initial expected failure: `AssertionError: 'method-core' not found in {'method-overview'}`.
  - After reviewer-identified gaps, added focused tests for caption locator, local action/material/value binding, missing source locator, and text-marked caption/TOC markers. RED observed for caption locator and missing/text-marked source guards before tightening the implementation.
- Implementation:
  - Added generic preparation-core value/material/action/source-body helper surface in Stage2 selector code.
  - Added local sentence-level preparation-core binding: source span must co-locate a preparation action, a PLGA-nanoparticle-domain material cue, and a typed numeric/unit value.
  - Added source-body guards requiring a concrete source locator and rejecting caption/figure/TOC/reference locators or text prefixes.
  - Extended `apply_minimal_evidence_floor()` to add a single best source-backed preparation-core method candidate when selected methods lack such a core candidate, recording `minimal_evidence_floor_added_preparation_core`, `floor_added_preparation_core`, and `added_source_backed_preparation_core` metadata.
  - Removed a reviewer-flagged paper-specific payload literal from the core material cues; no paper-key branches, GT-derived snippets, raw PDF/XML supplementation, or S5-3 changes were made by this tick.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1` -> 6 tests OK after implementation. Existing environment warnings from urllib3/Google packages/generative-ai deprecation emitted.
  - `PYTHONPATH=. python3 -m unittest tests.test_preparation_evidence_sufficiency_audit_v1` -> 4 tests OK.
  - `python3 -m py_compile src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py tests/test_stage2_preparation_core_selector_floor_v1.py` -> passed.
  - Read-only subagent spec/quality reviews were run. First review requested changes for caption/source/genericity issues; final quality review approved. Remaining spec-smoke concerns were addressed with additional failing tests and implementation guards before final verification.
- Data-source boundary: no benchmark/audit/workbook comparison consuming `data/results/` was run; no benchmark-valid final-output claim is made.
- Status: PASS, but uncommitted.
- Commit: none. Reason: commit scope ambiguous because repo had heavy pre-existing dirty WIP and same Stage2 file included unrelated pre-existing table changes; P3.1 test file was untracked in the dirty workspace.
- Next pending task: P4.1 — add table-role negative taxonomy and source-role separation.

### 2026-05-05 16:07:40 EDT — P4.1 table-role negative taxonomy and source-role separation

- Task attempted: P4.1 — generic table-role negative taxonomy and source-role separation for Stage2 selector/table payload authority.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: heavy pre-existing tracked and untracked WIP remained before this tick, including `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` already dirty with unrelated prior changes and `tests/test_stage2_preparation_core_selector_floor_v1.py` already untracked. Commit scope remained ambiguous.
- Files touched this tick:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - `tests/test_stage2_preparation_core_selector_floor_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
- TDD RED:
  - `PYTHONPATH=. python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1` initially failed with `ImportError: cannot import name 'classify_table_source_role'` before the taxonomy implementation.
  - Reviewer-regression RED after initial implementation: compact composition headers plus characterization columns failed as `characterization_result_table` instead of `formulation_composition_table`; execution/payload release-profile table failed as `mixed_table` instead of `non_formulation_table`.
  - Final reviewer-regression RED: targeting/intravenous administration result table failed as `formulation_composition_table` instead of `noise_or_nonformulation_table`.
- Implementation:
  - Added generic `classify_table_source_role()` taxonomy with `formulation_composition_table`, `preparation_parameter_table`, `characterization_result_table`, `release_profile_table`, `pharmacokinetic_table`, `tissue_distribution_table`, `reference_spillover_table`, and `noise_or_nonformulation_table`.
  - Added negative-family separation so pharmacokinetic/release/tissue/reference/noise tables are optional/non-formulation unless generic strong composition evidence exists.
  - Added compact composition override for common headers such as `PLGA (mg)`, `Drug (mg)`, and `PVA (%)` with characterization columns.
  - Integrated source-role checks into selector table inclusion, execution table type classification, and payload inclusion; persisted `table_source_role` in normalized table payloads and had payload guardrails consume the persisted role when present.
  - Added generic targeting/intravenous/cell-uptake/cytotoxicity negative cues to prevent result-only administration/biology tables from becoming formulation authority.
  - No new paper-specific runtime branches, GT-derived snippets, raw PDF/XML supplementation, or S5-3 changes were made by this tick. One read-only reviewer flagged a pre-existing `WFDTQ4VX` runtime branch elsewhere in the already-dirty Stage2 file; this tick did not create it and did not attempt broad removal under the one-task boundary.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1` -> 12 tests OK after implementation. Existing environment warnings from urllib3/Google packages/generative-ai deprecation emitted.
  - `PYTHONPATH=. python3 -m unittest tests.test_preparation_evidence_sufficiency_audit_v1 tests.test_stage2_preparation_core_selector_floor_v1` -> 16 tests OK. Same existing warnings emitted.
  - `PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py tests/test_stage2_preparation_core_selector_floor_v1.py` -> passed.
  - Read-only subagent spec/quality reviews were run. Initial reviews requested changes for payload-path drift, compact header false negatives, and targeting/intravenous negative cues; focused regression tests were added and passed after fixes.
- Data-source boundary: no benchmark/audit/workbook comparison consuming `data/results/` was run; no benchmark-valid final-output claim is made.
- Status: PASS, but uncommitted.
- Commit: none. Reason: commit scope ambiguous because repo had heavy pre-existing dirty WIP and same Stage2/test files contained pre-existing unrelated/untracked WIP.
- Next pending task: P5.1 — implement generic material-value binding helper surface.

### 2026-05-05 16:21:31 EDT — P5.1 generic material-value binding helper surface

- Task attempted: P5.1 — create a reusable, side-effect-free Stage5 helper surface for direct type validation, paper-local alias graph resolution, entity-bound value extraction, scope inference, and canonical promotion review/proposals.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: broad pre-existing tracked and untracked WIP remained before/through this tick, including governance/docs/src/test modifications and untracked root `.docx`/markdown files. The new P5 helper/test files were untracked in this workspace during verification, so commit scope remained ambiguous.
- Files touched this tick:
  - `src/stage5_benchmark/material_value_binding_v1.py`
  - `tests/test_material_value_binding_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-repair-plan.md`
- TDD RED:
  - `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1` initially failed with `ModuleNotFoundError: No module named 'src.stage5_benchmark.material_value_binding_v1'` before helper creation.
  - Reviewer-regression RED after initial helper implementation: importing `evaluate_canonical_promotions` failed before the conflict-aware promotion review API existed.
  - Final reviewer-regression RED: `typed_row_local_assignment` promoted to both `row-1` and `row-2`, and row-local missing-target rejection was absent before the row-local safety fix.
- Implementation:
  - Added `material_value_binding_v1.py` with side-effect-free helpers for numeric direct-value validation, source/row-hint material alias graph construction, preparation-context entity-bound mass/concentration extraction, negative-context filtering, scope inference, canonical field mapping, promotion proposal generation, and auditable promotion evaluation.
  - Added conflict-aware shared-scope review: conflicting shared values for the same canonical field/scope are rejected with `conflicting_shared_values_for_field` instead of donor-filling admitted rows.
  - Added row-local safety: `row_local` and `typed_row_local_assignment` require a target admitted row and only propose to that row; missing or unmatched targets produce auditable rejections.
  - No paper-key branches, DEV15 maps, GT lookup, data/results reads, raw PDF/XML/HTML consumption, LLM calls, row mutation, row creation, or S5-3 expansion were added.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1` -> 9 tests OK after implementation and reviewer fixes.
  - `PYTHONPATH=. python3 -m py_compile src/stage5_benchmark/material_value_binding_v1.py tests/test_material_value_binding_v1.py` -> passed.
  - Read-only spec review -> PASS.
  - Read-only quality/genericity review initially requested row-local/typed-row-local safety changes; after focused RED/GREEN fix, re-review -> APPROVED.
- Data-source boundary: no benchmark/audit/workbook comparison consuming `data/results/` was run; no benchmark-valid final-output claim is made.
- Status: PASS, but uncommitted.
- Commit: none. Reason: broad pre-existing dirty WIP and same-scope untracked files made commit scope ambiguous.
- Next pending task: P6.1 — integrate generic S5-2 shared preparation / admitted-row carrythrough.

### 2026-05-05 16:43:06 EDT — P6.1 generic S5-2 material-value carrythrough

- Task attempted: P6.1 — integrate generic material-value binding helper surface into Stage5 S5-2 shared preparation/admitted-row carrythrough.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: broad pre-existing tracked and untracked WIP remained before/through this tick, including `src/stage5_benchmark/build_minimal_final_output_v1.py`, `tests/test_compare_layer3_values_v1.py`, and untracked `src/stage5_benchmark/material_value_binding_v1.py` already present in the workspace. Commit scope remained ambiguous.
- Files touched this tick:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `tests/test_compare_layer3_values_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-repair-plan.md`
- TDD RED:
  - `PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests.test_stage5_uses_material_alias_binding_for_method_shared_direct_masses` initially failed with `AssertionError: 'drug_feed_amount_text' not found in {'organic_solvent', 'plga_mass_mg'}` before Stage5 consumed the material-value binding helper.
- Implementation:
  - Imported the side-effect-free `material_value_binding_v1` helper surface into `build_minimal_final_output_v1.py`.
  - Added `extract_material_value_binding_shared_masses()` adapter to build a paper-local alias graph from the same clean/source text plus row hints, extract direct entity-bound values, run admitted-row promotion review, and map generic `drug_mass_mg`/`polymer_mass_mg` proposals to existing Stage5 direct fields without row creation or overwriting populated row-local bundles.
  - Integrated the adapter as a fallback inside `apply_global_preparation_material_carrythrough()` after existing higher-authority direct shared-mass extraction and before scoped subtype fallback. Values produced by the adapter are tagged with `material_value_binding_direct_text` evidence provenance.
  - No paper-key branches, GT lookup, raw PDF/XML/HTML reads, live LLM calls, S5-3 use, row creation, or derived calculations were added.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests.test_stage5_uses_material_alias_binding_for_method_shared_direct_masses` -> RED before implementation, OK after implementation.
  - `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests` -> 53 tests OK. Existing environment warnings from urllib3/Google packages/generative-ai deprecation emitted.
  - `PYTHONPATH=. python3 -m py_compile src/stage5_benchmark/build_minimal_final_output_v1.py src/stage5_benchmark/material_value_binding_v1.py tests/test_compare_layer3_values_v1.py tests/test_material_value_binding_v1.py` -> passed.
  - Post-review refinement added a polymer provenance assertion; reran targeted unittest and py_compile -> passed.
  - Read-only spec review -> PASS. Read-only quality/genericity review -> APPROVED, with only minor optional notes addressed for polymer provenance coverage.
- Data-source boundary: no benchmark/audit/workbook comparison consuming `data/results/` was run; no benchmark-valid final-output claim is made.
- Status: PASS, but uncommitted.
- Commit: none. Reason: broad pre-existing dirty WIP and same-scope Stage5/test files made commit scope ambiguous.
- Next pending task: P7.1 — enforce type validators at Stage2 compatibility and Stage5 final boundaries.


### 2026-05-05 in-session — P7.1 Stage2/Stage5 typed validator boundaries

- Task attempted: P7.1 — enforce type validators at Stage2 compatibility and Stage5 final boundaries.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: broad pre-existing tracked and untracked WIP remained; commit scope remains ambiguous.
- Files touched this task:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `tests/test_compare_layer3_values_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
- TDD RED:
  - Added `test_stage5_final_boundary_blanks_invalid_volume_before_lawful_carrythrough`; it failed before Stage5 final-boundary typed validation because identity strings in volume fields made the fields look populated and blocked lawful clean-text carrythrough.
- Implementation:
  - Reused `validate_direct_value()` from the generic material-value binding helper in Stage5 final-output materialization.
  - Added `blank_invalid_final_typed_fields()` covering direct mass, volume, and concentration-like final bundles; invalid identity/wrong-unit values are blanked and annotated with typed invalid reasons before and after carrythrough.
  - Kept Stage2 compatibility validator intact; existing Stage2 tests continue rejecting identity strings, ratio-only, concentration-only, and invalid compressed segments in direct mass bundles.
  - Preserved pure numeric final values because final-field/header context can carry canonical units in row-local table bindings.
  - No paper-key branches, GT lookup, raw PDF/XML/HTML supplementation, row creation, live LLM calls, or S5-3 use were added.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests.test_stage5_final_boundary_blanks_invalid_volume_before_lawful_carrythrough` -> RED before implementation, OK after implementation.
  - `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests` -> 57 tests OK after preserving numeric final table bindings.
  - `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests tests.test_stage2_preparation_core_selector_floor_v1 tests.test_preparation_evidence_sufficiency_audit_v1` -> 73 tests OK. Existing environment warnings from urllib3/Google packages/generative-ai deprecation emitted.
  - `PYTHONPATH=. python3 -m py_compile src/stage5_benchmark/build_minimal_final_output_v1.py src/stage5_benchmark/material_value_binding_v1.py src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py tests/test_compare_layer3_values_v1.py tests/test_material_value_binding_v1.py` -> passed.
- Data-source boundary: no benchmark/audit/workbook comparison consuming `data/results/` was run; no benchmark-valid final-output claim is made.
- Status: PASS, but uncommitted.
- Commit: none. Reason: broad pre-existing dirty WIP and same-scope Stage5/test files made commit scope ambiguous.
- Next pending task: P8.1 — harden direct vs derived provenance separation.


### 2026-05-05 in-session — P8.1 direct vs derived provenance hardening

- Task attempted: P8.1 — harden direct vs derived provenance separation.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: broad pre-existing tracked and untracked WIP remained; commit scope remains ambiguous.
- Files touched this task:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `tests/test_compare_layer3_values_v1.py`
  - `docs/plans/2026-05-05-preparation-evidence-selector-and-typed-materialization-progress-ledger.md`
- TDD RED:
  - Extended `test_derived_mass_provenance_does_not_write_direct_mass_fields` to require explicit `direct_or_derived=derived`, `direct_field_write_allowed=no`, and `evidence_binding_status=derived_without_direct_text`; it failed before implementation with missing provenance keys.
  - Added `test_concentration_times_volume_stays_in_derived_sidecar_not_direct_mass`; it verified `mg/mL × mL` creates a derived sidecar/provenance candidate while leaving the direct mass field blank, and failed before the sidecar metadata implementation.
- Implementation:
  - Added `derived_mass_provenance_record()` helper to standardize derived-mass sidecar/provenance rows.
  - Ratio-derived and concentration×volume-derived mass candidates now explicitly carry `direct_or_derived=derived`, `direct_field_write_allowed=no`, and `evidence_binding_status=derived_without_direct_text`.
  - Derived values remain in `derived_mass_provenance_json`; direct canonical mass fields are not written by derived calculations.
  - Added an extra Stage5 validation pass after row-local table-cell binding so invalid identity strings from row-local numeric cells are blanked before lawful direct clean-text carrythrough.
  - Preserved existing direct row-local table numeric behavior: pure numeric values remain valid when the canonical final-field/header context supplies the unit.
  - No paper-key branches, GT lookup, raw PDF/XML/HTML supplementation, row creation, live LLM calls, or S5-3 use were added.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1.UniversalTableCellGridTests.test_derived_mass_provenance_does_not_write_direct_mass_fields tests.test_compare_layer3_values_v1.UniversalTableCellGridTests.test_concentration_times_volume_stays_in_derived_sidecar_not_direct_mass` -> RED before implementation, OK after implementation.
  - `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_compare_layer3_values_v1.UniversalTableCellGridTests tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests tests.test_stage2_preparation_core_selector_floor_v1 tests.test_preparation_evidence_sufficiency_audit_v1` -> 119 tests OK. Existing environment warnings from urllib3/Google packages/generative-ai deprecation emitted.
  - `PYTHONPATH=. python3 -m py_compile src/stage5_benchmark/build_minimal_final_output_v1.py src/stage5_benchmark/material_value_binding_v1.py tests/test_compare_layer3_values_v1.py` -> passed.
- Data-source boundary: no benchmark/audit/workbook comparison consuming `data/results/` was run; no benchmark-valid final-output claim is made.
- Status: PASS, but uncommitted.
- Commit: none. Reason: broad pre-existing dirty WIP and same-scope Stage5/test files made commit scope ambiguous.
- Next pending task: P9.1 — bounded diagnostic replay validation with lineage/legal boundary checks.


### 2026-05-05 in-session — P9.1 bounded diagnostic replay validation

- Task attempted: P9.1 — bounded diagnostic replay validation with lineage/legal boundary checks.
- Baseline / branch:
  - Branch: `feature/ee-coverage-rl`.
  - Baseline dirty state: broad pre-existing tracked and untracked WIP remained; commit scope remains ambiguous.
- Lawful lineage used for accepted P9 validation:
  - Stage2 artifact: `data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`.
  - Stage3 artifacts: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv` and `resolved_relation_fields_v1.tsv`.
  - Stage5 final table: `data/results/20260504_ab9f61e/049_stage5_p9_polymer_mgml_mass_guard_bounded_diagnostic/final_formulation_table_v1.tsv`.
  - Layer3 compare: `data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/`.
  - GT authority: `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`.
  - Scope manifest: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`.
- TDD / regression audit:
  - Initial P9 compare `041` improved core recall but raised a regression signal: new `extra_in_system` vs baseline `032` was `polymer_mass_mg +26`.
  - Root cause: row-local table binding interpreted concentration headers such as `PLGA mg/mL` as direct polymer mass.
  - Added generic header guard coverage in `tests/test_compare_layer3_values_v1.py` and generic code changes in `src/stage2_sampling_labels/table_structure_dictionary_v1.py` / `src/stage5_benchmark/build_minimal_final_output_v1.py`; no paper-key branch or GT lookup was added.
  - Exploratory directories `040`-`048` are documented as diagnostic/superseded; final accepted P9 diagnostic artifacts are `049` and `050`.
- Final diagnostic result:
  - Stage5 row count unchanged: 202 final rows.
  - Baseline `032` core fixed fields: system_on_gt=2330/3119, recall=0.747034, extra=34.
  - Final `050` core fixed fields: system_on_gt=2336/3119, recall=0.748958, extra=34.
  - New extra vs `032`: none.
  - Retained improvement vs `032`: `drug_mass_mg` missing->present_and_match for 6 cells with typed direct mass evidence.
  - `polymer_mass_mg` extra remains 0 in the final core summary.
- Verification commands:
  - `PYTHONPATH=. python3 -m unittest tests.test_compare_layer3_values_v1.UniversalTableCellGridTests.test_field_header_alias_lexicon_maps_common_measurement_headers tests.test_compare_layer3_values_v1.UniversalTableCellGridTests.test_table_cell_grid_rejects_mg_per_ml_polymer_header_as_direct_mass` -> 2 tests OK.
  - `PYTHONPATH=. python3 -m unittest tests.test_table_structure_dictionary_v1 tests.test_compare_layer3_values_v1.UniversalTableCellGridTests tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests tests.test_material_value_binding_v1 tests.test_stage2_preparation_core_selector_floor_v1 tests.test_preparation_evidence_sufficiency_audit_v1 tests.test_pdf2clean_html_fallback_v1` -> 136 tests OK. Existing environment warnings from urllib3/Google packages/generative-ai deprecation emitted.
  - `PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/table_structure_dictionary_v1.py src/stage5_benchmark/build_minimal_final_output_v1.py tests/test_compare_layer3_values_v1.py` -> passed.
- Data-source boundary: P9 is `benchmark_valid=no` and diagnostic-only. It uses a lawful Stage5 final table for compare, but it is a bounded replay and not a full benchmark-valid final output claim.
- Status: PASS, but uncommitted.
- Commit: none. Reason: broad pre-existing dirty WIP and same-scope Stage5/test/data-result modifications made commit scope ambiguous.
- Next pending task: none for this 9-phase plan; plan execution is complete at diagnostic scope.
