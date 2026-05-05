# Generic Preparation Evidence, Material-Value Binding, and Scoped Carrythrough Repair Plan

> **For Hermes:** Use subagent-driven-development only after this plan is accepted. Keep all runtime repairs generic; concrete papers, fields, and GT misses are validation cases only.

**Goal:** Improve generic Stage2/S3/S5-2 capabilities for preparation evidence selection, clean-text source-quality auditing, paper-local material alias resolution, entity-bound value extraction, scope-aware carrythrough, canonical field promotion, direct/derived provenance separation, and typed validation without paper-specific runtime rules.

**Architecture:** Clean text is the only Stage2 source text. If cleaned text lacks a method body or preparation-core sentence, fix the text conversion / cleaned-text extraction layer upstream; downstream stages must not search original PDFs, raw HTML/XML, protocol excerpts, GT notes, or user-confirmed snippets as alternate evidence. Stage2 selects and packages evidence from cleaned text; S3/S5-2 resolve material/value relations and carry source-backed direct values only inside lawful admitted-row scopes; S5-3 remains residual source-evidenced gap fill and must not compensate for missing S5-2/S3 generic materialization capability.

**Tech Stack:** Python 3 standard library, existing PLGA Stage2/Stage3/Stage5 TSV/JSON artifacts, `unittest`, diagnostic sidecars under `data/results/`, and maintained entrypoints from `project/ACTIVE_PIPELINE_RUNBOOK.md`.

---

## 0. Non-Negotiable Scope Rules

1. **No paper-specific runtime branches.**
   - Do not implement `if paper_key == ...`, paper-specific string maps, DEV15 lookup tables, or fixed expected-count logic.
   - Concrete papers such as PA3SPZ28, RHMJWZX8, and YGA8VQKU may appear only in tests, audits, and bounded validation reports.

2. **No field-only repair framing.**
   - `polymer_mass_mg`, `drug_mass_mg`, phase volumes, surfactant concentration, or a GT miss are symptoms.
   - The feature is generic material/entity/value binding with lawful scope inference and canonical promotion.

3. **Clean text is the only Stage2 source text.**
   - Allowed: governed cleaned text artifacts, candidate blocks derived from them, evidence blocks derived from them, and normalized table payloads already produced by the cleaned-data pipeline.
   - Not allowed downstream: original PDFs, raw HTML/XML outside cleaned artifacts, source-completed excerpts in docs, GT protocol excerpts, or user-provided snippets as replacement evidence.
   - If known source text is missing from cleaned text, the first repair target is text conversion / cleaned-text extraction, not selector bypass.

4. **No semantic-authority drift.**
   - Stage2 LLM remains semantic formulation-discovery authority.
   - Deterministic code may normalize, bind, validate, select, carry, and materialize only after lawful Stage2 handoff/admission.

5. **S5-3 is not a generic materialization backstop.**
   - S5-3 should not fill S5-2/S3-owned generic capability gaps such as direct row-local table cells, method-shared preparation masses, paper-local abbreviation binding, or typed carrythrough.
   - S5-3 target scope should shrink only because S5-2/S3 generic direct-value capability improves, not because blank slots are treated as database completion tasks.

6. **Direct vs derived separation.**
   - Direct source-backed values may fill direct fields.
   - Derived calculations (`mg/mL × mL`, ratios, formulas) must stay in derived provenance sidecars and must not contaminate direct Layer3 compare.

7. **Diagnostic-only until full lineage alignment.**
   - New audits and bounded replays are `benchmark_valid=no` unless the complete intended final pipeline is run and compared under the active data-source contract.

---

## 1. Allowed Generic Capability Model

Runtime repairs must be expressed through these abstractions:

1. **Paper-local material alias graphs**
   - Map full names, abbreviations, formulation-label tokens, table labels, and row hints to material entities.
   - Roles include `drug`, `polymer`, `surfactant`, `stabilizer`, `solvent`, `helper_material`, and `unknown`.

2. **Abbreviation / full-name / entity-role resolution**
   - Resolve `Full Term (ABBR)` and loaded-label abbreviations only when supported by cleaned text and/or existing row hints.
   - Unknown uppercase tokens remain unknown.

3. **Entity-bound value extraction**
   - Bind value expressions to material entities: `X mg of material`, `material (X mg)`, concentration expressions, phase volumes, etc.
   - Carry raw span/context and extraction reason.

4. **Row / table / method / footnote scope inference**
   - Classify candidate values as row-local, typed row-local assignment, table-footnote shared, table-scoped constant, method-paragraph shared, paper-global unique constant, or ambiguous/conflicted.

5. **Admitted-row carrythrough**
   - Carry only into rows already admitted by lawful Stage2/Stage5 row-universe logic.
   - Respect blank/control/helper eligibility and do not create rows.

6. **Canonical field promotion**
   - Promote entity role + value type into canonical fields only after scope and type validation.
   - Use dictionary/header/role mappings; do not use paper-specific field hacks.

7. **Type validators**
   - Prevent identity strings from entering numeric/mass/volume/concentration fields.
   - Invalid values become blank plus invalid-reason diagnostics and must not block lawful higher-quality carrythrough.

8. **Evidence/table source-quality and role audits**
   - Diagnose whether failures start in cleaned text extraction, candidate segmentation, evidence selection, table role classification, relation resolution, or final materialization.

---

## 2. Diagnostic Anchors, Not Runtime Rules

The following are validation anchors only:

- **Shared preparation scope gate validation:** a method sentence like `organic solution of PLGA (X mg) and drug (Y mg) in solvent was added to aqueous phase` should be admitted by generic preparation-action/entity/value/scope cues.
- **Preparation-core selector validation:** a method paragraph containing polymer/drug masses, organic/aqueous phase volumes, surfactant concentration, stirring conditions, and preparation method should outrank figure/result captions and nonformulation result tables.
- **Cleaned-text source-quality validation:** if active cleaned text preserves a method heading but loses the body, the first failure boundary is cleaned-text extraction.
- **Typed validation anchor:** an identity token such as `PLGA` or a drug abbreviation cannot survive as a mass value.

These anchors must not appear as runtime paper-key or literal-string exceptions.

---

## 3. Implementation Phases

### Phase 1: Generic source/evidence/materialization boundary audit

**Objective:** Add a diagnostic-only audit that localizes the first failure boundary without changing selector or materialization behavior.

**Create:**
- `src/stage2_sampling_labels/audit_preparation_evidence_sufficiency_v1.py`
- `tests/test_preparation_evidence_sufficiency_audit_v1.py`

**Inputs:**
- candidate block artifact(s): `candidate_blocks_v1.json`
- selected evidence artifact(s): `evidence_blocks_v1.json`
- cleaned text path resolved from candidate/evidence metadata
- optional normalized table payloads and table-selection debug artifacts

**Do not consume:** raw PDFs, raw HTML/XML outside cleaned artifacts, GT protocol excerpts, user snippets, or docs/methods source-completed excerpts as evidence.

**Audit outputs:**
- `preparation_evidence_sufficiency_audit_v1.tsv`
- `preparation_evidence_sufficiency_summary_v1.json`
- `RUN_CONTEXT.md`

**Audit columns:**

```text
paper_key
cleaned_text_path
cleaned_text_has_method_heading
cleaned_text_has_method_body_after_heading
cleaned_text_has_preparation_core
candidate_has_preparation_core
selected_has_preparation_core
preparation_core_candidate_block_ids
preparation_core_selected_block_ids
selected_method_block_ids
table_blocks_selected_count
table_noise_selected_count
pharmacokinetic_table_selected_as_formulation_count
release_or_tissue_table_selected_as_formulation_count
source_quality_status
evidence_selection_status
first_failure_boundary
abstraction_compliance_status
notes
```

**Status taxonomy:**

```text
pass
cleaned_text_missing_method_body
cleaned_text_missing_preparation_core
candidate_segmentation_missing_preparation_core
evidence_selection_missing_preparation_core
table_selector_noise_overselected
table_role_misclassified
materialization_or_carrythrough_boundary
```

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_preparation_evidence_sufficiency_audit_v1
PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/audit_preparation_evidence_sufficiency_v1.py
```

---

### Phase 2: Repair cleaned-text extraction when source-quality audit fails

**Objective:** If the audit reports method-heading/body loss or missing preparation core in cleaned text, fix upstream text conversion / cleaned-text extraction.

**Inspect:**
- maintained cleaned-text generation scripts registered in `docs/maintained_script_surface.tsv`
- active cleaned text artifacts under `data/cleaned/content/text/`
- Stage2 manifest fields resolving cleaned text paths

**Do not fix by:**
- injecting source-completed excerpts into Stage2 evidence
- reading original files from Stage5
- adding paper-key exceptions
- copying user-confirmed text into final materialization

**Acceptance:** cleaned text contains the missing method body; candidate segmentation can see it; selector can select it under Phase 3.

---

### Phase 3: Add preparation-core sufficiency floor to Stage2 selector

**Objective:** Ensure selected evidence includes at least one generic preparation-core block when candidate blocks contain source-backed preparation parameters.

**Generic signals:**
- preparation actions: prepared, dissolved, weighed, mixed, poured, added, dropwise added, emulsified, stirred, evaporated, centrifuged
- preparation contexts: organic phase, aqueous phase, solvent diffusion, nanoprecipitation, emulsion, solvent displacement, formulation
- entity/value patterns: material alias + mass/concentration/volume unit, phase volumes, surfactant/stabilizer concentrations, time/speed/temp conditions

**Behavior:**
- score candidate blocks for generic preparation-core density;
- promote the best source-backed preparation-core block if selected evidence lacks one;
- include adjacent method context when the core sentence is split;
- demote lower-value noisy tables to preserve token budget;
- use entity-role/scope features, not paper-specific strings.

---

### Phase 4: Add table-role negative taxonomy and source-role separation

**Objective:** Prevent pharmacokinetic, tissue-distribution, release-profile, reference-spillover, and result-only tables from being labeled/prioritized as formulation-composition sources.

**Generic negative cues:** AUC, Cmax, Tmax, MRT, t1/2, Te, pharmacokinetic(s), tissue, targeting parameter, organ distribution, intravenous administration, release profile, cell uptake, cytotoxicity, references, bibliography.

**Override requires strong composition evidence:** formulation row labels plus composition/process columns such as polymer, drug amount, surfactant/stabilizer, phase volumes, or preparation variables.

**Role taxonomy:**

```text
formulation_composition_table
preparation_parameter_table
characterization_result_table
release_profile_table
pharmacokinetic_table
tissue_distribution_table
reference_spillover_table
noise_or_nonformulation_table
```

**Boundary:** table preservation is not table prompt priority and is not formulation-universe authority.

---

### Phase 5: Implement generic material-value binding helper surface

**Objective:** Create reusable helpers for type validation, material alias graph, entity-bound extraction, scope inference, and promotion proposals before integrating with Stage5.

**Preferred file:** `src/stage5_benchmark/material_value_binding_v1.py` if reuse is clearer than adding file-local helpers.

**Tests:** `tests/test_material_value_binding_v1.py`

**Capabilities:**
- direct mass validator rejects identity-only text and concentration-only text for mass fields;
- paper-local alias graph resolves source-backed abbreviations/full names/roles;
- entity-bound extraction binds values to material entities in preparation-positive contexts;
- negative contexts reject release/animal-dose/cell/pharmacokinetic/sample-prep masses;
- scope inference classifies row/table/method/footnote/paper-global/ambiguous scopes;
- promotion proposals map role + type + scope to canonical fields without mutating rows.

---

### Phase 6: Integrate generic S5-2 shared preparation / admitted-row carrythrough

**Objective:** Use the helper surface inside `build_minimal_final_output_v1.py` to materialize direct source-backed values for admitted rows only.

**Rules:**
- build paper-local alias graph from cleaned text, existing Stage2/Stage3 row hints, table headers/footnotes, and relation artifacts;
- extract entity-bound values from lawful row/table/method/footnote scopes;
- apply authority ladder: row-local table cell > typed row-local assignment > table-footnote/table-scoped > method-shared > paper-global unique > ambiguous no-promotion;
- never overwrite higher-authority row-local direct values;
- record accepted/rejected candidates in decision trace or sidecar;
- S5-3 target scope must not expand to fill these generic gaps.

---

### Phase 7: Enforce type validators at Stage2 compatibility and Stage5 final boundaries

**Objective:** Stop invalid identity values from reaching final numeric fields and preserve invalid-reason diagnostics.

**Behavior:**
- mass fields require numeric + mass unit;
- volume fields require numeric + volume unit;
- concentration fields require numeric concentration/unit;
- invalid values become blank plus reason, e.g. `invalid_mass_no_numeric_value`;
- invalid values must not block later lawful direct carrythrough.

---

### Phase 8: Direct vs derived provenance hardening

**Objective:** Keep direct source-reported values separate from calculated values.

**Direct examples:** `X mg of polymer`, `polymer (X mg)`, typed table cell under `polymer mass (mg)`.

**Derived examples:** concentration × volume, ratio-derived amount, formula-derived mass/loading.

**Behavior:** direct values may fill direct canonical fields; derived values go to derived sidecars / provenance JSON and Evidence Binding statuses such as `derived_supported` or `derived_without_direct_text`.

---

### Phase 9: Bounded replay validation

**Objective:** Validate using concrete papers only as bounded cases, with no live LLM calls unless S2-4a/S2-4b changed and user approves.

**Minimum validation categories:**
- cleaned-text method-body source-quality failure;
- preparation-core selector miss;
- noisy table over-selection / role misclassification;
- shared method preparation mass carrythrough;
- invalid identity-in-mass typed guard;
- abbreviation/full-name material alias;
- negative downstream dose/release/cell/pharmacokinetic controls.

**Metrics:**

```text
cleaned_text_has_preparation_core
candidate_has_preparation_core
selected_has_preparation_core
table_noise_selected_count
pharmacokinetic_table_selected_as_formulation_count
invalid_mass_no_numeric_value_count
direct vs derived provenance counts
canonical promotion proposal counts
S5-3 target count for S5-2/S3-owned fields
Stage5 final row count delta
paper-key runtime branch count = 0
```

**Replay rule:** DEV15-style replays must explicitly set DOE execution per runbook:

```bash
STAGE2_DOE_ENUMERATION_MODE=explicit_only
STAGE2_ENABLE_NUMBERED_DOE_RECOVERY=1
```

---

## 4. Acceptance Criteria

1. No paper-specific runtime branches, fixed DEV15 value maps, or literal paper-string rules.
2. Concrete papers/fields/GT misses are validation cases only.
3. Clean text remains the only Stage2 evidence source; missing bodies are fixed upstream.
4. Selector repairs are based on generic preparation-core and table-role features.
5. Material alias graphs resolve full names, abbreviations, formulation labels, and roles only when source-backed.
6. Entity-bound values carry source span/context and negative-context guards.
7. Scope inference prevents uncontrolled paper-global donor fills.
8. Canonical promotion writes direct values only when role, value type, row admission, and scope are valid.
9. Type validators prevent identity strings from entering mass/volume/concentration fields.
10. Direct and derived values remain separated.
11. S5-3 does not expand to cover S5-2/S3 generic materialization gaps.
12. Bounded diagnostics record exact lineage and remain `benchmark_valid=no`.

---

## 5. Immediate Execution Order

1. Implement Phase 1 audit as diagnostic-only.
2. Run Phase 1 audit tests and py_compile.
3. Run the audit on the explicit current diagnostic Stage2 lineage for selected validation anchors.
4. Use `first_failure_boundary` to choose the first behavior-changing repair: text extraction, selector sufficiency, table-role scoring, material-value binding, or Stage5 carrythrough.


---

## 6. Execution Log

### 2026-05-05 Phase 1 refinement + Phase 2 generic HTML fallback start

- Refined `src/stage2_sampling_labels/audit_preparation_evidence_sufficiency_v1.py` so `cleaned_text_has_preparation_core` requires action/material/unit co-locality in the same local prose window, not document-wide matches across TOC, table captions, figure captions, or references.
- Added `tests/test_preparation_evidence_sufficiency_audit_v1.py::test_toc_and_table_captions_do_not_satisfy_cleaned_text_preparation_body`.
- Re-ran bounded diagnostic audit to `data/results/20260504_ab9f61e/039_preparation_evidence_sufficiency_audit_refined_diagnostic/`.
- Refined first-failure distribution: `cleaned_text_missing_preparation_core=1`, `table_selector_noise_overselected=1`, `materialization_or_carrythrough_boundary=1`.
- Started Phase 2 with a generic Stage1 HTML fallback repair in `src/stage1_cleaning/pdf2clean.py`: BeautifulSoup fallback now recovers content-bearing `article`/`main`/`section`/`div` prose blocks when publisher HTML stores article paragraphs outside `<p>` tags, while skipping navigation/header/footer/sidebar/chrome zones.
- Added `tests/test_pdf2clean_html_fallback_v1.py` for content-bearing div recovery without paper-specific strings.
- Verification: `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1 tests.test_preparation_evidence_sufficiency_audit_v1` -> 5 tests OK; py_compile passed for touched scripts/tests.
- Boundary note: the refined audit remains diagnostic-only (`benchmark_valid=no`). The Phase 2 code repair is generic; the current repo lacks accessible original raw HTML for `YGA8VQKU`, so no cleaned text artifact was overwritten in this step.

### 2026-05-05 Phase 2 quality-gated BS4 supplement

- Completed bounded P2.2 repair: `extract_text_from_html()` now uses a generic source-quality gate for successful-but-sparse trafilatura HTML extraction and supplements from BeautifulSoup blocks from the same HTML source when method/preparation-like headings lack enough paragraph/list body prose.
- Added generic helper functions in `src/stage1_cleaning/pdf2clean.py` for source-quality gating, DOM-order BS4 block recovery, nested-container suppression, and exact-text deduplicated supplement merging. No paper keys, GT snippets, expected-count logic, or downstream raw source supplementation were added.
- Added TDD coverage in `tests/test_pdf2clean_html_fallback_v1.py` proving successful sparse trafilatura output is supplemented with content-bearing body prose, navigation chrome remains excluded, nested div parent text is not amplified, BS4 fallback preserves mixed heading/div DOM order, and trafilatura+BS4 supplement preserves source DOM order when trafilatura retained a later heading.
- Verification: targeted RED failures were observed before implementation for sparse trafilatura supplementation, BS4 mixed-tag DOM order, and trafilatura+BS4 supplement DOM order; after implementation `PYTHONPATH=. python3 -m unittest tests.test_pdf2clean_html_fallback_v1` -> 5 tests OK; `PYTHONPATH=. python3 -m py_compile src/stage1_cleaning/pdf2clean.py tests/test_pdf2clean_html_fallback_v1.py` passed. Final read-only quality re-review verdict: APPROVED.
- Boundary note: no cleaned artifacts were regenerated and no benchmark/audit interpretation was performed; this remains a generic upstream code repair only.

### 2026-05-05 Phase 3 preparation-core selector floor

- Completed bounded P3.1 repair: Stage2 minimal evidence floor can now add one source-backed preparation-core method block when selected evidence has only weaker generic method/table context and ranked candidates contain a stronger preparation-core method span.
- Added generic preparation-core guards in `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`: local sentence-level action/material/value co-location, concrete source locator requirement, and caption/figure/TOC/reference locator or text-prefix rejection. Removed a reviewer-flagged paper-specific payload literal from the helper cues.
- Added TDD coverage in `tests/test_stage2_preparation_core_selector_floor_v1.py` for adding source-backed preparation core, rejecting caption locators, requiring local action/material/value binding, rejecting missing source locators, rejecting text-marked caption/TOC prefixes, and rejecting dispersed TOC/caption cues.
- Verification: targeted RED failures were observed before implementation/tightening; after implementation `PYTHONPATH=. python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1` -> 6 tests OK; `PYTHONPATH=. python3 -m unittest tests.test_preparation_evidence_sufficiency_audit_v1` -> 4 tests OK; `python3 -m py_compile src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py tests/test_stage2_preparation_core_selector_floor_v1.py` passed. Read-only review loop was used; requested changes were addressed.
- Boundary note: no live LLM calls, no diagnostic replay, no `data/results/` benchmark/audit/workbook comparison, and no benchmark-valid final-output claim were made; this remains a generic selector code repair only.

### 2026-05-05 Phase 4 table-role negative taxonomy and source-role separation

- Completed bounded P4.1 repair: Stage2 table authority now classifies table source role separately from formulation authority and demotes release-profile, pharmacokinetic, tissue-distribution, reference-spillover, targeting/intravenous/cell-uptake/cytotoxicity, characterization-only, and noise/result-only surfaces to optional/non-formulation context unless strong generic composition columns override the negative cue.
- Added generic source-role helpers in `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`, integrated them into selector table inclusion, execution table type classification, and payload inclusion, and persisted `table_source_role` in normalized table payload entries.
- Added/extended TDD coverage in `tests/test_stage2_preparation_core_selector_floor_v1.py` for pharmacokinetic and release negative taxonomy, compact formulation composition headers with characterization columns, execution/payload release-profile optional context, targeting/intravenous result-only demotion, and strong composition override.
- Verification: targeted RED failures were observed before implementation/tightening; after implementation `PYTHONPATH=. python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1` -> 12 tests OK; `PYTHONPATH=. python3 -m unittest tests.test_preparation_evidence_sufficiency_audit_v1 tests.test_stage2_preparation_core_selector_floor_v1` -> 16 tests OK; `PYTHONPATH=. python3 -m py_compile src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py tests/test_stage2_preparation_core_selector_floor_v1.py` passed. Read-only review loop was used; requested payload-path/compact-header/targeting-cue changes were addressed.
- Boundary note: no live LLM calls, no diagnostic replay, no `data/results/` benchmark/audit/workbook comparison, and no benchmark-valid final-output claim were made; this remains a generic selector/payload code repair only.

### 2026-05-05 Phase 5 generic material-value binding helper surface

- Completed bounded P5.1 helper surface: added `src/stage5_benchmark/material_value_binding_v1.py` with side-effect-free direct value validation, paper-local material alias graph resolution, preparation-context entity-bound extraction, scope inference, canonical promotion proposal generation, and auditable promotion evaluation.
- Added TDD coverage in `tests/test_material_value_binding_v1.py` for identity/concentration rejection in mass fields, abbreviation/full-name/row-hint alias roles, preparation-positive extraction with downstream/sample-prep negative guards, ambiguous direct-text scope, conflict-aware shared-scope promotion rejection, and row-local/typed-row-local admitted-row safety.
- Verification: targeted RED failures were observed before helper creation and before `evaluate_canonical_promotions`/row-local safety fixes; after implementation `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1` -> 9 tests OK; `PYTHONPATH=. python3 -m py_compile src/stage5_benchmark/material_value_binding_v1.py tests/test_material_value_binding_v1.py` passed. Read-only spec review passed; quality/genericity re-review approved after reviewer-regression fixes.
- Boundary note: no paper-specific runtime branches, DEV15 maps, GT lookup, raw PDF/XML/HTML consumption, `data/results` reads, LLM calls, row mutation, row creation, S5-3 expansion, diagnostic replay, or benchmark-valid final-output claim were made; this remains a generic helper-surface code repair only.

### 2026-05-05 Phase 6 generic S5-2 material-value carrythrough

- Completed bounded P6.1 integration: `build_minimal_final_output_v1.py` now consumes the side-effect-free `material_value_binding_v1` helper surface to materialize method-shared direct polymer/drug masses for already-admitted Stage5 rows when alias-bound cleaned/source text evidence supports a unique direct value.
- Added a Stage5 adapter that builds a paper-local alias graph from the same clean/source text plus row hints, extracts direct entity-bound values, runs admitted-row promotion review, maps generic helper fields onto existing Stage5 direct mass fields, and preserves higher-authority row-local values.
- Added TDD coverage in `tests/test_compare_layer3_values_v1.py` for `curcumin (CUR)` alias-bound direct preparation masses (`100 mg of PLGA` and `10 mg of CUR`) carrying through to `plga_mass_mg` and `drug_feed_amount_text` with `material_value_binding_direct_text` provenance.
- Verification: targeted RED failure was observed before implementation; after implementation `PYTHONPATH=. python3 -m unittest tests.test_material_value_binding_v1 tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests` -> 53 tests OK; py_compile passed for touched Stage5/helper/test files; read-only spec review passed and quality/genericity review approved.
- Boundary note: no paper-key branches, GT lookup, raw PDF/XML/HTML reads, live LLM calls, S5-3 use, row creation, derived calculation, diagnostic replay, `data/results` comparison, or benchmark-valid final-output claim were made; this remains generic S5-2 direct materialization only.


### 2026-05-05 Phases 7-9 typed boundaries, direct/derived hardening, and bounded P9 replay

- Completed Phase 7: Stage5 final-boundary typed validation now blanks invalid mass/volume/concentration-like values before they can block lawful source-backed carrythrough; Stage2 compatibility validator tests remain in scope.
- Completed Phase 8: derived mass calculations (`ratio × known mass`, `mg/mL × mL`) now remain in `derived_mass_provenance_json` with explicit `direct_or_derived=derived`, `direct_field_write_allowed=no`, and `evidence_binding_status=derived_without_direct_text`; direct canonical mass fields are not filled from calculations.
- Completed Phase 9 bounded diagnostic replay using DOE-explicit baseline lineage (`029` Stage2 + `030` Stage3), Stage5 output `049`, and Layer3 compare `050`.
- P9 regression audit found and fixed a generic overfill signal: initial compare `041` had `polymer_mass_mg extra_in_system +26` because `PLGA mg/mL` concentration headers were eligible for direct polymer mass materialization. The final generic guard rejects polymer `mg/mL` headers for direct mass binding.
- Final P9 diagnostic metrics: Stage5 final_rows=202; core fixed-field recall `0.747034 -> 0.748958`; system_on_gt `2330 -> 2336`; core extra unchanged at `34`; new extra vs `032` is empty; retained improvement is `drug_mass_mg` missing->present_and_match for 6 cells.
- Verification: targeted header/mass-guard tests OK; scoped suite `PYTHONPATH=. python3 -m unittest tests.test_table_structure_dictionary_v1 tests.test_compare_layer3_values_v1.UniversalTableCellGridTests tests.test_compare_layer3_values_v1.MinimalPlusSharedSemanticsTests tests.test_material_value_binding_v1 tests.test_stage2_preparation_core_selector_floor_v1 tests.test_preparation_evidence_sufficiency_audit_v1 tests.test_pdf2clean_html_fallback_v1` -> 136 tests OK; py_compile passed for touched Stage2/Stage5/test files.
- Boundary note: P9 artifacts are bounded diagnostics (`benchmark_valid=no`), not formal benchmark-valid final output. Exploratory directories `040`-`048` are documented as superseded; accepted P9 diagnostic artifacts are `049_stage5_p9_polymer_mgml_mass_guard_bounded_diagnostic` and `050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic`.
