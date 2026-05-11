# Stage1/Stage2 Clean Text and Evidence Selector Visibility Repair Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Use read-only specialists for audit/count verification before any runtime patch. Keep all repairs generic and all DEV15 paper examples as validation anchors only.

**Goal:** Repair source-visibility failures so source-present, formulation-relevant paragraphs/tables remain visible through Stage1 clean text and Stage2 evidence/full-table authority surfaces, without forcing complete numeric tables into LLM prompts.

**Architecture:** DEV15 user-provided original excerpts are governed validation anchors for all-corpus generic repairs. Stage2 keeps two legal surfaces: compact summary/prompt view for LLM semantic authorization, and execution-grade full table authority (`normalized_table_payloads_v1.json`, normalized CSVs, `table_cell_grid_v1.tsv/jsonl`) for deterministic downstream materialization. Evidence selector may rank/summarize for prompt, but must not be an irreversible authority filter.

**Tech Stack:** Python stdlib, existing PLGA Stage1/Stage2/Stage5 benchmark scripts, TSV/JSONL/JSON artifacts, unittest. Run commands from repo root with `PYTHONPATH=.`.

---

## Defaults Confirmed by User

```text
Q1 scope = B
DEV15 is a governed validation/regression anchor set, but repair logic must apply to the whole corpus.

Q2 = yes
User-provided original excerpts are hard validation anchors. Each anchored paragraph/table/value must be visible in clean text or execution-grade table authority.

Q3 = yes
HTML/PDF dual-source visibility audit is allowed. Do not blindly merge prose; record source-specific coverage and lineage.

Q4 = rank for prompt, not filter for authority
Selector may rank and summarize LLM-facing evidence. It must not decide what full-table/table-grid authority exists.

Q5 = yes
Diagnostic-only audit helpers may be added if needed.
```

---

## Non-Negotiable Rules

1. **Evidence visibility first, materialization second.** Do not repair Stage5 when the first failure is clean text, table authority, selector, prompt authorization, or Stage2 projection.
2. **Prompt summary is not full table authority.** It is legal and expected for prompt packs to be compact. Missing complete numeric rows from prompt is not automatically a bug.
3. **Full table authority must be preserved separately.** Downstream numeric/table materialization must use coordinate-preserving full-table artifacts, not prompt summaries.
4. **No paper-key runtime branches.** DEV15 paper keys (`INMUTV7L`, `BB3JUVW7`, etc.) may appear only in tests, fixtures, diagnostics, and validation reports.
5. **No source bypass.** Stage5/S5-3 must not mine raw PDF/HTML/user excerpts to fill values. If clean text or table authority is missing, repair Stage1/Stage2.
6. **No selector-as-veto.** Selector ranking may decide compact LLM prompt content, but must not erase candidate evidence/table authority from execution/audit surfaces.
7. **No benchmark-valid reporting from diagnostic runs.** Every planned replay/audit here is diagnostic-only unless later promoted through governed runbook procedures.
8. **Lineage must be explicit.** Every audit output must record resolved run dirs, source files, clean text paths, table payload paths, and compare artifact paths.

---

## Current Evidence and Baseline Facts

### Governed user-provided original excerpts

Primary repository location:

```text
docs/methods/layer3_field_gt_protocol_v1.md
```

Section:

```text
## User-Provided Original Source Excerpts For Field-GT Debugging
```

Known DEV15 anchor headers:

```text
1100  INMUTV7L
1130  BB3JUVW7
1168  BXCV5XWB
1258  L3H2RS2H
1326  PA3SPZ28
1418  QLYKLPKT
1455  RHMJWZX8
1472  UFXX9WXE
1548  V99GKZEI
1572  WFDTQ4VX
1640  WIVUCMYG
1692  YGA8VQKU
1748  7ZS858NS
1765  5ZXYABSU
1814  5GIF3D8W
```

### Stage2 authority design

From `project/ACTIVE_DATA_SOURCE_CONTRACT.md`:

```text
summary / prompt / evidence view:
  may be compacted to reduce LLM prompt size;
  prompt-only and diagnostic/evidence-facing;
  must not be consumed as benchmark-facing numeric materialization authority.

full / execution-grade table authority:
  normalized_table_payloads_v1.json;
  payload normalized CSVs;
  table_cell_grid_v1.tsv/jsonl;
  lawful Stage2 source for downstream mechanical table materialization.
```

### Diagnostic evidence already available

Existing diagnostic artifacts under:

```text
data/results/20260504_ab9f61e/
```

Relevant runs:

```text
050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/
055_characterization_metric_cleantext_selector_visibility_audit_diagnostic/
057_characterization_metric_selector_payload_visibility_audit_diagnostic/
060_characterization_metric_tail_binding_projection_audit_diagnostic/
```

Known `057/060` characterization visibility first-failure counts:

```text
llm_prompt_has_value_but_llm_did_not_authorize:       66
clean_text_has_value_but_selector_omitted:            34
endpoint_or_alignment_blocked:                        16
source_value_absent_from_clean_text:                  13
selector_included_summary_but_not_numeric_table:       9
```

Current `050` broad missing-cell clean-text token audit:

```text
missing_in_system with nonempty GT: 688
clean text token found:             524
clean text token absent:            164
found rate:                         76.2%
```

Characterization metric subset:

```text
missing cells: 111
clean text found: 106
clean text absent: 5
found rate: 95.5%
```

Known bounded source-conversion/table-payload gap examples:

```text
5GIF3D8W ee/pdi/zeta/lc optimized metric tail values visible in clean text but absent from cleaned table CSV / normalized payload.
L3H2RS2H particle_size_nm and pdi values visible in clean text but absent from cleaned table CSV / normalized payload.
```

---

## Success Criteria

### Audit success

A diagnostic audit can classify every residual/error cell or user anchor into exactly one first failing boundary:

```text
source_anchor_not_found_in_clean_text_or_table_authority
clean_text_present_table_payload_missing
table_payload_present_grid_missing
grid_present_selector_registry_missing
selector_registry_present_prompt_summary_insufficient
prompt_summary_has_signal_llm_no_authorization
llm_authorized_stage2_projection_lost
stage2_projected_stage3_lost
stage3_present_stage5_materialization_lost
alignment_or_endpoint_blocked
not_a_direct_value_or_gt_authority_review
```

### Repair success

1. User-provided DEV15 anchors are visible in either clean text or execution-grade table authority.
2. Clean text/source conversion failures are reduced or explicitly isolated with source-lineage reasons.
3. Selector no longer acts as irreversible filter for formulation-relevant tables/blocks.
4. Full-table/grid artifacts survive even when prompt summary is compact.
5. Downstream materialization consumes full-table/grid authority when LLM semantic authorization exists.
6. No paper-specific runtime code, no benchmark claims from diagnostics, no Stage5 source bypass.

---

## Planned Output Files

Plan:

```text
docs/plans/2026-05-05-stage1-stage2-cleantext-selector-visibility-repair-plan.md
```

Progress ledger:

```text
docs/plans/2026-05-05-stage1-stage2-cleantext-selector-visibility-progress.tsv
```

Potential diagnostic helper scripts, only if needed:

```text
src/stage1_cleaning/audit_source_anchor_cleantext_visibility_v1.py
src/stage2_sampling_labels/audit_table_authority_anchor_visibility_v1.py
src/stage2_sampling_labels/audit_selector_anchor_recall_v1.py
src/stage2_sampling_labels/audit_prompt_summary_semantic_adequacy_v1.py
```

Potential tests:

```text
tests/test_source_anchor_cleantext_visibility_v1.py
tests/test_table_authority_anchor_visibility_v1.py
tests/test_selector_anchor_recall_v1.py
tests/test_prompt_summary_semantic_adequacy_v1.py
```

---

## Implementation Tasks

### Task 1: Create progress ledger

**Objective:** Make unattended/resumable execution safe.

**Files:**
- Create: `docs/plans/2026-05-05-stage1-stage2-cleantext-selector-visibility-progress.tsv`

**Content:**

```tsv
task_id	status	commit	test_status	notes	updated_at
T01	pending			create progress ledger	
T02	pending			inventory governed DEV15 anchors	
T03	pending			write source-anchor parser tests	
T04	pending			implement source-anchor parser	
T05	pending			write clean-text visibility audit tests	
T06	pending			implement clean-text visibility audit helper	
T07	pending			write table-authority visibility tests	
T08	pending			implement table-authority visibility audit helper	
T09	pending			write selector recall registry tests	
T10	pending			repair selector as ranker not authority filter	
T11	pending			write prompt summary adequacy tests	
T12	pending			audit summary semantic adequacy not numeric completeness	
T13	pending			write Stage2 projection/downstream boundary audit tests	
T14	pending			implement boundary classifier report	
T15	pending			bounded DEV15 diagnostic replay/audit	
T16	pending			final delta report and governance/memory update	
```

**Verification:**

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('docs/plans/2026-05-05-stage1-stage2-cleantext-selector-visibility-progress.tsv')
assert p.exists()
assert p.read_text().splitlines()[0].split('\t') == ['task_id','status','commit','test_status','notes','updated_at']
PY
```

---

### Task 2: Inventory governed DEV15 source anchors

**Objective:** Produce a diagnostic inventory of user-provided original excerpts without changing runtime behavior.

**Files:**
- Read: `docs/methods/layer3_field_gt_protocol_v1.md`
- Create diagnostic output under a new run directory, e.g. `data/results/20260504_ab9f61e/061_source_anchor_inventory_diagnostic/`

**Rules:**

- Parse only the `User-Provided Original Source Excerpts For Field-GT Debugging` section.
- Preserve exact line ranges and raw text snippets.
- Do not summarize source content inside the anchor inventory except metadata fields.
- Record paper key, section start line, section end line, and whether the section includes paragraph/table markers.

**Expected output schema:**

```tsv
paper_key	anchor_start_line	anchor_end_line	has_method_paragraph	has_table	anchor_source_file
```

**Verification:**

Expected paper keys exactly include the 15 DEV15 anchors listed above.

---

### Task 3: Add source-anchor parser tests

**Objective:** Lock parser behavior before implementing helpers.

**Files:**
- Create: `tests/test_source_anchor_cleantext_visibility_v1.py`

**Test cases:**

1. Parser finds `INMUTV7L` anchor and line range includes `90 mg of PLGA` and `5 mg of dexibuprofen`.
2. Parser finds `BB3JUVW7` anchor and recognizes both nanosphere and nanorod table text.
3. Parser returns all 15 DEV15 paper keys.
4. Parser treats diagnostic repair notes after the anchor section as non-anchor notes unless explicitly requested.

**Run:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_source_anchor_cleantext_visibility_v1
```

Expected first run before implementation: fail due to missing helper.

---

### Task 4: Implement source-anchor parser utility

**Objective:** Provide reusable parsing for governed user-provided source anchors.

**Files:**
- Create: `src/stage1_cleaning/audit_source_anchor_cleantext_visibility_v1.py`

**Implementation requirements:**

- Function: `parse_user_source_anchor_sections(protocol_path: Path) -> list[AnchorSection]`
- Dataclass fields:
  - `paper_key`
  - `start_line`
  - `end_line`
  - `raw_text`
  - `has_table_marker`
  - `has_method_marker`
- CLI options:
  - `--protocol-md`
  - `--out-dir`
  - `--write-inventory-only`

**Do not:**

- Compare to GT.
- Infer values.
- Rewrite protocol file.

**Run tests:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_source_anchor_cleantext_visibility_v1
```

---

### Task 5: Add clean-text visibility audit tests

**Objective:** Verify that anchor tokens/lines are searched in active clean text with explicit lineage.

**Files:**
- Modify: `tests/test_source_anchor_cleantext_visibility_v1.py`

**Test cases:**

1. Given an anchor and a clean text path, normalized exact fragments are found when present.
2. Unicode normalization handles `−`, `μ`, `µ`, non-breaking spaces, and ligatures.
3. Numeric token checks do not claim row-local binding; they only report visibility.
4. Missing anchor text is reported as `anchor_not_found_in_clean_text`, not as Stage5 loss.

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_source_anchor_cleantext_visibility_v1
```

---

### Task 6: Implement clean-text visibility audit helper

**Objective:** Compare governed anchors against active clean text for the whole corpus, using DEV15 as validation anchors.

**Files:**
- Modify: `src/stage1_cleaning/audit_source_anchor_cleantext_visibility_v1.py`

**Inputs:**

```text
--protocol-md docs/methods/layer3_field_gt_protocol_v1.md
--key2txt data/cleaned/index/key2txt.tsv
--manifest data/cleaned/index/manifest_current.tsv
--out-dir data/results/<run>/source_anchor_cleantext_visibility_diagnostic
```

**Outputs:**

```text
source_anchor_cleantext_visibility_v1.tsv
source_anchor_cleantext_visibility_summary_v1.tsv
source_anchor_cleantext_visibility_metadata.json
```

**TSV schema:**

```tsv
paper_key	anchor_start_line	anchor_end_line	clean_text_path	text_source_type	anchor_visibility	matched_fragment_count	missing_fragment_count	first_missing_fragment	pdf_path	html_path	primary_source_type	secondary_source_available
```

**Important:**

- This audit may use HTML/PDF paths from manifest for visibility metadata, but it must not merge raw source into clean text.
- Secondary source checks are diagnostic only and must record lineage.

---

### Task 7: Add table-authority visibility tests

**Objective:** Verify anchor tables are checked against execution-grade table authority, not prompt summaries.

**Files:**
- Create: `tests/test_table_authority_anchor_visibility_v1.py`

**Test cases:**

1. Table-like anchor lines can be normalized to row/cell tokens.
2. `normalized_table_payloads_v1.json` presence satisfies full-table authority even if prompt summary is compact.
3. `table_cell_grid_v1.tsv/jsonl` preserves coordinate geometry and blank header placeholders.
4. Absence from prompt summary is not failure when payload/grid has table cells.

---

### Task 8: Implement table-authority anchor visibility audit

**Objective:** Determine whether user-anchored tables are visible in Stage2 full-table artifacts.

**Files:**
- Create: `src/stage2_sampling_labels/audit_table_authority_anchor_visibility_v1.py`

**Inputs:**

```text
--stage2-run-dir <explicit Stage2 run dir>
--anchor-inventory <source_anchor_cleantext_visibility_v1.tsv or protocol-md>
--out-dir <diagnostic out dir>
```

**Search surfaces:**

```text
normalized_table_payloads_v1.json
payload normalized CSVs
table_cell_grid_v1.tsv
table_cell_grid_v1.jsonl
```

**Outputs:**

```text
table_authority_anchor_visibility_v1.tsv
table_authority_anchor_visibility_summary_v1.tsv
table_authority_anchor_visibility_metadata.json
```

**Failure buckets:**

```text
visible_in_payload_and_grid
visible_in_clean_text_only_payload_missing
visible_in_payload_grid_missing
not_visible_in_clean_text_or_table_authority
source_locator_unresolved
```

---

### Task 9: Add selector recall registry tests

**Objective:** Ensure selector is tested as a ranker/summary producer, not an authority eraser.

**Files:**
- Create: `tests/test_selector_anchor_recall_v1.py`

**Test cases:**

1. Candidate evidence registry retains formulation-relevant blocks even if not selected for compact prompt.
2. Selected evidence can be top-k/ranked without deleting unselected blocks from audit/execution surfaces.
3. Non-noise tables are preserved into table authority even when selector score is low.
4. Confirmed pure noise may be excluded, but reason must be explicit.

---

### Task 10: Repair selector as ranker, not authority filter

**Objective:** Make selector behavior consistent with architecture.

**Files:**
- Inspect first; likely candidates under `src/stage2_sampling_labels/`.
- Modify only the maintained selector/prompt-pack script identified in `ACTIVE_PIPELINE_RUNBOOK.md` and `docs/maintained_script_surface.tsv`.

**Implementation requirements:**

- Produce or preserve a full candidate evidence registry.
- Mark selected prompt evidence separately from full candidate evidence.
- Ensure table authority generation does not depend on selected prompt evidence.
- Add explicit fields:
  - `selector_rank`
  - `selected_for_prompt`
  - `preserved_for_authority`
  - `exclusion_reason`

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_selector_anchor_recall_v1
```

---

### Task 11: Add prompt summary semantic adequacy tests

**Objective:** Prevent the same mistake: prompt summary completeness is semantic, not numeric-table completeness.

**Files:**
- Create: `tests/test_prompt_summary_semantic_adequacy_v1.py`

**Test cases:**

Prompt summary is adequate when it signals:

```text
this is a formulation/preparation table;
rows are formulation variants;
columns include preparation variables and/or characterization metrics;
table family/role is preserved;
loaded vs blank/control semantics are visible when present.
```

Prompt summary is inadequate when:

```text
it loses that a table is formulation-bearing;
it describes only generic results without row/formulation semantics;
it collapses loaded/blank/control identities;
it omits that full execution-grade table authority exists.
```

Do not require full numeric rows in prompt summary.

---

### Task 12: Implement prompt summary adequacy audit

**Objective:** Audit whether compact prompt summaries provide enough semantic signal for LLM authorization.

**Files:**
- Create: `src/stage2_sampling_labels/audit_prompt_summary_semantic_adequacy_v1.py`

**Outputs:**

```text
prompt_summary_semantic_adequacy_v1.tsv
prompt_summary_semantic_adequacy_summary_v1.tsv
```

**Buckets:**

```text
adequate_semantic_signal_full_table_authority_present
summary_lost_formulation_table_identity
summary_lost_loaded_blank_control_semantics
summary_lost_row_variant_semantics
summary_mentions_metric_context_but_not_formulation_role
not_applicable_no_table_anchor
```

---

### Task 13: Add downstream boundary classifier tests

**Objective:** Distinguish clean text/table/selector failures from Stage2 projection, Stage3, and Stage5 losses.

**Files:**
- Create: `tests/test_stage1_stage2_visibility_boundary_classifier_v1.py`

**Test cases:**

1. Clean text absent -> `source_anchor_not_found_in_clean_text_or_table_authority`.
2. Clean text present but payload absent -> `clean_text_present_table_payload_missing`.
3. Payload/grid present but selector registry absent -> `grid_present_selector_registry_missing`.
4. Selector/prompt has semantic signal but LLM raw response omitted -> `prompt_summary_has_signal_llm_no_authorization`.
5. LLM raw response has authorization but weak label/projection lacks row bindings -> `llm_authorized_stage2_projection_lost`.
6. Stage2 projection exists but final table lacks value -> `stage3_or_stage5_materialization_lost`.
7. `blocked_alignment` remains `alignment_or_endpoint_blocked` and must not trigger value fill.

---

### Task 14: Implement unified boundary classifier report

**Objective:** Produce one diagnostic table that answers “where was the first failure?” for residual cells and anchors.

**Files:**
- Create or extend an audit helper in the appropriate Stage2/Stage5 diagnostic directory.

**Inputs:**

```text
--compare-cells data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_value_compare_cells_v1.tsv
--anchor-visibility <source_anchor_cleantext_visibility_v1.tsv>
--table-authority-visibility <table_authority_anchor_visibility_v1.tsv>
--selector-audit existing or newly generated selector recall TSV
--stage2-run-dir <explicit run dir>
--stage5-run-dir <explicit run dir>
--out-dir <diagnostic out dir>
```

**Outputs:**

```text
stage1_stage2_visibility_boundary_audit_v1.tsv
stage1_stage2_visibility_boundary_summary_v1.tsv
RUN_CONTEXT.md
```

**Required metadata:**

- active/explicit run dirs
- exact source files
- GT authority file
- clean text index
- Stage2 full table artifacts
- selector artifacts
- compare cells artifact
- diagnostic-only status

---

### Task 15: Bounded DEV15 diagnostic replay/audit

**Objective:** Validate generic repairs using DEV15 anchors without making benchmark-valid claims.

**Run sequence:**

1. Resolve explicit Stage2 and Stage5 run dirs from `ACTIVE_RUN.json` or CLI.
2. Run source-anchor clean text visibility audit.
3. Run table-authority anchor visibility audit.
4. Run selector anchor recall audit.
5. Run prompt semantic adequacy audit.
6. Run boundary classifier.
7. If runtime repairs were made, run bounded replay only over DEV15 / affected diagnostic subset.
8. Compare diagnostic deltas against `050`, labeling all outputs diagnostic-only.

**Stop conditions:**

- If active source lineage is ambiguous, stop.
- If a maintained entrypoint cannot be identified, stop and report.
- If a repair requires raw PDF/HTML mining inside Stage5, reject it and stop.
- If tests fail after one focused fix attempt, update ledger and stop.

---

### Task 16: Final report and governance update

**Objective:** Record the outcome without overstating benchmark validity.

**Files:**
- Append only if validated: `docs/methods/layer3_field_gt_protocol_v1.md`
- Append only principle/decision, not raw excerpts: `project/4_DECISIONS_LOG.md`
- Update governed memory rows under `data/mem/v1/` only if a durable repair decision/error pattern was confirmed.

**Report must include:**

```text
clean text anchor visibility before/after
table authority visibility before/after
selector recall before/after
prompt semantic adequacy before/after
boundary classifier counts before/after
changed compare cells, if any
no benchmark-valid claim unless full terminal pipeline was lawfully executed
```

---

## Validation Commands

Targeted tests:

```bash
PYTHONPATH=. python3 -m unittest \
  tests.test_source_anchor_cleantext_visibility_v1 \
  tests.test_table_authority_anchor_visibility_v1 \
  tests.test_selector_anchor_recall_v1 \
  tests.test_prompt_summary_semantic_adequacy_v1 \
  tests.test_stage1_stage2_visibility_boundary_classifier_v1
```

Existing regression tests to run after any runtime patch:

```bash
PYTHONPATH=. python3 -m unittest tests.test_table_structure_dictionary_v1 tests.test_compare_layer3_values_v1
```

Memory/schema check if governed memory is updated:

```bash
PYTHONPATH=. python3 src/utils/check_mem_v1.py
```

---

## Expected Repair Impact

This plan is not expected to make every missing cell match immediately. It is expected to move residuals into lawful first-failure buckets and repair generic visibility loss.

Likely positive outcomes:

```text
- fewer source-present anchors absent from clean text/table authority;
- fewer clean_text_has_value_but_selector_omitted failures;
- fewer table visible only in prompt summary but missing from execution-grade payload/grid;
- clearer separation between LLM authorization misses and downstream projection/materialization losses;
- fewer future cases incorrectly assigned to Stage5 when the first failure is Stage1/Stage2.
```

Not expected / not allowed:

```text
- no requirement that LLM prompt contain full numeric rows;
- no Stage5 raw-source mining;
- no paper-key runtime patch;
- no forcing ambiguous or unaligned cells to match;
- no benchmark-valid performance claim from these diagnostics.
```
