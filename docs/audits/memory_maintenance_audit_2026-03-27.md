# Memory Maintenance Audit 2026-03-27

## 1. Last meaningful memory update

- Artifact: `data/mem/v1/idx.tsv`, `data/mem/v1/run.tsv`, and `data/mem/v1/dec.tsv`
- Date/time: `2026-03-26 10:33:26` local filesystem mtime on `idx.tsv` and `dec.tsv`
- Evidence:
  - `data/mem/v1/run.tsv` ends at `MRUN087`, which indexes the 2026-03-25 Layer3 representation-repair run rooted at `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/51_value_gt_annotation_representation_repair/run_20260325_163435_9d4c2ab_dev15_value_gt_annotation_representation_repair_v2/RUN_CONTEXT.md`.
  - `data/mem/v1/dec.tsv` ends at `MDEC076`, which records the Stage2 semantic-emitter authority decision sourced from `project/4_DECISIONS_LOG.md`.
  - `data/mem/v1/idx.tsv` ends at `MIDX904` and `MIDX905`, matching those final `run.tsv` and `dec.tsv` rows.
  - `python src/utils/check_mem_v1.py` passed before this maintenance pass and reported the pre-update table counts as `idx.tsv=572`, `run.tsv=87`, `lin.tsv=77`, `dec.tsv=16`, `err.tsv=366`, `prm.tsv=26`.
  - No `data/mem/v1/*.tsv` row referenced the 2026-03-27 workbook repair siblings `value_gt_annotation_workbook_representation_repaired_v3.*` or `value_gt_annotation_workbook_representation_repaired_v4.*`.

## 2. Important changes since then

### Ordered timeline

1. `2026-03-27 12:52:59`
   - Fact: `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v2.xlsx` exists at the active run root as the current sibling workbook surface after the 2026-03-25 representation-aware Layer3 repair run.
   - Source paths:
     - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v2.xlsx`
     - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/51_value_gt_annotation_representation_repair/run_20260325_163435_9d4c2ab_dev15_value_gt_annotation_representation_repair_v2/RUN_CONTEXT.md`

2. `2026-03-27 12:55:46` to `2026-03-27 13:17:26`
   - Fact: the WFDTQ4VX repair produced `value_gt_annotation_workbook_representation_repaired_v3.audit.md` and `value_gt_annotation_workbook_representation_repaired_v3.xlsx`.
   - Fact: the audit states that blank WFDTQ4VX X1/X2 concentration cells were filled from authoritative DOE reconciliation and cleaned table artifacts, X3 was left unchanged, and no unrelated workbook cells were modified.
   - Fact: the audit reports `60` edits across WFDTQ4VX rows only.
   - Source paths:
     - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v3.xlsx`
     - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v3.audit.md`

3. `2026-03-27 13:42:58` to `2026-03-27 13:43:21`
   - Fact: the WIVUCMYG repair produced `value_gt_annotation_workbook_representation_repaired_v4.xlsx` and `value_gt_annotation_workbook_representation_repaired_v4.audit.md`.
   - Fact: the audit states that all 26 WIVUCMYG rows had blank concentration value/unit cells in the target fields and that `156` cells were filled in the existing concentration value/unit columns only.
   - Fact: the audit concludes that pH is present in the paper tables but absent already in the Stage2 weak-label extraction output, and that the workbook has no dedicated pH column.
   - Fact: the audit records an advisory artifact disagreement for `F9`, where the weak-label artifact reports polymer concentration `10.00 mg/mL` while the cleaned paper tables decode `9.5 mg/mL`.
   - Source paths:
     - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`
     - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.audit.md`

### Facts

- The active governed memory tables stopped before the 2026-03-27 workbook repair artifacts.
- The workbook sibling lineage advanced from `...repaired_v2.xlsx` to `...repaired_v3.xlsx` and then to `...repaired_v4.xlsx`.
- WFDTQ4VX DOE concentration blanks were repaired in the v3 sibling workbook.
- WIVUCMYG concentration blanks and the pH omission audit were captured in the v4 sibling workbook and audit note.

### Reasonable inferences

- Memory maintenance obligations were skipped during the 2026-03-27 paper-scoped workbook repair tasks because the repair outputs materially changed the active review workbook lineage and surfaced reusable audit findings, but no corresponding `data/mem/v1` rows were appended.
- The skipped updates belong in `data/mem/v1/err.tsv` rather than `project/4_DECISIONS_LOG.md` because they are paper-specific repair and failure-localization findings, not new repository-wide architecture or governance decisions.

### Unresolved uncertainties

- The repo contains sibling workbook repair outputs at the active run root, but there is no new 2026-03-27 `RUN_CONTEXT.md` describing those paper-scoped follow-up edits as standalone runs.
- WIVUCMYG `F9` polymer concentration disagrees across artifacts:
  - cleaned paper tables decode `cPLGA` as `+1 -> 9.5 mg/mL`
  - legacy weak-label artifacts report `10.00 mg/mL`
  - the workbook repair used the cleaned paper tables as the authoritative source

## 3. Artifacts updated now

- `project/memory_maintenance_audit_2026-03-27.md`
  - Added this evidence-based audit summarizing the last meaningful memory update, the post-memory timeline, the skipped maintenance gap, and the minimal governed update decision.
- `data/mem/v1/err.tsv`
  - Appended memory rows for:
    - WFDTQ4VX representation-repaired v2 workbook leaving blank DOE-decoded concentration cells that were later repaired in v3
    - WIVUCMYG representation-repaired v3 workbook leaving blank core concentration fields that were later repaired in v4
    - WIVUCMYG pH being present in paper tables but absent by the Stage2 weak-label output and therefore absent from the workbook schema surface
- `data/mem/v1/idx.tsv`
  - Appended the corresponding searchable registry rows for those new `err.tsv` entries

## 4. Artifacts considered but not changed

- `project/4_DECISIONS_LOG.md`
  - Not changed because the 2026-03-27 findings are paper-scoped repair and audit outcomes, not new cross-repo policy or architecture decisions.
- `docs/snapshots/`
  - Not changed because existing snapshots capture broader architecture, governance, or benchmark-state milestones rather than sibling workbook repair follow-ups.
- `data/mem/v1/run.tsv`
  - Not changed because there is no new 2026-03-27 `RUN_CONTEXT.md`-backed run root for the sibling v3/v4 workbook repairs.
- `data/mem/v1/lin.tsv`
  - Not changed because no new governed run lineage edge exists; the sibling workbook versions are file-level follow-up outputs, not new run directories with lineage metadata.
- `data/mem/v1/dec.tsv`
  - Not changed because no new durable project-level decision was evidenced by the repair tasks.
- `docs/methods/mem_v1.md` and `docs/methods/mem_v1_audit.md`
  - Not changed because the memory maintenance workflow itself did not change; only missing current-state rows needed to be added.

## 5. Any unresolved ambiguity or disagreement across artifacts

- WIVUCMYG `F9` polymer concentration disagreement:
  - `data/cleaned/goren_2025/tables/WIVUCMYG/WIVUCMYG__table_01__html_table.csv`
  - `data/cleaned/goren_2025/tables/WIVUCMYG/WIVUCMYG__table_05__html_table.csv`
  - `data/cleaned/goren_2025/tables/WIVUCMYG/WIVUCMYG__table_13__pdf_table.csv`
  - versus
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/03_WIVUCMYG_10.1002_jps.24101.txt`
- The 2026-03-27 workbook repair artifacts exist and are authoritative for the sibling workbook lineage, but they are not represented by standalone `RUN_CONTEXT.md` files. That limits `run.tsv`/`lin.tsv` updates and is why this maintenance pass updates `err.tsv`/`idx.tsv` instead.

## 6. 2026-03-28: blank/unloaded drug contamination audit

### Facts

- Source audit:
  - `project/blank_unloaded_drug_contamination_audit_2026-03-28.md`
- Candidate blank/unloaded rows found in the current workbook:
  - `13`
- Confirmed erroneous current-workbook non-empty drug rows:
  - `11`
- Wrong current-workbook drug values:
  - `Etoposide`
  - `FITC`
  - `XAN`
- Additional upstream workbook-lineage contamination:
  - `QLYKLPKT_G002` and `RHMJWZX8_G002` carried `GAR` in `value_gt_annotation_workbook_representation_repaired_v2.xlsx` and `...repaired_v3.xlsx`
  - those two cells are blank in the current `...representation_repaired_v4.xlsx`
- Benchmark-valid and current authoritative surfaces stayed clean for the affected rows:
  - Stage5 final table kept `drug_name_value` blank
  - current v7 value-alignment rows kept `sys_drug_name` blank

### Root cause

- The audit traced the first visible non-empty reviewer-surface `drug_name` values to:
  - `value_gt_annotation_workbook_representation_repaired_v2.xlsx`
- The pipeline-level defect is downstream workbook materialization, not Stage2 or benchmark-valid Stage5 output.
- The main code path identified by the audit is:
  - `src/stage5_benchmark/build_value_gt_annotation_representation_repair_v2.py`
- The specific failure mode is:
  - when row-local `drug_name_candidates` are blank, the representation-repair layer falls back to `detect_drug_name(...)` over full paper text
  - the helper uses simple substring matching rather than formulation-level evidence or word-boundary-safe matching
  - this allows blank/unloaded rows to inherit paper-level or unrelated text tokens, including short-code collisions such as upstream `GAR`
- A secondary earlier contributor exists in the older workbook lineage:
  - several blank rows already carried incorrect `l2_gt_formulation_label` values before representation-repair surfaced them as visible `drug_name`

### Action taken now

- Appended new governed error rows to:
  - `data/mem/v1/err.tsv`
- Appended new governed index rows to:
  - `data/mem/v1/idx.tsv`
- Added one governed prevention rule to:
  - `data/mem/v1/dec.tsv`

### Rule added

- Blank or unloaded formulations must have a null drug field.
- Do not inherit drug identity from paper-level mentions, family-level loaded siblings, or historical workbook labels.
- Do not assign a drug by substring-based matching without formulation-level evidence.
- Representation-repair must respect and preserve null drug fields.

## 7. 2026-03-28: Layer 3 cross-audit framework

### Facts

- Method added:
  - `docs/methods/layer3_cross_audit_v1.md`
- Entrypoint added:
  - `src/stage5_benchmark/run_layer3_cross_audit_v1.py`
- Primary outputs for the audited DEV15 workbook:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_gt_cross_audit_report_v4.tsv`
  - `project/layer3_gt_cross_audit_report_v4.md`
- Deterministic run counts:
  - non-empty candidate cells scanned: `3636`
  - flagged cells in merged report: `1532`
  - rule-source rows merged: `1532`
  - Gemini rows merged in this first run: `0`
  - NVIDIA rows merged in this first run: `0`
- Leading risk classes in the initial report:
  - `ambiguity`: `807`
  - `unsupported_value`: `444`
  - `inheritance_contamination`: `230`
  - `direction_mismatch`: `36`
  - `blank_should_be_null`: `8`

### Root cause / need addressed

- Layer 3 already had workbook builders and evidence-handoff validation, but it
  did not have a governed one-row-per-flagged-cell reviewer-risk report for the
  compact value workbook.
- Recent workbook audits showed that reviewer surfaces can contain unsupported,
  inherited, normalized-only, or contaminated values even when benchmark-valid
  Stage5 outputs remain clean.
- A governed cross-audit layer was needed to surface these risks without
  editing workbook data or conflating audit signals with GT truth.

### Action taken now

- Added a governed Layer 3 cross-audit method note:
  - `docs/methods/layer3_cross_audit_v1.md`
- Added a governed Stage 5 audit-only entrypoint:
  - `src/stage5_benchmark/run_layer3_cross_audit_v1.py`
- Registered the entrypoint in:
  - `docs/src_script_registry.tsv`
  - `docs/maintained_script_surface.tsv`
  - `project/PIPELINE_SCRIPT_MAP.md`
  - `project/ACTIVE_PIPELINE_RUNBOOK.md`
- Appended new governed decision memory rows to:
  - `data/mem/v1/dec.tsv`
- Appended new governed index rows to:
  - `data/mem/v1/idx.tsv`

### Rules added / recorded

- Layer 3 cross-audit is a report-only post-annotation reviewer-risk surface.
- It may combine:
  - deterministic rule audit
  - Gemini auditor outputs
  - NVIDIA auditor outputs
- Gemini and NVIDIA are auditors only, not GT editors.
- Layer 3 reviewer surfaces retain only explicit source-supported values.
- Computed, normalized, inferred, inherited, or direction-inverted values do
  not count as direct support.
- The existing blank/unloaded null-drug rule remains in force.

## 2026-03-28: Layer 3 cross-audit runtime diagnosis

### What was reviewed

- `src/stage5_benchmark/run_layer3_cross_audit_v1.py`
- Interrupted command:
  - `python src/stage5_benchmark/run_layer3_cross_audit_v1.py run --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`
- Diagnosis note:
  - `project/layer3_cross_audit_runtime_diagnosis_2026-03-28.md`

### Findings

- The prior `run` path did execute live Gemini and NVIDIA auditing; it was not
  task-pack-only scaffolding.
- Invocation pattern before the fix:
  - deterministic rule audit
  - Gemini live calls
  - NVIDIA live calls
  - merged report
- Model execution was sequential, synchronous, and one request per candidate
  cell.
- The apparent hang was most likely ordinary wall-clock time in the live model
  loops, not workbook processing or merge time.
- The Google / gRPC ALTS lines were noisy Gemini stderr, not the root cause.
- The operational defect was unbounded live execution with weak observability:
  no candidate cap, no per-backend call cap, minimal progress logging, and no
  incremental partial checkpointing.

### Action taken

- Added bounded runtime controls to
  `src/stage5_benchmark/run_layer3_cross_audit_v1.py`:
  - `--rules-only`
  - `--skip-gemini`
  - `--skip-nvidia`
  - `--max-candidates`
  - `--max-gemini-calls`
  - `--max-nvidia-calls`
  - `--batch-size`
  - `--request-timeout-seconds`
  - `--max-retries`
  - `--write-partial-every-batch`
- Added per-batch progress logging and partial backend writes.
- Added governed memory rows:
  - `MDEC080`
  - `MERR597`

### Memory impact

- Recorded a workflow decision that live Layer 3 model auditing must be bounded
  and observable.
- Recorded the prior runtime defect as an error-level memory row so future
  debugging does not misclassify noisy Gemini stderr as the root cause.

## 2026-03-28: formulation-centered Layer 3 audit-system design clarification

### What was clarified

- Layer 3 reviewer-facing outputs are not only evaluation helpers.
- They are also part of the governed production audit and governance layer
  around the frozen formulation database.
- The benchmark-valid endpoint remains:
  - `final_formulation_table_v1.tsv`
- Reviewer-facing audit outputs remain downstream support surfaces and must not
  mutate benchmark-valid outputs.

### Design conclusion recorded

- The preferred reviewer entry object is one formulation row.
- Human review is split into two linked layers:
  - formulation existence and identity audit
  - value credibility audit
- These layers are strongly dependent, not parallel.
- Many apparent value errors are projections of structure or identity errors.
- Current repo capability is:
  - partially present but not unified

### Functional-unit direction recorded

- `Formulation Index Builder`
- `Structure Review Builder`
- `Value Risk Builder`
- `Evidence Handoff Builder`

### Files updated

- `project/2_ARCHITECTURE.md`
- `project/4_DECISIONS_LOG.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `docs/methods/layer3_field_gt_protocol_v1.md`
- `docs/methods/layer3_cross_audit_v1.md`

### Memory impact

- Added governed decision row:
  - `MDEC081`
- Added governed index row:
  - `MIDX917`
