# Deterministic Two-Step Baseline Audit (2026-04-15)

## 1. Executive Conclusion

### Recommended baseline design

Recommended design: `Option 2`

- Step 1 chain:
  - `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `src/stage5_benchmark/enforce_identity_freeze_v1.py`
- Step 2 chain:
  - start from the frozen Step 1 `final_formulation_table_v1.tsv`
  - reuse the existing frozen-final downstream audit/value surfaces as the implementation skeleton:
    - `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
    - `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
  - add one new deterministic Step 2-only backfill helper rather than reusing Stage5 closure itself

### Why this is the best evidence-backed choice

Facts:

- The March 14 no-LLM lineage (`data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/RUN_CONTEXT.md`) is not a true full-text deterministic baseline. It replays saved LLM raw responses through `auto_extract_weak_labels_v7pilot_r3_fixparse.py`.
- The only repo-tested no-LLM full-text comparator chain over `manifest_current.tsv`-derived scope is `data/results/run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1/RUN_CONTEXT.md`.
- Stable formulation IDs first appear at the Stage5 final table as `final_formulation_id` in `data/results/run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1/final_formulation_table_v1.tsv`.
- Existing downstream frozen-final surfaces already obey the right Step 2 directionality: they read the frozen final table and do not change membership. See `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py` and `src/stage5_benchmark/build_modeling_ready_sidecar_v1.py`.

Inference:

- The repo does not currently contain a clean deterministic Step 2-only value-backfill entrypoint that starts from frozen `final_formulation_id` and writes a non-review value table.
- The smallest lawful design is therefore not “one unchanged chain for both steps,” but “existing deterministic Step 1 chain plus one minimal frozen-final Step 2 helper.”

### Bottom line

- Best Step 1 baseline: the April 11 rules-only comparator skeleton, because it is the only repo-proven no-LLM full-text chain that reaches Stage5 final IDs.
- Best Step 2 baseline: a new frozen-final deterministic backfill helper built from the existing audit-ready/value-review downstream surfaces, not from Stage5 closure logic directly.
- Audit confidence: `medium`
  - strong on Step 1
  - moderate on Step 2 because the exact helper does not yet exist

## 2. Evidence-Backed Historical Timeline

### Phase A: 2026-03-06 = LLM responsibility tightening

Governance / documentation evidence:

- `project/4_DECISIONS_LOG.md`
  - records that the LLM owns semantic structure, including instance boundaries and shared-vs-instance interpretation
- `docs/snapshots/snapshot_2026-03-06_llm_rule_audit_and_contract.md`
  - summarizes the same split as:
    - LLM extraction layer = semantic structuring
    - deterministic arbitration layer = normalization, derivation, evidence binding, export

Implementation evidence:

- None found showing a runtime switch on this date.
- The active extractor remained the legacy wide-row path at this point.

Practical authority surface at this phase:

- documented contract narrowed
- implementation authority did not yet switch

Assessment:

- The repo supports “LLM responsibility tightening” on `2026-03-06`.

### Phase B: 2026-03-13 = deterministic engineering drift becomes operationally visible

Governance / documentation evidence:

- `docs/methods/doe_logic_audit_ufxx9wxe_2026-03-13.md`
  - states that in strong numbered DOE tables, deterministic Stage2 enumeration is primary and the LLM is judge rather than row counter
- `project/4_DECISIONS_LOG.md`
  - records the numbered-DOE Stage2 strategy and the new deterministic enumerator

Implementation evidence:

- `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`
  - deterministically enumerates explicit numbered DOE rows from Stage1 table assets
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  - imports and additively integrates the deterministic DOE enumerator
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/RUN_CONTEXT.md`
  - still uses the legacy Stage2 extractor
- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/RUN_CONTEXT.md`
  - replays saved raw responses, but now with deterministic DOE priority and guardrails active

Practical authority surface at this phase:

- active Stage2 still depended on an LLM-produced candidate universe
- deterministic handling became operationally visible inside the old path, especially for DOE recovery

Assessment:

- The repo supports “deterministic engineering drift becomes operationally visible” on `2026-03-13`.
- This was practical drift, not yet a clean authority switch.

### Phase C: 2026-03-26 = formal authority switch / documentation-level deterministic semantic authority

Evidence that supports the user’s interpretation:

- `docs/audits/dev15_semantic_authority_switch_timeline_2026-04-10.md`
  - explicitly argues that the earliest commit-level governance rewrite happened on `2026-03-26`
  - distinguishes:
    - `2026-03-26` commit-level switch
    - `2026-03-29` dated decision-log normalization
    - `2026-03-30` `ACTIVE_RUN.json` repointing
- `project/4_DECISIONS_LOG.md`
  - records:
    - approval of replacement direction on `2026-03-25` without switching active runtime
    - later normalization that deterministic semantic replacement must not be treated as active mainline after the corrective freeze
- `data/results/ACTIVE_RUN.json`
  - current note says authority promotion points to the semantic Stage2 emitter -> compatibility adapter -> Stage3 -> Stage5 lineage

Evidence that limits or complicates the interpretation:

- `project/4_DECISIONS_LOG.md`
  - says `2026-03-25` approved the replacement direction “without switching the active runtime”
- `data/results/run_20260325_1434_f17211_dev15_3paper_true_semantic_replacement_validation_no_llm_v1/RUN_CONTEXT.md`
  - calls the true semantic emitter path `diagnostic-only, not benchmark-valid final output`
- `project/2_ARCHITECTURE.md`
  - current authority now rejects deterministic semantic Stage2 authority and classifies emitters/lifts as fallback/comparator only after the corrective freeze
- `docs/snapshots/snapshot_2026-03-30_stage2_authority_transition_audit.md`
  - later corrective audit reasserts that LLM semantic discovery is the maintained authority boundary

Practical authority surfaces that changed:

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  - became the practical deterministic semantic emitter for replacement/comparator work
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - became the required bridge into unchanged Stage3/Stage5 consumers
- `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md`
  - shows the semantic-emitter path being used as the active experimental chain

Assessment:

- The repo partially supports the user’s “2026-03-26 formal authority switch” interpretation, but only as a temporary governance/engineering phase.
- Stronger statement:
  - `2026-03-26` marks a commit-level and documentation-level switch toward deterministic semantic authority
  - `2026-03-30` onward corrects that state and reclassifies deterministic semantic emitters as comparator/fallback infrastructure rather than active Stage2 authority

### Historical interpretation verdict

- `2026-03-06 = LLM responsibility tightening`: `supported`
- `2026-03-13 = deterministic engineering drift operationally visible`: `supported`
- `2026-03-26 = formal authority switch`: `supported as a temporary doc/governance switch, but not as the repo’s current lasting authority contract`

## 3. Candidate Chain Comparison

| Chain name | Exact scripts in order | Requires LLM calls | Can run on `manifest_current.tsv` | Can produce stable formulation IDs | Separates identity reconstruction from value backfill | Can Step 2 operate without changing Step 1 membership | Expected strengths | Expected failure modes | Alignment with current governance | Adaptation effort for two-step baseline | Better suited for |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Legacy replay refresh no-LLM | `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py --replay-raw-responses-dir` -> `src/stage5_benchmark/build_minimal_final_output_v1.py` -> compare | No fresh calls, but depends on prior LLM raw responses | No, not from full text alone | Yes, but only at Stage5 | No | Partly | Historical benchmark-valid lineage; deterministic replay; DOE guardrails active | Not a true bypass of LLM-first Stage2; depends on saved raw responses; no clean Step 2 separation | Low for the new experiment | High | Historical comparison only |
| Semantic emitter replacement slice | `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py` -> `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py` -> `src/stage3_relation/build_formulation_relation_artifacts_v1.py` -> `src/stage5_benchmark/build_minimal_final_output_v1.py` | No | Only for the paper keys hardcoded in the emitter | Yes, at Stage5 | Partly | Yes, if Step 2 stays downstream of frozen final | True no-LLM semantic generation; reaches unchanged downstream stack | Hardcoded paper support; diagnostic/comparator status; many blanks in value surfaces; not lawful current Stage2 mainline | Medium for audit baseline selection, low for current mainline adoption | Medium | Step 1 skeleton |
| Rules-only comparator fulltext lineage | Same as above, plus downstream surfaces already exercised in `run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1`: `export_final_formulation_audit_ready_v1.py`, `build_field_gt_review_workbook_v1.py`, compare | No | Yes for the eligible manifest-backed cleaned full-text subset; no for unsupported papers | Yes, `final_formulation_id` in Stage5 final table | Partly | Yes, downstream surfaces are frozen-final only | Best repo-proven no-LLM full-text chain; tested over current manifest-backed cleaned corpus subset; exposes audit/value surfaces | Emitter only supports 15 eligible keys; undercounts remain on 7 papers; no dedicated automatic Step 2 backfill helper | Medium as comparator, low as active mainline | Low for Step 1 reuse, medium for Step 2 completion | Step 1 best current baseline |
| Frozen-final audit/value downstream skeleton | `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py` -> `src/stage5_benchmark/build_field_gt_review_workbook_v1.py` -> `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py` | No | Yes, if a frozen final table already exists | Consumes stable IDs rather than producing them | Yes | Yes | Correct direction for identity-frozen downstream work; does not mutate membership; already shaped like Layer3 review surfaces | Review/export oriented, not a clean deterministic value-backfill table builder; workbook-centric; not sufficient as Step 1 | High | Medium | Step 2 skeleton only |

## 4. Recommended Baseline Design

### Recommendation

Use `Option 2`.

- Step 1:
  - reuse the April 11 rules-only comparator fulltext chain
  - add the identity freeze gate as mandatory end-of-step freeze
- Step 2:
  - do not reuse Stage5 closure
  - create a new deterministic frozen-final backfill helper that reads Step 1 `final_formulation_table_v1.tsv` and writes a value table keyed by `final_formulation_id`
  - use the existing audit-ready export and field-review seed surfaces as the implementation template only

### Why not Option 1

- No existing single chain cleanly does both:
  - Step 1 membership/identity freeze
  - Step 2 identity-frozen explicit-only value backfill
- `build_minimal_final_output_v1.py` is a closure/filter/collapse builder, not a pure value-attachment step.

### Why not recommend the March 14 replay chain

- It is not a deterministic full-text extraction baseline.
- It still depends on prior LLM raw responses and does not bypass the LLM-first Stage2 mainline in the sense required by the user.

### Why the April 11 comparator lineage is the best Step 1 anchor

Facts from `data/results/run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1/RUN_CONTEXT.md`:

- it starts from `manifest_current.tsv`
- it is explicitly `comparator_only_rules_based_stage2_run`
- it uses:
  - `emit_semantic_objects_from_cleaned_papers_v1.py`
  - `build_stage2_compatibility_projection_v1.py`
  - `build_formulation_relation_artifacts_v1.py`
  - `build_minimal_final_output_v1.py`
- it reaches:
  - `final_formulation_table_v1.tsv`
  - `final_formulation_table_audit_ready_v1.tsv`
  - field review seed/workbook surfaces

Inference:

- This is the strongest existing repo-backed skeleton for the new two-step deterministic experiment.

## 5. Minimal Required Patch Set

Do not implement yet. Minimum engineering required:

### A. Stable formulation ID generation

- Freeze Step 1 output at Stage5 final table and adopt `final_formulation_id` as the Step 1 canonical ID.
- Ensure the Step 1 runner always emits:
  - `final_formulation_table_v1.tsv`
  - `final_output_decision_trace_v1.tsv`
  - `downstream_variant_records_v1.tsv`
  - identity freeze diagnostics

### B. Identity freeze between Step 1 and Step 2

- Make `src/stage5_benchmark/enforce_identity_freeze_v1.py` mandatory in the Step 1 baseline.
- Step 2 must hard fail if identity freeze did not pass.
- Step 2 inputs should be only:
  - frozen `final_formulation_table_v1.tsv`
  - associated decision trace
  - associated relation artifacts
  - source text/tables

### C. Value fill limited to explicit paper support

- Add one Step 2-only deterministic helper that:
  - keys every output row by `final_formulation_id`
  - reuses Stage5 relation-resolved carry-through rules only for explicit supported fields
  - leaves fields blank when source support is absent
  - forbids donor-fill, modeling-ready derivation, or membership-changing logic
- The helper should borrow its field/evidence conventions from:
  - `export_final_formulation_audit_ready_v1.py`
  - `field_gt_review_seed_rows_v1.tsv`
  - current Stage5 resolved-field usage

### D. Manifest-current full-scope execution

- Generalize `emit_semantic_objects_from_cleaned_papers_v1.py` beyond the current hardcoded paper set.
- The April 11 comparator run proves the chain shape, but not true full-manifest readiness.
- Minimum requirement:
  - replace paper-specific builders with a manifest-driven deterministic emitter path that can process all cleaned full-text papers in `manifest_current.tsv`

### E. Reproducible outputs and run context

- Add a dedicated Step 1 run contract and Step 2 run contract, each with:
  - run purpose
  - exact input files
  - exact script order
  - benchmark-valid vs diagnostic-only status
- Step 2 output should record:
  - `source_run_id`
  - `source_run_dir`
  - `source_files`
  - exact frozen Step 1 final table path

## 6. Open Uncertainties

- The current deterministic semantic emitter is still comparator/fallback infrastructure under current governance, not lawful active Stage2 authority.
- The emitter is currently paper-key-specific and therefore not yet full-manifest-capable.
- The repo does not yet provide a dedicated deterministic Step 2 value-backfill entrypoint.
- Stable IDs appear at Stage5, not earlier. If the experiment wants a pre-Stage5 identity freeze, that would require a different ID-bearing boundary than any maintained current artifact.
- The April 11 comparator run proves the chain over the eligible cleaned full-text subset, not over all `manifest_current.tsv` rows.

## 7. Exact Next Execution Plan

No code changes yet. Next lawful execution plan:

1. Derive the exact Step 1 target scope from `data/cleaned/index/manifest_current.tsv` using `src/stage1_cleaning/derive_target_manifest_v1.py`.
2. Run a fresh comparator-only deterministic Step 1 lineage using the April 11 chain shape:
   - `emit_semantic_objects_from_cleaned_papers_v1.py`
   - `build_stage2_compatibility_projection_v1.py`
   - `build_formulation_relation_artifacts_v1.py`
   - `build_minimal_final_output_v1.py`
3. Run `src/stage5_benchmark/enforce_identity_freeze_v1.py` on that final table against the selected upstream scaffold.
4. Use the resulting frozen final table plus:
   - `final_formulation_table_audit_ready_v1.tsv`
   - `field_gt_review_seed_rows_v1.tsv`
   as the specification input for the missing Step 2 deterministic backfill helper.
5. Only after that audit-backed patch design is approved, implement:
   - emitter generalization for full-manifest coverage
   - the Step 2 frozen-final explicit-only backfill helper

## Facts, Inferences, And Recommendations Summary

### Facts

- March 14 replay baseline is not a true full-text no-LLM extraction baseline.
- April 11 rules-only comparator is the strongest tested no-LLM full-text chain in the repo.
- Stable formulation IDs first appear in the Stage5 final table.
- Existing downstream frozen-final surfaces already preserve the right Step 2 directionality.

### Inferences

- Step 1 should end at frozen Stage5 final IDs.
- Step 2 should not reuse Stage5 closure logic directly.

### Recommendations

- Adopt the April 11 comparator chain as Step 1 baseline skeleton.
- Add identity freeze as the mandatory freeze boundary.
- Build a minimal new Step 2 helper downstream of frozen `final_formulation_id`.
