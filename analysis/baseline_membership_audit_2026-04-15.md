# Baseline Membership Audit (2026-04-15)

The checked current operational baseline can prove downstream no-new-LLM replay coverage from saved `S2-4b` raw responses through `S2-5` -> `S2-6` -> `S2-7` and into `Stage3` -> `Stage5` in the frozen lineage at `data/frozen/dev15_full_pipeline_freeze_v1/`, and it can prove specific Stage5 filtering behavior where the frozen decision trace records rows excluded from the benchmark-facing main table. It cannot prove upstream `S2-2` evidence-construction behavior from that same baseline because the checked replay lineage starts at `S2-4b`, not at `S2-2`. Identity-freeze failure blocks benchmark-valid downstream use, but it does not erase stage-local evidence that a specific Stage5 filter rule fired. `WFDTQ4VX` is therefore a lineage-mismatch proof rather than a Stage5 failure proof: the checked delta is anchored at the raw-response freeze boundary (`data/frozen/dev15_full_pipeline_freeze_v1/s2_4b/analysis/s2_4b_request_summary_v1.tsv` shows a successful persisted `WFDTQ4VX` payload in the updated freeze), while the current Stage5 decision trace for `WFDTQ4VX` shows only generic keep decisions and no Stage5-specific corrective rule firing.

## Patch 1

**Patch key**
`s2_2_table_recovery_doe_signal_expand`

**Patch description**
`S2-2 table recovery enhancement / DOE signal -> go back to S2-2 table surface for expansion`

**A. Which stage boundary owns this fix?**

`S2-2 Evidence construction`, specifically the governed `candidate_blocks_v1.json` plus canonical `evidence_blocks_v1.json` boundary owned inside `S2-2`. If `normalized_table_payloads_v1.json` is mentioned, it should be treated here as design-intended execution-facing table preservation discussed in governance, not as the canonical S2-2 authority surface named by this audit.

**B. Is that boundary covered by the current operational baseline?**

No. The checked frozen operational lineage records stage coverage beginning at `S2-4b` and continuing through `S2-5` -> `S2-6` -> `S2-7` -> `Stage3` -> `Stage5`. It does not record an `S2-2` rerun as part of the current operational baseline.

**C. Does the current baseline prove the fix is part of the baseline?**

No. The current baseline proves replay from saved raw responses and downstream processing after `S2-4b`; it does not prove that the upstream `S2-2` table-recovery behavior itself was traversed and frozen as part of that baseline.

**D. If not proven, what is the missing evidence step?**

A lawful maintained lineage that actually re-executes `S2-2` from cleaned assets, writes the governed `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json` and canonical `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json` artifacts, and then carries that same lineage forward through `S2-4b` -> `S2-7` -> `Stage3` -> `Stage5` with `RUN_CONTEXT.md` explicitly recording `S2-2` coverage.

**E. What is the strongest evidence file or artifact path supporting the judgment?**

`data/frozen/dev15_full_pipeline_freeze_v1/RUN_CONTEXT.md`

**Why this judgment follows from checked evidence**

- `project/2_ARCHITECTURE.md`, `project/ACTIVE_PIPELINE_FLOW.md`, and `project/ACTIVE_PIPELINE_RUNBOOK.md` assign the governed `S2-2` boundary to `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json` plus canonical `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`, not to `S2-5+`.
- `data/frozen/dev15_full_pipeline_freeze_v1/RUN_CONTEXT.md` lists current checked stage coverage as `S2-4b` -> `S2-5` -> `S2-6` -> `S2-7` -> `Stage3` -> `Stage5` -> `identity_freeze`.
- `data/frozen/dev15_stage2_freeze_v1/FREEZE_MANIFEST.md` says the promoted `S2-5` -> `S2-7` coverage is bounded and that the portable freeze does not itself prove full `S2-2` re-execution inside the downstream replay lineage.
- `data/frozen/dev15_full_pipeline_freeze_v1/s2_7/RUN_CONTEXT.md` confirms the promoted `S2-7` run consumes validated `S2-5` artifacts and stops before `Stage3`, while its feature-activation table marks the `S2-2` evidence-contract units as missing in the checked replay subset.

## Patch 2

**Patch key**
`stage5_exclude_post_processing_from_main_table`

**Patch description**
`post-processing rows do not enter benchmark-facing main table`

**A. Which stage boundary owns this fix?**

`Stage5`, at `S5-2 Filtering / normalization` into `S5-3 Final table`, where benchmark-facing identity closure excludes downstream/post-processing descendants from `final_formulation_table_v1.tsv`.

**B. Is that boundary covered by the current operational baseline?**

Yes. The checked frozen operational lineage records coverage through `Stage5`.

**C. Does the current baseline prove the fix is part of the baseline?**

Yes for observed behavior in scope. The checked `Stage5` decision trace in the frozen full-pipeline lineage records rows classified as `post_processing_or_measurement_variant` with decision `filtered_non_formulation`, so the current checked baseline proves that this exclusion behavior was observed in the maintained Stage5 closure pass. Separately, the same lineage is not benchmark-valid overall because identity freeze failed.

**D. If not proven, what is the missing evidence step?**

Not required for the observed exclusion claim. If preservation-side proof is also needed, the same lineage should additionally pin the sibling `downstream_variant_records_v1.tsv` output.

**E. What is the strongest evidence file or artifact path supporting the judgment?**

`data/frozen/dev15_full_pipeline_freeze_v1/stage5/final_output_decision_trace_v1.tsv`

**Why this judgment follows from checked evidence**

- `project/ACTIVE_PIPELINE_RUNBOOK.md`, `project/2_ARCHITECTURE.md`, and `docs/maintained_script_surface.tsv` all assign this behavior to the maintained `Stage5` benchmark-final builder.
- `data/frozen/dev15_full_pipeline_freeze_v1/RUN_CONTEXT.md` shows that the checked operational lineage reaches `Stage5`.
- `data/frozen/dev15_full_pipeline_freeze_v1/stage5/final_output_decision_trace_v1.tsv` records concrete exclusions such as:
  - `L3H2RS2H form_004`
  - `L3H2RS2H form_006`
  - `L3H2RS2H form_007`
  - `5GIF3D8W form_ctrl`
  - `PA3SPZ28 fc_2`
- Those rows are marked `decision=filtered_non_formulation`, `variant_class=post_processing_or_measurement_variant`, `benchmark_default_include=no`, with decision rule `parent_linked_non_synthesis_descendant_variant`.
- This proves observed Stage5 exclusion behavior in the checked baseline scope.
- Separately, `data/frozen/dev15_full_pipeline_freeze_v1/identity_freeze/identity_freeze_summary_v1.tsv` shows that the same lineage remains non-benchmark-valid overall because identity freeze failed.
