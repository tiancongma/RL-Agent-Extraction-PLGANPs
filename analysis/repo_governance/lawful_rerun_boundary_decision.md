# Lawful Rerun Boundary Decision

## Scope
- baseline anchor: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- current retained repair set: `analysis/stage2_audits/post_09_repair_retention_set.md`
- inspected integrated run: `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1`

## What changed since the old baseline
Separate:
- Stage0
  - No Stage0 script delta was detected in the targeted post-anchor diff against `385b6e1`.
- Stage1
  - `src/stage1_cleaning/pdf2clean.py` changed, but the observed delta is path remapping for manifest-backed source resolution across machines.
  - This is an upstream path-resolution repair, not a semantic clean-text extraction change tied to the preserved post-`09` paper repairs.
- Stage2
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py` changed to refresh authoritative `text_path` and `table_dir` bindings before Stage2 execution.
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` changed substantially in candidate segmentation, role-aware table selection, sequential table supplementation, restoration-profile handling, and evidence/prompt-facing table rendering.
  - `src/stage2_sampling_labels/table_row_expansion_v1.py` also changed after the anchor.
  - These are LLM-facing changes because they alter the governed candidate space, evidence blocks, and prompt material that feed `S2-4b`.
- Stage3
  - No Stage3 code delta was detected in the targeted post-anchor diff.
- Stage5
  - `src/stage5_benchmark/enforce_identity_freeze_v1.py` and `src/stage5_benchmark/compare_final_table_to_gt_v1.py` changed to a diagnostic, non-blocking compare contract.
  - These are downstream deterministic reporting changes, not a basis to reuse stale Stage2 LLM outputs.

## Earliest lawful rerun boundary
- exact boundary
  - `Stage2 S2-2 clean text / extracted tables -> governed evidence package (pre-LLM)`
- why it is earliest lawful
  - The preserved post-`09` repairs that matter for `UFXX9WXE`, `5GIF3D8W`, and `QLYKLPKT` change the candidate/evidence handoff before prompt construction, and `run_stage2_composite_v1.py` now refreshes the text/table bindings that feed that same pre-LLM boundary.
  - Because `S2-4b` prompts are derived from `evidence_blocks_v1.json`, any change at `S2-2` invalidates replay equivalence for the old raw responses.
- why later reuse would be invalid or unproven
  - Reusing `06` or `09` raw responses would pair old prompts with new evidence-selection logic.
  - The inspected integrated run already proves that replay was used rather than a lawful fresh rerun, and no strict equivalence artifact exists showing that the current `S2-2` evidence package is identical to the replay source.

## Rerun policy
- whether fresh maintained S2-4 live calls are required
  - Yes. A lawful true baseline from the current retained repair set requires maintained fresh `S2-4b` live calls.
- whether Stage1 can be reused
  - Yes. Existing Stage1 clean-text and extracted-table assets can be reused because the decisive post-anchor behavior change is at Stage2 pre-LLM evidence construction, and the observed Stage1 delta is a path-resolution repair rather than a paper-proven semantic extraction change.
- which old artifacts are allowed to be reused
  - The declared DEV15 scope manifest.
  - Existing Stage1 clean-text assets and extracted tables that remain authoritative for that scope.
- which old artifacts must NOT be reused
  - `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/`
  - Any replayed Stage2 raw responses derived from pre-repair `S2-2` evidence packages.
  - Downstream Stage3 and Stage5 artifacts built from those replayed Stage2 outputs.

## Decision
The earliest lawful boundary for a true new baseline is Stage2 `S2-2` pre-LLM evidence construction, so a valid rerun must reuse Stage1 only, regenerate governed Stage2 evidence and prompts, and execute maintained fresh `S2-4b` live calls before rebuilding all downstream stages.
