# Authority Sidecar Replay Refactor Report

## Executive Conclusion

- Authority metadata is now reattached through a deterministic sidecar written at `S2-5` and consumed at `S2-7`; it is no longer dependent on the LLM to carry replay-critical reopen handles.
- The replay reused frozen raw responses from `data/results/20260421_43ed145/02_s2_4b` and avoided fresh live LLM calls.
- The replayed baseline materially improved final-row volume from `35` to `117`, while `benchmark_valid` remains `no` because identity freeze still fails.
- Compare status remains diagnostic-only: matched papers stayed `1 -> 1`, but the recovered rows now survive into Stage5 for `UFXX9WXE` and `WIVUCMYG`, and `5GIF3D8W` now retains explicit table-anchor plus single-variable recovery rows.

## What Was Refactored

- `S2-5` now resolves authority metadata deterministically from frozen request metadata plus the frozen `S2-4a` prompt provenance (`source_prompts_jsonl_path -> source_evidence_artifact_path -> accepted pre-LLM authority root`).
- `S2-5` writes `semantic_stage2_objects/authority_reattachment_sidecar_v1.json` containing per-paper `authority_run_dir`, `authority_payload_root`, resolution metadata, and table locator entries.
- `S2-7` now reads that sidecar and reattaches authority metadata before compatibility projection. Semantic JSONL authority fields remain advisory only; the sidecar is the deterministic runtime source of truth for replayed reopen handles.
- The row-emission repairs from the previous patch were then exercised against the replayed semantic surface without changing prompts, selector logic, or any LLM behavior.

## Deterministic Authority Handling

- Sidecar written at: `data/results/20260421_c8f4b61/01_s2_5/semantic_stage2_objects/authority_reattachment_sidecar_v1.json`
- Sidecar read at: `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection` via `run_stage2_s2_7_compatibility_projection_v1.py`
- Semantic JSONL still carries `authority_run_dir` and `authority_payload_root` when available, but those fields are now advisory transport only for replay convenience; the sidecar is authoritative for deterministic reopen.
- Sidecar resolution entries: `15`

## Replay Inputs And Execution

- source frozen S2-4b lineage: `data/results/20260421_43ed145/02_s2_4b`
- replay lineage root: `data/results/20260421_c8f4b61`
- fresh live LLM calls avoided: `yes`
- exact replay chain:
  1. `01_s2_5`
  2. `02_s2_6`
  3. `03_s2_7`
  4. `04_stage3`
  5. `05_stage5`
  6. `06_compare`

## Compare Result Vs Previous Diagnostic Baseline

- previous diagnostic baseline: `data/results/20260421_43ed145` -> total final rows `35`, matched papers `1`, benchmark_valid `no`
- replay baseline: `data/results/20260421_c8f4b61` -> total final rows `117`, matched papers `1`, benchmark_valid `no`
- identity freeze result: still failed (`RISKS_RECORDED_NON_BLOCKING`)
- replay classification: `diagnostic-only, not benchmark-valid final output`

## 5GIF3D8W

- Stage2 completed row count: `12`
- Stage5 final row count: `11`
- compare status: `under` (`11` vs GT `26`)
- explicit table-anchor rows present: `yes`, count `4`
- clear table-anchor labels retained include: `PLGA 50/50 / Empty`, `PLGA 50/50 / Drug loaded`, `PLGA 75/25 / Empty`, `PLGA 75/25 / Drug loaded`
- blank-control style explicit anchor rows retained: `2`
- retained blank-control anchors: `PLGA 50/50 / Empty`, `PLGA 75/25 / Empty`
- Are all 8 expected explicit anchor formulations present? `no`
  Reason: the replay now retains the explicit anchor rows it emitted, but current replay-time table-anchor extraction still materializes only 4 explicit Table 4 anchor rows, not 8. The shortfall is upstream of Stage5 retention.
- Premature exclusion because of missing EE: `no evidence`. The observed filtered `5GIF3D8W` row was `Drug-free PLGA/PCL nanoparticles` and it was excluded by `parent_linked_non_synthesis_descendant_variant`, not by any EE-based rule.

## UFXX9WXE

- Stage2 completed row count: `28`
- Stage5 final row count: `28`
- compare status: `over` (`28` vs GT `27`)
- recovered rows survive through Stage5: `yes`
- dominant retained recovery source: `table_row_expansion_v1`

## WIVUCMYG

- Stage2 completed row count: `29`
- Stage5 final row count: `29`
- compare status: `over` (`29` vs GT `26`)
- recovered rows survive through Stage5: `yes`
- dominant retained recovery source: `doe_numbered_table_row_recovery`

## Residual Limitations

- `benchmark_valid` remains `no` because compare is still diagnostic and identity freeze still fails.
- `5GIF3D8W` improved materially, but the replay still does not materialize all expected explicit anchor formulations from the characterization-heavy table family.
- The target replay blocker was resolved for `5GIF3D8W`, `UFXX9WXE`, and `WIVUCMYG`, but unrelated papers still show some `missing_table_authority_payload` skips where table-scope references remain unresolved against the sidecar locator set.
- The replay demonstrates transport and survival of recovered rows; it does not by itself solve all remaining paper-specific Stage5 identity and count mismatches.

## FACTS

- Frozen raw responses from `20260421_43ed145/02_s2_4b` were reused; no fresh live calls were made.
- `S2-5` wrote `authority_reattachment_sidecar_v1.json` with `15` entries.
- Stage5 total rows changed from `35` to `117`.
- `UFXX9WXE` changed from `2` to `28` final rows.
- `WIVUCMYG` changed from `3` to `29` final rows.
- `5GIF3D8W` changed from `1` to `11` final rows.

## INFERENCES

- The authority sidecar refactor solved the replay-compatibility gap that previously blocked `S2-7` with `missing_table_authority_payload`.
- The surviving row gains on `UFXX9WXE` and `WIVUCMYG` show that the repaired post-reopen emitters are now functioning inside the governed replay baseline, not just in bounded diagnostics.
- The remaining `5GIF3D8W` deficit is not explained by missing-EE pruning at formulation-existence / row-retention stages.

## UNCERTAINTIES

- The exact additional anchor formulations still missing for `5GIF3D8W` require a separate localized audit of table-anchor extraction breadth versus expected article-native instances.
- Identity-freeze violations remain broad enough that this replay cannot yet be treated as benchmark-valid evidence.
