# Fresh LLM Rerun Verification

## Scope
- inspected run path: `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1`
- baseline anchor path: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- question being answered: whether the inspected integrated baseline candidate executed a fresh maintained Stage2 `S2-4b` live LLM rerun

## Evidence reviewed
- `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1/RUN_CONTEXT.md`
- `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1/stage2_run_metadata_v1.json`
- `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1/semantic_stage2_objects/raw_responses/`
- `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/`
- `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`

## Verification result
- fresh_llm_rerun = no

## Evidence
- RUN_CONTEXT evidence
  - `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1/RUN_CONTEXT.md` records `source_mode: legacy_llm_replay` and names `replay_raw_responses_dir: data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses`.
- raw response evidence
  - The inspected run contains `semantic_stage2_objects/raw_responses/*.json`, but sampled files for `UFXX9WXE`, `5GIF3D8W`, and `QLYKLPKT` are byte-identical to the corresponding files in `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/`.
- request metadata evidence
  - The inspected run contains `stage2_run_metadata_v1.json`, but it does not contain the maintained `request_metadata/` directory or per-paper `__stage2_v2_request_metadata.json` files that `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py` writes for fresh `S2-4b` execution.
- replay/live-call evidence
  - `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1/stage2_run_metadata_v1.json` records `source_mode: legacy_llm_replay` and `legacy_raw_responses_dir: data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses`.
  - No request-summary TSV or live-call request artifacts are present under the inspected run.

## Conclusion
The inspected integrated run reused prior raw responses from `06_dev15_full_baseline_no_marker_live` and does not contain the maintained `S2-4b` request-metadata surface that a fresh live rerun would have produced, so it is not a fresh LLM rerun.
