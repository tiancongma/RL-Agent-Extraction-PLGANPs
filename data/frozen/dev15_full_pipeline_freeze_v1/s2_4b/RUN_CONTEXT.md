# RUN_CONTEXT

## 1. Run ID
`dev15_stage2_freeze_v1/s2_4b`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This frozen surface is `diagnostic-only, not benchmark-valid final output`.
- It is a consolidated `S2-4b` raw-response freeze only.
- It is not itself a lawful Stage3 resume boundary.
- It becomes downstream-usable Stage2 authority only if later replayed through the maintained composite Stage2 path.

## 3. Purpose
- Preserve the full DEV15 `S2-4b` raw-response surface for the frozen `S2-4a` prompt lineage.
- Reuse already-successful same-lineage raw responses when the frozen prompt SHA is unchanged.
- Add only the newly generated completion-pass responses needed to reach full DEV15 coverage.
- Stop at raw-response persistence with no semantic parsing, validation, projection, or downstream execution.

## 4. Stage Boundary
- current_stage_boundary: `S2-4b`
- upstream_frozen_dependency: `S2-4a`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::call_gemini_stream_collect`
- stop_boundary: `raw_response_payloads_written`
- next_lawful_step: `S2-5 semantic parsing`, but only through maintained composite Stage2 replay

## 5. Frozen Inputs
- prompts_jsonl: `data/frozen/dev15_stage2_freeze_v1/s2_4a/analysis/s2_4a_prompts_v1.jsonl`
- prompt_template: `data/frozen/dev15_stage2_freeze_v1/s2_4a/analysis/s2_4a_prompt_template_v1.txt`
- prompt_audit: `data/frozen/dev15_stage2_freeze_v1/s2_4a/analysis/s2_4a_prompt_audit_v1.tsv`
- freeze_manifest: `data/frozen/dev15_stage2_freeze_v1/FREEZE_MANIFEST.md`
- expected_dev15_paper_count: `15`

## 6. Consolidation Sources
- `data/results/20260412_8517d36/13_H_s2_4b_parallel2_t180_r1_sleep5_v1`
- `data/results/20260412_8517d36/04_s2_4b_live_llm_call_dev15_v1`
- `data/results/20260412_8517d36/14_s2_4b_completion_remaining5_v1`
- `data/results/20260412_8517d36/10_E_s2_4b_parallel2_t180_r1_v1`

## 7. Consolidation Summary
- reused_successful_raw_responses: `10`
- newly_generated_raw_responses: `5`
- unresolved_papers: `0`
- full_dev15_raw_response_coverage: `yes`

## 8. Outputs
- raw responses:
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/raw_responses`
- request metadata sidecars:
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/request_metadata`
- request summary:
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/analysis/s2_4b_request_summary_v1.tsv`
- run metadata:
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/stage2_s2_4b_run_metadata_v1.json`

## 9. Boundary Meaning
- This task is `S2-4b` only.
- It reuses frozen `S2-4a` prompts.
- It preserves existing successful raw responses when the LLM-facing input is unchanged.
- It reruns only missing or failed papers under the bounded call policy.
- The resulting full raw-response set is a frozen replayable input surface for the next step.
- It is `NOT yet a lawful Stage3 resume boundary until replayed through the maintained composite Stage2 path`.

