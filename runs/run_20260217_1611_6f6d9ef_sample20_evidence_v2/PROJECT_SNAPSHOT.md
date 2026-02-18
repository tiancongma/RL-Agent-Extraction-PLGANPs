PROJECT SNAPSHOT
RL-Agent-Extraction-PLGANPs
Date: 2026-02-17 (local)

Project Goal
Build a reproducible, evidence-grounded pipeline to extract formulation-level PLGA NP data with field-level evidence and an evidence-only verifier.

Current Branch
feature/evidence-spans-v1 (你当前正在开发的分支名以 git status 为准)

Latest Run (Extractor baseline)
run_20260217_1611_6f6d9ef_sample20_evidence_v2

Input TSV: data/results/run_20260217_1611_6f6d9ef_sample20_evidence_v2/weak_labels__gemini.tsv

Evidence contract present: field_evidence_json + text_sha256

Verifier Path-2 Implemented (NEW)
Added three additive scripts under src/stage4_eval/:

build_verifier_requests_v1.py

Builds verifier_requests__hybrid_v1.jsonl + verifier_rule_gates__hybrid_v1.jsonl

Deterministic request_id = sha256(run_id|key|formulation_id|field_name|extracted_value|text_sha256|span_start|span_end)

Conservative numeric gate: value_not_in_evidence

Bugfix: empty extracted_value is no longer gated as value_not_in_evidence (prevents inflated n_rule)

run_verifier_requests_v1.py

Evidence-only verifier runner with --resume

Deterministic prompt + strict JSON parsing

Added tqdm progress bar (so long runs show progress)

apply_verifier_responses_v1.py

Recomputes request_id and merges (responses + gates) back into TSV

Outputs verified TSV + conflict queue + per-field summary

Adds only meta columns to TSV: verifier_policy, verifier_model, verdict_json, verdict_source_json

Verifier Model Choice
Primary verifier switched to Gemini 2.5 Flash Lite for throughput and stability.
Output files are kept separate from Gemma runs to avoid mixing responses.

Current Outputs (for Flash Lite verifier)

verifier_requests__hybrid_v1.jsonl

verifier_rule_gates__hybrid_v1.jsonl

verifier_responses__hybrid_v1__flash_lite.jsonl

verified__hybrid_v1__flash_lite.tsv

conflict_queue__hybrid_v1__flash_lite.tsv

verifier_summary__hybrid_v1__flash_lite.tsv

Where you are now

Architecture STATE_2 is operational: Extractor → Evidence → (Rule gate + LLM verifier) → Aggregated verdicts

Pipeline is stable, resume-safe, and scalable beyond sample20.

Next Step (Tomorrow’s task)

Confirm Flash Lite verifier completed: n_responses == n_requests

Use verifier_summary__hybrid_v1__flash_lite.tsv to perform system-level analysis:

extractor abstention rate (empty values)

evidence-supported rate among non-empty values

rule-gate rate (value not in evidence) as hallucination proxy

verifier contradicted frequency and which fields trigger it

Decide whether to add a small “Gemma spot-check verifier” only on high-risk cases (contradicted + selected insufficient) for heterogeneity, without slowing the full run.