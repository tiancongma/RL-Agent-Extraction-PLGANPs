# LLM Reuse Replay Policy V1

## Context

Recent validation work in this repository has mixed two different activities:

- rerunning LLM extraction because model-facing input changed
- replaying downstream deterministic logic over already existing raw LLM outputs

This note records the repository policy for distinguishing those cases.

## Confirmed Repository Evidence

- The active pipeline separates LLM-driven Stage 2 extraction from deterministic
  downstream layers:
  - `project/ACTIVE_PIPELINE_FLOW.md`
  - `project/PIPELINE_SCRIPT_MAP.md`
- Recent DEV15 engineering work already produced bounded replay-style
  validation runs inside the `2026-03-13 / f4912f3` family and repeatedly
  compared deterministic downstream changes against an existing baseline.
- The repository also shows that some older runs do not always record every
  model/config detail with equal completeness, so strict equivalence claims
  must stay evidence-based and conservative.

## Policy

- Reuse existing raw LLM outputs whenever the code change does not alter the
  LLM-facing input.
- Rerun LLM calls whenever the LLM-facing input changes.

## LLM-Facing Input Definition

LLM-facing input includes:

- prompt text
- source window or context selection
- table/text context sent to the model
- model name or version
- sampling or generation configuration
- any upstream logic that changes what is sent to the model

## Reuse-Eligible Downstream Changes

Existing raw LLM outputs should be reused when changes are limited to:

- weak-label parsing
- deterministic candidate generation
- merge, overlap, or dedup logic
- relation or provenance processing
- final table generation
- audit export
- confidence or review export

## Naming Guidance

- If a run reuses previous raw LLM outputs, describe it as a replay or
  reuse-LLM validation run.
- Do not describe that run as a fresh full-regeneration run.

## Conservative Rule

- If repository artifacts do not prove that the full LLM-facing input is
  unchanged, treat strict raw-output reuse equivalence as unproven.
- In that case, documentation should say so explicitly rather than implying a
  stronger equivalence claim than the run metadata supports.
