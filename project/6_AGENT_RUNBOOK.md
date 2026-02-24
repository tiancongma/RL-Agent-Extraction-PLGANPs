# Agent Runbook

This document defines the default execution contract for automated agents working in this repository.

---

## Preflight Read Order

Before starting implementation work, read in this order:

1. `README.md`
2. `project/2_ARCHITECTURE.md`
3. `project/FEATURE_EE_COVERAGE_RL_SCOPE.md` (when working on EE coverage branch tasks)
4. `docs/tool_index.md`
5. Relevant active diagnostics under `docs/ee_coverage_rl/`

---

## Invariants

- Benchmark/view logic must not constrain Full DB upstream.
- Evidence gating must not reduce Full DB row counts unless explicitly fixing splitting/dedup.
- All human-review debug outputs must include DOI metadata: `reference_normalized_doi` and `doi_url`.
- Avoid committing `data/results/` outputs by default.
- Require regression checks (before/after) for any behavior change.
- Keep commits small and scoped.

---

## Workflow

1. Identify which layer the bug belongs to:
   - extraction
   - evidence
   - instance
   - confidence
   - view
2. Change only that layer.
3. When diagnosing, generate a human-auditable debug matrix.
4. Write a step report under `docs/ee_coverage_rl/` and update `docs/ee_coverage_rl/README.md`.
5. Update project docs when system-level behavior changes.

