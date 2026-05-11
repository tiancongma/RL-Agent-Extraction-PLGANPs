# Root Cause Summary

## What changed since baseline

- The baseline for this audit is:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- The current lineage under audit is:
  - `data/results/20260418_9538ec2`

Main changes since baseline:

- Stage2 selector logic changed in `extract_semantic_stage2_objects_v2.py`
  - role-aware table selection promoted into the full-table path
  - sequential-optimization local pack added
  - trimmed context fallback added
  - prompt layout switched to evidence-pack mode when selector strategy is role-aware

- Runtime mode changed
  - baseline used replayed raw responses
  - current lineage re-ran fresh live Gemini calls from `S2-2`
  - only `10/15` papers reached completed Stage2

## Why current system partially fails

- Failure mode 1:
  - fresh live coverage is incomplete
  - four papers disappear entirely in current downstream artifacts because the restart never completed them

- Failure mode 2:
  - selector-driven evidence packs are not uniformly precise
  - `UFXX9WXE` improved
  - `WFDTQ4VX` regressed sharply
  - `QLYKLPKT` improved only slightly and still leaks assay/noise

- Failure mode 3:
  - prompt construction improved structurally but did not fully solve burden
  - evidence-pack prompts remain large
  - `V99GKZEI` still falls back to raw-prefix style and truncates

## Core failure mode

- `mixed`

Reason:

- The current discrepancy cannot be explained by one layer alone.
- It is the combination of:
  - `selector issue`
  - `prompt construction issue`
  - `runtime/live-call incompleteness`

## Most critical regression point

- `WFDTQ4VX`

Why:

- It was a preserved restored behavior in the April 17 baseline.
- It collapses from `27` final rows to `2` in the current lineage.
- That is the clearest evidence that the new mainline did not preserve an important repaired capability.

## One biggest change

- The biggest change since baseline is the promotion of role-aware evidence-pack prompting into the default full-table path for selected papers, combined with a switch from replayed raw responses to fresh live S2-2 restart execution.

## S2-2 quality

- `mixed`
- Cleaner for `UFXX9WXE`
- Still noisy for `QLYKLPKT`
- Poor for `WFDTQ4VX`

## S2-3 inputs

- Mostly `full-table mixed evidence packs`, not summary tables
- `V99GKZEI` still uses fallback raw-prefix + table excerpts and truncates

## Recommended next action

- `peer-review selector/runtime split`

## Caveat

- This root-cause summary is diagnostic-only.
- It should not be used for major decisions or investment priorities without peer review.
