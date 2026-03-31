# Cleanup Wave 1 Post-State Summary

## What moved

- Five pre-reduction legacy scripts were moved from `src/legacy/` into `src/archive_methods/pre_reduction_legacy/`.
- Seven audited delete candidates were moved into `src/archive_methods/delete_candidates_pending_confirmation/`.
- No `run_*` directory was physically moved in wave 1; instead, non-compliant and current engineering runs were segregated with top-level results indexes.

## What was not moved and why

- Historical scripts already under `src/archive_methods/` and not marked for wave-1 action were left in place to avoid broad archive churn.
- Non-compliant run directories were not physically moved because many historical docs and investigations still reference their current paths.
- Active-path scripts and current evaluation artifacts were left untouched to keep pipeline risk low.

## Script disposition counts before vs after

- `keep_in_src`: before `80` -> after `85`
- `keep_but_mark_branch_only`: before `41` -> after `41`
- `move_to_archive`: before `5` -> after `0`
- `delete_candidate_after_confirmation`: before `7` -> after `7`

## Run reproducibility counts before vs after

- `fully_reproducible`: before `0` -> after `0`
- `partially_reproducible`: before `4` -> after `4`
- `minimally_documented`: before `0` -> after `0`
- `non_compliant`: before `44` -> after `44`

## Remaining cleanup backlog

- Update or prune additional historical scripts still classified as branch-only but not part of the current mainline.
- Decide whether the delete-candidate quarantine scripts should be fully removed after human confirmation.
- Backfill reproducibility-grade run specs for current engineering runs, then decide whether historical runs should be compressed or relocated physically.

## Recommended wave 2 focus

- Primary recommendation: `CLEANUP_WAVE2` with emphasis on deeper `src/` pruning and run history compression before final-output layer work.