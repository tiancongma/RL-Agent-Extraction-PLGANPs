# Cleanup Wave 2 Script State Summary

## Before vs after disposition counts

- `keep_in_src`: before `85` -> after `37`
- `keep_but_mark_branch_only`: before `41` -> after `41`
- `move_to_archive`: before `0` -> after `0`
- `delete_candidate_after_confirmation`: before `7` -> after `0`

- Files removed from `src/` active surface: `52`
- `src/` now contains only active or explicitly branch-active engineering code.
- Remaining ambiguity is limited to some branch/supporting utilities whose ownership is clear enough to stay in `src/`, but whose future value should still be revisited in later cleanup.