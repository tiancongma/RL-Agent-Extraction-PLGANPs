# Archived baselines and how to reproduce them

This document records baseline snapshots that are intentionally preserved for comparison and rollback during extraction pipeline development.

## baseline_pre_tablefirst

The `baseline_pre_tablefirst` baseline represents the repository state before the table-first evidence binding changes were introduced. It is preserved so that we can reproduce pre-change behavior, run fair before/after evaluations, and recover a known reference point for debugging.

The baseline is archived on origin using both a branch and a tag:

- Remote branch: `origin/baseline_pre_tablefirst`
- Remote tag: `baseline_pre_tablefirst_local`

To reproduce this baseline in a clean directory, use the exact commands below:

```bash
git clone https://github.com/tiancongma/RL-Agent-Extraction-PLGANPs.git
cd RL-Agent-Extraction-PLGANPs
git fetch --all --tags
git checkout baseline_pre_tablefirst
# or:
git checkout tags/baseline_pre_tablefirst_local
```

Use the branch when you want a baseline line that can still receive follow-up baseline-only tweaks. Use the tag when you need an immutable snapshot for strict reproducibility in reports, audits, or experiments that must not drift over time.
