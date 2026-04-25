# Discussion Governance Landing Audit

## Executive summary

The recent Stage2 debugging conclusions were landed into governance and memory in a focused way without changing pipeline code.

Landed conclusions:

- `S2-2b` now has explicit governance wording for a confirmed-noise-only irreversible table-drop policy.
- `S2-3 / S2-4a` now explicitly state that the normal maintained summary path is neutral across preserved tables and that the residual issue is lossy compression, not primary-table ranking.
- the summary contract now explicitly prioritizes header / column schema and first-column row identity surfaces over sample rows.
- authority reopen handles are now explicitly governed as deterministic execution-side metadata rather than LLM semantic content.
- replay preference is now explicit: if the LLM-facing contract is unchanged, prefer replay from frozen raw responses when deterministic metadata can be lawfully reattached.
- failure-family anchors were landed into memory for `5ZXYABSU` and `5GIF3D8W`.

No pipeline code was changed. No rerun occurred. No push occurred. `ACTIVE_RUN.json` was not changed.

## What conclusions were landed

### 1. S2-2b table policy correction

Landed as explicit contract language:

- only confirmed pure noise may be irreversibly removed
- if a table is not confirmed noise, preserve it in the pre-LLM authority surface
- rules must not downrank or suppress a table because another table seems more important

### 2. S2-3 / S2-4a summary neutrality

Landed as explicit contract language:

- the normal maintained summary path is neutral across preserved tables
- the residual problem is lossy summary compression, not cross-table importance bias

### 3. Summary contract clarification

Landed as explicit contract language:

- header / column schema and first-column row identity are the primary table-summary structure
- sample rows are optional aids only
- deterministic rules should not pre-explain cross-table semantic relationships

### 4. Authority metadata boundary correction

Landed as explicit contract language:

- `authority_run_dir`, `authority_payload_root`, and related reopen handles are deterministic execution-side metadata
- they must not be treated as LLM semantic content
- replay should reattach them through sidecar / reattachment surfaces

### 5. Failure-family records

Landed into memory:

- `5ZXYABSU` as the selector/preservation failure-family anchor
- `5GIF3D8W` as the non-DOE single-variable recovery failure-family anchor

### 6. Replay principle

Landed as explicit governance language:

- if the LLM-facing contract is unchanged, prefer replay from frozen raw responses over fresh live calls when lawful deterministic reattachment is available

## Files changed

Governance/docs changed:

- [AGENTS.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/AGENTS.md)
- [README.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/README.md)
- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md)
- [project/ACTIVE_PIPELINE_FLOW.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)
- [project/ACTIVE_PIPELINE_RUNBOOK.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_RUNBOOK.md)
- [project/4_DECISIONS_LOG.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/4_DECISIONS_LOG.md)

Memory changed:

- [data/mem/v1/dec.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/mem/v1/dec.tsv)
- [data/mem/v1/err.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/mem/v1/err.tsv)
- [data/mem/v1/idx.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/mem/v1/idx.tsv)

Audit package created:

- [analysis/repo_governance/discussion_governance_landing_audit.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/analysis/repo_governance/discussion_governance_landing_audit.md)
- [analysis/repo_governance/discussion_governance_landing_audit.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/analysis/repo_governance/discussion_governance_landing_audit.tsv)

## What was already present vs newly added

Already partially present before this task:

- summary-only `S2-4a` contract
- neutral stable order wording for preserved table summaries
- replay support from frozen raw responses
- split between semantic-facing summary view and execution-facing table authority

New or materially strengthened in this task:

- explicit confirmed-noise-only irreversible table-drop rule
- explicit statement that summary neutrality is not the main current problem and that lossy compression is
- explicit statement that header/schema and first-column row identity are the primary summary contract
- explicit authority-metadata boundary: execution-side, not semantic content
- explicit replay-preference rule tied to unchanged LLM-facing contract
- formalized failure-family anchors in memory

## Exact memory/update actions taken

Added decision memory rows:

- `MDEC094` — `S2-2b confirmed-noise-only table preservation`
- `MDEC095` — `S2-3/S2-4a summary neutrality and structure-first summary contract`
- `MDEC096` — `Authority reopen metadata is deterministic execution-side metadata`
- `MDEC097` — `Prefer replay from frozen raw responses when the LLM-facing contract is unchanged`

Added failure-family memory rows:

- `MERR1133` — `5ZXYABSU selector/preservation family`
- `MERR1134` — `5GIF3D8W non-DOE single-variable recovery family`

Index rows added:

- `MIDX2001`
- `MIDX2002`
- `MIDX2003`
- `MIDX2004`
- `MIDX2005`
- `MIDX2006`

Memory update path used:

- `python3 src/utils/update_mem_v1.py ...`

No full memory rebuild was run because this was a targeted governance landing rather than a schema or source-tree rebuild.

## Unresolved wording conflicts

One important residual remains:

- governance now states a confirmed-noise-only irreversible table-drop policy, but older maintained code and some earlier historical wording still expose `must_include` / `optional_context` / `hard_drop` terminology and selector behaviors that were designed before this correction

This means:

- the new governance language is now the authoritative target
- implementation is not yet fully aligned
- future selector work should treat this as a code/governance alignment task, not as a reason to weaken the policy wording again

There is also a bounded vocabulary tension:

- some docs still retain older “primary authority” terminology inside `S2-2a` structure-ranking discussion
- this task did not remove those historical terms because the current landing was scoped to policy clarification rather than full wording normalization

## Future impact

- Future Stage2 selector work now has an explicit contract against importance-based table loss.
- Future summary work now has an explicit target: preserve schema and row-identity surfaces rather than relying on sample rows.
- Future replay work now has explicit authority to prefer raw-response replay when the LLM-facing contract is unchanged.
- Future debugging sessions can recover these conclusions quickly from `mem_v1` rather than rediscovering them from analysis files.
