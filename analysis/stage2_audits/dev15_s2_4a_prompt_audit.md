# DEV15 S2-4a Prompt Audit

## Executive summary

Important scope note:

- The requested post-fix lineage [20260421_9c4a03f/04_s2_4a_prompt_construction_postfix](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/04_s2_4a_prompt_construction_postfix) is **not a full DEV15 freeze**.
- It contains **6 papers**, not 15:
  - `5ZXYABSU`
  - `WFDTQ4VX`
  - `INMUTV7L`
  - `WIVUCMYG`
  - `UFXX9WXE`
  - `5GIF3D8W`
- So this audit is a **bounded post-fix prompt-surface audit on all papers present in that lineage**, not a true DEV15-wide 15-paper audit.

Main finding:

- The post-fix `S2-4a` prompt surface remains **neutral** and **structurally field-complete** in the bounded lineage:
  - all selected blocks are included
  - table summaries still carry `key_columns`
  - table summaries still carry `row_identifier_pattern`
  - no prompt-level truncation is recorded
- The new dominant risk is **prompt bloat**, not primary-table bias.
- The residual structural issue is **summary degradation in many preserved weak tables**, especially on:
  - `5ZXYABSU`
  - `WFDTQ4VX`
  - `INMUTV7L`
  - `UFXX9WXE`

High-level conclusion:

- Prompt expansion is manageable for the bounded slice in the sense that prompts remain under control and audit says `healthy/pass`.
- But the surface now strongly suggests a future **neutral compaction rule** is needed.
- That compaction should be about **uniform structure-preserving condensation**, not semantic importance ranking.

## Prompt size distribution

Source:

- [s2_4a_prompts_v1.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/04_s2_4a_prompt_construction_postfix/analysis/s2_4a_prompts_v1.jsonl)
- [s2_4a_prompt_audit_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/04_s2_4a_prompt_construction_postfix/analysis/s2_4a_prompt_audit_v1.tsv)

Prompt sizes are reported here in **characters** because that is the explicit audit field available in the frozen prompt audit.

Distribution over available papers:

- min: `9805`
- median: `16040.5`
- max: `20261`

### Top 5 largest prompt papers

| rank | paper_key | prompt_size |
|---|---|---:|
| 1 | `WFDTQ4VX` | `20261` |
| 2 | `UFXX9WXE` | `19458` |
| 3 | `5ZXYABSU` | `16790` |
| 4 | `5GIF3D8W` | `15291` |
| 5 | `INMUTV7L` | `13036` |

## Table count distribution

Table counts are counted as `[TABLE]` blocks in the frozen prompts.

Distribution over available papers:

- min: `6`
- median: `12.5`
- max: `17`

### Top 5 highest table-count papers

| rank | paper_key | table_count |
|---|---|---:|
| 1 | `UFXX9WXE` | `17` |
| 2 | `WFDTQ4VX` | `16` |
| 3 | `5ZXYABSU` | `14` |
| 4 | `INMUTV7L` | `11` |
| 5 | `5GIF3D8W` | `7` |

Expected 1-2 table cases in this lineage:

- none

This matters because the post-fix preservation rule is now exposing many more tables per paper than the earlier bounded validation did.

## Per-paper classification table

| paper_key | prompt_size | table_count | structure_ok | row_identity_present | redundancy_level | classification | notes |
|---|---:|---:|---|---|---|---|---|
| `WFDTQ4VX` | `20261` | `16` | no | yes | medium | `PROMPT_BLOAT` | Largest prompt in the bounded set; key DOE tables survive, but many extra weak tables now travel with them. |
| `UFXX9WXE` | `19458` | `17` | no | yes | medium | `NEEDS_COMPACTION` | Highest table count; field-complete, but many additional weak tables create a very dense prompt pack. |
| `5ZXYABSU` | `16790` | `14` | no | yes | medium | `STRUCTURE_LOSS` | Anchor tables now survive, but many preserved tables are still weak or repair-insufficient, so structure quality is uneven. |
| `5GIF3D8W` | `15291` | `7` | no | yes | low | `LARGE_BUT_SAFE` | Moderate table count with mostly usable summaries; some degraded tables remain, but the pack is still readable. |
| `INMUTV7L` | `13036` | `11` | no | yes | medium | `NEEDS_COMPACTION` | Simple-paper prompt is now much broader than semantically necessary; structure fields are present, but many low-signal tables are included. |
| `WIVUCMYG` | `9805` | `6` | yes | yes | low | `LARGE_BUT_SAFE` | Best-balanced prompt in the slice: multiple tables, but still compact and structurally clear. |

## Structure quality: header + row identity

What still looks good:

- Every included table summary in this bounded lineage still carries:
  - `key_columns`
  - `row_identifier_pattern`
  - sample-row fields
- `row_identity_present=yes` for every paper in the bounded lineage.
- The prompt audit says:
  - `all_selected_blocks_included=yes`
  - `truncation_detected=no`
  - `status=pass`

What is degraded:

- Several papers now include many tables where `representation_status=repair_insufficient` and/or `selector_readiness=weak`.
- In those degraded tables, `key_columns` often collapse to generic or clearly low-value summaries such as:
  - `0`
  - bibliographic text
  - narrative spillover

Most affected papers:

- `5ZXYABSU`
  - `11` weak tables
  - several tables with `key_columns: 0` or bibliographic/narrative fragments
- `WFDTQ4VX`
  - `7` weak tables
  - still includes degraded non-DOE/supporting surfaces alongside the recovered DOE tables
- `INMUTV7L`
  - `7` weak tables
  - simple-paper prompt now includes many additional low-signal tables
- `UFXX9WXE`
  - `5` weak tables
  - still field-complete, but broad

Less affected:

- `WIVUCMYG`
  - only `2` weak tables
  - no degraded `key_columns` by the simple audit heuristic used here
- `5GIF3D8W`
  - only `1` weak table
  - still more readable than the larger packs above

## Redundancy patterns

Exact prompt-surface duplicate check:

- No repeated `table_id` values were found within any single prompt by the simple table-id duplicate scan.
- The prompt audit also records:
  - `exact_duplicate_block_count=0` for all six papers

So the main issue is **not exact duplication**.

What is happening instead:

- many papers now carry **distinct but low-value or weak** tables
- this creates **informational bloat**, not strict duplicate bloat
- the bloat is therefore mostly:
  - additional preserved weak tables
  - repair-insufficient summaries
  - multi-table breadth without prompt-level condensation

## Neutrality validation

Evidence for neutrality holding in this bounded lineage:

- `all_selected_blocks_included=yes` for every paper
- there is no sign in the frozen prompt audit of prompt-level truncation or selective omission
- the new broader table sets are carried into prompt construction rather than being re-ranked away
- ordering remains stable evidence-pack ordering, not a new “best table only” path

What did **not** reappear:

- no prompt-layer evidence of primary-table bias
- no prompt-layer evidence that preserved tables were silently removed again
- no prompt-layer evidence of selective table weakening beyond the inherited quality of the preserved table summaries

So the post-fix prompt surface is still **neutral**, but it is now **much wider**.

## Whether compaction is needed

Yes, at a high level.

Why:

- The preservation fix solved the upstream semantic-loss problem for key tables.
- But in the bounded prompt slice it also expanded table counts substantially:
  - several papers now carry `11-17` table blocks
  - prompt sizes rise to roughly `16k-20k` characters for the biggest cases
- The current summary contract remains field-complete, but many preserved weak tables make the pack harder to read.

What kind of compaction is indicated, at a high level only:

- a **neutral, structure-preserving compaction rule**
- likely based on:
  - uniform summary condensation
  - structural grouping
  - duplicate-equivalence handling only when exact
  - no semantic importance ranking
  - no dropping of non-noise tables

What is **not** indicated:

- reintroducing primary-table selection
- guessed importance ranking
- dropping tables because another looks more useful

## FACTS

- The requested post-fix lineage contains 6 papers, not 15.
- Prompt sizes range from `9805` to `20261` characters.
- Table counts range from `6` to `17`.
- All prompts record:
  - `uses_evidence_pack_only=yes`
  - `all_selected_blocks_included=yes`
  - `truncation_detected=no`
  - `status=pass`
- Every included table summary still carries `key_columns` and `row_identifier_pattern`.
- Every paper in the bounded lineage contains at least one weak or repair-insufficient table summary.
- No exact prompt-block duplicates were detected by the frozen prompt audit.

## INFERENCES

- The preservation fix did not break the prompt contract, but it shifted the main risk from semantic omission to prompt breadth.
- The structure-first summary contract is still technically satisfied because schema and row-identity surfaces remain present.
- The practical usability risk is now table-pack sprawl, especially for the anchor papers with many preserved weak tables.

## UNCERTAINTIES

- This audit is bounded to the six-paper post-fix lineage and does not establish prompt-surface behavior for the other 9 DEV15 papers.
- Prompt size is reported in characters rather than model token counts.
- The redundancy assessment is conservative; it detects exact duplication and obvious low-signal repetition, not deeper semantic equivalence.
