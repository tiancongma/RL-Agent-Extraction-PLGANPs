# S2-2b Confirmed-Noise-Only Implementation Report

## Executive conclusion

The bounded implementation succeeded on the target preservation goal.

After the `S2-2b` code change:

- `5ZXYABSU` `Table 1` and `Table 2` now survive into:
  - `evidence_blocks`
  - `normalized_table_payloads`
  - frozen `S2-4a` prompts
- `WFDTQ4VX` real DOE design surfaces now survive into:
  - `evidence_blocks`
  - `normalized_table_payloads`
  - frozen `S2-4a` prompts
- those anchor tables are no longer marked `hard_drop_table_noise`
- guard papers remain stable in the narrow sense that their key expected tables still survive

On this bounded six-paper validation slice, governance and implementation are now aligned on the confirmed-noise-only preservation rule.

## What code changed

Modified code:

- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)

Main changes:

1. Narrowed `table_hard_drop_reasons(...)`
- removed hard-drop triggers based on:
  - weak or corrupted-looking representation alone
  - prose-like row surfaces alone
  - low numeric density alone
  - no-structured-formulation-surface heuristics
- retained hard drop only for confirmed pure-noise style signals such as:
  - front-matter / bibliographic surfaces
  - page/review surfaces
  - explicit high-confidence non-content fragments
  - clearly unrepaired non-content fragments
  - clearly caption/narrative fragment surfaces

2. Narrowed `payload_hard_drop_reasons(...)`
- mirrored the same confirmed-noise-only logic at normalized payload stage

3. Changed table preservation semantics in selector evidence packing
- `must_include` tables are still selected first
- all other non-hard-drop tables are now also preserved into the pre-LLM authority surface
- table survival no longer depends on priority score thresholds
- table survival no longer depends on semantic-duplicate-like suppression

4. Kept only exact duplicate removal for tables
- suppression reason now records `exact_duplicate_removed`
- semantic-near-duplicate suppression no longer removes table surfaces

5. Added explicit audit label
- non-hard-drop tables now carry `table_preservation_status=preserved_non_noise`
- hard-dropped tables carry `table_preservation_status=confirmed_noise`

## Which hard-drop causes were removed or narrowed

Removed or narrowed from irreversible drop:

- guessed low semantic value
- weak-table status
- corrupted-looking but still scientific table surfaces
- lower-value / secondary table competition
- score-threshold suppression of optional tables
- semantic-near-duplicate table suppression
- pseudo-floor logic as the only route to preserve a table

Still allowed as confirmed noise:

- `front_matter_or_bibliographic_surface`
- `journal_page_or_review_surface`
- `high_confidence_non_content_fragment`
- `unrepaired_non_content_fragment`
- `caption_or_narrative_fragment_surface`
- exact duplicate table artifact removal

## Before/after preservation results for 5ZXYABSU and WFDTQ4VX

Validation lineages:

- before:
  - [20260421_9c4a03f/01_s2_4a_table_preservation_validation](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/01_s2_4a_table_preservation_validation)
  - [20260421_9c4a03f/02_s2_4a_prompt_construction_v2](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/02_s2_4a_prompt_construction_v2)
- after:
  - [20260421_9c4a03f/03_s2_4a_table_preservation_validation_postfix](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/03_s2_4a_table_preservation_validation_postfix)
  - [20260421_9c4a03f/04_s2_4a_prompt_construction_postfix](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/04_s2_4a_prompt_construction_postfix)

### 5ZXYABSU

Before:

- `Table 1` and `Table 2` present at `S2-2a`
- both lost at `S2-2b`
- both marked `hard_drop_table_noise`
- only `Table 14` reached evidence / normalized payload / prompt

After:

- `Table 1` preserved in evidence as `5ZXYABSU__table__04`
- `Table 2` preserved in evidence as `5ZXYABSU__table__05`
- both present in normalized payloads with:
  - `table_preservation_status=preserved_non_noise`
  - `hard_drop_reason=[]`
- both visible in frozen `S2-4a` prompt

### WFDTQ4VX

Before:

- DOE-like design candidates survived `S2-2a`
- key candidates such as `candidate_table__08`, `__02`, `__06` lost at `S2-2b`
- all marked `hard_drop_table_noise`
- only downstream `Table 8` reached evidence / normalized payload / prompt

After:

- real DOE design surface preserved:
  - `WFDTQ4VX__table__01` from `WFDTQ4VX__candidate_table__08`
  - appears in evidence as `Table 1`
- related DOE-bearing design surfaces also preserved:
  - `Table 14`
  - `Table 15`
- those preserved DOE tables now carry:
  - `table_preservation_status=preserved_non_noise`
  - `hard_drop_reason=[]`
- frozen `S2-4a` prompt now contains `Table 1`, `Table 14`, and `Table 15`

## Guard-paper results

### INMUTV7L

- expected simple formulation-bearing table still survives
- no regression on the anchor surface
- more non-noise tables are now also preserved, including additional low-confidence surfaces

### WIVUCMYG

- DOE path guard remains intact at the preservation boundary
- `Table 1` still survives
- additional non-noise tables also survive

### UFXX9WXE

- expected key formulation / DOE tables still survive
- more non-noise tables now remain preserved
- one exact duplicate was removed as `exact_duplicate_removed`

### 5GIF3D8W

- expected `Table 4` still survives
- additional non-noise tables now remain preserved

## Whether governance and implementation are now aligned

On this bounded validation slice: yes.

Evidence:

- The two known anchor failures (`5ZXYABSU`, `WFDTQ4VX`) are fixed at the preservation boundary.
- Their critical tables are no longer hard-dropped.
- Non-noise table preservation is now the mainline behavior.
- Remaining hard drops in the bounded slice are limited to tables still classified as confirmed noise or exact duplicates.

Important limit:

- This report validates alignment only on the bounded six-paper slice, not the entire repository.

## Residual risks

1. Prompt-size expansion
- preserving all non-noise tables increases table block counts substantially
- examples from `s2_4a_prompt_audit_v1.tsv`:
  - `5ZXYABSU`: `5 -> 18` evidence blocks
  - `WFDTQ4VX`: `6 -> 21` evidence blocks
- this task does not redesign prompt compaction or multi-table organization

2. Weak but preserved tables now reach prompts
- this is consistent with the new governance contract
- but it may increase summary noise and LLM burden

3. Table-id ambiguity remains
- some preserved tables still have repeated or messy derived `table_id` values
- this task preserves them rather than adjudicating them away

## What is still NOT solved

- This patch does not solve:
  - downstream row emission
  - DOE decode
  - simple-table enumeration
  - non-DOE sweep recovery
  - Stage3 / Stage5 retention
- It only fixes the `S2-2b` preservation boundary so key non-noise tables are not lost before the LLM boundary.

