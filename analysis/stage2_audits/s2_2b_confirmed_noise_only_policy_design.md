# S2-2b Confirmed-Noise-Only Policy Design

## Executive conclusion

The current `S2-2b` table policy is overreaching because it does not just remove confirmed noise. It also performs semantic-like importance filtering through authority ranking, hard-drop classes, competition between tables, duplicate-like suppression, and a minimal-evidence floor that can preserve one guessed-better table while irreversibly discarding others. The `5ZXYABSU` failure is a concrete example: `Table 1` and `Table 2` survived `S2-2a`, but both were dropped at `S2-2b` as `hard_drop_table_noise`, leaving only `Table 14`.

Recommended rewrite:

- replace the current mixed policy with a strict two-class contract:
  - `CONFIRMED_NOISE`
  - `PRESERVE`
- if a table is not confirmed pure noise, preserve it in the pre-LLM authority surface
- do not use guessed semantic importance to hard-drop, suppress, or downrank tables
- if prompt-facing organization is needed, use neutral, stable organization only

## Why current S2-2b policy is overreaching

The user-level principle is clear:

- rules may make one strong irreversible judgment: confirmed noise -> drop
- rules must not decide table importance
- rules must not demote tables based on guessed importance

Current `S2-2b` goes beyond that.

Observed implementation behaviors:

- authority-based ranking and tiering:
  - `rank_table_authority_payloads(...)` computes `authority_score`, `authority_rank`, `authority_tier`, `table_inclusion_class`, and `hard_drop_reason`
  - [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:3251)
- weak-table demotion based on unresolved status, corrupted status, and heuristic signal penalties:
  - [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:3282)
- hard drop at evidence selection for any table whose inclusion class is `TABLE_INCLUSION_HARD_DROP`:
  - [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:4943)
- duplicate-like suppression using both exact signatures and semantic-near-duplicate checks:
  - [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:3240)
  - [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:4954)
- minimal evidence floor that back-fills a single “best” table if no table survives:
  - [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:4793)

These behaviors are not just pure-noise removal. They act as rule-based semantic adjudication before the LLM sees the summary surfaces.

## Current behavior inventory

### 1. Authority ranking and inclusion classes

Current behavior:

- tables are scored and ranked
- tables are split into `primary`, `secondary`, `weak_secondary`
- tables also receive inclusion classes such as `must_include` or `hard_drop`

Policy assessment:

- keeping audit fields for debugging is acceptable
- using them to remove or suppress tables is not acceptable under the new policy

### 2. Hard-drop table noise

Current behavior:

- any table assigned `TABLE_INCLUSION_HARD_DROP` is removed in `build_evidence_priority_selection(...)`
- suppression reason is recorded as `hard_drop_table_noise`

Policy assessment:

- acceptable only when the table is confirmed pure noise
- not acceptable when “noise” is standing in for “weak-looking,” “corrupted,” “less useful,” “low-value,” or “not the best formulation surface”

### 3. Duplicate suppression

Current behavior:

- exact duplicate suppression exists
- semantic-near-duplicate suppression also exists

Policy assessment:

- exact duplicate removal can remain
- semantic-near-duplicate suppression is too strong for tables unless the evidence is equivalent enough to prove true duplication

### 4. Coverage competition and floor behavior

Current behavior:

- if no table survives, `apply_minimal_evidence_floor(...)` adds the single best surviving non-hard-drop table
- this can create a regime where many tables are dropped and one fallback table is preserved

Policy assessment:

- not acceptable as a substitute for preserving non-noise tables
- it hides upstream loss rather than preventing it

### 5. Role/quality-biased suppression

Current behavior:

- unresolved or corrupted-looking tables are treated as weak
- weak tables are more easily demoted or hard-dropped
- strong structural tables and higher scores are more likely to survive

Policy assessment:

- acceptable as audit metadata
- not acceptable as an irreversible preservation decision unless the table is confirmed non-table garbage

## New policy: confirmed_noise_only_drop

### Formal contract

Every `S2-2a` table candidate must be assigned exactly one preservation class:

- `CONFIRMED_NOISE`
- `PRESERVE`

Decision rule:

- assign `CONFIRMED_NOISE` only when the artifact is clearly not a usable scientific table surface
- if there is any meaningful uncertainty, assign `PRESERVE`

Hard rule:

- no importance-based downranking
- no guessed-best-table competition
- no guessed result-table preference
- no guessed formulation-table preference
- no guessed semantic usefulness filter

### Meaning of `PRESERVE`

`PRESERVE` means:

- include the table in pre-LLM authority preservation
- construct a governed summary surface if possible
- keep it available for downstream deterministic reopen
- allow later stages or the LLM to decide semantic importance

### Meaning of `CONFIRMED_NOISE`

`CONFIRMED_NOISE` means the artifact is not meaningfully a scientific table surface. This is the only class that may be hard-dropped.

## Legitimate removals

These removals remain legitimate under the new policy, if confirmed by artifact-backed evidence:

- parser garbage with no coherent table structure
- page headers and footers split as fake tables
- author / affiliation / address blocks mis-read as tables
- bibliography / references fragments mis-read as tables
- publisher boilerplate or submission footer tables
- exact duplicate table artifacts pointing to the same table content
- empty or effectively empty extraction outputs with no recoverable content

These are legitimate because they are confirmed non-table or pure-noise surfaces, not merely low-confidence scientific tables.

## Illegitimate removals

These actions are not legitimate under the new policy:

- hard-dropping a table because it seems weak, sparse, or less useful
- hard-dropping because another table seems more formulation-bearing
- suppressing a table because a result table, formulation table, or later table appears more valuable
- suppressing by authority score threshold
- suppressing by authority rank competition
- suppressing by guessed role bias
- suppressing by semantic-near-duplicate alone when exact equivalence is not proven
- preserving only one fallback formulation surface after other non-noise tables have already been discarded

If a decision depends on guessed semantic importance, it should be rejected.

## Neutral prompt-summary organization strategy

If many tables are preserved, prompt-facing summary construction should stay neutral.

Acceptable organization strategies:

- stable source order
- `Table 1`, `Table 2`, `Table 3` numeric order
- grouped presentation by table family only when grouping is explicit from the source
- flat parallel summaries, one table summary block per preserved table
- optional compact index listing preserved tables before their summaries

Unacceptable organization strategies:

- “best” table first because the selector thinks it is most important
- shrinking or dropping some table summaries because another table seems more relevant
- semantic priority ordering based on guessed formulation importance

Bounded neutral strategy:

- preserve all non-confirmed-noise tables
- emit one governed summary surface per preserved table
- order by stable source order or numeric table order
- if token pressure exists, compress formatting uniformly rather than selectively demoting certain tables

## Duplicate handling under the new policy

### Exact duplicates

A table may be removed as a duplicate only when there is strong evidence of exact duplication, such as:

- same canonical source table asset
- same normalized payload signature
- same source table reference and materially identical content

### Near-duplicates

Near-duplicates should usually both be preserved.

Reason:

- similar-looking formulation tables may represent different surfactants, different polymers, baseline vs loaded states, or formulation vs result views
- collapsing them based on semantic similarity is too risky before LLM interpretation

Required evidence for true duplicate removal:

- exact shared table authority source or exact normalized payload equivalence

Not sufficient:

- high text similarity alone
- shared numbers alone
- same variable family alone

## Minimal future implementation scope

### Must change

- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)
  - `rank_table_authority_payloads(...)`
  - `build_evidence_priority_selection(...)`
  - `apply_minimal_evidence_floor(...)`
  - exact-vs-near-duplicate table suppression logic

### Likely change

- `S2-2b` audit/debug outputs so they expose:
  - `CONFIRMED_NOISE`
  - `PRESERVE`
  - exact reason for confirmed-noise drop
  - exact reason when exact duplicate suppression is used

### Should not change

- selector segmentation in `S2-2a`
- prompt text itself
- `S2-5` semantic interpretation contract
- Stage3
- Stage5

This keeps the rewrite focused on table preservation policy only.

## Governance/doc implications

If this policy is adopted, the governance wording should eventually be updated in:

- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md)
- [project/ACTIVE_PIPELINE_FLOW.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)
- [project/ACTIVE_PIPELINE_RUNBOOK.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_RUNBOOK.md)
- [project/4_DECISIONS_LOG.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/4_DECISIONS_LOG.md)
- possibly [README.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/README.md) if it describes current `S2-2b` table preservation behavior

Key governance statement to add:

- `S2-2b` may hard-drop only confirmed pure-noise tables
- all other table candidates must be preserved into the pre-LLM authority surface
- authority ranking may remain as audit metadata only, not as a preservation gate

## FACTS

- Current `S2-2b` performs authority scoring, ranking, inclusion-class assignment, duplicate-like suppression, and minimal-evidence-floor fallback in maintained code.
- `5ZXYABSU` shows a concrete failure where `Table 1` and `Table 2` survive `S2-2a` but are suppressed at `S2-2b` as `hard_drop_table_noise`.
- Only `Table 14` is preserved downstream for that paper.

## INFERENCES

- Current `S2-2b` is acting as a semantic prefilter, not just a pure-noise filter.
- Because downstream LLM input is summary-only, this prefiltering creates irreversible loss.
- A conservative two-class contract is the smallest policy change that addresses the failure mode without redesigning Stage2.

## UNCERTAINTIES

- Some current `hard_drop` cases may in fact be confirmed garbage; those would need implementation-time review to preserve legitimate removals while retiring overreaching ones.
- The repo may want a narrow third audit-only label internally for “preserved but low-confidence,” but under this policy that label must not change preservation behavior.

## NOT RECOMMENDED

- any importance-based ranking used to decide preservation
- any guessed single-best-table policy
- semantic-near-duplicate table collapse before LLM interpretation
- replacing broad preservation with a single fallback formulation surface
