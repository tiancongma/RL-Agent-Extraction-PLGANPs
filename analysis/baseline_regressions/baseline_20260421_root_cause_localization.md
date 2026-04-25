# Baseline 20260421 Root-Cause Localization

## Executive conclusion

The dominant count loss in the diagnostic baseline lineage
`data/results/20260421_43ed145/` happens upstream in Stage2, not in Stage3,
Stage5, or identity freeze.

Artifact-backed count spine:

- GT total: `210`
- `S2-5` semantic candidate total: `45`
- `S2-7` completed Stage2 rows: `45`
- Stage5 final rows: `35`

So:

- `165` rows are already missing by `S2-5`
- `S2-7` does not recover any of those missing rows
- Stage5 contributes only `10` additional row losses across the full DEV15

The user suspicion that DOE is broken is partly correct, but not as a single
uniform bug:

1. `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W` all retain DOE or sweep signals into
   `S2-5`, yet still collapse to family-level candidates and never trigger
   row-level expansion in `S2-7`.
2. `WFDTQ4VX` is different:
   its first major loss happens at `S2-4a`, where the prompt carries only a
   downstream/non-formulation table summary and no usable formulation-bearing
   DOE table surface.

The single highest-value next repair target is therefore the `S2-5 -> S2-7`
DOE expansion handshake for papers where DOE scope survives into `S2-5`.
That is the narrowest boundary with the highest plausible recovery impact.

## Boundary-by-boundary localization summary

### Compare -> Stage5

From [final_table_vs_gt_counts.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/08_compare/final_table_vs_gt_counts.tsv):

- matched papers: `1`
- mismatched papers: `14`
- total final rows: `35`
- total GT rows: `210`

This confirms a severe undercount, but not where it begins.

### Stage5

From [final_formulation_table_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/07_stage5/final_formulation_table_v1.tsv) and [final_output_decision_trace_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/07_stage5/final_output_decision_trace_v1.tsv):

- Stage5 emits `35` rows from `45` incoming `S2-7` rows
- the Stage5 drop is therefore `10` rows total
- for the four primary target papers, Stage5 losses are:
  - `UFXX9WXE`: `2 -> 2`
  - `WIVUCMYG`: `3 -> 3`
  - `WFDTQ4VX`: `3 -> 3`
  - `5GIF3D8W`: `2 -> 1`

Conclusion:

- Stage5 is not the dominant cause of the count collapse
- it is a secondary loss boundary only

### Stage3

From [formulation_relation_records_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/06_stage3/formulation_relation_v1/formulation_relation_records_v1.tsv) and [resolved_relation_fields_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/06_stage3/formulation_relation_v1/resolved_relation_fields_v1.tsv):

- Stage3 relation rows remain present for all primary target papers:
  - `UFXX9WXE`: `9`
  - `WIVUCMYG`: `21`
  - `WFDTQ4VX`: `12`
  - `5GIF3D8W`: `9`

Stage3 operates over an already-reduced candidate universe. It does not show a
new large row-count collapse relative to `S2-7`.

### S2-7

From [weak_labels__v7pilot_r3_fixparse.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/05_s2_7/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv) and [compatibility_projection_summary_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/05_s2_7/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json):

- total projected rows: `45`
- `numbered_doe_recovered_rows: 0`
- `table_row_expansion_rows: 0`

For the primary target papers:

- `UFXX9WXE`
  - DOE unit considered and authorized
  - not called
  - `skip_reason = authorized_target_unresolved`
- `WIVUCMYG`
  - DOE unit considered and authorized
  - not called
  - `skip_reason = authorized_target_unresolved`
- `5GIF3D8W`
  - DOE unit considered and authorized
  - not called
  - `skip_reason = authorized_target_unresolved`
- `WFDTQ4VX`
  - DOE unit not authorized
  - `skip_reason = missing_llm_declared_doe_scope`

Conclusion:

- `S2-7` is not where the first numerical drop occurs, because `S2-7` row
  counts equal `S2-5` formulation-candidate counts
- but `S2-7` is where deterministic recovery fails to happen, even when DOE
  scope has been preserved

### S2-6

From [stage2_semantic_authority_contract_report_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/04_s2_6/analysis/stage2_semantic_authority_contract_report_v1.json):

- `status: pass`
- `error_count: 0`
- `warning_count: 0`

No evidence shows `S2-6` suppressing rows or blocking lawful handoff.

### S2-5

From [semantic_stage2_v2_summary.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/03_s2_5/semantic_stage2_objects/semantic_stage2_v2_summary.tsv) and [semantic_stage2_v2_objects.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/03_s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl):

- total `formulation_candidate_count`: `45`
- this exactly matches the `S2-7` completed Stage2 row count of `45`

For the primary target papers:

- `UFXX9WXE`: `2`
- `WIVUCMYG`: `3`
- `WFDTQ4VX`: `3`
- `5GIF3D8W`: `2`

This is the first boundary where the large row-count collapse is numerically
visible.

### S2-4a

From [s2_4a_prompts_v1.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompts_v1.jsonl), [s2_4a_prompt_audit_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompt_audit_v1.tsv), and [dev15_layer_a_hard_gate_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/dev15_layer_a_hard_gate_v1.tsv):

- all 15 papers passed
- summary-only contract held
- no truncation was detected
- the primary papers all had lawful prompt construction

But lawful does not mean sufficient for row-level enumeration:

- `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W` each carried table-based DOE or sweep
  evidence into `S2-4a`
- `WFDTQ4VX` did not: its frozen prompt included only a repair-insufficient
  downstream Table 8/9 summary rather than a formulation-bearing DOE table

## Primary target paper mini-cases

### UFXX9WXE

Boundary counts:

- `S2-4a`: prompt contains three table summaries; hard gate path1 satisfied
- `S2-5`: `2` candidates
- `S2-7`: `2` rows
- Stage3: `9` relation rows
- Stage5: `2` final rows
- GT: `27`

Last boundary where row-bearing structure is still present:

- `S2-4a`
- prompt includes a Box-Behnken design summary with
  `table_role_hint: design matrix` and `table_shape_hint: rows=22, cols=9`

First boundary where the rows are absent:

- `S2-5`
- only two family-level candidates remain:
  - `Formulations from Table 1`
  - `Formulations from Table 2`

Localization:

- first major loss boundary: `S2-5`
- dominant failure mode:
  - LLM semantic under-enumeration
  - followed by a non-firing `S2-7` DOE expansion handshake

### WIVUCMYG

Boundary counts:

- `S2-4a`: four table summaries in prompt
- `S2-5`: `3` candidates
- `S2-7`: `3` rows
- Stage3: `21` relation rows
- Stage5: `3` final rows
- GT: `26`

Last boundary where row-bearing structure is still present:

- `S2-4a`
- prompt includes Table 1 with:
  - `Experimental design variable table`
  - `row_identifier_pattern: F-numbered rows`
  - `table_shape_hint: rows=26, cols=9`

First boundary where the rows are absent:

- `S2-5`
- only three objects survive:
  - DOE family for Table 1
  - selected formulations before freeze-drying
  - selected formulations after freeze-drying

Localization:

- first major loss boundary: `S2-5`
- dominant failure mode:
  - LLM semantic under-enumeration
  - followed by a non-firing `S2-7` DOE expansion handshake

### WFDTQ4VX

Boundary counts:

- `S2-4a`: one table summary plus supporting prose
- `S2-5`: `3` candidates
- `S2-7`: `3` rows
- Stage3: `12` relation rows
- Stage5: `3` final rows
- GT: `30`

Last boundary where row-bearing structure is still present:

- not demonstrated in the current frozen prompt

First boundary where the relevant formulation table is absent:

- `S2-4a`
- the only prompt table block is:
  - `WFDTQ4VX__table__01`
  - a repair-insufficient summary combining Table 8 and Table 9
  - classified in `S2-5` as `downstream_variant_table`, not formulation-bearing

Localization:

- first major loss boundary: `S2-4a`
- dominant failure mode:
  - pre-LLM evidence omission / prompt assembly omission of the formulation-bearing DOE table

This paper is the clearest exception to the otherwise Stage2-semantic pattern.

### 5GIF3D8W

Boundary counts:

- `S2-4a`: one formulation table summary in prompt
- `S2-5`: `2` candidates
- `S2-7`: `2` rows
- Stage3: `9` relation rows
- Stage5: `1` final row
- GT: `26`

Last boundary where structured sweep evidence is still present:

- `S2-4a`
- prompt includes Table 4 as a formulation table with `rows=9`
- `S2-5` still marks the table scope as `doe_table` and `has_variable_sweep=yes`

First boundary where the full row universe is absent:

- `S2-5`
- only family plus control survive:
  - `Etoposide-loaded nanoparticles`
  - `Drug-free nanoparticles`

Localization:

- first major loss boundary: `S2-5`
- dominant failure mode:
  - LLM semantic under-enumeration
  - followed by a non-firing `S2-7` DOE expansion handshake

Important caveat:

- unlike `UFXX9WXE` and `WIVUCMYG`, the current prompt surface is itself more
  compact and does not obviously expose the full historical row universe
- so this case is still upstream Stage2, but prompt sufficiency for the full
  `26` GT count is less certain than in the two strong numbered DOE cases

## DOE-specific diagnosis

### What the artifacts prove

For `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W`:

- `S2-4a` prompt legality is fine
- prompt surfaces include DOE or sweep-supporting table summaries
- `S2-5` preserves:
  - `has_variable_sweep = yes`
  - `table_scope_count > 0`
  - `is_doe = true` for at least one table scope
- but `S2-5` still emits only family-level candidates instead of row-bearing
  formulation candidates
- `S2-7` then emits zero DOE recovery rows:
  - `numbered_doe_recovered_rows = 0`
  - `table_row_expansion_rows = 0`

For `WFDTQ4VX`:

- the DOE-supporting formulation table is already absent from the prompt
- `S2-5` records `has_variable_sweep = yes`, but only downstream tables are in
  table scope
- `S2-7` never authorizes DOE expansion

### What this means

The current lineage does not support the broad claim “DOE is globally broken in
exactly one way.”

What it does support is:

- a dominant Stage2 under-enumeration pattern across DOE-sensitive papers
- with two submodes:
  1. DOE signals survive into `S2-5`, but row expansion never materializes
  2. the DOE-supporting table never reaches the prompt in the first place

## Dominant failure ranking

### 1. Stage2 candidate-universe collapse by `S2-5`

Impact:

- global
- highest row-count impact
- `210` GT rows reduce to `45` Stage2 semantic candidates before Stage3 begins

Evidence:

- [semantic_stage2_v2_summary.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/03_s2_5/semantic_stage2_objects/semantic_stage2_v2_summary.tsv)
- [weak_labels__v7pilot_r3_fixparse.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/05_s2_7/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv)

### 2. Missing `S2-7` recovery/expansion for DOE-sensitive papers

Impact:

- high for DOE-sensitive papers
- especially `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W`

Evidence:

- `numbered_doe_recovered_rows = 0`
- `table_row_expansion_rows = 0`
- authorized-but-not-called DOE function units with
  `skip_reason = authorized_target_unresolved`

### 3. Paper-specific prompt evidence omission

Impact:

- high for `WFDTQ4VX`
- not the dominant global cause

Evidence:

- only downstream Table 8/9 summary appears in the frozen prompt
- no formulation-bearing DOE table survives into prompt scope

### 4. Stage5 filtering / collapse

Impact:

- real but secondary
- only `10` rows lost from `S2-7` to Stage5 across full DEV15

Evidence:

- compare `45` Stage2 rows vs `35` Stage5 rows

### 5. Identity freeze binding failure

Impact:

- diagnostic risk signal only
- not the count-loss origin

Evidence:

- upstream identity counts remain high, but final-table binding finds no
  selected scaffold matches
- the count collapse is already present before identity freeze runs

## Highest-value next repair target

The highest-value next repair target is the `S2-5 -> S2-7` DOE expansion
handshake, especially for papers where:

- the prompt already carried DOE-supporting structure
- `S2-5` preserved `is_doe = true` and `has_variable_sweep = yes`
- but `S2-7` emitted zero DOE rows

Why this is the best next target:

- it covers the dominant failure family for `UFXX9WXE`, `WIVUCMYG`, and
  `5GIF3D8W`
- it is upstream of Stage3 and Stage5, where the dominant loss is not
- it is more specific than a vague “Stage2 is weak” claim

`WFDTQ4VX` should remain a separate prompt/evidence-localization case.

## FACTS

- Compare is diagnostic-only and `benchmark_valid = no`.
- `S2-4a`, `S2-4b`, `S2-5`, `S2-6`, `S2-7`, Stage3, and Stage5 all completed.
- `S2-6` shows `0` errors and `0` warnings.
- Total counts:
  - GT `210`
  - `S2-5` candidates `45`
  - `S2-7` rows `45`
  - Stage5 final rows `35`
- For `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W`, DOE or sweep signals survive into
  `S2-5`.
- For those same papers, `S2-7` emits zero DOE recovery rows.
- `WFDTQ4VX` prompt scope lacks a formulation-bearing DOE table summary in the
  current frozen lineage.

## INFERENCES

- The dominant global failure is upstream of Stage3 and Stage5.
- The most likely high-impact seam is the `S2-5 -> S2-7` handshake, not Stage5
  collapse.
- `WFDTQ4VX` is not explained by the same failure mode as the other three
  primary papers.

## UNCERTAINTIES

- For `5GIF3D8W`, the current prompt surface is compact enough that prompt
  sufficiency for all `26` GT rows is less certain than for `UFXX9WXE` and
  `WIVUCMYG`.
- `UFXX9WXE` prompt table labeling is visibly noisy
  (`table_10` summary text referring to `Table 1`), although it still clearly
  preserves design-matrix structure.
- This localization does not by itself prove whether the next repair should
  change prompt content, semantic object formation, or `S2-7` authorization
  resolution logic. It only localizes the smallest high-value boundary family.

## NOT RECOMMENDED YET

- Do not start with a Stage3 repair.
- Do not start with a Stage5 repair as the mainline fix.
- Do not treat identity freeze as the primary cause of the count mismatch.
- Do not claim a single repo-wide “DOE is broken” bug without separating:
  - DOE-under-enumeration after valid DOE prompt evidence
  - prompt-time omission of DOE-supporting tables
- Do not reopen `ACTIVE_RUN.json` or rerun the full baseline until the
  localization outcome is converted into a bounded repair hypothesis.
