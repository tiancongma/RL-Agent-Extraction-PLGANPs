# DEV15/GPT35 Regression Repair Plan

## Status
`r0_negative_r1_failed_r2a_failed_r2b_positive_diagnostic`

## Scope
This plan covers the campaign-local regression created after GPT35/EE35 repair
experiments under `data/results/20260511_b069802`.

The immediate problem is:

- `139_compare_dev15_goldline_loaded_alias_duplicate_collapse` passes DEV15.
- GPT35/EE35 repairs improved campaign-local row recovery but left unresolved
  losses from the `168` S2-4b raw-response boundary.
- The later executable-scope authorization path `196 -> 202` can be consumed by
  downstream stages, but it regresses DEV15 badly on the five-paper probe.

This work is diagnostic-only. It must not be reported as benchmark-valid final
output.

## Governing Baselines

- DEV15 pass baseline:
  - `data/results/20260511_b069802/139_compare_dev15_goldline_loaded_alias_duplicate_collapse/gt_authority_v2_variantaware`
- GPT35/EE35 failed or partially repaired lineage:
  - S2-4b raw responses:
    `data/results/20260511_b069802/168_stage2_s2_4b_ee35_new_s2_4a_live_same_params`
  - latest repaired Stage5 diagnostic output:
    `data/results/20260511_b069802/186_stage5_ee35_condition_row_repair_final_output`
- DEV15 regression probe:
  - S2-4a prompt variant:
    `data/results/20260511_b069802/196_stage2_s2_4a_ee380_executable_scope_authorization_prompt_variant_no_live`
  - S2-4b live five-paper run:
    `data/results/20260511_b069802/197_stage2_s2_4b_196_dev15_5paper_deepseek_streaming`
  - S2-7 projection:
    `data/results/20260511_b069802/200_stage2_s2_7_196_dev15_5paper_projection`
  - Stage5 final diagnostic output:
    `data/results/20260511_b069802/202_stage5_196_dev15_5paper_final_output`

## Repair Strategy

Use `139` as the capability-preservation baseline. Treat `196` as a regression
diagnostic and salvage only the specific executable-scope signals that can be
lawfully mapped back into the existing downstream execution contract.

Do not replace the `139` semantic shape wholesale. Each repair must identify:

- the earliest break boundary,
- the exact old signal that disappeared or changed shape,
- the downstream function unit that consumes that signal,
- a bounded replay result,
- and whether DEV15 regression improves without inventing rows downstream.

## Round R0: 196 Executable-Scope Negative Diagnostic

### Diagnosis

`5GIF3D8W` in the `196` five-paper probe regresses from `26` final rows in
`139` to `1` final row. The S2-7 execution ledger shows:

- table: `Table 4`
- function unit: `table_row_expansion_v1`
- authorized: `yes`
- called: `yes`
- rows emitted: `0`
- skip reason: `missing_llm_variable_roles`

The S2-5 semantic object contains executable-scope authorization for the same
source table and records `composition_column_hints` such as `PLGA 50/50` and
`PLGA 75/25`, but the executable-scope normalizer only promotes
`factor_column_hints` into `semantic_signals.primary_variable_names`.

### Repair

Update the executable-scope adapter so formulation-bearing table scopes with
explicit row-universe authorization may promote source-declared composition or
preparation column hints into the legacy variable-role surface consumed by
`table_row_expansion_v1`.

This is not row creation. It preserves an LLM-declared scope and LLM-declared
column-role hints so the existing deterministic row materializer can inspect
the already preserved table authority.

### Validation

Run a bounded replay from the frozen `197` raw responses through:

`S2-5 -> S2-6 -> S2-7 -> Stage3 -> Stage5 -> DEV15 diagnostic compare`

Minimum acceptance for R1:

- `5GIF3D8W` no longer fails with `missing_llm_variable_roles`.
- The five-paper replay improves versus `202` without weakening the lawful
  boundary rule.
- Results remain labeled diagnostic-only.

### R0 Result

Status: `completed_adapter_hygiene_no_row_count_recovery`

Code and test:

- updated `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  so executable-scope `composition_column_hints` and
  `preparation_column_hints` can feed `semantic_signals.primary_variable_names`
  for formulation-bearing scopes with downstream expansion authorization.
- updated `materialize_execution_markers_from_shrunken_scopes` so non-DOE
  formulation-bearing scopes can receive `table_variable_roles` when the LLM
  has supplied primary variable hints.
- added regression coverage in
  `tests/test_stage2_table_row_expansion_scope_alias_roles_v1.py`.
- validation command:
  `PYTHONPATH=. python3 -m unittest tests.test_stage2_table_row_expansion_scope_alias_roles_v1 -q`

Replay lineage:

- S2-5:
  `data/results/20260511_b069802/204_stage2_s2_5_196_dev15_r1_composition_hint_reparse`
- S2-6:
  `data/results/20260511_b069802/205_stage2_s2_6_196_dev15_r1_composition_hint_validation`
- S2-7:
  `data/results/20260511_b069802/206_stage2_s2_7_196_dev15_r1_composition_hint_projection`
- Stage3:
  `data/results/20260511_b069802/207_stage3_196_dev15_r1_composition_hint_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/208_stage5_196_dev15_r1_composition_hint_final_output`
- diagnostic compare:
  `data/results/20260511_b069802/209_compare_196_dev15_r1_composition_hint_vs_gt`

Observed effect:

- `5GIF3D8W` S2-7 skip reason changed from `missing_llm_variable_roles` to
  `insufficient_column_header_rows`.
- `5GIF3D8W` final rows stayed `1` versus GT `26`.
- Five-paper S2-7 rows stayed `90`.
- Five-paper Stage5 rows stayed `77`.
- Full DEV15 diagnostic compare remains `77` final rows versus `202` GT rows
  because only the five-paper subset was rerun and the regression remains.

Interpretation:

R0 confirms that one adapter signal was lost, but also proves that the
`196` executable-scope raw responses do not contain enough row-universe
semantics to lawfully restore the `139` DEV15 behavior downstream. The next
repair must move back to the S2-4a/S2-4b contract: preserve the `139`
row-materialization semantic shape and add GPT35-specific signals only as
auxiliary fields, rather than replacing it with executable-scope authorization.

This round is not counted as a formal `139+` repair. It is retained only as
negative boundary evidence and adapter hygiene.

## Round R1: 139-Inherited GPT35 Auxiliary Signal Prompt

### Diagnosis

Prior EE35/GPT35 localization under `168 -> 186` shows remaining deficits
dominated by these classes:

- S2-7 table-row expansion under-materialization.
- S2-7 DOE expansion with no rows from authorized scopes.
- Stage5 filter or identity closure loss.
- Smaller raw S2-4b semantic underselection cases.

The `196` experiment showed that replacing the `139` semantic shape with a
summary-level executable-scope authorization schema regresses DEV15. Therefore
R1 must keep the `139` top-level schema and add GPT35-oriented instructions
only through existing fields.

### Repair

Create a new `139+` prompt variant from the `128` DEV15 goldline S2-4a prompt
surface. The prompt must preserve the same top-level JSON schema:

- `paper_key`
- `table_scopes`
- `semantic_signals`
- `formulation_candidates`
- `shared_semantics`
- `selection_markers`
- `context_inheritance_markers`
- `protocol_inheritance_markers`
- `relation_cues`
- `typed_inheritance_fields`
- `typed_doe_factors`
- `result_binding_candidates`

The only change is an auxiliary instruction block telling the model to:

- preserve primary formulation-table row universes before metric/supporting
  interpretation,
- expose table-row, column-sweep, DOE, condition-matrix, and supporting-table
  linkage signals using the existing fields,
- avoid broad family summaries when source rows or columns define independent
  formulations,
- and keep collapse/filter hints sparse and downstream-readable.

No `executable_scope_authorizations` top-level key is allowed in R1.

### Validation

Run a five-paper DEV15 gate first:

- `5GIF3D8W` sweep gate: final count should not regress from `26`.
- `UFXX9WXE` DOE gate: final count should not regress from `27`.
- `5ZXYABSU` collapse gate: final count should stay at `9`.
- `L3H2RS2H` over-retention gate: final count should stay near `21` and must
  not show the `196`-style `31` over-retention.
- `WFDTQ4VX` neutral gate: final count should stay at `30`.

Only if the DEV15 gate is healthy should the same prompt style be tested on a
small GPT35 subset.

### R1 Result

Status: `failed_dev15_gate_not_promoted`

Artifact lineage:

- S2-4a prompt variant:
  `data/results/20260511_b069802/210_stage2_s2_4a_dev15_139plus_gpt35_aux_signal_prompt_freeze`
- S2-4b live:
  `data/results/20260511_b069802/211_stage2_s2_4b_dev15_139plus_gpt35_aux_signal_deepseek_live`
- S2-5:
  `data/results/20260511_b069802/212_stage2_s2_5_dev15_139plus_gpt35_aux_signal_parse`
- S2-6:
  `data/results/20260511_b069802/213_stage2_s2_6_dev15_139plus_gpt35_aux_signal_validation`
- S2-7:
  `data/results/20260511_b069802/214_stage2_s2_7_dev15_139plus_gpt35_aux_signal_projection`
- Stage3:
  `data/results/20260511_b069802/215_stage3_dev15_139plus_gpt35_aux_signal_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/216_stage5_dev15_139plus_gpt35_aux_signal_final_output`
- Diagnostic compare:
  `data/results/20260511_b069802/217_compare_dev15_139plus_gpt35_aux_signal_vs_gt`

S2-4b settings:

- backend: `deepseek`
- model: `deepseek-v4-flash`
- thinking: `disabled`
- response format: `json_object`
- timeout: `180`
- retries: `0`
- max parallel requests: `1`

Five-paper Stage5 gate result:

| paper_key | `139` final | `139+ R1` final | GT | status |
|---|---:|---:|---:|---|
| `5GIF3D8W` | 26 | 6 | 26 | fail_under |
| `5ZXYABSU` | 9 | 9 | 9 | pass |
| `L3H2RS2H` | 21 | 14 | 21 | fail_under |
| `UFXX9WXE` | 27 | 26 | 27 | fail_minor_under |
| `WFDTQ4VX` | 30 | 30 | 30 | pass |

S2-7 showed the same failure before Stage5:

| paper_key | `139` S2-7 | `139+ R1` S2-7 |
|---|---:|---:|
| `5GIF3D8W` | 27 | 7 |
| `5ZXYABSU` | 12 | 18 |
| `L3H2RS2H` | 26 | 18 |
| `UFXX9WXE` | 28 | 28 |
| `WFDTQ4VX` | 60 | 60 |

Interpretation:

Even a schema-preserving auxiliary instruction block changed the live S2-4b
semantic output enough to regress DEV15. The repair is therefore rejected and
must not be used on GPT35. The next attempt should not alter the main `139`
prompt body globally. It should instead use one of these safer patterns:

- a separate GPT35-only diagnostic prompt branch with no DEV15 promotion claim,
- or a post-S2-4b deterministic adapter/materializer repair driven by signals
  already present in `139`-style raw responses,
- or a very narrow per-field prompt addition validated one feature gate at a
  time rather than one broad auxiliary instruction block.

## Round R2A: 139-Inherited Narrow Feature-Gate Prompt

Status: `failed_dev15_gate_not_promoted`

Rationale:

R1 proved that a broad auxiliary instruction block can damage the live S2-4b
semantic output even when the top-level schema is preserved. R2A therefore
tests only one narrow feature gate at a time, with `139`/`128` as the prompt
base and with no inheritance from `196`.

The first R2A gate is row-universe preservation:

- keep the original `139` schema and task wording,
- do not add `executable_scope_authorizations`,
- do not globally change helper/control/filter behavior,
- only remind the model that when it already identifies a formulation-bearing
  table, source-visible row/column identity axes should remain exposed through
  existing `table_scopes`, `semantic_signals`, and `formulation_candidates`
  rather than being summarized into family-level prose.

Validation:

- run the same five-paper DEV15 gate used for R1,
- pass/fail against the `139` Stage5 counts before any GPT35/EE35 run,
- if the DEV15 gate fails, reject R2A and do not promote the prompt branch.

### R2A Result

Artifact lineage:

- S2-4a prompt variant:
  `data/results/20260511_b069802/222_stage2_s2_4a_dev15_139plus_narrow_row_universe_prompt_freeze`
- S2-4b live:
  `data/results/20260511_b069802/223_stage2_s2_4b_dev15_139plus_narrow_row_universe_deepseek_live`
- S2-5:
  `data/results/20260511_b069802/224_stage2_s2_5_dev15_139plus_narrow_row_universe_parse`
- S2-6:
  `data/results/20260511_b069802/225_stage2_s2_6_dev15_139plus_narrow_row_universe_validation`
- S2-7:
  `data/results/20260511_b069802/226_stage2_s2_7_dev15_139plus_narrow_row_universe_projection`
- Stage3:
  `data/results/20260511_b069802/227_stage3_dev15_139plus_narrow_row_universe_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/228_stage5_dev15_139plus_narrow_row_universe_final_output`
- Diagnostic audit:
  `data/results/20260511_b069802/229_dev15_139plus_narrow_row_universe_audit`

S2-4b settings:

- backend: `deepseek`
- model: `deepseek-v4-flash`
- thinking: `disabled`
- streaming: `enabled`
- response format: `json_object`
- timeout: `300`
- retries: `0`
- max parallel requests: `1`
- parameter-lock deviation: user-requested `300s` timeout instead of campaign
  lock `180s`

Five-paper Stage5 gate result:

| paper_key | `139` final | `R1` final | `R2A` final | GT | status |
|---|---:|---:|---:|---:|---|
| `5GIF3D8W` | 26 | 6 | 20 | 26 | fail_under |
| `5ZXYABSU` | 9 | 9 | 9 | 9 | pass |
| `L3H2RS2H` | 21 | 14 | 14 | 21 | fail_under |
| `UFXX9WXE` | 27 | 26 | 27 | 27 | pass |
| `WFDTQ4VX` | 30 | 30 | 30 | 30 | pass |

S2-7 gate result:

| paper_key | `139` S2-7 | `R1` S2-7 | `R2A` S2-7 |
|---|---:|---:|---:|
| `5GIF3D8W` | 27 | 7 | 22 |
| `5ZXYABSU` | 12 | 18 | 18 |
| `L3H2RS2H` | 26 | 18 | 19 |
| `UFXX9WXE` | 28 | 28 | 28 |
| `WFDTQ4VX` | 60 | 60 | 33 |

Interpretation:

R2A is less damaging than R1 for `5GIF3D8W`, but it still fails the DEV15
capability-preservation gate and cannot be promoted to GPT35/EE35. The result
also shows that even narrow row-universe wording can perturb unrelated live
semantic discovery, especially `WFDTQ4VX` at S2-7 and `L3H2RS2H` at Stage5.
Prompt repair remains possible only as smaller, paper-class-specific
experiments after deterministic replay has been exhausted.

## Round R2B: GPT35 Deterministic Replay From Existing 168 Signals

Status: `positive_diagnostic_candidate_not_benchmark_valid`

Rationale:

R1 suggests live prompt changes are high risk. R2B therefore makes no new LLM
calls and does not alter the S2-4a prompt. It replays the existing GPT35/EE35
semantic boundary:

- S2-4b raw: `168_stage2_s2_4b_ee35_new_s2_4a_live_same_params`
- S2-5 parse: `169_stage2_s2_5_ee35_new_s2_4a_semantic_parse`
- S2-6 validation: `170_stage2_s2_6_ee35_new_s2_4a_validation`

through the current deterministic S2-7 adapter/materializer and then Stage3
and Stage5.

This branch tests whether the already-present raw/validated GPT35 signals can
recover rows after deterministic materializer fixes, especially where prior
loss localization showed S2-7 under-materialization.

Validation:

- compare S2-7 row count to
  `184_stage2_s2_7_ee35_condition_row_repair`,
- compare Stage5 final rows to
  `186_stage5_ee35_condition_row_repair_final_output`,
- compare paper-level row coverage against the campaign-local GPT Web master
  formulation reference under
  `17_ee35_master_formulation_gt_from_gpt_web`,
- report the result as diagnostic-only unless complete lineage and GT authority
  alignment are explicitly revalidated for benchmark use.

### R2B Result

Artifact lineage:

- S2-7 replay:
  `data/results/20260511_b069802/218_stage2_s2_7_ee35_deterministic_from_168_current_replay`
- Stage3:
  `data/results/20260511_b069802/219_stage3_ee35_deterministic_from_168_current_replay_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/220_stage5_ee35_deterministic_from_168_current_replay_final_output`
- Diagnostic audit:
  `data/results/20260511_b069802/221_ee35_deterministic_from_168_current_replay_audit`

Resolved input boundary:

- S2-6 validation source:
  `data/results/20260511_b069802/170_stage2_s2_6_ee35_new_s2_4a_validation`
- This is a lawful completed Stage2 replay path through S2-7, Stage3, and
  Stage5. No new S2-4b live call was made.

Aggregate result:

| layer | previous lineage | R2B lineage | delta |
|---|---:|---:|---:|
| S2-7 rows | 259 (`184`) | 285 (`218`) | +26 |
| Stage5 final rows | 212 (`186`) | 233 (`220`) | +21 |
| GPT Web master reference rows | 402 | 402 | n/a |

Paper-level result:

- improved Stage5 paper count: `3`
- regressed Stage5 paper count: `0`
- main improvements:
  - `JRMKHP5C`: `6 -> 21` final rows, master reference `45`
  - `TFT6JTT6`: `11 -> 16` final rows, master reference `16`
  - `9454P5IZ`: `8 -> 9` final rows, master reference `7`
- largest remaining gaps:
  - `JRMKHP5C`: remaining gap `24`
  - `XDIRIJ74`: remaining gap `21`
  - `2RNHC2M5`: remaining gap `18`
  - `YZYKTTFE`: remaining gap `17`
  - `KTNLRQZU`: remaining gap `16`

Interpretation:

R2B is the first positive repair direction in this round. It improves the
existing GPT35/EE35 chain without changing the prompt or making new live LLM
calls, and it introduces no paper-level final-row count regression against the
previous `186` diagnostic lineage. The remaining loss classes are now more
clearly split between unresolved S2-4b underselection, DOE/run-matrix
materialization, and Stage5 identity/filtering cases. This branch is a better
candidate base for the next repair than any R1/R2A prompt variant.

## Round R2B+ Iteration Strategy

Status: `active`

Use R2B as the full GPT35/EE35 no-live regression gate. The current R2B
baseline is:

- S2-7:
  `data/results/20260511_b069802/218_stage2_s2_7_ee35_deterministic_from_168_current_replay`
- Stage3:
  `data/results/20260511_b069802/219_stage3_ee35_deterministic_from_168_current_replay_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/220_stage5_ee35_deterministic_from_168_current_replay_final_output`
- Audit:
  `data/results/20260511_b069802/221_ee35_deterministic_from_168_current_replay_audit`

Do not make a new GPT35 live call for this repair phase. Each repair round
should:

1. select a small set of high-gap papers from the R2B audit,
2. localize the earliest loss boundary using S2-5/S2-6 semantic objects,
   S2-7 ledger rows, Stage3 relation artifacts, and Stage5 decision trace,
3. patch exactly one reusable deterministic function unit,
4. run a focused unit test or bounded replay for the target class,
5. run the full `170 -> S2-7 -> Stage3 -> Stage5 -> audit` GPT35/EE35 no-live
   replay,
6. accept the repair only if the full replay improves target losses without
   paper-level count regression versus the current R2B baseline.

Initial target set:

- `JRMKHP5C`: still `21/45`, but improved from `6`; likely table-row expansion
  under-materialization or Stage5 filtering on newly recovered table rows.
- `XDIRIJ74`: `8/29`, no R2B improvement; likely semantic underselection or
  unmaterialized source table/sweep rows.
- `2RNHC2M5`: `14/32`, DOE/run-matrix class; inspect whether Stage5 filters
  rows or S2-7 never materializes them.
- `YZYKTTFE`: `2/19`, likely DOE/authorization unresolved.
- `KTNLRQZU`: `1/17`, likely raw underselection or unresolved table
  materialization boundary.

The next concrete round is R2B-R3. It starts with `JRMKHP5C` because it has
the clearest positive partial recovery under R2B and therefore the best chance
of exposing a deterministic downstream loss class without modifying prompts.

## Round R2B-R3: Primary NP Sample-ID Guard

Status: `positive_diagnostic_candidate_not_benchmark_valid`

### Diagnosis

`JRMKHP5C` still had a large remaining gap after R2B: `21/45` Stage5 rows.
The S2-7 ledger showed `Table 1`, `Table 2`, and `Table 3` were all
LLM-authorized and called. Inspection localized one deterministic loss:

- `Table 1` normalized authority rows contained real sample IDs such as
  `HbNPs-5`, `HbNPs-15`, `HbNPs-24`, and `HbNPs-6`.
- `extract_first_column_identity_rows_from_authority` could recover these
  rows, but the non-primary table guard treated `HbNPs-5`-style labels as not
  primary enough.
- Because those rows mostly had measurement/result columns, the guard filtered
  them as `measurement_only_nonprimary_table:partial_formulation`.

This is a reusable identity-label classification bug, not a `JRMKHP5C`-only
row creation rule.

### Repair

Update `formulation_identity_label_looks_primary` so nanoparticle sample IDs
such as `HbNPs-5`, `HbNPs5`, and similar `*NP(s)-number` labels count as
primary formulation identity labels.

Code and test:

- `src/stage2_sampling_labels/table_row_expansion_v1.py`
- `tests/test_stage2_table_row_expansion_scope_alias_roles_v1.py`
- validation command:
  `PYTHONPATH=. python3 -m unittest tests.test_stage2_table_row_expansion_scope_alias_roles_v1 -q`

### Replay Lineage

- S2-7:
  `data/results/20260511_b069802/230_stage2_s2_7_ee35_r2b_r3_primary_np_id_replay`
- Stage3:
  `data/results/20260511_b069802/231_stage3_ee35_r2b_r3_primary_np_id_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/232_stage5_ee35_r2b_r3_primary_np_id_final_output`
- Audit:
  `data/results/20260511_b069802/233_ee35_r2b_r3_primary_np_id_audit`

### Result

Aggregate:

| layer | R2B baseline | R2B-R3 | delta |
|---|---:|---:|---:|
| S2-7 rows | 285 (`218`) | 296 (`230`) | +11 |
| Stage5 final rows | 233 (`220`) | 244 (`232`) | +11 |

Paper-level audit:

- improved Stage5 paper count: `1`
- regressed Stage5 paper count: `0`
- `JRMKHP5C`: `21 -> 32` final rows, remaining count gap `13` versus GPT Web
  master reference `45`.

Interpretation:

R2B-R3 is accepted as a positive diagnostic candidate. It recovers real
LLM-authorized source-table sample IDs without changing prompt, raw responses,
or semantic authorization. The remaining `JRMKHP5C` gap is now mostly packed
source-table recovery and identity-normalization quality, for example
`row_01__hbnps-7` should align to `HbNPs-7` and packed source CSV rows still
hide labels such as `HbNPs-1`, `HbNPs-2`, `HbNPs-3`, and several later
`HbNPs-*` rows. That next repair is wider and should be separated from R2B-R3.

## Recording Requirements

For each completed repair round:

- update this plan with result paths and counts,
- add or update a repair-index row in
  `docs/repair_index/success_pattern_index_v1.tsv`,
- update governed repo memory under `data/mem/v1/` using maintained memory
  tooling or record why a rebuild/manual update is deferred,
- and add an external Codex memory note only if requested by the user-facing
  memory system contract.

## R2B-R4 Rowwise Sample-ID Label Canonicalization

Status: `positive_diagnostic_identity_repair_not_benchmark_valid`

### Diagnosis

R2B-R3 recovered many `JRMKHP5C` rows, but several Table 3 formulation rows
entered Stage5 with synthetic labels such as `row_01__hbnps-7` instead of the
native sample IDs. This was not a row-count loss at Stage5; it was an identity
alignment loss that would make GT comparison and downstream row joining brittle.

The loss boundary was inside S2-7 rowwise table materialization:

- the first cell already contained native labels such as `HbNPs-7`;
- the row extractor generated row-index labels instead of preserving that
  native formulation identity;
- Stage3 and Stage5 carried the synthetic labels forward.

### Repair

Update `extract_rowwise_formulation_rows_from_authority` so a primary first-cell
sample label is parsed and preserved as the row label. The raw first-cell label
is also kept as a `formulation_identity_label` assignment for downstream trace.

Code and test:

- `src/stage2_sampling_labels/table_row_expansion_v1.py`
- `tests/test_stage2_table_row_expansion_scope_alias_roles_v1.py`
- validation command:
  `PYTHONPATH=. python3 -m unittest tests.test_stage2_table_row_expansion_scope_alias_roles_v1 -q`

### Replay Lineage

- S2-7:
  `data/results/20260511_b069802/234_stage2_s2_7_ee35_r2b_r4_rowwise_sample_id_label_replay`
- Stage3:
  `data/results/20260511_b069802/235_stage3_ee35_r2b_r4_rowwise_sample_id_label_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/236_stage5_ee35_r2b_r4_rowwise_sample_id_label_final_output`
- Audit:
  `data/results/20260511_b069802/237_ee35_r2b_r4_rowwise_sample_id_label_audit`

### Result

Aggregate counts were intentionally unchanged:

| layer | R2B-R3 | R2B-R4 | delta |
|---|---:|---:|---:|
| S2-7 rows | 296 (`230`) | 296 (`234`) | 0 |
| Stage5 final rows | 244 (`232`) | 244 (`236`) | 0 |

Identity diagnostics:

- `JRMKHP5C` final rows stayed `32`.
- Naive missing GT-label count for `JRMKHP5C` improved `24 -> 18`.
- Table 3 labels changed from row-index forms to native IDs, for example
  `row_01__hbnps-7 -> HbNPs7`.
- Paper-level count regressions versus R2B-R3: `0`.

Interpretation:

R2B-R4 is accepted as a positive identity repair. It does not claim benchmark
performance, but it makes the recovered row universe more consumable by GT
alignment and later relation logic without changing live LLM output.

## R2B-R5 Packed Source-Row Sample-ID Recovery

Status: `positive_diagnostic_candidate_with_overretention_residual_not_benchmark_valid`

### Diagnosis

After R2B-R4, `JRMKHP5C` still missed many native sample IDs from Tables 1 and
2. Inspection of normalized source CSVs showed packed first-column rows where a
single source line contained multiple real sample IDs, for example:

- `HbNPs-1 HbNPs-14 HbNPs-22 HbNPs-2 HbNPs-5`
- `HbNPs-23 HbNPs-3 HbNPs-15`
- `HbNPs-18 Sample ID HbNPs-26`

The LLM had authorized the table scopes, but S2-7 only materialized the visible
rowwise or first-column labels and did not split packed source-row sample-ID
surfaces.

### Repair

Add a bounded S2-7 augmentation that reads the already reattached table source
CSV and emits additional rows only when one source line contains two or more
packed prefixed sample IDs. The repair uses the same label canonicalization key
as duplicate detection, so `HbNPs-5` and `HbNPs5` are treated as the same
identity.

An initial dedup attempt was rejected:

- rejected S2-7/Stage5/audit:
  `242 -> 244 -> 245`
- reason:
  the duplicate guard applied to all non-primary direct rows and incorrectly
  removed legal repeated condition rows in `ZB76MB3J`, reducing Stage5
  `7 -> 5`.

The accepted R5 guarded variant narrows duplicate suppression to rows created
by the new `packed_prefixed_sample_id_source_row` augmentation. Existing
condition/sweep rows are no longer touched.

Code and test:

- `src/stage2_sampling_labels/table_row_expansion_v1.py`
- `tests/test_stage2_table_row_expansion_scope_alias_roles_v1.py`
- validation command:
  `PYTHONPATH=. python3 -m unittest tests.test_stage2_table_row_expansion_scope_alias_roles_v1 -q`

### Replay Lineage

- S2-7:
  `data/results/20260511_b069802/246_stage2_s2_7_ee35_r2b_r5_packed_sample_id_guarded_dedup_replay`
- Stage3:
  `data/results/20260511_b069802/247_stage3_ee35_r2b_r5_packed_sample_id_guarded_dedup_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/248_stage5_ee35_r2b_r5_packed_sample_id_guarded_dedup_final_output`
- Audit:
  `data/results/20260511_b069802/249_ee35_r2b_r5_packed_sample_id_guarded_dedup_audit`

### Result

Aggregate:

| layer | R2B-R4 | R2B-R5 guarded | delta |
|---|---:|---:|---:|
| S2-7 rows | 296 (`234`) | 314 (`246`) | +18 |
| Stage5 final rows | 244 (`236`) | 262 (`248`) | +18 |

Paper-level audit:

- improved Stage5 paper count: `1`
- regressed Stage5 paper count: `0`
- `ZB76MB3J`: restored to `7` after the rejected dedup attempt had reduced it
  to `5`.
- `JRMKHP5C`: `32 -> 50` Stage5 rows versus GPT Web master reference `45`.
- `JRMKHP5C` naive missing GT-label count: `18 -> 2`.
- remaining missing labels: `PLGA-NPs`, `HbNPs-12`.
- residual over-retention: `+5` rows, including two prose-fragment labels
  (`Hb concentration=i.e` and
  `Hb concentration=a 20.6% for HbNPs-6 when using 100 mg mL−1 of Hb`) and
  four duplicate groups.

Interpretation:

R2B-R5 guarded is a positive diagnostic candidate because it recovers the
packed source-row sample-ID class across the full EE35 no-live replay with no
paper-level count regressions. It is not a finished benchmark repair: the next
repair should target over-retention/collapse after packed recovery, especially
malformed single-variable prose labels and duplicate identity groups in
`JRMKHP5C`.
