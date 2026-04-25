# DOE Scope Construction Audit

## Source resolution

`data/results/ACTIVE_RUN.json` was not the right primary source for this audit.
It currently points to `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`, whose Stage2 surface is `paper_driven_deterministic_semantic_emitter_v1`, not the maintained `llm_first_composite` Stage2 path requested here.

This audit therefore resolved the maintained evidence from the explicit April 6 diagnostic Stage2 child runs:

- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`

Exact maintained source artifacts consumed:

- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/semantic_stage2_objects/raw_responses/WIVUCMYG__stage2_v2_raw_response.json`
- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/semantic_stage2_objects/raw_responses/5GIF3D8W__stage2_v2_raw_response.json`
- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final/RUN_CONTEXT.md`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/semantic_stage2_objects/raw_responses/UFXX9WXE__stage2_v2_raw_response.json`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix/RUN_CONTEXT.md`
- `data/results/20260406_ced19d6/08_doe_trigger_path_audit/doe_trigger_diagnostics_v1.json`
- `data/results/20260406_ced19d6/08_doe_trigger_path_audit/doe_trigger_path_instrumentation_report.md`

These runs explicitly declare `stage2_semantic_source_mode: llm_first_composite`, use the maintained composite Stage2 entrypoint, and isolate the DOE function-unit path under the maintained contract.

## Exact code path inspected

Maintained Stage2 scope construction and gating path:

1. `src/stage2_sampling_labels/run_stage2_composite_v1.py`
   - maintained composite Stage2 entrypoint
2. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
   - `DOE_TOKENS` at lines 68-77
   - identity-variable -> `variable_role` assignment at lines 1169-1187
   - `infer_semantic_scope_declarations(...)` at lines 1418-1535
   - semantic summary fields at lines 1618-1635
3. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
   - DOE scope resolution and function-unit call at lines 891-935
4. `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
   - `resolve_llm_declared_doe_scope(...)` at lines 89-105
   - execution preconditions and early exits at lines 222-267
   - deterministic DOE row materialization with governed scope ref at lines 285-335
5. `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
   - document-level DOE scope detection at lines 59-68
   - downstream legality checks for DOE expansion rows at lines 143-161

## Exact files inspected

- `AGENTS.md`
- `project/0_PROJECT_CHARTER.md`
- `project/1_REQUIREMENTS.md`
- `project/2_ARCHITECTURE.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `project/ACTIVE_DATA_SOURCE_CONTRACT.md`
- `docs/maintained_script_surface.tsv`
- `data/results/ACTIVE_RUN.json`

## DOE scope-construction decision path

The exact maintained scope-construction logic is in `infer_semantic_scope_declarations(...)`.

It always emits a default document scope:

- `scope_kind = document_semantic_scope`
- `declared_by = llm_semantic_discovery`
- `row_enumeration_required = no`
- `table_scope_refs = []`

It emits an additional DOE scope only if all of the following are true:

1. `doe_factor_names` is non-empty.
   - Built from `variable_candidates` whose `variable_role == "doe_factor"`.
2. `table_scope_refs` is non-empty.
   - First preference: table-linked evidence spans attached to those `doe_factor` variables.
   - Fallback only when `strong_doe_language` is true: table evidence whose locator/support text contains one of:
     - `box-behnken`
     - `experimental design`
     - `independent variable`
     - `dependent variable`
     - `effect of independent`
     - `design`
     - `level`
3. `(strong_doe_language or numbered_formulation_count >= 8)` is true.
   - `strong_doe_language` is driven by evidence span / unassigned observation text containing DOE-like tokens such as:
     - `box-behnken`
     - `response surface`
     - `factorial`
     - `experimental design`
     - `design matrix`
     - `design expert`
     - `run order`
     - `doe`
   - `numbered_formulation_count` counts formulation labels matching `F1`, `F-1`, `11.`, etc.

If those three conditions all hold, the DOE declaration is:

- `scope_kind = doe_table_row_enumeration_scope`
- `declared_by = llm_semantic_discovery`
- `authorizes_row_materialization_modes = [llm_semantic_discovery, deterministic_row_expansion_within_llm_scope]`
- `row_enumeration_required = yes`
- `table_scope_refs = [...]`
- `declared_doe_factors = sorted(doe_factor_names)`
- `declaration_basis =`
  - `llm_detected_strong_doe_language_plus_doe_factor_candidates_plus_table_scopes`, or
  - `llm_detected_numbered_formulation_sweep_plus_doe_factor_candidates_plus_table_scopes`

The adapter then always calls the DOE function unit, but the function unit exits immediately with `missing_llm_declared_doe_scope` if that DOE scope was not constructed.

## Per-paper comparison summary

### UFXX9WXE

Raw LLM evidence:

- Strong DOE/design language is explicit in title and text:
  - `Box-Behnken design`
  - `four independent factors`
  - `26 confirmatory runs`
  - `Table 2: Effect of independent process variables on dependent variable`
- Table-linked DOE cues are explicit.

Semantic objects:

- `variable_candidates` include 4 `doe_factor` objects:
  - `factor X1`
  - `factor X2`
  - `factor X3`
  - `factor X4`
- Table-linked evidence spans include:
  - `UFXX9WXE_table_13_caption_0`
  - `UFXX9WXE_table_14_caption_0`
  - `UFXX9WXE__table_13__pdf_table.csv | sample_row_1`

Scope construction outcome:

- DOE scope created:
  - `scope_kind = doe_table_row_enumeration_scope`
  - `declared_by = llm_semantic_discovery`
  - `row_enumeration_required = yes`
  - `table_scope_refs_count = 3`
  - `declared_doe_factors = ["factor_x1", "factor_x2", "factor_x3", "factor_x4"]`
  - `declaration_basis = llm_detected_strong_doe_language_plus_doe_factor_candidates_plus_table_scopes`
- Adapter resolved `semantic_scope_ref = UFXX9WXE__llm_declared_doe_scope__01`
- Function unit emitted 26 deterministic DOE rows.

### WIVUCMYG

Raw LLM evidence:

- DOE-like variable grid is explicit in table structure:
  - header row with `cPF`, `cPVA`, `cPLGA`, `pH`
  - numbered formulations `F1` through `F26`
- The decisive path is a numbered formulation sweep rather than a prose `Box-Behnken` phrase in the final declaration basis.

Semantic objects:

- `formulation_candidates` contain a large numbered sweep (`F1`-`F26`).
- Identity variables were normalized into `variable_candidates` with `variable_role = doe_factor` because names normalized into `DOE_TOKENS`:
  - `cpf_coded_level`
  - `cpva_coded_level`
  - `cplga_coded_level`
  - `ph_coded_level`
- `table_scope_refs` were built directly from table-linked factor evidence and numbered row spans.

Scope construction outcome:

- DOE scope created:
  - `scope_kind = doe_table_row_enumeration_scope`
  - `declared_by = llm_semantic_discovery`
  - `row_enumeration_required = yes`
  - `table_scope_refs_count = 19`
  - `declared_doe_factors = ["cpf_coded_level", "cplga_coded_level", "cpva_coded_level", "ph_coded_level"]`
  - `declaration_basis = llm_detected_numbered_formulation_sweep_plus_doe_factor_candidates_plus_table_scopes`
- Function unit ran but emitted 0 rows because the LLM output already covered the numbered formulation rows; no additional recovery rows were missing.

### 5GIF3D8W

Raw LLM evidence:

- Weak DOE-like study language exists:
  - `Effect of formulation variables like stabilizer concentration, amount of polymer, and drug was studied.`
  - `amount of drug, amount of polymer, and concentration of stabilizer were optimized to 5 mg, 50 mg, and 1.0% w/v.`
- The raw response preserves those as unassigned shared-context observations:
  - `U_DOE_StabilizerConc`
  - `U_DOE_PolymerAmount`
  - `U_DOE_DrugAmount`
  - `U_DOE_LactideContent`
- The paper has optimized-formulation tables, but not an explicit numbered DOE design table or per-row factor grid.

Semantic objects:

- `variable_candidates` exist, but none are `doe_factor`.
- Observed roles are:
  - `shared_context`
  - `identity_signal`
- There are no factor objects whose normalized names land in `DOE_TOKENS`.
- There are no table-linked DOE factor spans available for the first `table_scope_refs` pass.
- The fallback table-anchor pass also finds nothing because the relevant table captions/cells are optimized formulation outputs, not table text containing the required DOE anchor tokens.

Scope construction outcome:

- No DOE scope created.
- Only the default document scope remains:
  - `scope_kind = document_semantic_scope`
  - `declared_by = llm_semantic_discovery`
  - `row_enumeration_required = no`
  - `table_scope_refs = []`
- Adapter reaches the DOE function-unit callsite, but `resolve_llm_declared_doe_scope(document)` returns `None`.
- Function unit exits with `notes = missing_llm_declared_doe_scope`.
- No `semantic_scope_ref` for governed DOE expansion exists in the completed Stage2 artifact.

## DOE scope-construction condition matrix

| Condition | Implemented requirement | UFXX9WXE | WIVUCMYG | 5GIF3D8W |
|---|---|---|---|---|
| C1 | At least one `variable_candidate` with `variable_role = doe_factor` | yes | yes | no |
| C2 | At least one DOE-capable `table_scope_ref` | yes | yes | no |
| C3a | `strong_doe_language` | yes | not needed | likely yes |
| C3b | `numbered_formulation_count >= 8` | no | yes | no |
| C3 | `(C3a or C3b)` | yes | yes | yes or irrelevant |
| C4 | Final DOE declaration append requires `C1 and C2 and C3` | yes | yes | no |
| C5 | Function unit requires resolved LLM DOE scope | yes | yes | no |
| C6 | Function unit additionally requires `llm_first_composite` | yes | yes | yes |
| C7 | Function unit additionally requires DOE recovery enabled / explicit-only mode | yes | yes | yes |
| C8 | Function unit additionally requires source text path | yes | yes | yes |

Notes:

- `C1` and `C2` are the decisive gates.
- `5GIF3D8W` can still contain DOE-like language at the observation layer without ever reaching governed DOE scope.
- `WIVUCMYG` succeeds through the numbered-sweep branch.
- `UFXX9WXE` succeeds through the strong-DOE-language branch.

## First failing condition for 5GIF3D8W

The first failing condition is:

- `C1: at least one qualifying doe_factor object`

Exact reason:

- In the maintained semantic objects for `5GIF3D8W`, DOE-like variable-study evidence stays in `unassigned_observations` and `shared_context` notes.
- No `variable_candidate` is promoted to `variable_role = doe_factor`.
- Because `doe_factor_names` is empty, the first pass that tries to build `table_scope_refs` from DOE-factor evidence cannot populate anything.
- The document therefore arrives at the final scope append with:
  - `doe_factor_names = empty`
  - `table_scope_refs = empty`
  - no governed DOE scope

Upstream evidence that exists but is not promoted:

- `stabilizer concentration was studied`
- `amount of polymer was studied`
- `drug amount was studied`
- optimized values `5 mg`, `50 mg`, `1.0% w/v`

Why that evidence is not promoted:

- It is preserved only as shared-context / unassigned DOE-like observations.
- It is not attached to row-level DOE factor objects.
- It is not tied to a table anchor that the maintained scope builder treats as a governed DOE table scope.

Downstream consequence:

- `resolve_llm_declared_doe_scope(document)` returns `None`
- `run_doe_row_expansion_function_unit(...)` exits at the `missing_llm_declared_doe_scope` branch
- deterministic DOE expansion is never lawful for this paper in the maintained path

## Governance judgment

Classification: `conservative contract choice`

Reasoning:

- The maintained implementation is not merely checking for any weak DOE-like phrase.
- It requires DOE-factor objects plus table-anchored scope evidence before it will create a governed DOE scope.
- That is consistent with current governance language that:
  - deterministic DOE row enumeration is allowed only within LLM-declared DOE scope
  - row-level benchmark objects should come from concrete reported formulation instances, not abstract design-space discussion
- `5GIF3D8W` presents variable-study and optimization context, but not an explicit numbered DOE row matrix or per-formulation factor grid in the maintained artifacts.

Strict answer to the task question:

`5GIF3D8W` is not excluded because of a clear accidental bottleneck in the maintained code path. It is excluded because the current maintained scope-construction logic conservatively requires promoted DOE-factor objects plus DOE-capable table anchors before a governed DOE scope may exist, and `5GIF3D8W` never satisfies those conditions.`

Put differently:

- It is not enough that the paper says variables were studied.
- Under the current governed implementation, that signal must become concrete DOE-factor semantic objects and table-scoped anchors.
- For `5GIF3D8W`, it never does.

## Uncertainties / blocked evidence

- The semantic summary TSV field `doe_factor_count` is not reliable for this audit. It is recomputed from normalized `variable_name` membership in `DOE_TOKENS`, so factor names like `factor X1` still produce `doe_factor_count = 0` even when the semantic objects carry `variable_role = doe_factor`. I therefore treated the semantic JSONL objects, raw responses, and scope declarations as authoritative for factor eligibility.
- I did not widen prompts, validators, or contracts, per task constraint. This report localizes the current maintained behavior only.
