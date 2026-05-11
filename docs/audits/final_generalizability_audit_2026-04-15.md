# Final Generalizability Audit (2026-04-15)

Diagnostic-only, not benchmark-valid final output.

## 1. Executive Conclusion

For the remaining EE-positive Core Set A failures in `YGA8VQKU` and `BB3JUVW7`, the checked paper-local evidence does **not** support an additional clean reusable deterministic recovery rule that would lawfully move more rows into Core Set A.

Bottom line:

- remaining EE-positive rows still failing Core Set A: `21`
- missing field cases across those rows: `26`
- clearly generalizable missing cases: `0`
- borderline missing cases: `0`
- not generalizable missing cases: `26`
- estimated additional Core Set A rows recoverable by implementing **only** generalizable deterministic rules: `0`

Recommendation:

- `STOP`

Reason:

- `YGA8VQKU` still lacks checked paper-local `la_ga_ratio` support for the retained EE-positive family.
- `BB3JUVW7` still lacks checked paper-local `polymer_mw_kDa` support for the EE-positive nanosphere rows.
- `BB3JUVW7` does contain `la_ga_ratio` values, but only on the non-EE nanorod branch, with multiple values and no safe deterministic attachment path back to the EE-positive nanosphere rows.

The remaining gap is therefore best classified as **structurally unavoidable under a clean deterministic baseline**, not as a near-miss fixable by one more reusable rule.

## 2. Checked Surfaces

This audit used existing checked artifacts only:

- Step 1 final table:
  - `data/results/20260415_targeted_core_a_repair_round2_codepath_v1/final_formulation_table_v1.tsv`
- Step 2 value table:
  - `data/results/20260415_8a2502a/03_deterministic_step2_baseline/step2_value_backfill_table_v1.tsv`
- parameter binding output:
  - `data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_parameter_binding_resolved_v1.tsv`
- overlap audit artifacts:
  - `docs/audits/modeling_core_overlap_failure_audit_2026-04-15.md`
  - `docs/audits/modeling_core_overlap_failure_audit_2026-04-15_artifacts/ee_positive_core_overlap_failures_v1.tsv`
  - `docs/audits/modeling_core_overlap_failure_audit_2026-04-15_artifacts/paper_level_core_overlap_failure_summary_v1.tsv`
- paper-local table assets:
  - `data/cleaned/goren_2025/tables/YGA8VQKU/YGA8VQKU__table_01__html_table.csv`
  - `data/cleaned/goren_2025/tables/YGA8VQKU/YGA8VQKU__table_06__html_table.csv`
  - `data/cleaned/goren_2025/tables/YGA8VQKU/YGA8VQKU__table_07__html_table.csv`
  - `data/cleaned/goren_2025/tables/BB3JUVW7/BB3JUVW7__table_01__html_table.csv`
  - `data/cleaned/goren_2025/tables/BB3JUVW7/BB3JUVW7__table_02__html_table.csv`

Checked limitation:

- the manifest for `YGA8VQKU` points to `data/cleaned/content_goren_2025/text/YGA8VQKU.html.txt`, but that text surface is not present locally in the checked workspace
- no checked cleaned-text file for `BB3JUVW7` was present locally either
- accordingly, the generalizability decision below is anchored to the actual checked table assets and current pipeline artifacts that are present in the workspace

## 3. Remaining EE-Positive Row Inventory

### 3.1 `YGA8VQKU`

All `16` EE-positive rows still fail Core Set A only because `la_ga_ratio` is missing. `drug_name`, `encapsulation_efficiency_percent`, and `polymer_mw_kDa` are already present.

| final_formulation_id | missing_fields | already_present |
|---|---|---|
| `YGA8VQKU__fo__026349273ded` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__089ee49d17a4` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__22c0ae5347bd` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__353007a25802` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__3fea1371f4da` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__40e7ca936556` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__7b67ff8e4342` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__7c20b325dbdd` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__8b60bac4f61f` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__a4c0728f59cf` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__a979221be31a` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__b91bf8081fce` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__baa1875e0434` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__c6fef35aca8a` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__daa0a4eb88db` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |
| `YGA8VQKU__fo__f989e971a84d` | `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent`, `polymer_mw_kDa` |

### 3.2 `BB3JUVW7`

All `5` EE-positive rows still fail Core Set A because both `polymer_mw_kDa` and `la_ga_ratio` are missing. `drug_name` and `encapsulation_efficiency_percent` are already present.

| final_formulation_id | missing_fields | already_present |
|---|---|---|
| `BB3JUVW7__fo__65a902a4d2a7` | `polymer_mw_kDa`, `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent` |
| `BB3JUVW7__fo__7809efe8eb0a` | `polymer_mw_kDa`, `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent` |
| `BB3JUVW7__fo__809b2974391a` | `polymer_mw_kDa`, `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent` |
| `BB3JUVW7__fo__a1dda6a726f4` | `polymer_mw_kDa`, `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent` |
| `BB3JUVW7__fo__e853ad99c310` | `polymer_mw_kDa`, `la_ga_ratio` | `drug_name`, `encapsulation_efficiency_percent` |

## 4. Source-Evidence Check And Binding Feasibility

### 4.1 `YGA8VQKU`

Checked paper-local evidence:

- `YGA8VQKU__table_01__html_table.csv` gives the retained F1-F16 EE-positive formulation rows
- `YGA8VQKU__table_06__html_table.csv` and `YGA8VQKU__table_07__html_table.csv` explicitly distinguish low-viscosity versus high-viscosity PLGA families
- no checked selected HTML table contains `50:50`, `75:25`, `lactide`, `glycolide`, or an explicit LA:GA ratio surface for the retained EE-positive family

Conclusion:

- `polymer_mw_kDa` already has a lawful family-level deterministic path and is not a remaining blocker
- `la_ga_ratio` is not present in the checked source surfaces for the retained EE-positive family

### 4.2 `BB3JUVW7`

Checked paper-local evidence:

- `BB3JUVW7__table_01__html_table.csv` is the EE-positive nanosphere table and contains no `polymer_mw_kDa` or `la_ga_ratio` field
- `BB3JUVW7__table_02__html_table.csv` is the nanorod table and explicitly contains `PLGA type (lactide:glycolide)` with values including `75:25` and `50:50`
- no checked selected HTML table contains `kDa`, `molecular weight`, `MW`, or a polymer-grade surface

Conclusion:

- `polymer_mw_kDa` is absent in the checked source for the EE-positive nanosphere rows
- `la_ga_ratio` exists only for a different formulation family, with multiple values, and cannot be safely attached back to the EE-positive nanosphere rows by a clean deterministic rule

## 5. Binding Feasibility Table

| paper_key | field | target rows | evidence type in checked source | binding_feasibility | generalizable_rule | why |
|---|---|---:|---|---|---|---|
| `YGA8VQKU` | `la_ga_ratio` | 16 | not present in checked retained-family tables | `absent_in_source` | `no` | No explicit global, row, or retained-family LA:GA ratio was present in the checked paper-local surfaces. |
| `BB3JUVW7` | `polymer_mw_kDa` | 5 | not present in checked EE-positive nanosphere table or sibling nanorod table | `absent_in_source` | `no` | No checked polymer MW text, grade, or numeric surface exists to attach deterministically. |
| `BB3JUVW7` | `la_ga_ratio` | 5 | explicit per-row values exist only on nanorod branch (`75:25`, `50:50`) | `ambiguous_or_conflicting` | `no` | The target EE-positive rows are nanospheres, while the available ratios belong to a separate branch with multiple values and no safe deterministic mapping back. |

For completeness:

- the already-landed `YGA8VQKU` low-viscosity polymer recovery is an example of a **generalizable family-level deterministic rule** that was worth implementing
- it did not resolve the remaining overlap because the unresolved gap moved to `la_ga_ratio`, where the checked paper-local evidence no longer supports a reusable rule

## 6. Per-Paper Breakdown

### 6.1 `YGA8VQKU`

- total EE-positive rows: `16`
- rows missing `polymer_mw_kDa`: `0`
- rows missing `la_ga_ratio`: `16`
- `safe_global_attachable`: `0`
- `safe_family_attachable`: `0`
- `requires_branch_resolution`: `0`
- `ambiguous_or_conflicting`: `0`
- `absent_in_source`: `16`

Decision:

- remaining failures are **not generalizable**
- additional deterministic gain from one more reusable rule: `0`

### 6.2 `BB3JUVW7`

- total EE-positive rows: `5`
- rows missing `polymer_mw_kDa`: `5`
- rows missing `la_ga_ratio`: `5`
- `safe_global_attachable`: `0`
- `safe_family_attachable`: `0`
- `requires_branch_resolution`: `0`
- `ambiguous_or_conflicting`: `5`
- `absent_in_source`: `5`

Decision:

- remaining failures are **not generalizable**
- additional deterministic gain from one more reusable rule: `0`

## 7. Generalizable vs Non-Generalizable Split

Across both papers:

- total remaining EE-positive rows: `21`
- total remaining missing field cases: `26`
- clearly generalizable: `0`
- borderline: `0`
- not generalizable: `26`

Not-generalizable split:

- `16` cases: checked source does not contain a retained-row LA:GA signal at all (`YGA8VQKU`)
- `5` cases: checked source does not contain polymer MW support at all (`BB3JUVW7`)
- `5` cases: value exists only on a different branch, with multiple branch values and no safe deterministic attachment (`BB3JUVW7` `la_ga_ratio`)

## 8. Estimated Recoverable Core Set A Gain

If we implement **only** clean reusable deterministic rules:

- additional Core Set A rows recovered: `0`
- estimated Core Set A after patch for `YGA8VQKU` + `BB3JUVW7`: `0`

Interpretation:

- the current target-paper gain ceiling under a clean deterministic contract has already been reached
- any further gain would require paper-specific heuristics, manual interpretation, or multi-branch identity reasoning without an explicit attachment boundary

## 9. Final Recommendation

Recommendation:

- `STOP`

Why:

- there is no remaining generalizable deterministic patch with a lawful expected gain
- one more paper-specific patch would violate the audit decision rule for generalizability
- the residual gap is best treated as structurally unavoidable for the deterministic baseline rather than as unfinished engineering work

Practical readout:

- baseline status: sufficient for deterministic-baseline characterization
- residual misses: acceptable and expected under a strict explicit-only contract
- continuation value: low
