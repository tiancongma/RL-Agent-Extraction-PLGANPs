# DEV15 Identity-Freeze Root-Cause Analysis v1

Scope:
- run under analysis: `data/results/20260414_0011ee7/14_full_pipeline_patched_stage2_dev15_v1`
- Stage2 source: `data/results/20260414_0011ee7/01_non_doe_table_row_repair_v1/outputs/repaired_full_freeze/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 evidence: `data/results/20260414_0011ee7/14_full_pipeline_patched_stage2_dev15_v1/formulation_relation_v1/formulation_relation_records_v1.tsv`
- Stage5 evidence: `data/results/20260414_0011ee7/14_full_pipeline_patched_stage2_dev15_v1/final_formulation_table_v1.tsv`
- identity-freeze evidence: `data/results/20260414_0011ee7/14_full_pipeline_patched_stage2_dev15_v1/audit/identity_freeze_guardrail_v1/identity_freeze_report_v1.tsv`
- GT counts authority: `data/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv`

Method:
- For each paper, compare Stage2 candidate count, Stage5 final count, and GT count.
- Inspect whether Stage5 actually collapsed anything.
- Inspect whether identity-freeze failure is only a binding artifact or whether count drift is already present upstream.
- Assign exactly one primary failure type:
  - `TYPE A` = Stage2 over-generation
  - `TYPE B` = Stage5 insufficient collapse
  - `TYPE C` = identity reassignment / binding error
  - `TYPE D` = Stage2 under-generation
  - `TYPE E` = mixed

## Executive Conclusion

The dominant failure is Stage2-driven count drift, not Stage5 collapse.

Identity-freeze also fails for every paper at the binding surface in this run: `selected_binding_count=0` and `identity_reassignment_detected=yes` for all 15 papers. But that binding failure is not the primary root cause for most papers, because 14 of 15 papers already show count drift before identity-freeze can even be meaningfully evaluated.

Clear statement:

`Stage2 repair is invalid relative to identity contract.`

Reason:
- the repaired Stage2 artifact changed the formulation universe for many papers, including non-target papers
- 14 of 15 papers already disagree with GT at the Stage2 or Stage5 count layer
- Stage5 did not materially collapse these rows back into the reviewed identity universe
- only `PA3SPZ28` reaches the GT count and still fails primarily because the identity-binding surface is not aligned

Required next-step recommendation:

`next step MUST be upstream rollback`

This recommendation is a classification outcome, not a fix proposal: the patched Stage2 lineage is not lawful enough to hand off to Stage5 or identity-binding redesign as the main repair target because the candidate universe itself is already unstable.

## Count Surface

| paper_key | Stage2 candidates | Stage5 final | GT count | primary_type |
| --- | ---: | ---: | ---: | --- |
| 5GIF3D8W | 8 | 8 | 26 | TYPE D |
| 5ZXYABSU | 9 | 6 | 9 | TYPE D |
| 7ZS858NS | 7 | 7 | 1 | TYPE A |
| BB3JUVW7 | 13 | 13 | 12 | TYPE A |
| BXCV5XWB | 4 | 3 | 9 | TYPE D |
| INMUTV7L | 14 | 13 | 12 | TYPE A |
| L3H2RS2H | 8 | 6 | 21 | TYPE D |
| PA3SPZ28 | 6 | 5 | 5 | TYPE C |
| QLYKLPKT | 3 | 2 | 7 | TYPE D |
| RHMJWZX8 | 2 | 1 | 2 | TYPE D |
| UFXX9WXE | 29 | 29 | 27 | TYPE A |
| V99GKZEI | 7 | 7 | 6 | TYPE A |
| WFDTQ4VX | 3 | 2 | 30 | TYPE D |
| WIVUCMYG | 29 | 29 | 26 | TYPE A |
| YGA8VQKU | 13 | 13 | 17 | TYPE D |

## Per-Paper Classification

### 5GIF3D8W

- Counts: Stage2 `8` -> Stage5 `8` -> GT `26`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits only eight coarse identities: `FC_PCL_DrugLoaded`, `FC_PCL_Empty`, `FC_PLGA5050_DrugLoaded`, `FC_PLGA5050_Empty`, `FC_PLGA7525_DrugLoaded`, `FC_PLGA7525_Empty`, `FC_PLGA8515_DrugLoaded`, `FC_PLGA8515_Empty`
  - Stage5 keeps all eight with `final_output_rule=kept_without_collapse`
  - GT authority expects 26 formulation instances
  - Identity-freeze shows final rows as unbound, for example `FC_PCL_DrugLoaded`
- Why this classification is correct:
  - The failure is not that Stage5 duplicated rows; Stage5 preserved the eight rows it received.
  - The paper-level formulation sweep space present in GT is missing from the Stage2 candidate universe itself.

### 5ZXYABSU

- Counts: Stage2 `9` -> Stage5 `6` -> GT `9`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 rows are `NPB1-3`, `NPG1-3`, `NPR1-3`
  - Stage5 filters `NPB1`, `NPB2`, `NPB3` with `decision=filtered_non_formulation` and `decision_rule=explicit_candidate_non_formulation`
  - Stage5 keeps only `NPG1-3` and `NPR1-3`
  - Identity-freeze still shows unbound finals such as `NPG1`
- Why this classification is correct:
  - Raw Stage2 row count happens to equal GT, but three benchmark-relevant candidates are emitted in a form that Stage5 discards as non-formulation.
  - The effective benchmark-eligible formulation universe is already too small before identity freeze.

### 7ZS858NS

- Counts: Stage2 `7` -> Stage5 `7` -> GT `1`
- Primary type: `TYPE A — Stage2 over-generation`
- Evidence:
  - Stage2 emits seven rows: `F1` through `F7`
  - Stage5 keeps all seven with `kept_without_collapse`
  - `collapsed_variant_count_sum=0`
  - Identity-freeze reports all seven as unbound final rows; example representative source ID: `F1`
- Why this classification is correct:
  - The overcount exists fully at Stage2 and survives unchanged through Stage5.
  - There is no sign that Stage5 created the surplus; it simply retained the Stage2 candidate universe.

### BB3JUVW7

- Counts: Stage2 `13` -> Stage5 `13` -> GT `12`
- Primary type: `TYPE A — Stage2 over-generation`
- Evidence:
  - Stage2 emits 13 formulation IDs including `FC_NR_T2_R1-7`, `FC_NS_T1_R1-5`, and `FC_NS_GENERIC`
  - Stage5 keeps all 13 with `kept_without_collapse`
  - Relation layer records extensive membership and parent-link structure, but no benchmark-final collapse occurs
  - Identity-freeze shows unbound rows such as `FC_NR_T2_R1`
- Why this classification is correct:
  - The paper is already over GT before Stage5 closure.
  - Stage5 does not add rows; it merely fails to reduce the Stage2 surplus, so the primary failure remains upstream over-generation.

### BXCV5XWB

- Counts: Stage2 `4` -> Stage5 `3` -> GT `9`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits only `F1`, `F2`, `F3`, `F4`
  - Stage5 filters `F4` as `filtered_non_formulation`
  - Final kept set is only `F1`, `F2`, `F3`
  - GT authority expects 9 formulation instances
- Why this classification is correct:
  - Even before the Stage5 filter, the candidate universe is far below GT.
  - The paper is missing formulation instances, not suffering from excess duplicate rows.

### INMUTV7L

- Counts: Stage2 `14` -> Stage5 `13` -> GT `12`
- Primary type: `TYPE A — Stage2 over-generation`
- Evidence:
  - Stage2 emits `FC001` through `FC014`
  - Stage5 keeps 13 rows and collapses none
  - Relation records show 14 candidates and 13 parent links, indicating a dense candidate universe already present upstream
  - Identity-freeze shows unbound final IDs such as `FC001`
- Why this classification is correct:
  - The row surplus exists at Stage2 and remains above GT after Stage5.
  - This is not a binding-only problem because the benchmark count is already too high before any scaffold alignment.

### L3H2RS2H

- Counts: Stage2 `8` -> Stage5 `6` -> GT `21`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits only eight coarse identities: nanocapsules, nanoemulsions, and nanospheres for a few drug states
  - Stage5 filters `nanoemulsion_3MeOXAN_L3H2RS2H` and `nanoemulsion_XAN_L3H2RS2H` as `explicit_candidate_non_formulation`
  - Final keeps only six rows
  - GT authority expects 21 formulation instances
- Why this classification is correct:
  - The candidate universe is much too small relative to GT.
  - Stage5 pruning worsens the shortfall, but the primary problem is that Stage2 never produced a sufficiently granular formulation set.

### PA3SPZ28

- Counts: Stage2 `6` -> Stage5 `5` -> GT `5`
- Primary type: `TYPE C — identity reassignment / binding error`
- Evidence:
  - Stage5 count matches GT exactly
  - The only removed Stage2 row is `F_Blank_NPs`, filtered as `explicit_candidate_non_formulation`
  - Final rows include `F_GAR_NPs_1_10_NP`, `F_GAR_NPs_1_10_Stored`, `F_GAR_NPs_1_20_NP`, `F_GAR_NPs_1_6_66_NP`, `F_GAR_NPs_General`
  - Identity-freeze still fails because final rows are treated as unbound
- Why this classification is correct:
  - This is the one paper where count drift has been effectively corrected by Stage5 and the remaining failure is the inability to align final identities to the scaffold.
  - Therefore the primary failure here is binding, not Stage2 count drift.

### QLYKLPKT

- Counts: Stage2 `3` -> Stage5 `2` -> GT `7`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits only `FC_PLGA_ITZ_NS_General`, `FC_Sporanox`, `FC_PLGA_ITZ_NS_Optimal`
  - Stage5 filters `FC_Sporanox` as `explicit_candidate_non_formulation`
  - Final keeps only two rows
  - GT authority expects seven formulation instances
- Why this classification is correct:
  - The benchmark failure is driven by missing formulation instances, not duplication.
  - Identity-freeze misbinding is present, but the count deficit is already decisive upstream.

### RHMJWZX8

- Counts: Stage2 `2` -> Stage5 `1` -> GT `2`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits `FC1` and `FC2`
  - Stage5 filters `FC2` as `explicit_candidate_non_formulation`
  - Final keeps only `FC1`
  - Identity-freeze shows `FC1` as unbound
- Why this classification is correct:
  - Stage2 nominal count equals GT, but one of the two candidates is not carried as a benchmark-valid formulation into Stage5.
  - The effective formulation universe is under-generated once Stage2 semantics are applied.

### UFXX9WXE

- Counts: Stage2 `29` -> Stage5 `29` -> GT `27`
- Primary type: `TYPE A — Stage2 over-generation`
- Evidence:
  - Stage2 contains the DOE series `UFXX9WXE_DOE_Row_01` through `UFXX9WXE_DOE_Row_26` plus `FC001-003`
  - Stage5 keeps all 29 rows with `kept_without_collapse`
  - `collapsed_variant_count_sum=0`
  - Identity-freeze shows unbound final IDs such as `UFXX9WXE_DOE_Row_15`
- Why this classification is correct:
  - The two-row surplus is already present in the Stage2 candidate surface.
  - Stage5 does not create new rows; it preserves the Stage2 over-generation.

### V99GKZEI

- Counts: Stage2 `7` -> Stage5 `7` -> GT `6`
- Primary type: `TYPE A — Stage2 over-generation`
- Evidence:
  - Stage2 emits `FC_01` through `FC_07`
  - Stage5 keeps all seven with `kept_without_collapse`
  - `collapsed_variant_count_sum=0`
  - Identity-freeze shows unbound finals such as `FC_01`
- Why this classification is correct:
  - There is a one-row surplus already upstream.
  - No Stage5 collapse occurs, so the primary fault remains Stage2 over-generation.

### WFDTQ4VX

- Counts: Stage2 `3` -> Stage5 `2` -> GT `30`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits only `FC1_Lopinavir_PLGA_NPs_General`, `FC2_Coumarin_PLGA_NPs`, `FC3_Plain_Drug_Suspension`
  - Stage5 filters `FC3_Plain_Drug_Suspension` as `explicit_candidate_non_formulation`
  - Final keeps only two rows
  - GT authority expects 30 formulation instances
- Why this classification is correct:
  - This is a massive missing-instance failure.
  - Neither identity binding nor Stage5 collapse can explain a gap of 28 rows; the formulation universe is missing at Stage2.

### WIVUCMYG

- Counts: Stage2 `29` -> Stage5 `29` -> GT `26`
- Primary type: `TYPE A — Stage2 over-generation`
- Evidence:
  - Stage2 emits 29 formulation IDs including `F1`, `F10-26`, `F11_freeze_dried`, `F19_freeze_dried`, `F20_freeze_dried`
  - Stage5 keeps all 29 with `kept_without_collapse`
  - `collapsed_variant_count_sum=0`
  - Identity-freeze shows unbound finals such as `F1`
- Why this classification is correct:
  - The paper is over GT already at Stage2.
  - Stage5 does not materially alter the count, so the primary problem is upstream over-generation.

### YGA8VQKU

- Counts: Stage2 `13` -> Stage5 `13` -> GT `17`
- Primary type: `TYPE D — Stage2 under-generation`
- Evidence:
  - Stage2 emits 13 identities: `F1-F11` plus `low_viscosity_PLGA_ns` and `high_viscosity_PLGA_ns`
  - Stage5 keeps all 13 with `kept_without_collapse`
  - GT authority expects 17 formulation instances
  - Identity-freeze shows all final rows as unbound
- Why this classification is correct:
  - The count deficit exists before identity binding.
  - There is no Stage5 collapse to blame because Stage5 simply preserves the undersized candidate set.

## Cross-Paper Pattern

Classification counts:
- `TYPE A — Stage2 over-generation`: 6 papers
- `TYPE B — Stage5 insufficient collapse`: 0 papers
- `TYPE C — identity reassignment / binding error`: 1 paper
- `TYPE D — Stage2 under-generation`: 8 papers
- `TYPE E — mixed`: 0 papers

Dominant pattern:
- The dominant failure type is `TYPE D`, with `TYPE A` close behind.
- Combined, `TYPE A + TYPE D = 14/15` papers.
- That makes the run primarily `Stage2-driven`, not `Stage5-driven`.

Important nuance:
- The identity-freeze gate surface is universally binding-broken in this run.
- Every paper shows `selected_binding_count=0`, so the gate cannot validate membership alignment.
- But that universal binding failure does not erase the upstream count evidence.
- For 14 of 15 papers, the formulation universe is already wrong before identity-freeze has a fair chance to act.

Why `TYPE B` is not primary anywhere:
- Kept final rows almost uniformly use `final_output_rule=kept_without_collapse`
- `collapsed_variant_count_sum=0` across all 15 papers
- Where counts are wrong, they are already wrong in the Stage2 candidate surface
- Stage5 is mostly passing through or filtering Stage2 rows, not creating a new duplicate-collapse failure family

## Final Assessment

`Stage2 repair is invalid relative to identity contract.`

Interpretation:
- the patched Stage2 artifact is not a lawful benchmark continuation boundary for DEV15
- the candidate universe shifted too far from the reviewed identity universe
- the identity-freeze failure is not primarily a downstream Stage5 collapse issue
- identity binding is broken at the gate, but that is secondary for 14 of 15 papers because the incoming candidate set is already wrong

Required next-step recommendation:

`next step MUST be upstream rollback`

Reason for this recommendation:
- the failure is primarily Stage2-driven
- a Stage5 patch would act on the wrong candidate universe
- an identity-binding redesign would only relabel a candidate universe that is still wrong for 14 of 15 papers
