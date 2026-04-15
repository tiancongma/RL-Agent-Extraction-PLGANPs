# WFDTQ4VX Loss Localization

## Scope

This file localizes where WFDTQ4VX rows are lost between:

- Lineage A:
  - older frozen raw-response replay lineage
- Lineage B:
  - newer live/current Stage2 recovery lineage

The question is whether the missing WFDTQ4VX rows are:

- already absent in the older raw response
- or lost later in parsing, validation, projection, or Stage5 closure

## A. Raw response layer

### Evidence for loss at this layer

- Lineage A raw-response artifact:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
  - explicitly contains only `3` `formulation_candidates`
  - they are:
    - `FC1_Lopinavir_PLGA_NPs_General`
    - `FC2_Coumarin_PLGA_NPs`
    - `FC3_Plain_Drug_Suspension`
- Lineage B raw-response artifact:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
  - textually shows:
    - `F_T2_B1` through batch-style DOE rows
    - repeated `3^3 factorial design` row-level structure
    - a much larger candidate universe

### Evidence against loss at this layer

- none strong enough to negate the direct candidate-count contrast

### Classification

- `upstream absence`

### Judgment

The missing WFDTQ4VX rows are already absent in the older replay raw response.

That is directly proven by artifact content.

## B. S2-5 semantic parsing layer

### Evidence for loss at this layer

- Lineage A S2-5 semantic summary:
  - `formulation_count = 3`
  - `doe_scope_declared = no`
- Lineage A semantic objects:
  - still only the same three candidates as the raw response
- Lineage B S2-5 semantic summary:
  - `formulation_count = 34`
- Lineage B semantic objects:
  - include many explicit batch-level candidates:
    - `F_T2_B1`
    - `F_T2_B2`
    - ...

### Evidence against loss at this layer

- the counts match the raw-response candidate-universe contrast almost exactly:
  - A raw = `3`, A S2-5 = `3`
  - B raw = `34`, B S2-5 = `34`

### Classification

- `unproven` as an independent loss layer

### Judgment

S2-5 reflects the upstream difference, but the repo does **not** prove that
S2-5 itself is where the missing rows are first lost.

The stronger reading is:

- the loss is already present before or at raw response

## C. S2-6 validation layer

### Evidence for loss at this layer

- none proven

### Evidence against loss at this layer

- Lineage A S2-6 report:
  - `status = pass`
  - `error_count = 0`
  - `warning_count = 0`
- Lineage B S2-6 report:
  - `status = pass`
  - `error_count = 0`
  - `warning_count = 0`
- no WFDTQ4VX-specific suppression reason is recorded in either validation
  report

### Classification

- `validator loss` = `unproven`

### Judgment

The repo does not support a claim that S2-6 suppresses WFDTQ4VX rows.

## D. S2-7 compatibility projection layer

### Evidence for loss at this layer

- Lineage A weak labels contain `3` WFDTQ4VX rows
- Lineage B weak labels contain `34` WFDTQ4VX rows

### Evidence against loss at this layer

- the S2-7 counts still mirror the S2-5 counts:
  - A S2-5 = `3`, A S2-7 = `3`
  - B S2-5 = `34`, B S2-7 = `34`
- no evidence shows additional WFDTQ4VX rows being dropped during projection
- no S2-7 guard or trace row shows WFDTQ4VX candidate suppression after the
  larger candidate universe already exists

### Classification

- `projection loss` = `unproven`

### Judgment

S2-7 does not look like the primary loss point.
It mostly preserves the candidate universe it receives.

## E. Stage5 layer

### Evidence for loss at this layer

- Lineage A Stage5 final table has `2` WFDTQ4VX rows while S2-7 has `3`
- Lineage A decision trace shows:
  - `FC1_Lopinavir_PLGA_NPs_General` kept
  - `FC2_Coumarin_PLGA_NPs` kept
  - `FC3_Plain_Drug_Suspension` filtered as explicit non-formulation

### Evidence against loss at this layer

- that single-row drop is expected and explicit:
  - `FC3_Plain_Drug_Suspension` is `candidate_non_formulation`
- Lineage B Stage5 final table has `33` rows while decision trace has `34`
- the one dropped row in B is also explicit non-formulation:
  - `F_Shared_NP_Preparation`
- the maintained Stage5 rule did **not** collapse away the missing DOE batch
  rows in A because those rows were never present upstream

### Classification

- `downstream collapse` = `minor but not root cause`

### Judgment

Stage5 performs only a small explicit non-formulation filter in both lineages.
It is not the root cause of the `2 vs 33` contrast.

## Overall localization

### Strongest evidence-supported class

- `upstream absence`

### What is proven

- the older replay lineage raw-response artifact already contains only a
  3-candidate WFDTQ4VX universe
- the newer live/current lineage raw-response artifact already contains a much
  larger batch-level DOE candidate universe
- later Stage2, Stage3, and Stage5 surfaces largely preserve that difference

### What is not proven

- that S2-5 independently destroyed a larger row universe in Lineage A
- that S2-6 suppressed WFDTQ4VX candidates
- that S2-7 dropped a previously present full DOE row set
- that Stage5 caused the large count gap

## Final localization statement

The repo supports this specific claim:

- WFDTQ4VX fails in the older replay lineage primarily because the older
  raw-response artifact already lacks the DOE batch-level formulation universe
  that appears in the newer live/current lineage.
