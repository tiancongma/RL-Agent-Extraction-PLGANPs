# L3H2RS2H Formulation-Instance Regression Audit (2026-03-10)

## Paper
- DOI: `10.1016/j.ejpb.2004.09.002`
- Key: `L3H2RS2H`

## Prior artifact for 22-instance expectation
- Fixed GT workbook: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx` (`review_formulations` sheet has 22 reviewed GT rows for this DOI).
- Candidate scaffold: `data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/L3H2RS2H__10_1016_j_ejpb_2004_09_002__candidates.jsonl` enumerates the same 22 candidate formulation rows.
- This older path is the DEV15 formulation-skeleton candidate/review workflow, not the new v7 pilot extractor.

## Regression localization
- Baseline probe with full 50k text window still returned only 8 predicted instances for this paper.
- Raw model output already contained only 8 family-level records (`NS-XAN-50`, `NC-XAN`, etc.), so the collapse happened in the LLM extraction step before flattening.
- Parser/flattening was not the main cause here: the TSV preserved the same 8 objects that appeared in the raw response.
- Truncation was not the main cause of the original 8-count regression either: the full-window baseline still returned 8. The earlier 20k run did additionally hide `Table 3/4/5`, but the family-level collapse was already present without truncation.

## Root cause
- The prompt let the model summarize repeated sweep-table rows into family-level parent/variant groups instead of enumerating every synthesis row first.
- For this paper, that meant separate nanosphere and nanocapsule concentration rows were collapsed into a few abstract variants, especially around the `Table 1/2` and `Table 3/4/5` sweeps.

## Minimal fix applied
- Added a table-heavy enumeration prompt hint in `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` that activates when the input looks like a multi-table sweep paper.
- The hint explicitly requires row-by-row formulation enumeration before parent/variant abstraction, preserves crystallized/ND sweep rows as formulation candidates, and forbids standalone `Global Parameters` pseudo-instances.

## Result
- Old pilot predicted count: `8`
- New single-paper rerun predicted count: `21`
- GT count: `22`
- Main improvement: row-level sweep enumeration was largely restored; the remaining gap is one omitted empty nanocapsule baseline row for the `0.5 mL Myritol` block.

## Remaining unresolved item
- Missing row: the empty nanocapsule baseline corresponding to the `Table 3/4` `0.5 mL Myritol 318` block.
- The successful rerun no longer collapses the sweep into 8 family summaries, but it still skips that one baseline row.