# Snapshot: DEV15 Reviewed-Boundary GT Closed (2026-03-18)

## Objective of Layer 2

Layer 2 boundary GT exists to validate the benchmark-facing formulation-row
object after Stage 5 closure.

The review target is not raw Stage 2 candidate recall and not field-level
parameter correctness. The Layer 2 object is:

- one benchmark-facing Stage 5 final row per observable formulation instance
- with correct keep/drop decisions
- correct row boundaries
- correct family/variant retention behavior
- correct benchmark inclusion semantics for controls, assay-only variants, and
  external comparators

## Final Outcome

Layer 2 reviewed-boundary GT is considered complete for the current DEV15
benchmark cycle.

Closed reviewed-boundary benchmark state:

- benchmark object: `final_formulation_table_v1.tsv`
- reviewed-boundary GT total: `210`
- reviewed-boundary compare result: `15/15` papers matching
- remaining reviewed-boundary mismatches: `none`

Benchmark-valid closing child run:

- `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/10_qlyk_commercial_reference_filter/run_20260318_1347_ae5599d_dev15_qlyk_commercial_reference_filter_no_llm_v1`

Primary reviewed-boundary compare artifact:

- `gt_compare_boundary_review_v1/final_table_vs_gt_counts_by_doi.tsv`

## Governing Counting Rules Now Frozen

Layer 2 is frozen around the following benchmark-governing rules:

- Count only observable formulation instances.
- Methods-only design-space levels do not count unless the paper reports them
  as realized formulation instances with attributable result evidence.
- Trend-only statements do not create formulation rows.
- Post-processing or measurement-only variants do not count as new formulation
  rows unless they are independently reported as formulation instances.
- Assay-only derivative particles do not count unless independently reported as
  benchmark-relevant formulation instances.
- External commercial or marketed comparator products do not count as internal
  formulation rows.
- Blank or control rows may remain in the final table for context, but their
  benchmark include semantics must be explicit and must not be inferred only
  from the presence of numeric outcomes.

## Key Closing Fixes

### 5GIF3D8W class: design-space inflation

Closed rule:

- narrative-only sweep sections cannot expand one row per declared methods level
  without explicit identity-level result support

Effect:

- `5GIF3D8W` dropped from over-counted sweep expansion to the reviewed-boundary
  authority count of `26`

### QLYKLPKT class: external commercial comparator inflation

Closed rule:

- comparative commercial or marketed reference rows without internal
  preparation identity are excluded from benchmark-facing Stage 5 formulation
  closure

Effect:

- `QLYKLPKT` commercial comparator row (`Sporanox`) is no longer counted
- reviewed-boundary DEV15 compare moved from `14/15` to `15/15`

## What Is Frozen

Frozen for the current DEV15 reviewed-boundary benchmark:

- Layer 2 formulation-row boundary authority
- row-count semantics for reviewed DEV15 papers
- benchmark-valid Stage 5 reviewed-boundary compare target
- counting rules for design-only levels, trend-only evidence, assay-only
  derivatives, and external commercial comparators

## What Remains Future Work

Not covered by this Layer 2 freeze:

- field-level correctness of retained formulation rows
- unit normalization and semantic normalization policy
- reported-vs-derived field provenance policy
- numeric tolerance policy for field-level evaluation
- formula-based validation and reconciliation rules

Those items belong to the next stage:

- Layer 3 GT for field-level correctness
