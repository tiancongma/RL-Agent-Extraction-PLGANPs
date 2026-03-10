# Formulation-Instance Pilot Evaluation (2026-03-10)

## Fixed 3-paper set reused
- `10.1016/j.ejpb.2004.09.002` | key `L3H2RS2H` | segmentation/alignment difficulty (top EE mismatch and contamination risk)

## Per-paper result summary
- `10.1016/j.ejpb.2004.09.002`: GT=22 formulation rows, pred=17; over-seg=no, under-seg=yes, boundary=broken, non-form suppression=not_applicable, inheritance separation=yes.

## Engineering readout
- Predicted formulation rows: 17 vs GT 22.
- Predicted candidate_non_formulation rows: 7 vs GT non-formulation rows 0.
- Compressed enum design appears workable: **mixed**.
- Next bottleneck: instance boundary enumeration in dense table blocks.