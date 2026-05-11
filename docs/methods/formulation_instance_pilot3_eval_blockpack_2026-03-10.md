# Formulation-Instance Pilot Evaluation (2026-03-10)

## Fixed 3-paper set reused
- `10.1002/jps.24101` | key `WIVUCMYG` | table-heavy multi-formulation (high row multiplicity and under-enumeration risk)
- `10.1016/j.ejpb.2004.09.002` | key `L3H2RS2H` | segmentation/alignment difficulty (top EE mismatch and contamination risk)
- `10.2147/ijn.s130908` | key `5ZXYABSU` | shared/global baseline condition risk (global inheritance seen in diagnostics)

## Per-paper result summary
- `10.1002/jps.24101`: GT=26 formulation rows, pred=26; over-seg=no, under-seg=no, boundary=preserved, non-form suppression=yes, inheritance separation=no.
- `10.1016/j.ejpb.2004.09.002`: GT=22 formulation rows, pred=28; over-seg=yes, under-seg=no, boundary=broken, non-form suppression=not_applicable, inheritance separation=no.
- `10.2147/ijn.s130908`: GT=9 formulation rows, pred=9; over-seg=no, under-seg=no, boundary=preserved, non-form suppression=not_applicable, inheritance separation=yes.

## Engineering readout
- Predicted formulation rows: 63 vs GT 57.
- Predicted candidate_non_formulation rows: 3 vs GT non-formulation rows 3.
- Compressed enum design appears workable: **yes**.
- Next bottleneck: parent-link and synthesis-change attribution quality for variant rows.