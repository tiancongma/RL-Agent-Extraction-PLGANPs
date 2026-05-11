# Formulation-Instance Pilot Evaluation (2026-03-10)

## Fixed 3-paper set reused
- `10.1007/s10439-019-02430-x` | key `BXCV5XWB` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1016/j.colsurfb.2009.03.028` | key `YGA8VQKU` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1016/j.ijpharm.2021.120820` | key `BB3JUVW7` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1021/acsomega.0c00111` | key `7ZS858NS` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1038/s41598-017-00696-6` | key `PA3SPZ28` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1039/c5ra27386b` | key `V99GKZEI` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1080/10717540802174662` | key `5GIF3D8W` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1080/10717544.2016.1199605` | key `WFDTQ4VX` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1111/jphp.12481` | key `RHMJWZX8` | remaining_dev15_expansion_after_fixed_tuning3
- `10.1155/2014/156010` | key `UFXX9WXE` | remaining_dev15_expansion_after_fixed_tuning3
- `10.2147/ijn.s54040` | key `QLYKLPKT` | remaining_dev15_expansion_after_fixed_tuning3
- `10.3390/nano10040720` | key `INMUTV7L` | remaining_dev15_expansion_after_fixed_tuning3

## Per-paper result summary
- `10.1007/s10439-019-02430-x`: GT=3 formulation rows, pred=3; over-seg=no, under-seg=no, boundary=mixed, non-form suppression=no, inheritance separation=no.
- `10.1016/j.colsurfb.2009.03.028`: GT=16 formulation rows, pred=18; over-seg=yes, under-seg=no, boundary=broken, non-form suppression=no, inheritance separation=no.
- `10.1016/j.ijpharm.2021.120820`: GT=12 formulation rows, pred=12; over-seg=no, under-seg=no, boundary=preserved, non-form suppression=not_applicable, inheritance separation=yes.
- `10.1021/acsomega.0c00111`: GT=1 formulation rows, pred=1; over-seg=no, under-seg=no, boundary=mixed, non-form suppression=partial, inheritance separation=not_observed.
- `10.1038/s41598-017-00696-6`: GT=3 formulation rows, pred=5; over-seg=yes, under-seg=no, boundary=broken, non-form suppression=partial, inheritance separation=yes.
- `10.1039/c5ra27386b`: GT=6 formulation rows, pred=6; over-seg=no, under-seg=no, boundary=mixed, non-form suppression=no, inheritance separation=yes.
- `10.1080/10717540802174662`: GT=32 formulation rows, pred=6; over-seg=no, under-seg=yes, boundary=broken, non-form suppression=not_applicable, inheritance separation=no.
- `10.1080/10717544.2016.1199605`: GT=27 formulation rows, pred=30; over-seg=yes, under-seg=no, boundary=broken, non-form suppression=no, inheritance separation=no.
- `10.1111/jphp.12481`: GT=1 formulation rows, pred=1; over-seg=no, under-seg=no, boundary=mixed, non-form suppression=no, inheritance separation=not_observed.
- `10.1155/2014/156010`: GT=26 formulation rows, pred=5; over-seg=no, under-seg=yes, boundary=broken, non-form suppression=not_applicable, inheritance separation=no.
- `10.2147/ijn.s54040`: GT=7 formulation rows, pred=8; over-seg=yes, under-seg=no, boundary=mixed, non-form suppression=partial, inheritance separation=yes.
- `10.3390/nano10040720`: GT=12 formulation rows, pred=12; over-seg=no, under-seg=no, boundary=preserved, non-form suppression=not_applicable, inheritance separation=no.

## Engineering readout
- Predicted formulation rows: 107 vs GT 146.
- Predicted candidate_non_formulation rows: 20 vs GT non-formulation rows 79.
- Compressed enum design appears workable: **yes**.
- Next bottleneck: instance boundary enumeration in dense table blocks.