# DEV15 Error Diagnostic Summary (2026-03-10)

- Total papers: 15
- Exact matches: 8
- Near matches: 4
- Under-segmentation papers: 3
- Over-segmentation papers: 4

## Worst 5 Papers By Absolute Error
- `5GIF3D8W` | `10.1080/10717540802174662` | GT=32 pred=6 diff=-26 | severe_error
- `UFXX9WXE` | `10.1155/2014/156010` | GT=26 pred=5 diff=-21 | severe_error
- `WFDTQ4VX` | `10.1080/10717544.2016.1199605` | GT=27 pred=30 diff=3 | moderate_error
- `L3H2RS2H` | `10.1016/j.ejpb.2004.09.002` | GT=22 pred=20 diff=-2 | near_match
- `PA3SPZ28` | `10.1038/s41598-017-00696-6` | GT=3 pred=5 diff=2 | near_match

## Common Patterns
- Under-segmentation remains concentrated in dense multi-formulation and DOE-style papers.
- Over-segmentation still appears in table-heavy and post-processing-heavy papers.
- Post-processing-heavy papers still show weak suppression of non-formulation rows.
- Sweep-series papers remain a systematic stress case for stable formulation-instance boundaries.
