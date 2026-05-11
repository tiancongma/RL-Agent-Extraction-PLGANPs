# DEV15 Remaining-12 Expansion (2026-03-10)

## Why only 12 papers were newly run
- The fixed 3-paper tuning subset was excluded from the new extraction job to preserve a clean separation between repeatedly tuned papers and newly expanded DEV diagnostics.
- The frozen tuning subset remained: `5ZXYABSU`, `L3H2RS2H`, `WIVUCMYG`.
- The new run used only the other 12 DEV-15 benchmark papers from `data/cleaned/goren_2025/index/splits/dev_manifest_v1.tsv`.

## How the combined DEV-15 view was built
- Remaining-12 rows come from the new schema-first stage2 extraction run on `dev_manifest_remaining12_2026-03-10.tsv`.
- Tuned-3 rows come from the latest existing tuned result set: `formulation_instance_pilot3_eval_synthmethod_2026-03-10`.
- The combined DEV-15 TSV merges both summaries and labels each row with `source_group = tuned_3paper` or `remaining_12paper`.

## Major error patterns in the remaining 12
- Under-segmentation remains concentrated in: 5GIF3D8W, UFXX9WXE.
- Over-segmentation appears in: YGA8VQKU, PA3SPZ28, WFDTQ4VX, QLYKLPKT.
- Roughly matched count behavior appears in: BXCV5XWB, BB3JUVW7, 7ZS858NS, V99GKZEI, RHMJWZX8, INMUTV7L.
- Non-formulation suppression is still weak on: BXCV5XWB, YGA8VQKU, 7ZS858NS, PA3SPZ28, V99GKZEI, WFDTQ4VX, RHMJWZX8, QLYKLPKT.
- Clear inheritance-aware separation within the remaining 12 was observed on: BB3JUVW7, PA3SPZ28, V99GKZEI, QLYKLPKT.
