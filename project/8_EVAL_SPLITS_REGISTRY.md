# Eval Splits Registry

Purpose: prevent leakage across iterative development and holdout evaluation.

- DEV splits are used for iterative pipeline improvement.
- TEST splits are used for evaluation and must exclude all registered DEV keys.

## goren_2025

- split_name: `dev_v1`
- dev_keys_file: `data/cleaned/goren_2025/index/splits/dev_keys_v1.tsv`
- dev_manifest_file: `data/cleaned/goren_2025/index/splits/dev_manifest_v1.tsv`
- dev_coverage_file: `data/cleaned/goren_2025/index/splits/dev_tables_extraction_coverage_v1.tsv`
- selection_rule: include all `html_found=True`, then fill with `pdf_found=True && html_found=False`
- seed: `13`

Exact DEV keys (15):

```tsv
dataset_id	split_name	zotero_key
goren_2025	dev_v1	5GIF3D8W
goren_2025	dev_v1	5ZXYABSU
goren_2025	dev_v1	7ZS858NS
goren_2025	dev_v1	BB3JUVW7
goren_2025	dev_v1	BXCV5XWB
goren_2025	dev_v1	INMUTV7L
goren_2025	dev_v1	L3H2RS2H
goren_2025	dev_v1	PA3SPZ28
goren_2025	dev_v1	QLYKLPKT
goren_2025	dev_v1	RHMJWZX8
goren_2025	dev_v1	UFXX9WXE
goren_2025	dev_v1	V99GKZEI
goren_2025	dev_v1	WFDTQ4VX
goren_2025	dev_v1	WIVUCMYG
goren_2025	dev_v1	YGA8VQKU
```

Binding rule:

- Any TEST split builder must exclude all DEV keys listed in this registry.

## How Tools Must Enforce This

- Any script that builds a TEST split must accept:
  - `--exclude-keys-file <path>`
- In strict mode, TEST split builders must either:
  - require `--exclude-keys-file`, or
  - load DEV keys from this registry automatically.
