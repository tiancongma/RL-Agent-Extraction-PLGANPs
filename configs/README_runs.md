# Standard Commands

## Clean from manifest
python .\src\pdf2clean.py --manifest .\data\cleaned\manifest.tsv --out-dir .\data\cleaned --prefer html --overwrite --verbose

## Step1 static smoke (2 keys, no LLM)
pwsh -File .\configs\run_step1_static_smoke_2keys.ps1 `
  -RunId run_20260227_1016_a8d884b_goren2025_step1dev_v1 `
  -OutSubdir step1_dev_smoke_2keys `
  -DatasetId goren_2025 `
  -Keys "5GIF3D8W,7ZS858NS"
