# RUN Summary

Run ID:run_20260130_0913_9436e6c_sample10
Date (local):20260130
Git commit:9436e6c
Branch:main

## Purpose
Baseline run on sample10 under the new repo structure (reproducibility checkpoint).

## Inputs
- manifest_current: data/cleaned/index/manifest_current.tsv
- key2txt: data/cleaned/index/key2txt.tsv
- sample definition: data/cleaned/samples/sample10_htmlfirst.jsonl
- optional: data/cleaned/samples/sample10_htmlfirst.tsv

## Configuration
- config_used: configs/<name>.yaml (or "N/A" if CLI-only)

## Commands executed
1) Convert sample10 JSONL to TSV adapter for key2txt
python .\src\utils\convert_sample_manifest_to_tsv.py --in-jsonl data\cleaned\samples\sample10_htmlfirst.jsonl --out-tsv data\cleaned\samples\sample10_for_key2txt.tsv
2) Build key2txt from sample10 (TSV adapter)
python .\src\stage2_sampling_labels\build_key2txt_from_sample_manifest.py --sample-manifest data\cleaned\samples\sample10_for_key2txt.tsv --output data\cleaned\index\key2txt.tsv

3)

## Outputs
- results dir: data/results/run_20260130_0913_9436e6c_sample10/
- key artifacts:
  - formulations*.tsv/jsonl
  - qc report (if any)
  - metrics.json (if any)

## Notes / Issues
- 

## Validation
- [ ] Script(s) completed without error
- [ ] Output files exist in data/results/run_20260130_0913_9436e6c_sample10/
- [ ] This run is reproducible from this commit + config + inputs
