# Memory Validation Report v2

## Final Commands
1. `python3 src/utils/build_mem_v1.py`
2. `python3 src/utils/check_mem_v1.py`

## Final Outputs

### Build
```text
mem_dir=/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/mem/v1
status=initialized
mode=rebuild
refreshing=dec.tsv,err.tsv,idx.tsv,lin.tsv,prm.tsv,run.tsv
sources=112
runs=46
lineage=39
decisions=16
errors=462
prompts=35
index=598
status=pass
```

### Check
```text
mem_dir=/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/mem/v1
idx.tsv=598
run.tsv=46
lin.tsv=39
dec.tsv=16
err.tsv=462
prm.tsv=35
status=pass
```

## Validation Interpretation
- `build_mem_v1.py`: passed
- `check_mem_v1.py`: passed
- current rebuilt memory tables contain `0` missing `source_file` references
- initial stale state was removed by rebuild from the current governed source inventory
- validation behavior remains strict:
  - missing files still fail
  - lineage parent/child mismatches still fail
  - schema/header/id mismatches still fail

## Counts
- `historical_missing_source`: `0`
- `stale_memory_entry` removed during repair: `1332`
- `path_normalization_bug`: `0`
- `invalid_contract_reference`: `0`
- fixed tooling path/lineage bugs: `3`
  - `Path.parents` slicing crash
  - legacy-only run-id parsing against current v2 bucket/child layout
  - missing synthetic bucket-parent run rows for v2 lineage parents
