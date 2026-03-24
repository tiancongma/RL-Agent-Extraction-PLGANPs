# mem_v1

## Purpose

`mem_v1` is a governed supporting memory layer for recalling prior runs,
decisions, prompt fragments, and recurring failure patterns without relying on
chat context.

It is not a Stage 0-5 pipeline stage.

## Structure

Root:

- `data/mem/v1/`

Files:

- `sch.json`: schema manifest
- `idx.tsv`: searchable registry across all memory units
- `run.tsv`: run registry
- `lin.tsv`: run lineage links
- `dec.tsv`: decisions
- `err.tsv`: failure patterns
- `prm.tsv`: prompt recipes

## Workflow

1. Build or rebuild memory from governed sources:

```powershell
python src/utils/build_mem_v1.py
```

2. Query memory before complex debugging:

```powershell
python src/utils/mem_bootstrap_v1.py --query "investigate stage2 parsing regression for 5GIF3D8W"
python src/utils/query_mem_v1.py --query "collapse"
python src/utils/query_mem_v1.py --type error --stage stage2 --format json
```

Default complex-task pattern:

1. identify task class
2. query memory or run the bootstrap helper
3. inspect top memory-linked files
4. then read local source code or run artifacts
5. then act

3. Append a small manual correction or addition when rebuild is not needed:

```powershell
python src/utils/update_mem_v1.py --type decision --field title="Memory rule" --field decision="Query memory before complex debugging." --field source_file="AGENTS.md"
```

4. Validate the memory surface:

```powershell
python src/utils/check_mem_v1.py
```

## Rebuild vs update

- Rebuild when `docs/snapshots/`, `docs/methods/`, `project/*.md`, or
  `data/results/**/RUN_CONTEXT.md` changed materially.
- Use targeted update only for small manual additions or corrections that
  should append to the existing memory tables.

## Common task templates

- Debugging:
  - `python src/utils/mem_bootstrap_v1.py --query "collapse in stage2 for 5GIF3D8W"`
- Regression investigation:
  - `python src/utils/mem_bootstrap_v1.py --query "DOE regression in current dev15 run"`
- Run comparison:
  - `python src/utils/mem_bootstrap_v1.py --query "compare run lineage for deterministic refresh"`
- Pipeline modification:
  - `python src/utils/mem_bootstrap_v1.py --query "pipeline change for family variant handling"`
- GT mismatch analysis:
  - `python src/utils/mem_bootstrap_v1.py --query "identity mismatch for BB3JUVW7"`
- Lineage tracing:
  - `python src/utils/mem_bootstrap_v1.py --query "trace run lineage for targeted5 stage2 regression"`
