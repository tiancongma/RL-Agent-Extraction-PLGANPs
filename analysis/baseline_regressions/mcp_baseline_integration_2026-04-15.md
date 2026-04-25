# MCP Baseline Integration Audit (2026-04-15)

## Summary

The existing repo-local MCP server `repo-mcp` was extended to expose the governed baseline object model as read-only MCP resources and tools. The integration is additive only: it does not change pipeline logic, baseline semantics, or the registry/manifest contract. Baseline lookup now flows through `src/utils/baseline_registry_v1.py` as the single authority layer.

## What Tools Were Added

Added to `mcp_server/server.py`:

- `list_baselines`
- `resolve_baseline_query`
- `show_baseline`
- `validate_baseline`
- `get_baseline_artifacts`

Also added one read-only resource:

- `repo://baseline_registry`

## How Baseline Is Resolved

Resolution path:

1. MCP tool receives a query such as `20260415`
2. `mcp_server/server.py` delegates baseline loading and resolution to `src/utils/baseline_registry_v1.py`
3. The utility reads:
   - `data/baselines/BASELINE_REGISTRY.tsv`
   - the resolved `BASELINE_MANIFEST.json`
4. The utility validates:
   - required registry fields
   - required manifest fields
   - registry/manifest cross-reference consistency
5. MCP returns the resolved governed baseline object

Important constraint preserved:

- no baseline lookup is performed from `data/frozen/` names
- no baseline lookup is performed from `data/results/` recency
- no filesystem glob guessing is used

## Example Queries

### `list_baselines`

Returns the governed registry rows with the requested fields:

- `baseline_id`
- `baseline_type`
- `baseline_date`
- `stage_coverage`
- `benchmark_validity`
- `active_status`

Observed result includes:

- `baseline_20260415_operational_replay_v1`

### `show_baseline 20260415`

`show_baseline(query="20260415")` resolved correctly to:

- `baseline_20260415_operational_replay_v1`

Returned:

- full manifest content
- `authority_root = data/frozen/dev15_full_pipeline_freeze_v1`
- full `artifact_chain`
- full `lineage_chain`
- explicit `limitations`

## Proof That Registry Is Used

Code path evidence in [server.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\server.py):

- imports:
  - `from src.utils import baseline_registry_v1 as baseline_registry`
- helper usage:
  - `_load_registry_rows()`
  - `_resolve_baseline_row()`
  - `_load_resolved_baseline_object()`

Those helpers call the utility functions from [baseline_registry_v1.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\utils\baseline_registry_v1.py):

- `read_registry`
- `resolve_baseline_row`
- `read_manifest`
- `validate_registry_row`
- `validate_manifest`
- `validate_cross_reference`
- `build_baseline_object`

This means the MCP server does not implement a second baseline-resolution logic path.

## Proof That No Filesystem Inference Is Used

Negative proof from implementation:

- no `glob`
- no directory-recency sorting
- no `data/frozen/` scan for baseline resolution
- no `data/results/` scan for baseline resolution

Positive proof from structured error/results:

- error responses for baseline tools include:
  - `resolution_mode = baseline_registry_only`
  - `filesystem_inference_used = False`

## Validation Results

### MCP server starts successfully

Verified by launching the registered repo-local server command and observing:

- process remained alive under stdio wait state
- log line:
  - `INFO:mcp_server.server:Starting read-only repository MCP server over STDIO`

### `list_baselines` returns the 20260415 baseline

Verified:

- returned `baseline_20260415_operational_replay_v1`

### `show_baseline 20260415` resolves correctly

Verified:

- `resolved_baseline_id = baseline_20260415_operational_replay_v1`
- `authority_root = data/frozen/dev15_full_pipeline_freeze_v1`

### `validate_baseline` passes for 20260415

Verified:

- `validation_status = pass`
- `missing_fields = []`

### `resolve_baseline_query("20260415")` returns the correct baseline id

Verified:

- `baseline_id = baseline_20260415_operational_replay_v1`

### Optional `get_baseline_artifacts`

Verified:

- `stage2_completed = data/frozen/dev15_full_pipeline_freeze_v1/s2_7/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `stage3_relation = data/frozen/dev15_full_pipeline_freeze_v1/stage3/formulation_relation_records_v1.tsv`
- `stage5_final = data/frozen/dev15_full_pipeline_freeze_v1/stage5/final_formulation_table_v1.tsv`
- `comparison_outputs = []`

## Limitations

- read-only only
- no baseline creation via MCP
- no baseline mutation via MCP
- no automatic baseline detection outside the governed registry
- no pipeline execution triggers

## Files Updated

- [server.py](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\server.py)
- [README_MCP.md](d:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\README_MCP.md)
