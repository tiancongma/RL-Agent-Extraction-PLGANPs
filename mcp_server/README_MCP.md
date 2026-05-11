# Repository MCP Server v1

This directory contains a read-only MCP server skeleton for this repository.

## Scope

The server exposes only repository-governance resources and validation tools.

It does not:

- execute pipeline scripts
- write repository files
- edit `data/results/ACTIVE_RUN.json`
- mutate run artifacts

## Resources

- `repo://active_pipeline_authority`
- `repo://active_data_source_authority`
- `repo://maintained_entrypoint_registry`
- `repo://stage2_contract_and_frozen_substeps`
- `repo://naming_and_run_layout_contract`
- `repo://baseline_registry`

## Tools

- `resolve_active_run_source`
- `list_maintained_entrypoints_for_stage`
- `check_script_selection_legality`
- `validate_new_file_path`
- `get_authoritative_gt`
- `check_lawful_resume_boundary`
- `list_baselines`
- `resolve_baseline_query`
- `show_baseline`
- `validate_baseline`
- `get_baseline_artifacts`

## Baseline Query Tools

The server now exposes the governed baseline object model as read-only MCP tools.

Baseline source of truth:

- `data/baselines/BASELINE_REGISTRY.tsv`
- `data/baselines/<baseline_id>/BASELINE_MANIFEST.json`
- `src/utils/baseline_registry_v1.py`

Important rule:

- baseline lookup uses the registry utility as the only logic layer
- the server does not infer baselines from folder names, filesystem recency, or glob guessing

Available baseline tools:

- `list_baselines`
  - returns governed baseline registry entries with:
    - `baseline_id`
    - `baseline_type`
    - `baseline_date`
    - `stage_coverage`
    - `benchmark_validity`
    - `active_status`
- `resolve_baseline_query`
  - resolves a strict baseline query such as `20260415` into one governed `baseline_id`
- `show_baseline`
  - resolves a baseline query and returns the full governed manifest plus:
    - resolved `authority_root`
    - `artifact_chain`
    - `lineage_chain`
    - `limitations`
- `validate_baseline`
  - validates one governed baseline object using the registry utility logic
- `get_baseline_artifacts`
  - returns the key artifact paths exposed by the governed baseline object when present

Example queries:

- `list_baselines`
- `resolve_baseline_query("20260415")`
- `show_baseline(query="20260415")`
- `validate_baseline("baseline_20260415_operational_replay_v1")`

## Expected dependency

This skeleton expects the MCP Python SDK with FastMCP support to be available
in the environment.

## Local STDIO run example

```powershell
python -m mcp_server.server
```
