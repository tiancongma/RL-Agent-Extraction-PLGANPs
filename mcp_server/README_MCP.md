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

## Tools

- `resolve_active_run_source`
- `list_maintained_entrypoints_for_stage`
- `check_script_selection_legality`
- `validate_new_file_path`
- `get_authoritative_gt`
- `check_lawful_resume_boundary`

## Expected dependency

This skeleton expects the MCP Python SDK with FastMCP support to be available
in the environment.

## Local STDIO run example

```powershell
python -m mcp_server.server
```
