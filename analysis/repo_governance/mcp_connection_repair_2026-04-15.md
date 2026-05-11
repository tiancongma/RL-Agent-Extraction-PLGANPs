# MCP Connection Repair (2026-04-15)

## Summary

The machine-local `repo-mcp` connection layer is now repaired. The missing Python MCP SDK was installed into the user Python environment, a minimal repo-local launcher was added at `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\launch_repo_mcp.py`, and an explicit `repo-mcp` registration was added to `C:\Users\tiancong\.codex\config.toml`. The registered command now starts successfully, the repo-local MCP server completes an actual stdio MCP handshake, and the governed repo resources/tools are exposed to a local MCP client. The only remaining limitation observed is that this already-running Codex thread still reports `unknown MCP server 'repo-mcp'`, which is consistent with the session having been initialized before the new registration was written.

## Starting Failure State

- `C:\Users\tiancong\.codex\config.toml` had no `repo-mcp` registration.
- `python -m mcp_server.server` failed importability because the local Python environment did not have the `mcp` package.
- The repo-governance MCP server name `repo-mcp` was not available to the current thread MCP resource surface.

## Exact Changes Made

1. Installed the minimum runtime dependency into the user Python environment:
   - package: `mcp`
   - version: `1.27.0`
   - install location: `C:\Users\tiancong\AppData\Roaming\Python\Python314\site-packages`

2. Added a minimal repo-local launcher:
   - path: `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\launch_repo_mcp.py`
   - role: bootstrap the existing `mcp_server.server` module from a stable absolute-path entrypoint without relying on uncertain client-side `cwd` behavior

3. Added the shared Codex registration:
   - path: `C:\Users\tiancong\.codex\config.toml`
   - block added:

```toml
[mcp_servers.repo-mcp]
command = 'C:\Program Files\Python314\python.exe'
args = ['D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\launch_repo_mcp.py']
```

## Registration Path and Config Used

The client-side registration was written to the shared Codex config file:

- `C:\Users\tiancong\.codex\config.toml`

This path was chosen because:

- the prior audit identified Codex Desktop as the visible MCP client surface on this machine
- OpenAI Codex MCP configuration is shared through `~/.codex/config.toml`
- there was no repo-local `.vscode/mcp.json` and no prior MCP registration in VS Code user settings

## Runtime Dependency Changes

- Installed `mcp==1.27.0` with `python -m pip install --user mcp`
- After install, these imports succeeded:
  - `mcp`
  - `mcp.server.fastmcp`
  - `mcp_server.server`

## Startup Verification

The registered command was verified directly using:

- `C:\Program Files\Python314\python.exe D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\launch_repo_mcp.py`

Observed startup evidence:

- process remained alive after 3 seconds under stdio waiting state
- stderr logged:
  - `INFO:mcp_server.server:Starting read-only repository MCP server over STDIO`

That confirms the repaired command starts without the previous import failure.

## Tool/Resource Exposure Verification

A local Python MCP client was then used against the same registered command outside the sandbox. Verification succeeded.

Observed MCP handshake evidence:

- initialized server name:
  - `rl-agent-extraction-plganps-repo`
- exposed resources:
  - `repo://active_data_source_authority`
  - `repo://active_pipeline_authority`
  - `repo://maintained_entrypoint_registry`
  - `repo://naming_and_run_layout_contract`
  - `repo://stage2_contract_and_frozen_substeps`
- exposed tools:
  - `check_lawful_resume_boundary`
  - `check_script_selection_legality`
  - `get_authoritative_gt`
  - `list_maintained_entrypoints_for_stage`
  - `resolve_active_run_source`
  - `validate_new_file_path`

The verification also completed a real governed tool call:

- `resolve_active_run_source`
- result: `ok: true`
- resolved run: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`

## Remaining Blockers, If Any

There is no remaining server-start or tool-exposure blocker for the repaired `repo-mcp` command itself.

The only remaining limitation observed is session-local visibility in this already-running Codex thread:

- `list_mcp_resources(server="repo-mcp")` still returned `unknown MCP server 'repo-mcp'`

This is best explained by the current thread MCP registry not hot-reloading after `C:\Users\tiancong\.codex\config.toml` was changed. It does not contradict the successful external MCP client handshake against the same command.

## Final Status Classification

`connected_and_verified`

Machine-local success criteria now met:

- explicit client-side `repo-mcp` registration exists
- required MCP SDK imports succeed
- the repo-local server starts successfully
- governed repo tools/resources are exposed over MCP

Session-local note:

- the current Codex thread likely needs a restart or fresh session to pick up the new shared registration automatically
