# MCP Baseline Audit (2026-04-15)

## Executive Summary

This machine shows evidence of a Codex Desktop client context, and this repository contains an intended repo-local MCP server surface under `mcp_server/`, but the intended repository-governance MCP system is not shown as actually connected on this computer today. The strongest local pattern is: repo-side MCP design exists, machine-side registration/launch wiring is missing, and the local Python environment cannot currently import the MCP SDK needed by the repo-local server. Separately, the repository is partially ready for MCP-managed baselines because it already has a machine-readable active-run authority pointer, governed frozen artifact roots, `RUN_CONTEXT.md` lineage tracking, and lawful replay-boundary governance; however, it does not yet show a first-class machine-readable baseline registry and baseline manifest contract that would let MCP manage baselines as governed objects rather than as run/freeze conventions.

## Machine MCP Connection Status

### Classification

`repo_has_design_but_machine_not_connected`

### Strongest local evidence

- Repo-local MCP server code exists at `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\server.py`.
- Repo-local MCP setup instructions exist at `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\README_MCP.md`.
- Local Codex config visible at `C:\Users\tiancong\.codex\config.toml` does not register any MCP servers.
- Local VS Code settings visible at `C:\Users\tiancong\AppData\Roaming\Code\User\settings.json` do not register any MCP servers.
- The repo-local server is not currently importable from the local Python environment because `mcp_server.server` fails on `ModuleNotFoundError: No module named 'mcp'`.
- The repo-governance server name `repo-mcp` is not currently available to the local MCP resource tool surface in this session.

### Missing piece

The missing layer is machine-local MCP wiring for the intended repo server: a registered client-side server entry for `repo-mcp`, plus a locally runnable server environment that can import the MCP SDK and start successfully.

### Confidence

`high`

### What MCP client appears to be in use, if any?

The visible client is `Codex Desktop`, supported by process environment variables such as `CODEX_INTERNAL_ORIGINATOR_OVERRIDE=Codex Desktop` and by `C:\Users\tiancong\.codex\.codex-global-state.json`.

### What servers are configured, if any?

No repo-local MCP server registration was found in the inspected machine-local config surfaces:

- `C:\Users\tiancong\.codex\config.toml`
- `C:\Users\tiancong\AppData\Roaming\Code\User\settings.json`
- `C:\Users\tiancong\AppData\Roaming\Code\User\mcp.json` was not present
- no workspace `.vscode` directory exists in this repo

The repository does contain a candidate server implementation under `mcp_server/`, but that is design/code presence, not machine-local registration.

### Which parts are repo-local vs machine-local?

Repo-local:

- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\server.py`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\README_MCP.md`
- governance instructions in `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\AGENTS.md` that expect `repo-mcp` when available

Machine-local:

- `C:\Users\tiancong\.codex\config.toml`
- `C:\Users\tiancong\.codex\.codex-global-state.json`
- `C:\Users\tiancong\AppData\Roaming\Code\User\settings.json`
- `C:\Users\tiancong\.vscode\argv.json`
- current process environment variables

### Is there evidence this repo was designed to be governed by MCP, or only by docs/prompt discipline?

There is evidence of both, but MCP governance is only partially realized. `AGENTS.md` explicitly instructs Codex to use the repo-local MCP server `repo-mcp` when available for source resolution, script legality, GT authority, file-path validation, and lawful resume-boundary checks. That is stronger than prompt discipline alone. However, the machine-local connection layer needed to make that governance executable was not found on this computer.

## Repo MCP-Readiness Findings

### Repo-side MCP design exists

The repo contains a concrete read-only MCP server skeleton rather than only aspirational discussion:

- `mcp_server/server.py` defines repository resources and governance tools.
- `mcp_server/README_MCP.md` describes local stdio startup via `python -m mcp_server.server`.
- `AGENTS.md` expects agents to call specific repo-governance MCP tools when the server is available.

### Repo-side MCP design is not yet machine-usable here

The current local environment does not show that this server is presently usable on this machine:

- no visible registration for `repo-mcp`
- no visible startup wrapper or launch config binding the repo server into Codex or VS Code
- local Python import failure for the `mcp` package required by `mcp_server.server`

## Baseline Integration Readiness Findings

### Readiness classification

`partially_ready_contract_exists_but_registry_missing`

### What already exists

The repository already has several contract pieces that help baseline management:

- A machine-readable authority pointer at `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\data\results\ACTIVE_RUN.json`
- Governed run reproducibility through `RUN_CONTEXT.md`
- Frozen artifact entrypoints under `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\data\frozen\`
- Explicit lawful replay-boundary governance in `README.md`, `project/2_ARCHITECTURE.md`, and `project/ACTIVE_PIPELINE_RUNBOOK.md`
- Feature activation and governed intervention metadata via `project/feature_units/feature_unit_registry.json`, `project/feature_units/feature_intervention_matrix.tsv`, and `feature_activation_report_v1.tsv` surfaces
- A read-oriented utility for baseline audits at `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\utils\daily_baseline_audit_v1.py`

### What is not yet explicit enough for MCP-managed baselines

The checked repo does not show a first-class machine-readable baseline object model with all of the following in one governed contract surface:

- explicit baseline registry
- explicit baseline manifest contract
- stable baseline identity mapping for all governed baselines
- machine-readable mapping from baseline ID to artifact chain
- machine-readable mapping from baseline ID to run lineage

`ACTIVE_RUN.json` is a machine-readable authority pointer for the currently active run lineage, not a general baseline registry. `docs/archive_project/ARCHIVED_BASELINES.md` records historical baselines, but it is narrative documentation and Git-reference guidance, not a machine-readable governed registry. `src/utils/daily_baseline_audit_v1.py` can snapshot and compare explicit baselines, but its own module docstring says it is additive and read-oriented and does not define a hidden orchestration layer.

### Specific inspection against the requested readiness checklist

- Explicit machine-readable baseline registry: not found
- Explicit baseline manifest contract: not found as a general baseline contract
- Stable baseline identity naming: partially present
  - examples exist such as `dev15_full_pipeline_freeze_v1` and `baseline_pre_tablefirst`
  - but no single governed machine-readable baseline registry was found
- Baseline root entrypoint: partially present
  - frozen roots under `data/frozen/`
  - active authority pointer under `data/results/ACTIVE_RUN.json`
- Machine-readable mapping from baseline -> artifact chain: not found as a generalized registry contract
- Machine-readable mapping from baseline -> run lineage: partially present through run-local `RUN_CONTEXT.md` and freeze-local manifests, but not via a central baseline registry object
- Boundary coverage metadata sufficient for MCP enforcement: partially present
  - boundary classes and feature activation exist
  - enforcement would still need a baseline registry/object layer to bind those checks to a named baseline

## Gap Classification

### Machine connection gap

The intended repo MCP server is not shown as connected on this machine right now.

### MCP client config gap

No local config registration for `repo-mcp` was found in Codex or VS Code surfaces inspected for this machine.

### MCP server availability gap

The repo-local server code exists, but the local Python environment does not currently import the required MCP package, so the server is not shown as runnable here yet.

### Repo contract gap

Repo governance is strong for run authority, resume boundaries, and frozen artifacts, but it does not yet expose a single machine-readable baseline registry/manifest contract.

### Baseline object model gap

Baseline is currently represented as a mix of active-run pointers, frozen directories, run contexts, audit utilities, and historical docs rather than a first-class governed baseline object with stable machine-readable fields.

### Registry/index gap

No central machine-readable index was found that maps baseline identity to its authority root, artifact chain, lineage chain, benchmark-validity state, and lawful replay coverage.

### Automation gap

No machine-visible MCP automation or startup wiring was found that would automatically register and start the repo-local server for this repository on this computer.

## Smallest Safe Next Step

The minimum safe next step is to establish machine-local connectivity for the existing repo-local MCP server before attempting any baseline-management redesign: define one explicit client-side registration for `repo-mcp` on this machine, point it at `python -m mcp_server.server`, and verify that the local environment can import the required MCP package and expose the governed tools. Only after that connection layer is real should the baseline object-model gap be addressed.

## Evidence Appendix With Exact Local File Paths

- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\AGENTS.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\README.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\project\2_ARCHITECTURE.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\project\ACTIVE_PIPELINE_FLOW.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\project\ACTIVE_PIPELINE_RUNBOOK.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\project\4_DECISIONS_LOG.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\README_MCP.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\mcp_server\server.py`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\data\results\ACTIVE_RUN.json`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\data\frozen\dev15_full_pipeline_freeze_v1\RUN_CONTEXT.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\data\frozen\dev15_stage2_freeze_v1\FREEZE_MANIFEST.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\src\utils\daily_baseline_audit_v1.py`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\docs\archive_project\ARCHIVED_BASELINES.md`
- `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs\project\feature_units\feature_unit_registry.json`
- `C:\Users\tiancong\.codex\config.toml`
- `C:\Users\tiancong\.codex\.codex-global-state.json`
- `C:\Users\tiancong\AppData\Roaming\Code\User\settings.json`
- `C:\Users\tiancong\AppData\Roaming\Code\User\mcp.json` not present
- `C:\Users\tiancong\.vscode\argv.json`

## What must be true before baseline management can be safely integrated into MCP?

- The intended repo-local MCP server must be actually registered and callable on this machine.
- The server runtime must successfully import the MCP SDK and expose the governed repository tools.
- A governed machine-readable baseline registry must exist, not just run folders and narrative docs.
- Each baseline entry must bind a stable baseline ID to an authority root, artifact chain, lineage chain, benchmark-validity state, and lawful replay boundary coverage.
- MCP tools must read that registry rather than infer baseline identity from directory names, recency, or free-text documentation.
