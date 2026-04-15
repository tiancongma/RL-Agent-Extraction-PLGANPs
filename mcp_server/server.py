from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .loaders import (
    check_resume_boundary as load_check_resume_boundary,
    check_script_selection as load_check_script_selection,
    get_authoritative_gt as load_get_authoritative_gt,
    list_maintained_entrypoints as load_list_maintained_entrypoints,
    load_active_data_source_authority,
    load_active_pipeline_authority,
    load_maintained_entrypoint_registry_resource,
    load_naming_and_run_layout_contract,
    load_stage2_contract_and_frozen_substeps,
    resolve_active_run as load_resolve_active_run,
    validate_file_path as load_validate_file_path,
)
from .schemas import error_response, success_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="rl-agent-extraction-plganps-repo",
    instructions=(
        "Read-only repository governance server for RL-Agent-Extraction-PLGANPs. "
        "It exposes repo authority resources and validation tools only. "
        "It never executes pipeline scripts or edits repository state."
    ),
)


@mcp.resource("repo://active_pipeline_authority")
def active_pipeline_authority() -> dict:
    logger.info("Loading resource repo://active_pipeline_authority")
    return success_response(resource=load_active_pipeline_authority())


@mcp.resource("repo://active_data_source_authority")
def active_data_source_authority() -> dict:
    logger.info("Loading resource repo://active_data_source_authority")
    return success_response(resource=load_active_data_source_authority())


@mcp.resource("repo://maintained_entrypoint_registry")
def maintained_entrypoint_registry() -> dict:
    logger.info("Loading resource repo://maintained_entrypoint_registry")
    return success_response(resource=load_maintained_entrypoint_registry_resource())


@mcp.resource("repo://stage2_contract_and_frozen_substeps")
def stage2_contract_and_frozen_substeps() -> dict:
    logger.info("Loading resource repo://stage2_contract_and_frozen_substeps")
    return success_response(resource=load_stage2_contract_and_frozen_substeps())


@mcp.resource("repo://naming_and_run_layout_contract")
def naming_and_run_layout_contract() -> dict:
    logger.info("Loading resource repo://naming_and_run_layout_contract")
    return success_response(resource=load_naming_and_run_layout_contract())


@mcp.tool()
def resolve_active_run_source(run_dir: str | None = None) -> dict:
    logger.info("Tool resolve_active_run_source called")
    result = load_resolve_active_run(run_dir=run_dir)
    if result.get("resolution_mode") == "error":
        return error_response(result["error"], details=result)
    return success_response(result=result)


@mcp.tool()
def list_maintained_entrypoints_for_stage(stage_or_workflow: str) -> dict:
    logger.info("Tool list_maintained_entrypoints_for_stage called for %s", stage_or_workflow)
    return success_response(result=load_list_maintained_entrypoints(stage_or_workflow))


@mcp.tool()
def check_script_selection_legality(script_path: str, task_type: str) -> dict:
    logger.info("Tool check_script_selection_legality called for %s", script_path)
    result = load_check_script_selection(script_path=script_path, task_type=task_type)
    if not result.get("allowed"):
        return error_response("Script selection is not allowed for the requested task type.", result=result)
    return success_response(result=result)


@mcp.tool()
def validate_new_file_path(proposed_path: str, intended_role: str | None = None) -> dict:
    logger.info("Tool validate_new_file_path called for %s", proposed_path)
    result = load_validate_file_path(proposed_path=proposed_path, intended_role=intended_role)
    if not result.get("allowed"):
        return error_response("Proposed path violates repo governance.", result=result)
    return success_response(result=result)


@mcp.tool()
def get_authoritative_gt(layer: str, explicit_path: str | None = None) -> dict:
    logger.info("Tool get_authoritative_gt called for %s", layer)
    result = load_get_authoritative_gt(layer=layer, explicit_path=explicit_path)
    if not result.get("allowed"):
        return error_response("Requested GT resolution is not allowed under the current contract.", result=result)
    return success_response(result=result)


@mcp.tool()
def check_lawful_resume_boundary(boundary_name: str) -> dict:
    logger.info("Tool check_lawful_resume_boundary called for %s", boundary_name)
    result = load_check_resume_boundary(boundary_name=boundary_name)
    if not result.get("known"):
        return error_response("Boundary is unknown to the v1 MCP skeleton.", result=result)
    return success_response(result=result)


def main() -> None:
    logger.info("Starting read-only repository MCP server over STDIO")
    mcp.run()


if __name__ == "__main__":
    main()
