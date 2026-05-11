from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.utils import baseline_registry_v1 as baseline_registry

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

BASELINE_REGISTRY_PATH = baseline_registry.DEFAULT_REGISTRY_PATH.resolve()

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


@mcp.resource("repo://baseline_registry")
def baseline_registry_resource() -> dict:
    logger.info("Loading resource repo://baseline_registry")
    return success_response(resource=_list_baselines_result())


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


def _baseline_query_text(baseline_id: str | None = None, query: str | None = None) -> str:
    baseline_id_text = str(baseline_id or "").strip()
    query_text = str(query or "").strip()
    if baseline_id_text and query_text:
        if baseline_id_text != query_text:
            raise ValueError("Provide either baseline_id or query, not both with different values.")
        return baseline_id_text
    if baseline_id_text:
        return baseline_id_text
    if query_text:
        return query_text
    raise ValueError("A non-empty baseline_id or query is required.")


def _extract_missing_fields(error_message: str) -> list[str]:
    prefixes = (
        "is missing required fields: ",
        "is missing required top-level fields: ",
    )
    for prefix in prefixes:
        if prefix in error_message:
            suffix = error_message.split(prefix, 1)[1].strip()
            try:
                parsed = json.loads(suffix.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
    if "has blank required field:" in error_message:
        return [error_message.rsplit(":", 1)[1].strip()]
    return []


def _load_registry_rows() -> list[dict[str, str]]:
    rows = baseline_registry.read_registry(BASELINE_REGISTRY_PATH)
    seen_ids: set[str] = set()
    for row in rows:
        baseline_registry.validate_registry_row(row, BASELINE_REGISTRY_PATH)
        baseline_id = row["baseline_id"]
        if baseline_id in seen_ids:
            raise SystemExit(f"Duplicate baseline_id found in registry: {baseline_id}")
        seen_ids.add(baseline_id)
    return rows


def _resolve_baseline_row(query_text: str) -> dict[str, str]:
    rows = _load_registry_rows()
    return baseline_registry.resolve_baseline_row(rows, query_text)


def _load_resolved_baseline_object(query_text: str) -> dict[str, Any]:
    row = _resolve_baseline_row(query_text)
    manifest_path = baseline_registry.PROJECT_ROOT / Path(row["manifest_path"])
    manifest = baseline_registry.read_manifest(manifest_path)
    baseline_registry.validate_manifest(manifest, manifest_path)
    baseline_registry.validate_cross_reference(row, manifest, manifest_path)
    return baseline_registry.build_baseline_object(row, manifest, BASELINE_REGISTRY_PATH)


def _list_baselines_result() -> dict[str, Any]:
    rows = _load_registry_rows()
    return {
        "registry_path": baseline_registry.repo_rel(BASELINE_REGISTRY_PATH),
        "baselines": [
            {
                "baseline_id": row["baseline_id"],
                "baseline_type": row["baseline_type"],
                "baseline_date": row["baseline_date"],
                "stage_coverage": row["stage_coverage"],
                "benchmark_validity": row["benchmark_validity"],
                "active_status": row["active_status"],
            }
            for row in rows
        ],
    }


def _build_baseline_artifact_summary(baseline_object: dict[str, Any]) -> dict[str, Any]:
    artifact_chain = baseline_object.get("artifact_chain", [])
    stage2_completed = None
    stage3_relation = None
    stage5_final = None
    comparison_outputs: list[str] = []

    for artifact in artifact_chain:
        boundary = artifact.get("boundary")
        artifact_role = artifact.get("artifact_role", "")
        path = artifact.get("path")
        if boundary == "S2-7" and stage2_completed is None:
            stage2_completed = path
        if boundary == "Stage3" and "relation materialization" in artifact_role and stage3_relation is None:
            stage3_relation = path
        if boundary == "Stage5" and "final table" in artifact_role and stage5_final is None:
            stage5_final = path
        if boundary == "Benchmark" and path:
            comparison_outputs.append(path)

    source_artifacts = baseline_object.get("source_artifacts", {})
    for key, path in source_artifacts.items():
        if "compare" in key or "comparison" in key:
            comparison_outputs.append(path)

    return {
        "baseline_id": baseline_object["baseline_id"],
        "stage2_completed": stage2_completed,
        "stage3_relation": stage3_relation,
        "stage5_final": stage5_final,
        "comparison_outputs": comparison_outputs,
    }


@mcp.tool()
def list_baselines() -> dict:
    logger.info("Tool list_baselines called")
    try:
        return success_response(result=_list_baselines_result())
    except SystemExit as exc:
        return error_response(str(exc), result={"registry_path": baseline_registry.repo_rel(BASELINE_REGISTRY_PATH)})


@mcp.tool()
def resolve_baseline_query(query: str) -> dict:
    logger.info("Tool resolve_baseline_query called for %s", query)
    try:
        query_text = _baseline_query_text(query=query)
        row = _resolve_baseline_row(query_text)
        return success_response(
            result={
                "query": query_text,
                "baseline_id": row["baseline_id"],
                "baseline_date": row["baseline_date"],
                "baseline_type": row["baseline_type"],
                "manifest_path": row["manifest_path"],
            }
        )
    except (SystemExit, ValueError) as exc:
        return error_response(
            str(exc),
            result={
                "query": str(query or "").strip(),
                "resolution_mode": "baseline_registry_only",
                "filesystem_inference_used": False,
            },
        )


@mcp.tool()
def show_baseline(baseline_id: str | None = None, query: str | None = None) -> dict:
    logger.info("Tool show_baseline called for baseline_id=%s query=%s", baseline_id, query)
    try:
        query_text = _baseline_query_text(baseline_id=baseline_id, query=query)
        baseline_object = _load_resolved_baseline_object(query_text)
        manifest_path = baseline_registry.PROJECT_ROOT / Path(baseline_object["manifest_path"])
        manifest = baseline_registry.read_manifest(manifest_path)
        return success_response(
            result={
                "query": query_text,
                "resolved_baseline_id": baseline_object["baseline_id"],
                "authority_root": baseline_object["authority_root"],
                "artifact_chain": baseline_object["artifact_chain"],
                "lineage_chain": baseline_object["lineage_chain"],
                "limitations": manifest["limitations"],
                "manifest": manifest,
            }
        )
    except (SystemExit, ValueError) as exc:
        return error_response(
            str(exc),
            result={
                "query": str(baseline_id or query or "").strip(),
                "resolution_mode": "baseline_registry_only",
                "filesystem_inference_used": False,
            },
        )


@mcp.tool()
def validate_baseline(baseline_id: str) -> dict:
    logger.info("Tool validate_baseline called for %s", baseline_id)
    try:
        query_text = _baseline_query_text(baseline_id=baseline_id)
        row = _resolve_baseline_row(query_text)
        manifest_path = baseline_registry.PROJECT_ROOT / Path(row["manifest_path"])
        manifest = baseline_registry.read_manifest(manifest_path)
        baseline_registry.validate_registry_row(row, BASELINE_REGISTRY_PATH)
        baseline_registry.validate_manifest(manifest, manifest_path)
        baseline_registry.validate_cross_reference(row, manifest, manifest_path)
        return success_response(
            result={
                "baseline_id": row["baseline_id"],
                "validation_status": "pass",
                "missing_fields": [],
                "validated_registry": baseline_registry.repo_rel(BASELINE_REGISTRY_PATH),
                "validated_manifest": baseline_registry.repo_rel(manifest_path),
            }
        )
    except (SystemExit, ValueError) as exc:
        error_message = str(exc)
        return error_response(
            "Baseline validation failed.",
            result={
                "baseline_id": str(baseline_id or "").strip(),
                "validation_status": "fail",
                "missing_fields": _extract_missing_fields(error_message),
                "error": error_message,
            },
        )


@mcp.tool()
def get_baseline_artifacts(baseline_id: str) -> dict:
    logger.info("Tool get_baseline_artifacts called for %s", baseline_id)
    try:
        query_text = _baseline_query_text(baseline_id=baseline_id)
        baseline_object = _load_resolved_baseline_object(query_text)
        manifest_path = baseline_registry.PROJECT_ROOT / Path(baseline_object["manifest_path"])
        manifest = baseline_registry.read_manifest(manifest_path)
        baseline_object["source_artifacts"] = manifest["source_artifacts"]
        return success_response(result=_build_baseline_artifact_summary(baseline_object))
    except (SystemExit, ValueError) as exc:
        return error_response(
            str(exc),
            result={
                "baseline_id": str(baseline_id or "").strip(),
                "resolution_mode": "baseline_registry_only",
                "filesystem_inference_used": False,
            },
        )


def main() -> None:
    logger.info("Starting read-only repository MCP server over STDIO")
    mcp.run()


if __name__ == "__main__":
    main()
