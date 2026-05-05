#!/usr/bin/env python3
from __future__ import annotations

"""
Resolve the authority surface required before building Stage5 evidence binding packs.

This is a Phase-0 gate. It reads ACTIVE_RUN.json-style pointer data, checks all
known aliases for each semantic artifact, and fails on alias conflicts unless an
explicit semantic_name=pointer_key override is provided. It does not build packs,
create rows, create values, or inspect evidence content.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from src.utils.paths import ACTIVE_RUN_POINTER_FILE, PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import ACTIVE_RUN_POINTER_FILE, PROJECT_ROOT


class AuthorityResolutionError(RuntimeError):
    """Raised when the active authority surface is ambiguous or missing."""


@dataclass(frozen=True)
class SemanticArtifactSpec:
    semantic_name: str
    canonical_key: str
    aliases: tuple[str, ...]
    required: bool = True
    description: str = ""


SEMANTIC_ARTIFACT_SPECS: dict[str, SemanticArtifactSpec] = {
    "frozen_final_table": SemanticArtifactSpec(
        semantic_name="frozen_final_table",
        canonical_key="stage5_final_table_tsv",
        aliases=("stage5_final_table_tsv", "stage5_final_table", "active_final_table_tsv", "active_stage5_final_table"),
        description="Frozen Stage5 final formulation table. Evidence packs explain this table; they do not replace it.",
    ),
    "final_output_decision_trace": SemanticArtifactSpec(
        semantic_name="final_output_decision_trace",
        canonical_key="stage5_decision_trace_tsv",
        aliases=("stage5_decision_trace_tsv", "stage5_decision_trace", "active_stage5_decision_trace"),
        description="Stage5 final-output decision trace for row retention/filtering lineage.",
    ),
    "stage3_relation_records": SemanticArtifactSpec(
        semantic_name="stage3_relation_records",
        canonical_key="stage3_relation_records_tsv",
        aliases=("stage3_relation_records_tsv", "stage3_relation_records", "active_stage3_relation_records"),
        description="Stage3 relation records consumed by relation-visible value assignments.",
    ),
    "stage3_resolved_relation_fields": SemanticArtifactSpec(
        semantic_name="stage3_resolved_relation_fields",
        canonical_key="resolved_relation_fields_tsv",
        aliases=("resolved_relation_fields_tsv", "stage3_resolved_relation_fields_tsv", "active_stage3_resolved_relation_fields"),
        description="Stage3 resolved relation-backed fields.",
    ),
    "stage2_compatibility_tsv": SemanticArtifactSpec(
        semantic_name="stage2_compatibility_tsv",
        canonical_key="stage2_compatibility_tsv",
        aliases=("stage2_compatibility_tsv", "active_stage2_compatibility_tsv", "stage2_candidate_tsv"),
        description="Completed Stage2 compatibility TSV consumed by Stage3/Stage5.",
    ),
    "stage2_projection_trace": SemanticArtifactSpec(
        semantic_name="stage2_projection_trace",
        canonical_key="stage2_projection_trace_tsv",
        aliases=("stage2_projection_trace_tsv", "active_stage2_projection_trace_tsv"),
        required=False,
        description="Optional Stage2 projection trace sidecar when available.",
    ),
    "layer3_compare_cells": SemanticArtifactSpec(
        semantic_name="layer3_compare_cells",
        canonical_key="layer3_compare_cells_tsv",
        aliases=("layer3_compare_cells_tsv", "layer3_compare_cells", "active_compare_cells_tsv", "active_layer3_compare_cells"),
        required=False,
        description="Optional Layer3 compare cells for diagnostic cross-reference.",
    ),
    "layer3_risk_review_queue": SemanticArtifactSpec(
        semantic_name="layer3_risk_review_queue",
        canonical_key="layer3_risk_review_queue_tsv",
        aliases=("layer3_risk_review_queue_tsv", "layer3_risk_review_queue", "active_layer3_risk_review_queue"),
        required=False,
        description="Optional Layer3 risk queue for later risk assessment integration.",
    ),
}

DEFAULT_SPEC_ORDER = tuple(SEMANTIC_ARTIFACT_SPECS.keys())

MANIFEST_FIELDS = [
    "semantic_name",
    "selected_key",
    "selected_path",
    "status",
    "required",
    "alias_conflict",
    "alias_keys_seen",
    "alias_paths_seen",
    "description",
]


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _repo_or_pointer_relative_path(value: str, *, pointer_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (PROJECT_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (pointer_path.parent / path).resolve()


def read_pointer(pointer_path: Path) -> dict[str, Any]:
    if not pointer_path.exists():
        raise FileNotFoundError(f"ACTIVE_RUN pointer not found: {pointer_path}")
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    required = ["active_run_id", "active_run_dir", "authoritative_terminal_files", "updated_at", "note"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise AuthorityResolutionError(f"ACTIVE_RUN pointer missing required keys: {missing}")
    if not isinstance(payload.get("authoritative_terminal_files"), dict):
        raise AuthorityResolutionError("ACTIVE_RUN authoritative_terminal_files must be an object")
    return payload


def _candidate_alias_values(payload: dict[str, Any], spec: SemanticArtifactSpec, pointer_path: Path) -> dict[str, Path]:
    terminal = payload.get("authoritative_terminal_files") or {}
    candidates: dict[str, Path] = {}
    for key in spec.aliases:
        raw = terminal.get(key)
        if raw is None and key in payload:
            raw = payload.get(key)
        raw_text = _norm(raw)
        if not raw_text:
            continue
        candidates[key] = _repo_or_pointer_relative_path(raw_text, pointer_path=pointer_path)
    return candidates


def _dedup_paths(candidates: dict[str, Path]) -> dict[str, list[str]]:
    by_path: dict[str, list[str]] = {}
    for key, path in candidates.items():
        by_path.setdefault(str(path), []).append(key)
    return by_path


def resolve_one_artifact(
    payload: dict[str, Any],
    spec: SemanticArtifactSpec,
    *,
    pointer_path: Path,
    authority_overrides: dict[str, str],
    require_exists: bool = True,
) -> dict[str, str]:
    candidates = _candidate_alias_values(payload, spec, pointer_path)
    paths_seen = _dedup_paths(candidates)
    alias_conflict = len(paths_seen) > 1
    override_key = _norm(authority_overrides.get(spec.semantic_name))

    if override_key:
        if override_key not in candidates:
            raise AuthorityResolutionError(
                f"authority override for {spec.semantic_name!r} selects {override_key!r}, "
                f"but available aliases are {sorted(candidates)}"
            )
        selected_key = override_key
        selected_path = candidates[override_key]
        status = "resolved_with_explicit_override" if alias_conflict else "resolved"
    else:
        if alias_conflict:
            raise AuthorityResolutionError(
                "authority_alias_conflict: "
                f"semantic_name={spec.semantic_name} aliases={{{', '.join(f'{k}={v}' for k, v in candidates.items())}}}. "
                f"Provide --authority-field {spec.semantic_name}=<pointer_key>."
            )
        if spec.canonical_key in candidates:
            selected_key = spec.canonical_key
            selected_path = candidates[spec.canonical_key]
            status = "resolved"
        elif candidates:
            selected_key, selected_path = next(iter(candidates.items()))
            status = "resolved_from_alias"
        elif spec.required:
            raise AuthorityResolutionError(
                f"missing_required_authority: semantic_name={spec.semantic_name} canonical_key={spec.canonical_key}"
            )
        else:
            selected_key = ""
            selected_path = Path("")
            status = "optional_missing"

    selected_path_text = str(selected_path) if selected_key else ""
    if selected_key and require_exists and not selected_path.exists():
        raise FileNotFoundError(
            f"resolved authority path for {spec.semantic_name!r} does not exist: {selected_path}"
        )

    return {
        "semantic_name": spec.semantic_name,
        "selected_key": selected_key,
        "selected_path": selected_path_text,
        "status": status,
        "required": "yes" if spec.required else "no",
        "alias_conflict": "yes" if alias_conflict else "no",
        "alias_keys_seen": ";".join(candidates.keys()),
        "alias_paths_seen": json.dumps(paths_seen, sort_keys=True),
        "description": spec.description,
    }


def resolve_authority_manifest(
    *,
    pointer_path: Path,
    semantic_specs: Iterable[SemanticArtifactSpec] | None = None,
    authority_overrides: dict[str, str] | None = None,
    require_exists: bool = True,
) -> dict[str, Any]:
    pointer_path = pointer_path.resolve()
    payload = read_pointer(pointer_path)
    specs = list(semantic_specs or [SEMANTIC_ARTIFACT_SPECS[name] for name in DEFAULT_SPEC_ORDER])
    overrides = authority_overrides or {}
    artifacts = [
        resolve_one_artifact(
            payload,
            spec,
            pointer_path=pointer_path,
            authority_overrides=overrides,
            require_exists=require_exists,
        )
        for spec in specs
    ]
    active_run_dir_raw = _norm(payload.get("active_run_dir"))
    active_run_dir = _repo_or_pointer_relative_path(active_run_dir_raw, pointer_path=pointer_path)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/resolve_evidence_binding_authority_v1.py",
        "boundary_class": "diagnostic_authority_gate",
        "benchmark_valid": "no",
        "active_run_id": _norm(payload.get("active_run_id")),
        "active_run_dir": str(active_run_dir),
        "pointer_path": str(pointer_path),
        "artifacts": artifacts,
    }


def parse_authority_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise AuthorityResolutionError(
                f"Invalid --authority-field {value!r}; expected semantic_name=pointer_key"
            )
        semantic_name, pointer_key = value.split("=", 1)
        semantic_name = semantic_name.strip()
        pointer_key = pointer_key.strip()
        if not semantic_name or not pointer_key:
            raise AuthorityResolutionError(
                f"Invalid --authority-field {value!r}; expected semantic_name=pointer_key"
            )
        overrides[semantic_name] = pointer_key
    return overrides


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_outputs(manifest: dict[str, Any], out_dir: Path) -> None:
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(analysis_dir / "evidence_binding_authority_manifest_v1.tsv", manifest["artifacts"], MANIFEST_FIELDS)
    (analysis_dir / "evidence_binding_authority_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Resolve and lock the authority surface required before building Evidence Binding Packs.",
        "",
        "## Run type",
        "Diagnostic authority-resolution gate. Not benchmark-valid final output.",
        "",
        "## Resolved active source",
        f"- pointer_path: `{manifest['pointer_path']}`",
        f"- active_run_id: `{manifest['active_run_id']}`",
        f"- active_run_dir: `{manifest['active_run_dir']}`",
        "",
        "## Output files",
        "- `analysis/evidence_binding_authority_manifest_v1.tsv`",
        "- `analysis/evidence_binding_authority_manifest_v1.json`",
        "",
        "## Authority artifacts",
    ]
    for row in manifest["artifacts"]:
        lines.append(
            f"- {row['semantic_name']}: status={row['status']} key={row['selected_key']} path=`{row['selected_path']}` conflict={row['alias_conflict']}"
        )
    (out_dir / "RUN_CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve Evidence Binding Pack authority inputs.")
    parser.add_argument("--active-run-json", type=Path, default=ACTIVE_RUN_POINTER_FILE)
    parser.add_argument("--out-dir", type=Path, required=True, help="Run-scoped child output directory.")
    parser.add_argument(
        "--authority-field",
        action="append",
        default=[],
        help="Explicit semantic_name=pointer_key override for alias conflicts. May repeat.",
    )
    parser.add_argument(
        "--allow-missing-paths",
        action="store_true",
        help="Resolve manifest without checking file existence. Intended for contract tests only.",
    )
    args = parser.parse_args(argv)

    overrides = parse_authority_overrides(args.authority_field)
    manifest = resolve_authority_manifest(
        pointer_path=args.active_run_json,
        authority_overrides=overrides,
        require_exists=not args.allow_missing_paths,
    )
    write_manifest_outputs(manifest, args.out_dir.resolve())

    print(f"resolved_active_run_dir={manifest['active_run_dir']}")
    for row in manifest["artifacts"]:
        print(f"{row['semantic_name']}\t{row['status']}\t{row['selected_key']}\t{row['selected_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
