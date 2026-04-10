from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import ACTIVE_RUN_POINTER_FILE, DATA_RESULTS_DIR, PROJECT_ROOT
from src.utils.run_id import classify_results_path, is_valid_legacy_run_id


REQUIRED_POINTER_KEYS = [
    "active_run_id",
    "active_run_dir",
    "authoritative_terminal_files",
    "lineage_policy",
    "updated_at",
    "note",
]

GLOBAL_AUTHORITY_POINTER_KEYS = {
    "layer1_gt_path",
    "layer2_gt_path",
    "layer3_gt_path",
    "gt_skeleton_tsv",
    "alignment_scaffold_tsv",
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def gt_authority_lock_enabled(pointer_payload: dict[str, Any] | None) -> bool:
    if not pointer_payload:
        return False
    return bool(pointer_payload.get("gt_authority_lock"))


def resolve_pointer_artifact_path(
    *,
    run_context: dict[str, Any],
    pointer_key: str,
    required: bool = True,
) -> Path | None:
    pointer_payload = run_context.get("pointer_payload") or {}
    pointer_files = pointer_payload.get("authoritative_terminal_files") or {}
    pointer_value = _normalize_text(pointer_files.get(pointer_key))
    if not pointer_value:
        if required:
            raise FileNotFoundError(
                f"ACTIVE_RUN.json does not define authoritative_terminal_files[{pointer_key!r}]."
            )
        return None
    resolved = _resolve_repo_path(pointer_value)
    if required and not resolved.exists():
        raise FileNotFoundError(
            f"ACTIVE_RUN.json pointer target for {pointer_key!r} not found: {resolved}"
        )
    return resolved


def enforce_gt_authority_lock_for_explicit_path(
    *,
    explicit_path: Path | None,
    run_context: dict[str, Any],
    pointer_key: str,
) -> None:
    if explicit_path is None:
        return
    pointer_payload = run_context.get("pointer_payload") or {}
    if not pointer_payload:
        try:
            pointer_payload, _ = read_active_run_pointer()
        except Exception:
            pointer_payload = {}
    if not gt_authority_lock_enabled(pointer_payload):
        return
    pointer_files = pointer_payload.get("authoritative_terminal_files") or {}
    pointer_value = _normalize_text(pointer_files.get(pointer_key))
    if not pointer_value:
        raise FileNotFoundError(
            f"GT authority lock is enabled, but ACTIVE_RUN.json does not define {pointer_key!r}."
        )
    contracted_path = _resolve_repo_path(pointer_value)
    if not contracted_path.exists():
        raise FileNotFoundError(
            f"GT authority lock contracted path for {pointer_key!r} not found: {contracted_path}"
        )
    resolved_explicit = explicit_path.resolve()
    if resolved_explicit != contracted_path:
        raise ValueError(
            "GT authority lock violation: explicit path does not match the contracted "
            f"{pointer_key!r} path. explicit={resolved_explicit} contracted={contracted_path}"
        )


def read_active_run_pointer(pointer_path: Path | None = None) -> tuple[dict[str, Any], Path]:
    path = (pointer_path or ACTIVE_RUN_POINTER_FILE).resolve()
    if not path.exists():
        raise FileNotFoundError(
            "Active data-source pointer not found. Provide --run-dir explicitly or create "
            f"{path}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_POINTER_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Active data-source pointer missing required keys: {missing}")
    if not isinstance(payload.get("authoritative_terminal_files"), dict):
        raise ValueError("ACTIVE_RUN.json field 'authoritative_terminal_files' must be an object.")
    active_run_id = _normalize_text(payload.get("active_run_id"))
    if not active_run_id:
        raise ValueError("ACTIVE_RUN.json contains empty active_run_id.")
    active_run_dir = _resolve_repo_path(_normalize_text(payload.get("active_run_dir")))
    if not active_run_dir.exists():
        raise FileNotFoundError(
            f"ACTIVE_RUN.json active_run_dir does not exist: {active_run_dir}"
        )
    if not active_run_dir.is_dir():
        raise NotADirectoryError(
            f"ACTIVE_RUN.json active_run_dir is not a directory: {active_run_dir}"
        )
    path_info = classify_results_path(active_run_dir, results_dir=DATA_RESULTS_DIR)
    payload["_resolved_pointer_path"] = str(path)
    payload["_resolved_active_run_dir"] = str(active_run_dir)
    payload["_resolved_active_run_dir_kind"] = path_info["path_kind"]
    return payload, path


def resolve_run_context(
    *,
    explicit_run_dir: Path | None = None,
    explicit_run_id: str = "",
    pointer_path: Path | None = None,
) -> dict[str, Any]:
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Explicit --run-dir not found: {run_dir}")
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Explicit --run-dir is not a directory: {run_dir}")
        path_info = classify_results_path(run_dir, results_dir=DATA_RESULTS_DIR)
        run_id = _normalize_text(explicit_run_id) or run_dir.name
        if explicit_run_id and explicit_run_id != run_dir.name and is_valid_legacy_run_id(explicit_run_id):
            raise ValueError(
                f"Explicit --run-id {explicit_run_id!r} does not match explicit --run-dir basename {run_dir.name!r}."
            )
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "run_dir_kind": path_info["path_kind"],
            "resolution_source": "explicit_run_dir",
            "pointer_payload": None,
            "pointer_path": "",
        }

    explicit_run_id = _normalize_text(explicit_run_id)
    if explicit_run_id:
        if not is_valid_legacy_run_id(explicit_run_id):
            raise ValueError(f"Invalid explicit legacy --run-id: {explicit_run_id}")
        run_dir = (DATA_RESULTS_DIR / explicit_run_id).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Explicit --run-id resolved run directory not found: {run_dir}")
        path_info = classify_results_path(run_dir, results_dir=DATA_RESULTS_DIR)
        return {
            "run_id": explicit_run_id,
            "run_dir": run_dir,
            "run_dir_kind": path_info["path_kind"],
            "resolution_source": "explicit_run_id_compat",
            "pointer_payload": None,
            "pointer_path": "",
        }

    payload, resolved_pointer_path = read_active_run_pointer(pointer_path)
    resolved_run_dir = _resolve_repo_path(_normalize_text(payload["active_run_dir"]))
    return {
        "run_id": _normalize_text(payload["active_run_id"]),
        "run_dir": resolved_run_dir,
        "run_dir_kind": _normalize_text(payload.get("_resolved_active_run_dir_kind")),
        "resolution_source": "active_run_pointer",
        "pointer_payload": payload,
        "pointer_path": str(resolved_pointer_path),
    }


def resolve_artifact_path(
    *,
    explicit_path: Path | None,
    run_context: dict[str, Any],
    pointer_key: str,
    canonical_relative: str | None = None,
    preferred_run_local_names: list[str] | None = None,
    required: bool = True,
) -> Path | None:
    if explicit_path is not None:
        enforce_gt_authority_lock_for_explicit_path(
            explicit_path=explicit_path,
            run_context=run_context,
            pointer_key=pointer_key,
        )
        resolved = explicit_path.resolve()
        if required and not resolved.exists():
            raise FileNotFoundError(f"Explicit artifact path not found: {resolved}")
        return resolved

    resolved = resolve_pointer_artifact_path(
        run_context=run_context,
        pointer_key=pointer_key,
        required=False,
    )
    if resolved is None and pointer_key in GLOBAL_AUTHORITY_POINTER_KEYS:
        try:
            active_payload, _ = read_active_run_pointer()
        except Exception:
            active_payload = {}
        pointer_files = active_payload.get("authoritative_terminal_files") or {}
        pointer_value = _normalize_text(pointer_files.get(pointer_key))
        if pointer_value:
            resolved = _resolve_repo_path(pointer_value)
            if required and not resolved.exists():
                raise FileNotFoundError(
                    f"ACTIVE_RUN.json pointer target for {pointer_key!r} not found: {resolved}"
                )
    if resolved is not None:
        return resolved

    run_dir = Path(run_context["run_dir"])
    if canonical_relative:
        candidate = (run_dir / canonical_relative).resolve()
        if candidate.exists() or not required:
            return candidate
    if preferred_run_local_names:
        for name in preferred_run_local_names:
            candidate = (run_dir / name).resolve()
            if candidate.exists():
                return candidate

    if required:
        raise FileNotFoundError(
            f"Could not resolve required artifact {pointer_key!r}. Provide an explicit CLI path "
            "or add the authoritative path to data/results/ACTIVE_RUN.json."
        )
    return None


def build_artifact_metadata(
    *,
    source_run_context: dict[str, Any],
    source_files: dict[str, str],
    generated_by: str,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_run_dir": str(source_run_context["run_dir"]),
        "source_run_id": str(source_run_context["run_id"]),
        "source_run_dir_kind": str(source_run_context.get("run_dir_kind") or ""),
        "source_resolution": str(source_run_context["resolution_source"]),
        "active_run_pointer_path": str(source_run_context.get("pointer_path") or ""),
        "source_files": source_files,
        "generated_by": generated_by,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    return payload


def write_artifact_metadata_json(artifact_path: Path, metadata: dict[str, Any]) -> Path:
    target = artifact_path.with_name(artifact_path.name + ".metadata.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return target
