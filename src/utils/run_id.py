#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path


LEGACY_RUN_ID_REGEX = r"^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$"
RUN_ID_REGEX = LEGACY_RUN_ID_REGEX
LEGACY_RUN_ID_PATTERN = re.compile(
    r"^run_(?P<date>\d{8})_(?P<time>\d{4})_(?P<git_hash>[0-9a-f]{7})_(?P<suffix>.+)$"
)
V2_BUCKET_REGEX = r"^\d{8}_[0-9a-f]{7}$"
V2_BUCKET_PATTERN = re.compile(V2_BUCKET_REGEX)
V2_CHILD_REGEX = r"^\d{2,3}_[a-z0-9][a-z0-9_]*$"
V2_CHILD_PATTERN = re.compile(V2_CHILD_REGEX)
RUN_TOKEN_FRAGMENT_REGEX = r"(?:^|_)run_\d{8}_\d{4}_[0-9a-f]{7}(?:_|$)"
TIMESTAMP_HASH_FRAGMENT_REGEX = r"(?:^|_)\d{8}_\d{4}_[0-9a-f]{7}(?:_|$)"


def is_valid_legacy_run_id(s: str) -> bool:
    return bool(re.fullmatch(LEGACY_RUN_ID_REGEX, str(s or "").strip()))


def is_valid_run_id(s: str) -> bool:
    return is_valid_legacy_run_id(s)


def is_valid_v2_bucket_name(s: str) -> bool:
    return bool(V2_BUCKET_PATTERN.fullmatch(str(s or "").strip().lower()))


def is_valid_v2_child_name(s: str) -> bool:
    return bool(V2_CHILD_PATTERN.fullmatch(str(s or "").strip().lower()))


def classify_results_basename(name: str) -> str:
    value = str(name or "").strip()
    lowered = value.lower()
    if is_valid_legacy_run_id(lowered):
        return "legacy_run_root"
    if is_valid_v2_bucket_name(lowered):
        return "v2_bucket"
    if is_valid_v2_child_name(lowered):
        return "v2_child"
    return "other"


def classify_results_path(path: str | Path, *, results_dir: Path | None = None) -> dict[str, str]:
    candidate = Path(path)
    resolved = candidate.resolve(strict=False)
    results_root = results_dir.resolve() if results_dir is not None else None
    basename = resolved.name
    basename_kind = classify_results_basename(basename)
    parent = resolved.parent
    parent_name = parent.name
    parent_kind = classify_results_basename(parent_name)
    relation_to_results = ""
    path_kind = "other"

    if results_root is not None:
        if parent == results_root:
            relation_to_results = "top_level_results_child"
            if basename_kind == "legacy_run_root":
                path_kind = "legacy_run_root"
            elif basename_kind == "v2_bucket":
                path_kind = "v2_bucket_root"
            elif basename_kind == "v2_child":
                path_kind = "v2_child_top_level_invalid"
            else:
                path_kind = "results_top_level_other"
        elif parent.parent == results_root and parent_kind == "v2_bucket":
            relation_to_results = "v2_bucket_child"
            if basename_kind == "v2_child":
                path_kind = "v2_child_execution"
            else:
                path_kind = "v2_bucket_nested_other"
    if path_kind == "other":
        path_kind = basename_kind

    return {
        "path_kind": path_kind,
        "basename": basename,
        "basename_kind": basename_kind,
        "parent_name": parent_name,
        "parent_kind": parent_kind,
        "relation_to_results": relation_to_results,
        "resolved_path": str(resolved),
    }


def parse_v2_bucket_name(bucket_name: str) -> dict[str, str]:
    value = str(bucket_name or "").strip().lower()
    if not is_valid_v2_bucket_name(value):
        raise ValueError(f"Invalid v2 bucket name: {bucket_name!r}")
    date_token, git_hash = value.split("_", 1)
    return {
        "bucket_name": value,
        "date": date_token,
        "git_hash": git_hash,
    }


def parse_v2_child_name(child_name: str) -> dict[str, str]:
    value = str(child_name or "").strip().lower()
    if not is_valid_v2_child_name(value):
        raise ValueError(f"Invalid v2 child name: {child_name!r}")
    ordinal_text, cue = value.split("_", 1)
    return {
        "child_name": value,
        "ordinal_text": ordinal_text,
        "ordinal": str(int(ordinal_text)),
        "cue": cue,
    }


def validate_artifact_subdir(subdir: str, *, param_name: str = "out-subdir") -> str:
    """
    Validate a run-scoped artifact subdirectory under a governed results root.

    Contract:
    - governed identity appears only at the declared root or child execution level
    - nested artifact directories must not repeat legacy run ids, timestamp/hash
      fragments, or future bucket/child identity tokens
    - nested directories describe functional layout only
    """
    value = str(subdir or "").strip().replace("\\", "/")
    if not value:
        raise ValueError(f"{param_name} is required.")
    if Path(value).is_absolute():
        raise ValueError(f"{param_name} must be a relative path.")
    parts = [part.strip() for part in value.split("/") if part.strip()]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"{param_name} cannot contain path traversal.")
    for part in parts:
        lowered = part.lower()
        if is_valid_legacy_run_id(lowered):
            raise ValueError(
                f"{param_name} must not repeat a legacy run_id below a governed results root: {part}"
            )
        if is_valid_v2_bucket_name(lowered) or is_valid_v2_child_name(lowered):
            raise ValueError(
                f"{param_name} must not embed governed bucket/child identity tokens: {part}"
            )
        if re.search(RUN_TOKEN_FRAGMENT_REGEX, lowered) or re.search(TIMESTAMP_HASH_FRAGMENT_REGEX, lowered):
            raise ValueError(
                f"{param_name} must not embed timestamp/hash or nested run_id tokens: {part}"
            )
        if re.search(r'[<>:"|?*]', part):
            raise ValueError(f"{param_name} contains invalid path characters: {part}")
    return "/".join(parts)


def sanitize_token(s: str) -> str:
    """Normalize a token for legacy run_id or future child cue usage."""
    x = str(s).strip().lower().replace(" ", "_")
    x = re.sub(r"[^a-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def parse_legacy_run_id(run_id: str) -> dict[str, str]:
    value = str(run_id or "").strip().lower()
    match = LEGACY_RUN_ID_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid legacy run_id: {run_id!r}")
    payload = match.groupdict()
    suffix = payload["suffix"]
    subset = ""
    stage = ""
    if "_" in suffix:
        subset, stage = suffix.split("_", 1)
    else:
        subset = suffix
    payload["subset"] = subset
    payload["stage"] = stage
    return payload


def get_git_short_hash(project_root: Path | None = None) -> str:
    """Return git short hash (7 lowercase hex chars)."""
    try:
        cmd = ["git", "rev-parse", "--short=7", "HEAD"]
        out = subprocess.check_output(
            cmd,
            cwd=str(project_root) if project_root else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip().lower()
        if re.fullmatch(r"[0-9a-f]{7}", out):
            return out
    except Exception:
        pass
    raise RuntimeError("Failed to resolve git short hash (expected 7 hex chars).")


def build_run_id(
    subset: str,
    stage: str,
    version: int = 1,
    dt: datetime | None = None,
    git_hash: str | None = None,
) -> str:
    """Legacy run_id generation is disabled.

    Historical legacy ids remain parseable for reuse and audit, but new run roots
    must use the future-facing YYYYMMDD_<shortcode> format.
    """
    raise RuntimeError(
        "Legacy run_id generation is disabled. Use build_future_run_root_name() "
        "or resolve_results_write_target() for new results roots."
    )


def build_future_run_root_name(
    *,
    dt: datetime | None = None,
    git_hash: str | None = None,
) -> str:
    """Build the future-facing top-level results root name as YYYYMMDD_<short_hash>."""
    return build_v2_bucket_name(dt=dt, git_hash=git_hash)


def build_v2_bucket_name(
    *,
    dt: datetime | None = None,
    git_hash: str | None = None,
) -> str:
    """Build future bucket name as YYYYMMDD_<git_short_hash>."""
    now = dt or datetime.now()
    date_token = now.strftime("%Y%m%d")
    g = (str(git_hash).strip().lower() if git_hash is not None else get_git_short_hash()).strip()
    if not re.fullmatch(r"[0-9a-f]{7}", g):
        raise ValueError(f"Invalid git short hash for v2 bucket: {g!r}")
    bucket = f"{date_token}_{g}"
    if not is_valid_v2_bucket_name(bucket):
        raise ValueError(f"Generated invalid v2 bucket name: {bucket}")
    return bucket


def normalize_v2_cue(cue: str) -> str:
    token = sanitize_token(cue)
    if not token:
        raise ValueError("v2 child cue is empty after normalization.")
    return token


def build_v2_child_name(
    ordinal: int,
    cue: str,
    *,
    width: int | None = None,
) -> str:
    """Build future child execution folder name as NN_<cue> or NNN_<cue>."""
    ord_value = int(ordinal)
    if ord_value < 0 or ord_value > 999:
        raise ValueError(f"Invalid child ordinal for v2 child name: {ordinal!r}")
    cue_token = normalize_v2_cue(cue)
    width_value = int(width) if width is not None else (2 if ord_value < 100 else 3)
    if width_value not in (2, 3):
        raise ValueError(f"Invalid v2 child width: {width_value!r}")
    child = f"{ord_value:0{width_value}d}_{cue_token}"
    if not is_valid_v2_child_name(child):
        raise ValueError(f"Generated invalid v2 child name: {child}")
    return child


def build_v2_bucket_path(
    root_dir: str | Path,
    *,
    dt: datetime | None = None,
    git_hash: str | None = None,
    bucket_name: str = "",
) -> Path:
    root = Path(root_dir)
    resolved_root = root.resolve(strict=False)
    if str(bucket_name).strip():
        final_bucket_name = parse_v2_bucket_name(str(bucket_name).strip())["bucket_name"]
    else:
        final_bucket_name = build_v2_bucket_name(dt=dt, git_hash=git_hash)
    return (resolved_root / final_bucket_name).resolve(strict=False)


def validate_v2_bucket_dir(
    bucket_dir: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> Path:
    candidate = Path(bucket_dir).resolve(strict=False)
    parse_v2_bucket_name(candidate.name)
    if root_dir is not None:
        expected_parent = Path(root_dir).resolve(strict=False)
        if candidate.parent != expected_parent:
            raise ValueError(
                f"v2 bucket directory must live directly under {expected_parent}, got {candidate}"
            )
    return candidate


def iter_existing_v2_child_dirs(bucket_dir: str | Path) -> list[Path]:
    bucket_path = validate_v2_bucket_dir(bucket_dir)
    if not bucket_path.exists():
        return []
    if not bucket_path.is_dir():
        raise NotADirectoryError(f"v2 bucket path is not a directory: {bucket_path}")
    children: list[Path] = []
    for child in sorted(bucket_path.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and is_valid_v2_child_name(child.name):
            children.append(child)
    return children


def next_v2_child_ordinal(bucket_dir: str | Path) -> int:
    existing_children = iter_existing_v2_child_dirs(bucket_dir)
    if not existing_children:
        return 1
    ordinals = [int(parse_v2_child_name(path.name)["ordinal"]) for path in existing_children]
    return max(ordinals) + 1


def build_next_v2_child_name(bucket_dir: str | Path, cue: str) -> str:
    bucket_path = validate_v2_bucket_dir(bucket_dir)
    next_ordinal = next_v2_child_ordinal(bucket_path)
    return build_v2_child_name(ordinal=next_ordinal, cue=cue)


def build_next_v2_child_path(bucket_dir: str | Path, cue: str) -> Path:
    bucket_path = validate_v2_bucket_dir(bucket_dir)
    child_name = build_next_v2_child_name(bucket_path, cue)
    child_path = (bucket_path / child_name).resolve(strict=False)
    validate_v2_child_dir(child_path, bucket_dir=bucket_path)
    return child_path


def validate_v2_child_dir(
    child_dir: str | Path,
    *,
    bucket_dir: str | Path | None = None,
) -> Path:
    candidate = Path(child_dir).resolve(strict=False)
    parse_v2_child_name(candidate.name)
    if bucket_dir is not None:
        expected_bucket = validate_v2_bucket_dir(bucket_dir)
        if candidate.parent != expected_bucket:
            raise ValueError(
                f"v2 child directory must live directly under bucket {expected_bucket}, got {candidate}"
            )
    else:
        parse_v2_bucket_name(candidate.parent.name)
    return candidate


def is_within_directory(candidate: str | Path, root_dir: str | Path) -> bool:
    candidate_path = Path(candidate).resolve(strict=False)
    root_path = Path(root_dir).resolve(strict=False)
    try:
        candidate_path.relative_to(root_path)
        return True
    except ValueError:
        return False


def resolve_results_write_target(
    *,
    results_root: str | Path,
    default_child_cue: str,
    explicit_run_dir: str | Path | None = None,
    explicit_legacy_run_id: str = "",
    dt: datetime | None = None,
    git_hash: str | None = None,
) -> dict[str, str]:
    """
    Resolve the governed write target for a new or explicit results run surface.

    Selection order:
    1. explicit run directory
    2. explicit legacy run_id compatibility mode
    3. default future-facing v2 bucket + child allocation
    """
    results_root_path = Path(results_root).resolve(strict=False)

    if explicit_run_dir is not None and str(explicit_run_dir).strip():
        run_dir = Path(explicit_run_dir).resolve(strict=False)
        if not is_within_directory(run_dir, results_root_path):
            raise ValueError(
                f"Explicit run directory must stay under results root {results_root_path}: {run_dir}"
            )
        classification = classify_results_path(run_dir, results_dir=results_root_path)
        if classification["path_kind"] not in {"legacy_run_root", "v2_child_execution"}:
            raise ValueError(
                "Explicit run directory must be either a legacy run root or a v2 child execution path. "
                f"Got {run_dir} classified as {classification['path_kind']}."
            )
        if classification["path_kind"] == "legacy_run_root" and not run_dir.exists():
            raise ValueError(
                "Creating new legacy run roots is disabled. Use a v2 bucket/child path instead: "
                f"{run_dir}"
            )
        bucket_dir = run_dir.parent if classification["path_kind"] == "v2_child_execution" else run_dir
        return {
            "run_dir": str(run_dir),
            "run_basename": run_dir.name,
            "path_kind": classification["path_kind"],
            "selection_mode": "explicit_run_dir",
            "bucket_dir": str(bucket_dir),
        }

    legacy_run_id = str(explicit_legacy_run_id or "").strip()
    if legacy_run_id:
        raise ValueError(
            "Legacy run_id write targets are disabled for new runs. Use --run-dir "
            "for existing historical directories or let the governed v2 allocator "
            "create a compliant root."
        )

    bucket_dir = build_v2_bucket_path(results_root_path, dt=dt, git_hash=git_hash)
    child_dir = build_next_v2_child_path(bucket_dir, default_child_cue)
    return {
        "run_dir": str(child_dir),
        "run_basename": child_dir.name,
        "path_kind": "v2_child_execution",
        "selection_mode": "default_v2_child",
        "bucket_dir": str(bucket_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build governed future run identity tokens.")
    parser.add_argument("--subset", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--note", default="", help="Optional note for caller metadata.")
    parser.add_argument(
        "--format",
        choices=["v2-bucket", "v2-child", "v2-bucket-path", "v2-next-child"],
        default="v2-bucket",
        help="Identity format to generate. Legacy generation is disabled.",
    )
    parser.add_argument("--ordinal", type=int, default=0, help="Ordinal for --format v2-child.")
    parser.add_argument("--cue", default="", help="Cue token for --format v2-child.")
    parser.add_argument("--root-dir", default="", help="Root directory for --format v2-bucket-path.")
    parser.add_argument("--bucket-dir", default="", help="Bucket directory for --format v2-next-child.")
    args = parser.parse_args()

    if args.format == "v2-bucket":
        print(build_v2_bucket_name())
        return

    if args.format == "v2-child":
        print(build_v2_child_name(ordinal=args.ordinal, cue=args.cue))
        return

    if args.format == "v2-bucket-path":
        if not str(args.root_dir).strip():
            raise SystemExit("--root-dir is required for --format v2-bucket-path.")
        print(build_v2_bucket_path(args.root_dir))
        return

    if args.format == "v2-next-child":
        if not str(args.bucket_dir).strip():
            raise SystemExit("--bucket-dir is required for --format v2-next-child.")
        print(build_next_v2_child_path(args.bucket_dir, args.cue))
        return

    raise SystemExit(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    main()
