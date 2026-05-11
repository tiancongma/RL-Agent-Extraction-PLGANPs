#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import DOCS_DIR, PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DOCS_DIR, PROJECT_ROOT


SECTION_HEADING = "## Feature Governance Signal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-blocking run-level governance signal from the feature execution ledger."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Target run directory. May be supplied multiple times.",
    )
    parser.add_argument(
        "--schema-tsv",
        default=str(DOCS_DIR / "feature_governance" / "feature_applicability_schema_v1.tsv"),
        help="Path to the adjudicated feature applicability schema TSV.",
    )
    parser.add_argument(
        "--append-run-context",
        action="store_true",
        help="Append a compact Feature Governance Signal section to RUN_CONTEXT.md when absent.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_merged_target(notes: str) -> str | None:
    match = re.search(r"merged_into:\s*([A-Za-z0-9_]+)", notes or "")
    return match.group(1) if match else None


def is_positive_mismatch(row: dict[str, str]) -> bool:
    return str(row.get("mismatch_flag", "")).strip().lower() in {"yes", "true", "1"}


def dedupe_mismatches(
    ledger_rows: list[dict[str, str]],
    schema_by_feature: dict[str, dict[str, str]],
    tier: str,
) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for row in ledger_rows:
        if not is_positive_mismatch(row):
            continue
        feature_key = row["feature_key"]
        schema_row = schema_by_feature.get(feature_key, {})
        if schema_row.get("mismatch_severity_tier", "") != tier:
            continue
        primary_key = parse_merged_target(schema_row.get("notes", "")) or feature_key
        existing = deduped.get(primary_key)
        if existing is None:
            deduped[primary_key] = row
            continue
        # Prefer the primary row over a trace-only merged row when both appear.
        if existing["feature_key"] != primary_key and row["feature_key"] == primary_key:
            deduped[primary_key] = row
    return [deduped[key] for key in sorted(deduped)]


def build_signal_payload(
    run_dir: Path,
    ledger_rows: list[dict[str, str]],
    schema_by_feature: dict[str, dict[str, str]],
) -> dict[str, Any]:
    run_id = ledger_rows[0]["run_id"] if ledger_rows else run_dir.name
    blocking_rows = dedupe_mismatches(ledger_rows, schema_by_feature, "blocking")
    secondary_rows = dedupe_mismatches(ledger_rows, schema_by_feature, "secondary")
    observability_rows = dedupe_mismatches(ledger_rows, schema_by_feature, "observability")
    return {
        "artifact_version": "v1",
        "run_id": run_id,
        "run_dir": repo_rel(run_dir),
        "source_ledger_path": repo_rel(run_dir / "analysis" / "feature_execution_ledger_v1.tsv"),
        "source_schema_path": repo_rel(Path(DOCS_DIR / "feature_governance" / "feature_applicability_schema_v1.tsv")),
        "governance_blocking_mismatch_present": bool(blocking_rows),
        "blocking_mismatch_count": len(blocking_rows),
        "blocking_mismatch_feature_keys": [row["feature_key"] for row in blocking_rows],
        "secondary_mismatch_count": len(secondary_rows),
        "secondary_mismatch_feature_keys": [row["feature_key"] for row in secondary_rows],
        "observability_mismatch_count": len(observability_rows),
        "observability_mismatch_feature_keys": [row["feature_key"] for row in observability_rows],
        "recommended_next_step": (
            "check input contract before semantic analysis"
            if blocking_rows
            else "proceed to secondary / semantic analysis"
        ),
    }


def render_signal_markdown(payload: dict[str, Any]) -> str:
    blocking = ", ".join(f"`{key}`" for key in payload["blocking_mismatch_feature_keys"]) or "`none`"
    lines = [
        "# Feature Governance Signal v1",
        "",
        "Diagnostic-only, signal-only governance support artifact.",
        "",
        f"- `governance_blocking_mismatch_present`: `{str(payload['governance_blocking_mismatch_present']).lower()}`",
        f"- `blocking_mismatch_count`: `{payload['blocking_mismatch_count']}`",
        f"- `blocking_mismatch_feature_keys`: {blocking}",
        f"- `secondary_mismatch_count`: `{payload['secondary_mismatch_count']}`",
        f"- `observability_mismatch_count`: `{payload['observability_mismatch_count']}`",
        f"- `recommended_next_step`: `{payload['recommended_next_step']}`",
    ]
    return "\n".join(lines) + "\n"


def append_run_context_signal(run_dir: Path, payload: dict[str, Any]) -> bool:
    run_context_path = run_dir / "RUN_CONTEXT.md"
    if not run_context_path.exists():
        return False
    current_text = run_context_path.read_text(encoding="utf-8")
    if SECTION_HEADING in current_text:
        return False
    lines = [
        SECTION_HEADING,
        "",
        f"- `governance_blocking_mismatch_present`: `{str(payload['governance_blocking_mismatch_present']).lower()}`",
        f"- `blocking_mismatch_count`: `{payload['blocking_mismatch_count']}`",
        f"- `blocking_mismatch_feature_keys`: `{json.dumps(payload['blocking_mismatch_feature_keys'])}`",
    ]
    appended = current_text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    run_context_path.write_text(appended, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    schema_rows = read_tsv(Path(args.schema_tsv))
    schema_by_feature = {row["feature_key"]: row for row in schema_rows}
    for run_dir_text in args.run_dir:
        run_dir = Path(run_dir_text)
        ledger_path = run_dir / "analysis" / "feature_execution_ledger_v1.tsv"
        if not ledger_path.exists():
            raise FileNotFoundError(f"Feature execution ledger not found: {ledger_path}")
        ledger_rows = read_tsv(ledger_path)
        payload = build_signal_payload(run_dir, ledger_rows, schema_by_feature)
        json_path = run_dir / "analysis" / "feature_governance_signal_v1.json"
        md_path = run_dir / "analysis" / "feature_governance_signal_v1.md"
        write_json(json_path, payload)
        write_text(md_path, render_signal_markdown(payload))
        appended = append_run_context_signal(run_dir, payload) if args.append_run_context else False
        print(
            json.dumps(
                {
                    "run_dir": repo_rel(run_dir),
                    "signal_json": repo_rel(json_path),
                    "signal_md": repo_rel(md_path),
                    "run_context_appended": appended,
                    "blocking_mismatch_present": payload["governance_blocking_mismatch_present"],
                    "blocking_mismatch_count": payload["blocking_mismatch_count"],
                    "blocking_mismatch_feature_keys": payload["blocking_mismatch_feature_keys"],
                }
            )
        )


if __name__ == "__main__":
    main()
