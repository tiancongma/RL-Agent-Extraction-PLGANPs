#!/usr/bin/env python3
"""Build diagnostic S2-5b semantic-signal to S2-2 authority reattachment sidecars.

This helper is diagnostic-only. It does not create semantic authorization, does
not call an LLM, and does not emit completed Stage2 rows. It binds existing
Stage2 semantic table signals to preserved S2-2 full-table authority records so
S2-6/S2-7 can later validate/consume explicit locators.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SIDECAR_REL = "semantic_stage2_objects/authority_reattachment"
AUDIT_REL = "analysis/semantic_authority_reattachment_audit_v1.tsv"
SIDECAR_NAME = "semantic_authority_reattachment_v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_alias_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    # Treat file paths as table identities, not as opaque path strings.
    text = text.replace("\\", "/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"\.(csv|tsv|json|jsonl)$", "", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _add_alias(aliases: set[str], value: Any) -> None:
    token = normalize_alias_token(value)
    if token:
        aliases.add(token)


@dataclass(frozen=True)
class PayloadRecord:
    payload: dict[str, Any]
    aliases: frozenset[str]
    payload_artifact_path: str

    def selected_record(self) -> dict[str, Any]:
        fields = [
            "table_id",
            "source_table_id",
            "source_table_asset_id",
            "source_table_reference",
            "source_csv_path",
            "normalized_csv_path",
            "source_caption_or_title",
            "table_type",
            "selector_readiness_label",
            "representation_status",
            "authority_rank",
            "authority_score",
            "authority_tier",
            "table_inclusion_class",
            "row_count",
            "normalized_row_count",
            "raw_row_count",
        ]
        out = {name: self.payload.get(name, "") for name in fields if name in self.payload}
        out["payload_artifact_path"] = self.payload_artifact_path
        out["aliases"] = sorted(self.aliases)
        return out


def payload_aliases(payload: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for field in (
        "table_id",
        "source_table_id",
        "source_table_asset_id",
        "source_table_reference",
        "source_csv_path",
        "source_filename",
        "source_caption_or_title",
        "normalized_csv_path",
        "file_derived_table_id",
    ):
        _add_alias(aliases, payload.get(field))
    for field in ("table_identity_aliases", "identity_aliases", "authority_aliases"):
        values = payload.get(field)
        if isinstance(values, list):
            for value in values:
                _add_alias(aliases, value)
    locators = payload.get("table_scope_locators")
    if isinstance(locators, dict):
        for value in locators.values():
            _add_alias(aliases, value)
    return aliases


def load_payload_records(payload_root: Path, paper_key: str) -> list[PayloadRecord]:
    artifact_path = payload_root / paper_key / "normalized_table_payloads_v1.json"
    if not artifact_path.exists():
        return []
    artifact = read_json(artifact_path)
    records: list[PayloadRecord] = []
    for payload in artifact.get("normalized_table_payloads", []):
        if isinstance(payload, dict):
            records.append(
                PayloadRecord(
                    payload=payload,
                    aliases=frozenset(payload_aliases(payload)),
                    payload_artifact_path=str(artifact_path),
                )
            )
    return records


def extract_semantic_table_signals(document: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for family in ("table_formulation_scopes", "table_scopes"):
        values = document.get(family) or []
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            # Bind existing semantic table/scope signals only. Do not infer new
            # formulation-bearing status from payload text.
            is_bearing = bool(
                item.get("is_formulation_table")
                or item.get("is_formulation_bearing")
                or normalize_alias_token(item.get("scope_kind")) == "formulation table"
            )
            if not is_bearing:
                continue
            scope_id = str(item.get("scope_id") or f"{family}_{index:02d}")
            table_id = str(item.get("table_id") or item.get("source_table_id") or "")
            key = (family, scope_id, table_id)
            if key in seen:
                continue
            seen.add(key)
            signals.append({"signal_family": family, "signal_index": index, "scope_id": scope_id, "signal": item})
    return signals


def signal_aliases(signal: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for _, values in signal_alias_groups(signal):
        aliases.update(values)
    return aliases


def signal_alias_groups(signal: dict[str, Any]) -> list[tuple[str, set[str]]]:
    """Return target aliases ordered from strongest locator to broad label.

    A semantic table signal may carry both a broad logical label (for example
    `Table 1`) and an execution-side locator (`source_table_asset_id` or
    `table_scope_locators.source_table_reference`) that points to the repaired
    S2-2 authority table.  Reattachment must prefer exact locators over broad
    labels.  Relationship hints such as `parent_table_hint` are intentionally
    not target aliases here; they remain preserved inside the semantic signal
    but require explicit relation-aware logic before they can target authority.
    """
    item = signal.get("signal") if isinstance(signal.get("signal"), dict) else signal
    if not isinstance(item, dict):
        return []

    def values_for(*fields: str) -> set[str]:
        out: set[str] = set()
        for field in fields:
            _add_alias(out, item.get(field))
        return out

    locator_refs: set[str] = set()
    locator_assets: set[str] = set()
    locator_labels: set[str] = set()
    locators = item.get("table_scope_locators")
    if isinstance(locators, dict):
        for field in ("source_table_reference", "source_csv_path", "normalized_csv_path"):
            _add_alias(locator_refs, locators.get(field))
        for field in ("source_table_asset_id", "source_filename"):
            _add_alias(locator_assets, locators.get(field))
        for field in ("table_id", "source_table_id"):
            _add_alias(locator_labels, locators.get(field))

    groups = [
        ("exact_source_reference", values_for("source_table_reference", "source_csv_path")),
        ("exact_source_asset", values_for("source_table_asset_id", "source_filename")),
        ("locator_source_reference", locator_refs),
        ("locator_source_asset", locator_assets),
        ("caption_or_title", values_for("caption", "title")),
        ("locator_table_label", locator_labels),
        ("logical_table_label", values_for("table_id", "source_table_id")),
    ]
    return [(name, values) for name, values in groups if values]


def _rank_key(record: PayloadRecord) -> tuple[float, float]:
    rank = record.payload.get("authority_rank")
    try:
        rank_value = float(rank)
    except (TypeError, ValueError):
        rank_value = 1_000_000.0
    score = record.payload.get("authority_score")
    try:
        score_value = -float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    return (rank_value, score_value)


def resolve_signal(signal: dict[str, Any], records: list[PayloadRecord]) -> tuple[str, PayloadRecord | None, list[PayloadRecord], set[str]]:
    alias_groups = signal_alias_groups(signal)
    aliases: set[str] = set()
    for _, values in alias_groups:
        aliases.update(values)
    if not aliases:
        return "unresolved", None, [], aliases

    strongest_name = ""
    matches: list[PayloadRecord] = []
    for group_name, group_aliases in alias_groups:
        group_matches = [record for record in records if group_aliases.intersection(record.aliases)]
        if group_matches:
            strongest_name = group_name
            matches = group_matches
            break
    if not matches:
        return "unresolved", None, [], aliases

    sorted_matches = sorted(matches, key=_rank_key)
    source_refs = {normalize_alias_token(m.payload.get("source_table_reference") or m.payload.get("source_csv_path")) for m in sorted_matches}
    source_refs.discard("")
    # Exact source locator/asset matches are allowed to disambiguate away from a
    # broad logical `Table N` label. If several payload rows share one source,
    # choose the best authority-ranked record within that source.
    if strongest_name in {"exact_source_reference", "exact_source_asset", "locator_source_reference", "locator_source_asset"}:
        if len(source_refs) <= 1:
            return "resolved", sorted_matches[0], sorted_matches, aliases
    if len(sorted_matches) > 1 and len(source_refs) > 1:
        return "ambiguous", None, sorted_matches, aliases
    return "resolved", sorted_matches[0], sorted_matches, aliases


def build_reattachment_for_document(
    document: dict[str, Any],
    payload_root: Path,
    grid_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paper_key = str(document.get("paper_key") or document.get("document_key") or "")
    records = load_payload_records(payload_root, paper_key)
    signals = extract_semantic_table_signals(document)
    reattachments: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for signal in signals:
        status, selected, candidates, aliases = resolve_signal(signal, records)
        selected_record = selected.selected_record() if selected else {}
        candidate_records = [candidate.selected_record() for candidate in candidates[:10]]
        item = {
            "paper_key": paper_key,
            "scope_id": signal.get("scope_id", ""),
            "signal_family": signal.get("signal_family", ""),
            "signal_index": signal.get("signal_index", ""),
            "semantic_signal": signal.get("signal", {}),
            "signal_aliases": sorted(aliases),
            "resolution_status": status,
            "selected_authority_record": selected_record,
            "candidate_authority_records": candidate_records,
            "ambiguity_count": len(candidates) if status == "ambiguous" else 0,
            "unresolved_reason": "no_matching_s2_2_authority_alias" if status == "unresolved" else "",
            "grid_path": str(grid_path) if grid_path else "",
            "notes": "diagnostic-only; existing semantic signal only; no semantic authorization invented",
        }
        reattachments.append(item)
        audit_rows.append({
            "paper_key": paper_key,
            "scope_id": item["scope_id"],
            "signal_family": item["signal_family"],
            "semantic_table_id": item["semantic_signal"].get("table_id", "") if isinstance(item["semantic_signal"], dict) else "",
            "resolution_status": status,
            "selected_table_id": selected_record.get("table_id", ""),
            "selected_source_table_reference": selected_record.get("source_table_reference", ""),
            "selected_payload_artifact_path": selected_record.get("payload_artifact_path", ""),
            "selected_normalized_csv_path": selected_record.get("normalized_csv_path", ""),
            "candidate_match_count": len(candidates),
            "signal_aliases": ";".join(sorted(aliases)),
            "grid_path": str(grid_path) if grid_path else "",
            "benchmark_valid": "no",
            "diagnostic_only": "yes",
        })
    summary = {
        "semantic_signal_count": len(signals),
        "payload_record_count": len(records),
        "resolved_signal_count": sum(1 for item in reattachments if item["resolution_status"] == "resolved"),
        "ambiguous_signal_count": sum(1 for item in reattachments if item["resolution_status"] == "ambiguous"),
        "unresolved_signal_count": sum(1 for item in reattachments if item["resolution_status"] == "unresolved"),
    }
    sidecar = {
        "contract_version": "s2_5b_semantic_authority_reattachment_v1",
        "diagnostic_only": True,
        "benchmark_valid": False,
        "paper_key": paper_key,
        "source_semantic_document_key": document.get("document_key") or paper_key,
        "payload_root": str(payload_root),
        "grid_path": str(grid_path) if grid_path else "",
        "authority_rule": "bind existing S2-5 semantic table signals to preserved S2-2 full-table authority; do not invent semantic scope",
        "summary": summary,
        "reattachments": reattachments,
    }
    return sidecar, audit_rows


def iter_semantic_documents(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "paper_key",
        "scope_id",
        "signal_family",
        "semantic_table_id",
        "resolution_status",
        "selected_table_id",
        "selected_source_table_reference",
        "selected_payload_artifact_path",
        "selected_normalized_csv_path",
        "candidate_match_count",
        "signal_aliases",
        "grid_path",
        "benchmark_valid",
        "diagnostic_only",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_run_context(out_dir: Path, args: argparse.Namespace, audit_rows: list[dict[str, Any]]) -> None:
    context = f"""# RUN_CONTEXT\n\nrun_purpose: Diagnostic-only S2-5b semantic signal to S2-2 authority reattachment sidecar generation.\nrun_type: diagnostic-only, not benchmark-valid.\nbenchmark_valid: no\ngenerated_at: {utc_now()}\ngenerated_by: src/stage2_sampling_labels/build_semantic_authority_reattachment_v1.py\n\n## Exact inputs\n\n- semantic_jsonl: {args.semantic_jsonl}\n- payload_root: {args.payload_root}\n- table_cell_grid: {args.table_cell_grid or ''}\n\n## Exact outputs\n\n- sidecar_root: {out_dir / SIDECAR_REL}\n- audit_tsv: {out_dir / AUDIT_REL}\n\n## Summary\n\n- semantic_table_signals: {len(audit_rows)}\n- resolved: {sum(1 for row in audit_rows if row.get('resolution_status') == 'resolved')}\n- ambiguous: {sum(1 for row in audit_rows if row.get('resolution_status') == 'ambiguous')}\n- unresolved: {sum(1 for row in audit_rows if row.get('resolution_status') == 'unresolved')}\n\nNo live LLM call was made. This helper does not create Stage3 inputs, Stage5 inputs, or benchmark-valid outputs.\n"""
    (out_dir / "RUN_CONTEXT.md").write_text(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-jsonl", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--table-cell-grid", type=Path, default=None)
    parser.add_argument("--paper-key", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_keys = {str(key) for key in args.paper_key if str(key).strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    paper_count = 0
    for document in iter_semantic_documents(args.semantic_jsonl):
        paper_key = str(document.get("paper_key") or document.get("document_key") or "")
        if selected_keys and paper_key not in selected_keys:
            continue
        paper_count += 1
        sidecar, rows = build_reattachment_for_document(document, args.payload_root, args.table_cell_grid)
        write_json(args.out_dir / SIDECAR_REL / paper_key / SIDECAR_NAME, sidecar)
        all_rows.extend(rows)
    write_tsv(args.out_dir / AUDIT_REL, all_rows)
    write_run_context(args.out_dir, args, all_rows)
    print(f"diagnostic_only=yes benchmark_valid=no")
    print(f"semantic_jsonl={args.semantic_jsonl}")
    print(f"payload_root={args.payload_root}")
    print(f"table_cell_grid={args.table_cell_grid or ''}")
    print(f"out_dir={args.out_dir}")
    print(f"papers={paper_count} signals={len(all_rows)} resolved={sum(1 for r in all_rows if r.get('resolution_status') == 'resolved')} ambiguous={sum(1 for r in all_rows if r.get('resolution_status') == 'ambiguous')} unresolved={sum(1 for r in all_rows if r.get('resolution_status') == 'unresolved')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
