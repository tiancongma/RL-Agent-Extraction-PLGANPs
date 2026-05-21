#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT

csv.field_size_limit(sys.maxsize)

OUT_TSV = "semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv"
OUT_JSONL = "semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fields})


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def stable_final_id(row: dict[str, str]) -> str:
    paper_key = row.get("paper_key", "").strip()
    raw = "|".join(
        [
            paper_key,
            row.get("canonical_formulation_id", "").strip(),
            row.get("formulation_label", "").strip(),
            row.get("preparation_evidence_quote", "").strip(),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{paper_key}__fu__{digest}"


def stable_local_id(row: dict[str, str], index: int) -> str:
    value = row.get("canonical_formulation_id", "").strip()
    if value:
        return value
    return f"FU{index:04d}"


def normalize_bool(value: str) -> str:
    return "yes" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "no"


def promote_rows(universe_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for index, row in enumerate(universe_rows, start=1):
        final_id = stable_final_id(row)
        local_id = stable_local_id(row, index)
        label = row.get("formulation_label", "").strip() or local_id
        paper_key = row.get("paper_key", "").strip()
        evidence = row.get("preparation_evidence_quote", "").strip()
        locator = row.get("source_locator", "").strip()
        promoted.append(
            {
                "final_formulation_id": final_id,
                "representative_source_formulation_id": local_id,
                "representative_source_raw_formulation_label": label,
                "source_candidate_count": "1",
                "source_candidate_ids": json.dumps([local_id], ensure_ascii=False),
                "source_candidate_labels": json.dumps([label], ensure_ascii=False),
                "source_candidate_sources": json.dumps(["274_formulation_universe_frozen_v1"], ensure_ascii=False),
                "collapsed_variant_count": "0",
                "collapsed_variant_source_ids": "[]",
                "collapsed_variant_classes": "[]",
                "retention_reason": "Retained from frozen formulation-universe row authority 274.",
                "review_needed": normalize_bool(row.get("requires_human_review", "")),
                "family_id": f"{paper_key}::{local_id}",
                "parent_core_row_id": local_id,
                "variant_role": "family_core",
                "payload_state": "",
                "benchmark_default_include": "yes",
                "collapse_signature": "",
                "loaded_state_final": "",
                "polymer_identity_final": "",
                "final_output_rule": "kept_from_frozen_formulation_universe",
                "relation_graph_ids": "[]",
                "relation_method_group_ids": "[]",
                "relation_parent_candidate_ids": "[]",
                "relation_record_count": "0",
                "field_source_type": "formulation_universe_identity_only",
                "derived_mass_provenance_json": "",
                "key": paper_key,
                "model": "deepseek-v4-flash",
                "local_instance_id": local_id,
                "formulation_id": local_id,
                "raw_formulation_label": label,
                "polymer_identity": "",
                "polymer_name_raw": "",
                "instance_kind": "single_formulation",
                "parent_instance_id": "",
                "change_descriptions": "[]",
                "change_role": "",
                "instance_context_tags": json.dumps(["synthesis_core"], ensure_ascii=False),
                "change_context_tags": "[]",
                "supporting_evidence_refs": json.dumps(
                    [
                        {
                            "source_region_type": "formulation_universe_preparation_evidence",
                            "source_locator_text": locator,
                            "supporting_snippet": evidence,
                        }
                    ],
                    ensure_ascii=False,
                ),
                "formulation_role": row.get("row_role", "").strip(),
                "instance_confidence": row.get("confidence", "").strip(),
                "candidate_source": "274_formulation_universe_frozen_v1",
                "stage2_semantic_source_mode": "llm_formulation_universe_promoted",
                "semantic_universe_authority": "274_formulation_universe_frozen_v1",
                "row_materialization_mode": "frozen_universe_identity_only",
                "semantic_scope_authority": "formulation_universe_discovery_gate",
                "semantic_scope_ref": f"{paper_key}|{local_id}",
                "table_id": locator if re.search(r"\btable\b", locator, flags=re.I) else "",
                "table_row_id": label,
                "instance_evidence_region_type": "formulation_universe_preparation_evidence",
                "evidence_section": locator,
                "evidence_span_text": evidence,
                "evidence_span_start": "",
                "evidence_span_end": "",
                "instance_kind_raw": "prepared_formulation_row",
                "instance_kind_inferred": "single_formulation",
                "instance_kind_reconciliation_note": "promoted_from_274_formulation_universe",
                "universe_paper_key": paper_key,
                "universe_canonical_formulation_id": row.get("canonical_formulation_id", "").strip(),
                "universe_formulation_label": label,
                "universe_aliases_json": row.get("aliases_json", "").strip(),
                "universe_row_role": row.get("row_role", "").strip(),
                "universe_identity_basis": row.get("identity_basis", "").strip(),
                "universe_preparation_evidence_quote": evidence,
                "universe_source_locator": locator,
                "universe_confidence": row.get("confidence", "").strip(),
                "universe_requires_human_review": normalize_bool(row.get("requires_human_review", "")),
            }
        )
    return promoted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_context(out_dir: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("universe_row_role", "") for row in rows)
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Promote frozen formulation-universe rows from run 274 into a completed Stage2-compatible diagnostic surface.",
        "",
        "## Run type",
        "Diagnostic promoted compatibility surface. Not benchmark-valid final output.",
        "",
        "## Boundary",
        "This run does not call DeepSeek and does not create rows beyond the frozen formulation-universe input. Downstream stages may fill values only for these row IDs or raise review candidates.",
        "",
        "## Inputs",
        f"- formulation_universe_tsv: `{repo_path(args.formulation_universe_tsv)}`",
        "",
        "## Outputs",
        f"- `{OUT_TSV}`",
        f"- `{OUT_JSONL}`",
        "- `analysis/formulation_universe_promotion_ledger_v1.tsv`",
        "",
        "## Counts",
        f"- promoted_rows: {len(rows)}",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- row_role.{key or '<blank>'}: {value}")
    lines.append("")
    lines.append(f"generated_at: `{datetime.now(timezone.utc).isoformat()}`")
    (out_dir / "RUN_CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote formulation-universe rows into a Stage2-compatible diagnostic surface.")
    parser.add_argument("--formulation-universe-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = repo_path(args.out_dir)
    universe_rows = read_tsv(repo_path(args.formulation_universe_tsv))
    rows = promote_rows(universe_rows)
    fields = list(rows[0].keys()) if rows else []
    write_tsv(out_dir / OUT_TSV, rows, fields)
    write_jsonl(out_dir / OUT_JSONL, rows)
    write_tsv(out_dir / "analysis" / "formulation_universe_promotion_ledger_v1.tsv", rows, fields)
    write_context(out_dir, args=args, rows=rows)
    print(f"promoted_rows={len(rows)}")
    print(f"out_tsv={out_dir / OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
