#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT


COUNT_FIELDS = [
    "paper_key",
    "model_name",
    "source_run_dir",
    "status",
    "included_count",
    "excluded_count",
    "review_count",
    "gt_count",
    "count_delta_vs_gt",
]

DISAGREEMENT_FIELDS = [
    "paper_key",
    "model_names",
    "included_counts_by_model_json",
    "gt_count",
    "count_range",
    "label_overlap_summary_json",
    "adjudication_status",
    "recommended_next_action",
]

ADJUDICATED_FIELDS = [
    "paper_key",
    "canonical_formulation_id",
    "formulation_label",
    "aliases_json",
    "row_role",
    "identity_basis",
    "preparation_evidence_quote",
    "source_locator",
    "confidence",
    "requires_human_review",
    "provisional_source_model",
    "adjudication_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare formulation-universe model outputs and emit diagnostic adjudication surfaces."
    )
    parser.add_argument(
        "--model-run",
        action="append",
        required=True,
        help="Repeatable model_name=run_dir. run_dir must contain parsed/*__formulation_universe.json.",
    )
    parser.add_argument("--gt-counts-tsv", default="", help="Optional Layer1 GT counts TSV for diagnostic comparison only.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fields})


def key_of(row: dict[str, str]) -> str:
    for name in ["paper_key", "key", "paper_id", "zotero_key"]:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_gt_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for row in load_tsv(path):
        key = key_of(row)
        if not key:
            continue
        try:
            out[key] = int(str(row.get("gt_count", "")).strip())
        except ValueError:
            pass
    return out


def parse_model_runs(values: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-run must be model_name=run_dir: {value}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        path = repo_path(path_text.strip())
        if not name:
            raise ValueError(f"Empty model name in --model-run: {value}")
        if not (path / "parsed").exists():
            raise FileNotFoundError(f"Missing parsed/ directory for {name}: {path}")
        runs.append((name, path))
    return runs


def load_model_payloads(model_name: str, run_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "parsed").glob("*__formulation_universe.json")):
        paper_key = path.name.split("__", 1)[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        payload["_model_name"] = model_name
        payload["_source_run_dir"] = to_repo_rel(run_dir)
        payloads[paper_key] = payload
    return payloads


def norm_label(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def included_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("included_formulations") or []
    return [row for row in rows if isinstance(row, dict)]


def excluded_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("excluded_candidates") or []
    return [row for row in rows if isinstance(row, dict)]


def review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("unresolved_candidate_review") or []
    return [row for row in rows if isinstance(row, dict)]


def label_set(payload: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for row in included_rows(payload):
        label = norm_label(str(row.get("formulation_label", "")))
        if label:
            labels.add(label)
    return labels


def choose_provisional_source(paper_payloads: dict[str, dict[str, Any]]) -> tuple[str, str]:
    counts = {name: len(included_rows(payload)) for name, payload in paper_payloads.items()}
    if not counts:
        return "", "no_model_output"
    unique_counts = set(counts.values())
    if len(unique_counts) == 1:
        return sorted(counts)[0], "model_count_consensus"
    # Provisional only: prefer the model that made the narrowest row-universe
    # call while preserving an explicit exclusion ledger. This prevents value
    # stages from inheriting obvious over-inclusion, but remains review-only.
    ranked = sorted(
        counts,
        key=lambda name: (
            counts[name],
            -len(excluded_rows(paper_payloads[name])),
            name,
        ),
    )
    return ranked[0], "model_count_disagreement_precision_candidate"


def build_adjudicated_rows(
    paper_key: str,
    source_model: str,
    status: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(included_rows(payload), start=1):
        rows.append(
            {
                "paper_key": paper_key,
                "canonical_formulation_id": item.get("canonical_formulation_id") or f"F{index:03d}",
                "formulation_label": item.get("formulation_label", ""),
                "aliases_json": item.get("aliases", []),
                "row_role": item.get("row_role", ""),
                "identity_basis": item.get("identity_basis", ""),
                "preparation_evidence_quote": item.get("preparation_evidence_quote", ""),
                "source_locator": item.get("source_locator", ""),
                "confidence": item.get("confidence", ""),
                "requires_human_review": item.get("requires_human_review", ""),
                "provisional_source_model": source_model,
                "adjudication_status": status,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_runs = parse_model_runs(args.model_run)
    gt_counts = load_gt_counts(repo_path(args.gt_counts_tsv)) if args.gt_counts_tsv else {}

    by_model = {name: load_model_payloads(name, run_dir) for name, run_dir in model_runs}
    paper_keys = sorted({key for payloads in by_model.values() for key in payloads})
    count_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    adjudicated_rows: list[dict[str, Any]] = []

    for paper_key in paper_keys:
        paper_payloads = {name: payloads[paper_key] for name, payloads in by_model.items() if paper_key in payloads}
        counts = {name: len(included_rows(payload)) for name, payload in paper_payloads.items()}
        gt_count = gt_counts.get(paper_key)
        for name, payload in paper_payloads.items():
            included_count = len(included_rows(payload))
            count_rows.append(
                {
                    "paper_key": paper_key,
                    "model_name": name,
                    "source_run_dir": payload.get("_source_run_dir", ""),
                    "status": "parsed",
                    "included_count": included_count,
                    "excluded_count": len(excluded_rows(payload)),
                    "review_count": len(review_rows(payload)),
                    "gt_count": "" if gt_count is None else gt_count,
                    "count_delta_vs_gt": "" if gt_count is None else included_count - gt_count,
                }
            )

        overlap: dict[str, Any] = {}
        names = sorted(paper_payloads)
        if len(names) >= 2:
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    lset = label_set(paper_payloads[left])
                    rset = label_set(paper_payloads[right])
                    union = lset | rset
                    overlap[f"{left}__vs__{right}"] = {
                        "left_count": len(lset),
                        "right_count": len(rset),
                        "intersection_count": len(lset & rset),
                        "union_count": len(union),
                        "jaccard": round(len(lset & rset) / len(union), 4) if union else 1.0,
                    }
        unique_counts = set(counts.values())
        if len(paper_payloads) < len(model_runs):
            status = "missing_model_output"
            action = "complete missing model output before final adjudication"
        elif len(unique_counts) == 1:
            status = "model_count_consensus"
            action = "inspect label/evidence overlap; consensus count can be reviewed as provisional"
        else:
            status = "model_count_disagreement"
            action = "Codex/human adjudication required before row-universe freeze"
        source_model, source_status = choose_provisional_source(paper_payloads)
        if source_model:
            adjudicated_rows.extend(
                build_adjudicated_rows(
                    paper_key,
                    source_model,
                    source_status,
                    paper_payloads[source_model],
                )
            )
        disagreement_rows.append(
            {
                "paper_key": paper_key,
                "model_names": names,
                "included_counts_by_model_json": counts,
                "gt_count": "" if gt_count is None else gt_count,
                "count_range": "" if not counts else max(counts.values()) - min(counts.values()),
                "label_overlap_summary_json": overlap,
                "adjudication_status": status,
                "recommended_next_action": action,
            }
        )

    write_tsv(out_dir / "model_count_comparison_v1.tsv", count_rows, COUNT_FIELDS)
    write_tsv(out_dir / "model_disagreement_ledger_v1.tsv", disagreement_rows, DISAGREEMENT_FIELDS)
    write_tsv(out_dir / "provisional_adjudicated_formulation_universe_v1.tsv", adjudicated_rows, ADJUDICATED_FIELDS)
    (out_dir / "RUN_CONTEXT.md").write_text(
        "\n".join(
            [
                "# formulation_universe_adjudication_v1",
                "",
                f"generated_at: {now_iso()}",
                "generated_by: `src/stage2_sampling_labels/build_formulation_universe_adjudication_v1.py`",
                "benchmark_valid: `no`",
                "compare_mode: `diagnostic-only, not benchmark-valid final output`",
                "",
                "Boundary:",
                "- This compares model-proposed formulation universes.",
                "- The provisional adjudicated table is review-only.",
                "- It is not a completed Stage2 artifact, not a Stage3 input, and not a Stage5 final table.",
                "- GT counts, when supplied, are used only for post-hoc diagnostic comparison.",
                "",
                "Model inputs:",
                *[f"- `{name}`: `{to_repo_rel(path)}`" for name, path in model_runs],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {to_repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
