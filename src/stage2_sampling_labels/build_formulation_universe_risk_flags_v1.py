#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT


RISK_FIELDS = [
    "paper_key",
    "risk_level",
    "risk_flags_json",
    "risk_reasons",
    "recommended_route",
    "model_counts_json",
    "model_count_range",
    "source_label_candidate_count",
    "matrix_signal_count",
    "included_control_like_count",
    "excluded_count_max",
    "review_count_total",
    "gpt35_stage2_candidate_count",
    "gpt35_stage5_final_count",
]


COUNTLIKE_FIELDS = ["key", "paper_key", "paper_id", "zotero_key"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign no-GT formulation-universe risk flags from model outputs, "
            "source/prompt signals, and optional GPT35 lineage row counts."
        )
    )
    parser.add_argument(
        "--model-run",
        action="append",
        required=True,
        help="Repeatable model_name=discovery_run_dir.",
    )
    parser.add_argument(
        "--adjudication-run",
        default="",
        help="Optional adjudication run directory with model_disagreement_ledger_v1.tsv.",
    )
    parser.add_argument("--stage2-tsv", default="", help="Optional GPT35/other Stage2 TSV for observation only.")
    parser.add_argument("--stage5-final-table", default="", help="Optional GPT35/other Stage5 final table for observation only.")
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


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def row_key(row: dict[str, str]) -> str:
    for field in COUNTLIKE_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return ""


def parse_model_runs(values: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-run must be model_name=run_dir: {value}")
        name, path_text = value.split("=", 1)
        path = repo_path(path_text.strip())
        if not (path / "parsed").exists():
            raise FileNotFoundError(f"Missing parsed directory for {name}: {path}")
        out.append((name.strip(), path))
    return out


def load_payloads(run_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "parsed").glob("*__formulation_universe.json")):
        key = path.name.split("__", 1)[0]
        payloads[key] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def included(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("included_formulations") or []
    return [row for row in rows if isinstance(row, dict)]


def excluded(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("excluded_candidates") or []
    return [row for row in rows if isinstance(row, dict)]


def review(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("unresolved_candidate_review") or []
    return [row for row in rows if isinstance(row, dict)]


def count_rows_by_key(path_text: str) -> dict[str, int]:
    if not path_text:
        return {}
    path = repo_path(path_text)
    if not path.exists():
        return {}
    counts: Counter[str] = Counter()
    for row in load_tsv(path):
        key = row_key(row)
        if not key:
            continue
        counts[key] += 1
    return dict(counts)


LABEL_RE = re.compile(
    r"\b(?:F|NP|NLC|NC|NS|PNP|Batch|Run|Formulation)\s*[-_]?\s*\d{1,3}\b",
    re.IGNORECASE,
)
MATRIX_TERMS = [
    "formulation variables",
    "formulation variable",
    "formulation matrix",
    "experimental design",
    "box-behnken",
    "factorial",
    "doe",
    "optimization",
    "optimized formulation",
    "blank formulation",
    "drug loaded",
    "empty",
    "polymer type",
    "polymer ratio",
    "variable",
]
CONTROL_TERMS = [
    "blank",
    "fitc",
    "commercial",
    "free drug",
    "standard",
    "calibration",
    "process_control",
    "commercial_comparator",
]


def prompt_text_for_key(model_runs: list[tuple[str, Path]], paper_key: str) -> str:
    texts: list[str] = []
    for _, run_dir in model_runs:
        path = run_dir / "prompts" / f"{paper_key}__formulation_universe_prompt.txt"
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts)


def source_label_candidate_count(text: str) -> int:
    return len(set(m.group(0).lower().replace(" ", "") for m in LABEL_RE.finditer(text or "")))


def matrix_signal_count(text: str) -> int:
    lowered = (text or "").lower()
    return sum(1 for term in MATRIX_TERMS if term in lowered)


def control_like_count(payloads: list[dict[str, Any]]) -> int:
    max_count = 0
    for payload in payloads:
        count = 0
        for row in included(payload):
            hay = " ".join(
                [
                    str(row.get("formulation_label", "")),
                    str(row.get("row_role", "")),
                    str(row.get("identity_basis", "")),
                ]
            ).lower()
            if any(term in hay for term in CONTROL_TERMS):
                count += 1
        max_count = max(max_count, count)
    return max_count


def risk_level(flags: list[str]) -> str:
    high_flags = {
        "model_consensus_matrix_underexpansion_risk",
        "high_model_count_disagreement",
        "control_or_assay_overinclusion_risk",
    }
    if any(flag in high_flags for flag in flags):
        return "HIGH"
    if flags:
        return "MEDIUM"
    return "LOW"


def main() -> int:
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_runs = parse_model_runs(args.model_run)
    by_model = {name: load_payloads(path) for name, path in model_runs}
    paper_keys = sorted({key for payloads in by_model.values() for key in payloads})
    stage2_counts = count_rows_by_key(args.stage2_tsv)
    stage5_counts = count_rows_by_key(args.stage5_final_table)

    risk_rows: list[dict[str, Any]] = []
    for paper_key in paper_keys:
        payloads_by_name = {name: payloads[paper_key] for name, payloads in by_model.items() if paper_key in payloads}
        payload_list = list(payloads_by_name.values())
        model_counts = {name: len(included(payload)) for name, payload in payloads_by_name.items()}
        count_values = list(model_counts.values())
        count_range = max(count_values) - min(count_values) if count_values else 0
        text = prompt_text_for_key(model_runs, paper_key)
        label_candidates = source_label_candidate_count(text)
        matrix_signals = matrix_signal_count(text)
        included_control_like = control_like_count(payload_list)
        excluded_count_max = max((len(excluded(payload)) for payload in payload_list), default=0)
        review_count_total = sum(len(review(payload)) for payload in payload_list)

        flags: list[str] = []
        reasons: list[str] = []
        min_count = min(count_values) if count_values else 0
        max_count = max(count_values) if count_values else 0
        consensus = len(set(count_values)) == 1 and len(count_values) >= 2
        if consensus and matrix_signals >= 8 and min_count <= 10:
            flags.append("model_consensus_matrix_underexpansion_risk")
            reasons.append("models agree on a small row universe while source/prompt has multiple formulation-matrix signals")
        if count_range >= 4:
            flags.append("high_model_count_disagreement")
            reasons.append("model row counts disagree by at least four rows")
        if included_control_like >= 4 and min_count <= 10:
            flags.append("control_or_assay_overinclusion_risk")
            reasons.append("included rows contain many control/assay/comparator-like labels or roles")
        if label_candidates >= max_count + 5 and label_candidates >= 12:
            flags.append("source_label_surplus_risk")
            reasons.append("source text contains substantially more article-native label candidates than model rows")
        if excluded_count_max >= max(6, min_count):
            flags.append("large_exclusion_ledger_review")
            reasons.append("large excluded-candidate ledger indicates boundary ambiguity")
        if review_count_total:
            flags.append("unresolved_candidate_review_present")
            reasons.append("at least one model emitted unresolved candidates")
        if len(payloads_by_name) < len(model_runs):
            flags.append("missing_model_output")
            reasons.append("at least one configured model run did not emit this paper")

        route = "routine_value_binding_ok"
        level = risk_level(flags)
        if level == "HIGH":
            route = "hold_for_formulation_universe_review"
        elif level == "MEDIUM":
            route = "review_before_value_binding"

        risk_rows.append(
            {
                "paper_key": paper_key,
                "risk_level": level,
                "risk_flags_json": flags,
                "risk_reasons": "; ".join(reasons),
                "recommended_route": route,
                "model_counts_json": model_counts,
                "model_count_range": count_range,
                "source_label_candidate_count": label_candidates,
                "matrix_signal_count": matrix_signals,
                "included_control_like_count": included_control_like,
                "excluded_count_max": excluded_count_max,
                "review_count_total": review_count_total,
                "gpt35_stage2_candidate_count": stage2_counts.get(paper_key, ""),
                "gpt35_stage5_final_count": stage5_counts.get(paper_key, ""),
            }
        )

    write_tsv(out_dir / "formulation_universe_risk_flags_v1.tsv", risk_rows, RISK_FIELDS)
    summary = Counter(row["risk_level"] for row in risk_rows)
    (out_dir / "RUN_CONTEXT.md").write_text(
        "\n".join(
            [
                "# formulation_universe_risk_flags_v1",
                "",
                f"generated_at: {now_iso()}",
                "generated_by: `src/stage2_sampling_labels/build_formulation_universe_risk_flags_v1.py`",
                "benchmark_valid: `no`",
                "gt_free_risk_detection: `yes`",
                "",
                "Boundary:",
                "- This risk layer does not consume GT counts.",
                "- Optional Stage2/Stage5 row counts are observational only.",
                "- It assigns review routing for formulation-universe discovery before value binding.",
                "- It does not create rows, create values, or replace Stage2/Stage5 artifacts.",
                "",
                "Model inputs:",
                *[f"- `{name}`: `{to_repo_rel(path)}`" for name, path in model_runs],
                "",
                f"risk_summary: `{dict(summary)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {to_repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
