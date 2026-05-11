#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.utils.paths import DATA_RESULTS_DIR
    from src.utils.run_id import resolve_governed_results_artifact_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_RESULTS_DIR
    from src.utils.run_id import resolve_governed_results_artifact_path


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_CURATED_GOLD = "data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark alignment evaluation on projected curated-schema outputs."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--projected-tsv",
        default="",
        help="Defaults to data/results/<run_id>/benchmark_goren_2025/projected_to_curated.tsv",
    )
    parser.add_argument("--curated-tsv", default=DEFAULT_CURATED_GOLD)
    parser.add_argument(
        "--modes",
        default="strict,relaxed,canonicalized",
        help="Comma-separated: strict,relaxed,canonicalized",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Defaults to data/results/<run_id>/benchmark_goren_2025",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[^\w\s./:+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_doi(value: Any) -> str:
    s = normalize_text(value)
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s


def parse_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_laga_to_scalar(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(" ", "")
    if text == "":
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                la = float(parts[0])
                ga = float(parts[1])
            except ValueError:
                return None
            if ga == 0:
                return None
            return la / ga
    return parse_float(text)


def compare_numeric(
    left: Any,
    right: Any,
    *,
    mode: str,
    field_name: str,
) -> bool:
    lv = parse_laga_to_scalar(left) if field_name == "LA/GA" else parse_float(left)
    rv = parse_laga_to_scalar(right) if field_name == "LA/GA" else parse_float(right)
    if lv is None or rv is None:
        return False
    diff = abs(lv - rv)
    if mode == "strict":
        return diff == 0
    if mode == "relaxed":
        if field_name == "EE":
            return diff <= 5.0
        return diff <= max(0.1 * max(abs(lv), abs(rv), 1.0), 1e-9)
    # canonicalized
    if field_name == "EE":
        return diff <= 5.0
    return diff <= max(0.05 * max(abs(lv), abs(rv), 1.0), 1e-9)


def compare_categorical(left: Any, right: Any, *, mode: str) -> bool:
    ltxt = normalize_text(left)
    rtxt = normalize_text(right)
    if ltxt == "" or rtxt == "":
        return False
    if mode in {"strict", "relaxed"}:
        return ltxt == rtxt
    # canonicalized mode: token-set equality.
    ltoks = sorted(set(ltxt.split()))
    rtoks = sorted(set(rtxt.split()))
    return ltoks == rtoks


def build_doi_column(df: pd.DataFrame) -> pd.Series:
    if "doi_norm" in df.columns:
        return df["doi_norm"].map(normalize_doi)
    if "reference" in df.columns:
        return df["reference"].map(normalize_doi)
    return pd.Series([""] * len(df))


def row_match_score(
    projected_row: pd.Series,
    curated_row: pd.Series,
    *,
    mode: str,
    categorical_fields: list[str],
    numeric_fields: list[str],
) -> tuple[int, list[str]]:
    matched_count = 0
    mismatches: list[str] = []
    for field in categorical_fields:
        if field not in projected_row.index or field not in curated_row.index:
            continue
        left = projected_row[field]
        right = curated_row[field]
        if str(left).strip() == "" or str(right).strip() == "":
            continue
        if compare_categorical(left, right, mode=mode):
            matched_count += 1
        else:
            mismatches.append(field)
    for field in numeric_fields:
        if field not in projected_row.index or field not in curated_row.index:
            continue
        left = projected_row[field]
        right = curated_row[field]
        if str(left).strip() == "" or str(right).strip() == "":
            continue
        if compare_numeric(left, right, mode=mode, field_name=field):
            matched_count += 1
        else:
            mismatches.append(field)
    return matched_count, mismatches


def align_mode(projected: pd.DataFrame, curated: pd.DataFrame, *, mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    categorical_fields = ["small_molecule_name", "surfactant_name", "solvent"]
    numeric_fields = ["polymer_MW", "LA/GA", "drug/polymer", "surfactant_concentration", "aqueous/organic", "pH", "particle_size", "EE", "LC"]

    projected = projected.copy()
    curated = curated.copy()
    projected["doi_eval"] = build_doi_column(projected)
    curated["doi_eval"] = build_doi_column(curated)
    projected["projected_row_id"] = projected.index.astype(str)
    curated["curated_row_id"] = curated.index.astype(str)

    align_rows: list[dict[str, Any]] = []
    matched_projected_keys: set[str] = set()
    matched_curated_keys: set[str] = set()

    for doi, curated_group in curated.groupby("doi_eval", dropna=False):
        doi = str(doi)
        if doi == "":
            for _, c_row in curated_group.iterrows():
                align_rows.append(
                    {
                        "mode": mode,
                        "doi_norm": doi,
                        "curated_row_id": c_row["curated_row_id"],
                        "projected_row_id": "",
                        "matched": False,
                        "failure_type": "missing_doi_curated",
                        "matched_fields_count": 0,
                        "mismatched_fields": "",
                    }
                )
            continue

        projected_group = projected[projected["doi_eval"] == doi]
        if projected_group.empty:
            for _, c_row in curated_group.iterrows():
                align_rows.append(
                    {
                        "mode": mode,
                        "doi_norm": doi,
                        "curated_row_id": c_row["curated_row_id"],
                        "projected_row_id": "",
                        "matched": False,
                        "failure_type": "doi_not_found_in_projected",
                        "matched_fields_count": 0,
                        "mismatched_fields": "",
                    }
                )
            continue

        used_projected: set[str] = set()
        for _, c_row in curated_group.iterrows():
            best_pid = ""
            best_score = -1
            best_mismatches: list[str] = []
            for _, p_row in projected_group.iterrows():
                pid = str(p_row["projected_row_id"])
                if pid in used_projected:
                    continue
                score, mismatches = row_match_score(
                    p_row,
                    c_row,
                    mode=mode,
                    categorical_fields=categorical_fields,
                    numeric_fields=numeric_fields,
                )
                if score > best_score:
                    best_score = score
                    best_pid = pid
                    best_mismatches = mismatches

            threshold = 1 if mode == "relaxed" else 2
            is_match = best_score >= threshold and best_pid != ""
            failure_type = ""
            if is_match:
                used_projected.add(best_pid)
                matched_projected_keys.add(best_pid)
                matched_curated_keys.add(str(c_row["curated_row_id"]))
            else:
                failure_type = "field_mismatch_or_insufficient_overlap"

            align_rows.append(
                {
                    "mode": mode,
                    "doi_norm": doi,
                    "curated_row_id": c_row["curated_row_id"],
                    "projected_row_id": best_pid if is_match else "",
                    "matched": bool(is_match),
                    "failure_type": failure_type,
                    "matched_fields_count": int(best_score if best_score > 0 else 0),
                    "mismatched_fields": ",".join(sorted(set(best_mismatches))),
                }
            )

    # Unmatched projected rows for precision diagnostics.
    for _, p_row in projected.iterrows():
        pid = str(p_row["projected_row_id"])
        if pid in matched_projected_keys:
            continue
        doi = str(p_row["doi_eval"])
        align_rows.append(
            {
                "mode": mode,
                "doi_norm": doi,
                "curated_row_id": "",
                "projected_row_id": pid,
                "matched": False,
                "failure_type": "unmatched_projected_row",
                "matched_fields_count": 0,
                "mismatched_fields": "",
            }
        )

    align_df = pd.DataFrame(align_rows)
    total_curated = int(len(curated))
    total_projected = int(len(projected))
    matched_curated = int(len(matched_curated_keys))
    matched_projected = int(len(matched_projected_keys))
    recall = (matched_curated / total_curated) if total_curated else 0.0
    precision = (matched_projected / total_projected) if total_projected else 0.0

    failure_counts = (
        align_df.loc[align_df["matched"] == False, "failure_type"]
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    metrics = {
        "mode": mode,
        "total_curated_rows": total_curated,
        "total_projected_rows": total_projected,
        "matched_curated_rows": matched_curated,
        "matched_projected_rows": matched_projected,
        "recall": recall,
        "precision": precision,
        "failure_type_counts": failure_counts,
    }
    return align_df, metrics


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025")
    resolve_governed_results_artifact_path(
        out_dir,
        results_root=DATA_RESULTS_DIR,
        require_existing_governed_root=True,
    )
    projected_tsv = Path(args.projected_tsv) if args.projected_tsv else out_dir / "projected_to_curated.tsv"
    curated_tsv = Path(args.curated_tsv)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    if not projected_tsv.exists():
        raise FileNotFoundError(f"Projected TSV not found: {projected_tsv}")
    if not curated_tsv.exists():
        raise FileNotFoundError(f"Curated TSV not found: {curated_tsv}")
    if not modes:
        raise RuntimeError("At least one alignment mode is required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    projected = pd.read_csv(projected_tsv, sep="\t", dtype=str).fillna("")
    curated = pd.read_csv(curated_tsv, sep="\t", dtype=str).fillna("")

    allowed_modes = {"strict", "relaxed", "canonicalized"}
    invalid = [m for m in modes if m not in allowed_modes]
    if invalid:
        raise RuntimeError(f"Unsupported modes: {invalid}. Allowed: {sorted(allowed_modes)}")

    all_rows: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    for mode in modes:
        mode_rows, mode_metrics = align_mode(projected, curated, mode=mode)
        all_rows.append(mode_rows)
        metrics[mode] = mode_metrics

    alignment_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    alignment_out = out_dir / "alignment_rows.tsv"
    alignment_rows.to_csv(alignment_out, sep="\t", index=False)

    summary = {
        "run_id": run_id,
        "projected_tsv": str(projected_tsv),
        "curated_tsv": str(curated_tsv),
        "modes": modes,
        "metrics_by_mode": metrics,
        "alignment_rows_tsv": str(alignment_out),
    }
    summary_out = out_dir / "metrics_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failure_rows = []
    for mode in modes:
        mode_counts = metrics[mode]["failure_type_counts"]
        for failure_type, count in mode_counts.items():
            failure_rows.append({"mode": mode, "failure_type": failure_type, "count": int(count)})
    pd.DataFrame(failure_rows).to_csv(out_dir / "failure_types.tsv", sep="\t", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
