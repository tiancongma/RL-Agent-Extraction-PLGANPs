#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

try:
    from src.utils.paths import DATA_RESULTS_DIR, dataset_text_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_RESULTS_DIR, dataset_text_root


DEFAULT_PAPER_KEY = "WFDTQ4VX"
DEFAULT_DOI = "10.1080/10717544.2016.1199605"
DEFAULT_EXPECTED_COUNT = 29


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_doi(value: Any) -> str:
    s = normalize_text(value).lower()
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic Stage4 experiment: reconcile DoE checkpoint rows by coordinate signature."
    )
    p.add_argument("--paper-key", default=DEFAULT_PAPER_KEY)
    p.add_argument("--doi", default=DEFAULT_DOI)
    p.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    p.add_argument(
        "--out-dir",
        default=str(DATA_RESULTS_DIR / "doe_coordinate_reconciliation_v1"),
    )
    return p.parse_args()


def discover_weak_label_candidates() -> List[Path]:
    return sorted(DATA_RESULTS_DIR.rglob("weak_labels__v7pilot_r3_fixparse.tsv"))


def select_weak_label_input(paper_key: str, doi: str) -> Tuple[Path, List[Path]]:
    checked = discover_weak_label_candidates()
    wanted_doi = normalize_doi(doi)
    matches: List[Tuple[float, Path]] = []
    for path in checked:
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        except Exception:
            continue
        if "key" not in df.columns or "doi" not in df.columns:
            continue
        mask = df["key"].astype(str).eq(paper_key) & df["doi"].map(normalize_doi).eq(wanted_doi)
        if mask.any():
            matches.append((path.stat().st_mtime, path.resolve()))
    if not matches:
        checked_str = "\n".join(str(p) for p in checked)
        raise FileNotFoundError(
            "No weak-label TSV contained the requested paper.\n"
            f"paper_key={paper_key}\n"
            f"doi={wanted_doi}\n"
            f"candidate_files_checked:\n{checked_str}"
        )
    matches.sort(key=lambda item: (item[0], str(item[1])))
    return matches[-1][1], checked


def load_predicted_rows(path: Path, paper_key: str, doi: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    out = df[df["key"].astype(str).eq(paper_key) & df["doi"].map(normalize_doi).eq(normalize_doi(doi))].copy()
    if out.empty:
        raise RuntimeError(f"Selected weak-label TSV does not contain paper after filtering: {path}")
    return out.reset_index(drop=True)


def clean_ocr_token(token: str) -> str:
    s = token.replace("", "-").replace("–", "-").replace("−", "-").replace("—", "-")
    s = re.sub(r"[^\x20-\x7E]+", " ", s)
    return normalize_text(s)


def parse_numeric(token: str) -> float | None:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", clean_ocr_token(token))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_percent(token: str) -> float | None:
    return parse_numeric(token)


def parse_mass_mg(token: str) -> float | None:
    s = clean_ocr_token(token).lower()
    if not s:
        return None
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mg|g|ug|μg|mcg)?", s)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "mg").replace("μ", "u")
    if unit == "g":
        value *= 1000.0
    elif unit in {"ug", "mcg"}:
        value /= 1000.0
    return value


def format_num(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def signature_string(x1_drug_mg: float | None, x2_polymer_mg: float | None, x3_surfactant_pct: float | None) -> str:
    return "|".join(
        [
            f"x1_drug_mg={format_num(x1_drug_mg)}",
            f"x2_polymer_mg={format_num(x2_polymer_mg)}",
            f"x3_surfactant_pct={format_num(x3_surfactant_pct)}",
        ]
    )


def parse_organic_phase_volume_ml(text: str) -> float:
    m = re.search(
        r"dissolving\s+plga\s*\([^)]*\)\s+and\s+drug\s*\([^)]*\)\s+in\s+(\d+(?:\.\d+)?)\s*ml\s+of\s+acetone",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        raise RuntimeError("Could not parse the fixed organic-phase volume from the paper text.")
    return float(m.group(1))


def percent_wv_to_mg(percent_value: float, volume_ml: float) -> float:
    # % w/v means g per 100 mL.
    return percent_value * 10.0 * volume_ml


def extract_table1_level_map(lines: List[str]) -> Dict[str, Dict[float, float]]:
    try:
        anchor = next(i for i, line in enumerate(lines) if "Table 1. Factorial design parameters" in line)
    except StopIteration as exc:
        raise RuntimeError("Could not locate Table 1 factor levels in the paper text.") from exc

    level_map: Dict[str, Dict[float, float]] = {}
    for factor_name in ["X1", "X2", "X3"]:
        found = False
        for idx in range(anchor, min(anchor + 60, len(lines))):
            if lines[idx].startswith(factor_name):
                vals: List[float] = []
                for probe in lines[idx + 1 : idx + 8]:
                    val = parse_numeric(probe)
                    if val is not None:
                        vals.append(val)
                    if len(vals) == 3:
                        break
                if len(vals) != 3:
                    raise RuntimeError(f"Could not extract three factor levels for {factor_name}.")
                level_map[factor_name] = {-1.0: vals[0], 0.0: vals[1], 1.0: vals[2]}
                found = True
                break
        if not found:
            raise RuntimeError(f"Could not locate factor level row for {factor_name}.")
    return level_map


def interpolate_from_coded(levels: Dict[float, float], coded_value: float) -> float:
    ordered = sorted(levels.items())
    xs = [k for k, _ in ordered]
    ys = [v for _, v in ordered]
    if coded_value in levels:
        return levels[coded_value]
    if coded_value < xs[0] or coded_value > xs[-1]:
        raise RuntimeError(f"Coded value {coded_value} is outside supported interpolation range {xs}.")
    for left_idx in range(len(xs) - 1):
        x0, x1 = xs[left_idx], xs[left_idx + 1]
        if x0 <= coded_value <= x1:
            y0, y1 = ys[left_idx], ys[left_idx + 1]
            frac = (coded_value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    raise RuntimeError(f"Could not interpolate coded value {coded_value}.")


def extract_checkpoint_rows(lines: List[str]) -> List[Dict[str, Any]]:
    try:
        anchor = next(i for i, line in enumerate(lines) if "Checkpoint batches with their predicted and measured values of PS and EE" in line)
    except StopIteration as exc:
        raise RuntimeError("Could not locate Table 7 checkpoint rows in the paper text.") from exc

    rows: List[Dict[str, Any]] = []
    idx = anchor + 1
    while idx + 7 < len(lines):
        batch_label = clean_ocr_token(lines[idx])
        if not re.fullmatch(r"\d+", batch_label):
            break
        rows.append(
            {
                "batch_no": int(batch_label),
                "x1_raw": clean_ocr_token(lines[idx + 1]),
                "x2_raw": clean_ocr_token(lines[idx + 2]),
                "x3_raw": clean_ocr_token(lines[idx + 3]),
                "observed_ps_nm": parse_numeric(lines[idx + 4]),
                "predicted_ps_nm": parse_numeric(lines[idx + 5]),
                "observed_ee_percent": parse_numeric(lines[idx + 6]),
                "predicted_ee_percent": parse_numeric(lines[idx + 7]),
            }
        )
        idx += 8
    if not rows:
        raise RuntimeError("Checkpoint table anchor found, but no checkpoint rows were parsed.")
    return rows


def parse_coded_cell(cell: str) -> Tuple[float, str]:
    cleaned = clean_ocr_token(cell)
    negative_prefix = cell[:1] and ord(cell[:1]) < 32
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\(([^)]*)\)", cleaned)
    if m:
        coded_str = m.group(1)
        actual_str = m.group(2)
    else:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if len(nums) < 2:
            raise RuntimeError(f"Could not parse coded checkpoint cell: {cell!r}")
        coded_str = nums[0]
        actual_str = nums[1]
    coded_value = float(coded_str)
    if negative_prefix and coded_value > 0:
        coded_value = -coded_value
    return coded_value, actual_str


def build_design_signatures(pred: pd.DataFrame, organic_phase_volume_ml: float) -> pd.DataFrame:
    design_rows = pred[~pred["local_instance_id"].astype(str).str.contains("Checkpoint", case=False, na=False)].copy()
    design_rows["x1_drug_mg"] = design_rows["drug_feed_amount_text_value_text"].map(parse_mass_mg)
    design_rows["x2_polymer_mg"] = design_rows["plga_mass_mg_value_text"].map(parse_mass_mg)
    design_rows["x3_surfactant_pct"] = design_rows["surfactant_concentration_text_value_text"].map(parse_percent)
    design_rows["coordinate_signature"] = design_rows.apply(
        lambda r: signature_string(r["x1_drug_mg"], r["x2_polymer_mg"], r["x3_surfactant_pct"]),
        axis=1,
    )
    design_rows["coordinate_source"] = "weak_labels_design_tuple"
    design_rows["coordinate_notes"] = (
        "drug_feed_amount_text_value_text + plga_mass_mg_value_text + surfactant_concentration_text_value_text"
    )
    needed = ["x1_drug_mg", "x2_polymer_mg", "x3_surfactant_pct"]
    missing_mask = design_rows[needed].isna().any(axis=1)
    if missing_mask.any():
        bad_ids = design_rows.loc[missing_mask, "local_instance_id"].tolist()
        raise RuntimeError(f"Missing design coordinate fields for rows: {bad_ids}")
    return design_rows


def build_checkpoint_signatures(
    pred: pd.DataFrame,
    table1_levels: Dict[str, Dict[float, float]],
    checkpoint_rows: List[Dict[str, Any]],
    organic_phase_volume_ml: float,
) -> pd.DataFrame:
    checkpoint_pred = pred[pred["local_instance_id"].astype(str).str.contains("Checkpoint", case=False, na=False)].copy()
    checkpoint_map = {int(re.search(r"(\d+)$", str(v)).group(1)): idx for idx, v in checkpoint_pred["local_instance_id"].items()}

    for row in checkpoint_rows:
        batch_no = row["batch_no"]
        if batch_no not in checkpoint_map:
            raise RuntimeError(f"Checkpoint batch {batch_no} was parsed from paper text but not found in predicted rows.")
        pred_idx = checkpoint_map[batch_no]
        x1_coded, _ = parse_coded_cell(row["x1_raw"])
        x2_coded, _ = parse_coded_cell(row["x2_raw"])
        x3_coded, _ = parse_coded_cell(row["x3_raw"])

        x1_pct = interpolate_from_coded(table1_levels["X1"], x1_coded)
        x2_pct = interpolate_from_coded(table1_levels["X2"], x2_coded)
        x3_pct = interpolate_from_coded(table1_levels["X3"], x3_coded)

        x1_mg = percent_wv_to_mg(x1_pct, organic_phase_volume_ml)
        x2_mg = percent_wv_to_mg(x2_pct, organic_phase_volume_ml)

        checkpoint_pred.loc[pred_idx, "x1_drug_mg"] = x1_mg
        checkpoint_pred.loc[pred_idx, "x2_polymer_mg"] = x2_mg
        checkpoint_pred.loc[pred_idx, "x3_surfactant_pct"] = x3_pct
        checkpoint_pred.loc[pred_idx, "checkpoint_x1_coded"] = x1_coded
        checkpoint_pred.loc[pred_idx, "checkpoint_x2_coded"] = x2_coded
        checkpoint_pred.loc[pred_idx, "checkpoint_x3_coded"] = x3_coded
        checkpoint_pred.loc[pred_idx, "coordinate_source"] = "table7_coded_levels_plus_table1_map"
        checkpoint_pred.loc[pred_idx, "coordinate_notes"] = (
            f"coded=({format_num(x1_coded)},{format_num(x2_coded)},{format_num(x3_coded)})"
        )

    checkpoint_pred["coordinate_signature"] = checkpoint_pred.apply(
        lambda r: signature_string(r["x1_drug_mg"], r["x2_polymer_mg"], r["x3_surfactant_pct"]),
        axis=1,
    )
    return checkpoint_pred


def build_reconciled_instances(pred: pd.DataFrame, source_text: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    organic_phase_volume_ml = parse_organic_phase_volume_ml(source_text)
    lines = [clean_ocr_token(line) for line in source_text.splitlines()]
    table1_levels = extract_table1_level_map(lines)
    checkpoint_rows = extract_checkpoint_rows(lines)

    design_df = build_design_signatures(pred=pred, organic_phase_volume_ml=organic_phase_volume_ml)
    checkpoint_df = build_checkpoint_signatures(
        pred=pred,
        table1_levels=table1_levels,
        checkpoint_rows=checkpoint_rows,
        organic_phase_volume_ml=organic_phase_volume_ml,
    )

    all_rows = pd.concat([design_df, checkpoint_df], ignore_index=True)
    all_rows["duplicate_of_existing_signature"] = all_rows["coordinate_signature"].duplicated(keep="first")
    all_rows["reconciled_group_id"] = (
        all_rows.groupby("coordinate_signature", dropna=False).ngroup().map(lambda n: f"coord_{int(n)+1:03d}")
    )
    all_rows["coordinate_signature_count"] = all_rows.groupby("coordinate_signature")["coordinate_signature"].transform("size")

    summary = {
        "organic_phase_volume_ml": organic_phase_volume_ml,
        "table1_levels": table1_levels,
        "checkpoint_rows_parsed": len(checkpoint_rows),
        "design_rows": int(len(design_df)),
        "checkpoint_rows": int(len(checkpoint_df)),
        "raw_rows": int(len(all_rows)),
        "unique_coordinate_signatures": int(all_rows["coordinate_signature"].nunique()),
    }
    return all_rows.sort_values(["duplicate_of_existing_signature", "local_instance_id"], kind="stable"), summary


def build_summary_row(
    *,
    paper_key: str,
    doi: str,
    input_path: Path,
    all_rows: pd.DataFrame,
    summary_meta: Dict[str, Any],
    expected_count: int,
) -> pd.DataFrame:
    reconciled_count = int(summary_meta["unique_coordinate_signatures"])
    raw_row_count = int(summary_meta["raw_rows"])
    direction = "yes" if abs(reconciled_count - expected_count) < abs(raw_row_count - expected_count) else "no"
    notes = (
        "Checkpoint rows reconciled using Table 7 coded levels mapped through Table 1 factor levels; "
        "design rows keyed by drug mg + polymer mg + surfactant percent."
    )
    return pd.DataFrame(
        [
            {
                "zotero_key": paper_key,
                "doi": normalize_doi(doi),
                "input_predicted_rows_tsv": str(input_path),
                "raw_instance_rows": raw_row_count,
                "unique_coordinate_signatures": reconciled_count,
                "reconciled_formulation_count": reconciled_count,
                "expected_interpretation_count": expected_count,
                "moved_toward_expected": direction,
                "notes": notes,
            }
        ]
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weak_label_path, checked_inputs = select_weak_label_input(args.paper_key, args.doi)
    pred = load_predicted_rows(weak_label_path, args.paper_key, args.doi)

    text_path = dataset_text_root("goren_2025") / args.paper_key / f"{args.paper_key}.pdf.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Source text not found: {text_path}")
    source_text = text_path.read_text(encoding="utf-8", errors="replace")

    all_rows, summary_meta = build_reconciled_instances(pred=pred, source_text=source_text)

    detail_cols = [
        "key",
        "doi",
        "local_instance_id",
        "raw_formulation_label",
        "instance_kind",
        "instance_context_tags",
        "coordinate_source",
        "coordinate_notes",
        "x1_drug_mg",
        "x2_polymer_mg",
        "x3_surfactant_pct",
        "coordinate_signature",
        "coordinate_signature_count",
        "duplicate_of_existing_signature",
        "reconciled_group_id",
        "size_nm_value_text",
        "encapsulation_efficiency_percent_value_text",
        "loading_content_percent_value_text",
    ]
    detail_cols = [c for c in detail_cols if c in all_rows.columns]
    detail_df = all_rows[detail_cols].copy()

    summary_df = build_summary_row(
        paper_key=args.paper_key,
        doi=args.doi,
        input_path=weak_label_path,
        all_rows=all_rows,
        summary_meta=summary_meta,
        expected_count=args.expected_count,
    )

    summary_path = out_dir / "reconciliation_summary.tsv"
    detail_path = out_dir / f"{args.paper_key}_reconciled_instances.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    detail_df.to_csv(detail_path, sep="\t", index=False)

    reconciled_count = int(summary_df.iloc[0]["reconciled_formulation_count"])
    raw_count = int(summary_df.iloc[0]["raw_instance_rows"])
    moved = summary_df.iloc[0]["moved_toward_expected"]

    print(f"paper_key={args.paper_key}")
    print(f"doi={normalize_doi(args.doi)}")
    print(f"input_predicted_rows_tsv={weak_label_path}")
    print(f"source_text_path={text_path}")
    print(f"candidate_input_files_checked={len(checked_inputs)}")
    print(f"raw_row_count={raw_count}")
    print(f"reconciled_count={reconciled_count}")
    print(f"expected_interpretation_for_{args.paper_key}={args.expected_count}")
    print(f"moved_toward_expected={moved}")
    print(f"output_summary_tsv={summary_path}")
    print(f"output_detail_tsv={detail_path}")
    print(f"coordinate_signature_fields=x1_drug_mg|x2_polymer_mg|x3_surfactant_pct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
