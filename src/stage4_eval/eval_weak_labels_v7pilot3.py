#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from src.utils.paths import PROJECT_ROOT, dataset_text_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT, dataset_text_root


FORMULATION_INSTANCE_KINDS = {"new_formulation", "variant_formulation"}
NON_FORMULATION_KIND = "candidate_non_formulation"
WFDTQ4VX_DOI = "10.1080/10717544.2016.1199605"


def normalize_doi(v: Any) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def parse_json_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in re.split(r"[|,;\n]+", s) if p.strip()]


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def clean_ocr_token(token: str) -> str:
    s = token.replace("\x04", "-")
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
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mg|g|ug|mcg)?", s)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2) or "mg"
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
        raise RuntimeError("Could not parse fixed organic-phase volume for WFDTQ4VX reconciliation.")
    return float(m.group(1))


def percent_wv_to_mg(percent_value: float, volume_ml: float) -> float:
    return percent_value * 10.0 * volume_ml


def extract_table1_level_map(lines: List[str]) -> Dict[str, Dict[float, float]]:
    try:
        anchor = next(i for i, line in enumerate(lines) if "Table 1. Factorial design parameters" in line)
    except StopIteration as exc:
        raise RuntimeError("Could not locate Table 1 factor levels.") from exc

    level_map: Dict[str, Dict[float, float]] = {}
    for factor_name in ["X1", "X2", "X3"]:
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
                break
        if factor_name not in level_map:
            raise RuntimeError(f"Could not locate factor row for {factor_name}.")
    return level_map


def interpolate_from_coded(levels: Dict[float, float], coded_value: float) -> float:
    if coded_value in levels:
        return levels[coded_value]
    ordered = sorted(levels.items())
    xs = [k for k, _ in ordered]
    ys = [v for _, v in ordered]
    if coded_value < xs[0] or coded_value > xs[-1]:
        raise RuntimeError(f"Coded value {coded_value} is outside interpolation range {xs}.")
    for left_idx in range(len(xs) - 1):
        x0, x1 = xs[left_idx], xs[left_idx + 1]
        if x0 <= coded_value <= x1:
            y0, y1 = ys[left_idx], ys[left_idx + 1]
            frac = (coded_value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    raise RuntimeError(f"Could not interpolate coded value {coded_value}.")


def extract_checkpoint_rows(lines: List[str]) -> List[Dict[str, Any]]:
    try:
        anchor = next(
            i for i, line in enumerate(lines) if "Checkpoint batches with their predicted and measured values of PS and EE" in line
        )
    except StopIteration as exc:
        raise RuntimeError("Could not locate Table 7 checkpoint rows.") from exc

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
            }
        )
        idx += 8
    if not rows:
        raise RuntimeError("Checkpoint table anchor found, but no checkpoint rows were parsed.")
    return rows


def parse_coded_cell(cell: str) -> Tuple[float, str]:
    cleaned = clean_ocr_token(cell)
    negative_prefix = bool(cell[:1] and ord(cell[:1]) < 32)
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


def resolve_source_text_path(m: pd.Series) -> Path:
    candidates: List[Path] = []
    raw_text_path = str(m.get("text_path", "")).strip()
    if raw_text_path:
        candidates.append(PROJECT_ROOT / Path(raw_text_path))
        candidates.append(PROJECT_ROOT / Path(raw_text_path.replace("\\", "/")))
    key = str(m.get("key", "")).strip()
    candidates.append(dataset_text_root("goren_2025") / key / f"{key}.pdf.txt")
    candidates.append(dataset_text_root("goren_2025") / key / f"{key}.html.txt")
    candidates.append(PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "text" / f"{key}.pdf.txt")
    candidates.append(PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "text" / f"{key}.html.txt")
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve source text path for {key}. Checked:\n{checked}")


def reconcile_wfdtq4vx_coordinate_count(pred_form: pd.DataFrame, manifest_row: pd.Series) -> Tuple[int, str]:
    source_text = resolve_source_text_path(manifest_row).read_text(encoding="utf-8", errors="replace")
    organic_phase_volume_ml = parse_organic_phase_volume_ml(source_text)
    lines = [clean_ocr_token(line) for line in source_text.splitlines()]
    table1_levels = extract_table1_level_map(lines)
    checkpoint_rows = extract_checkpoint_rows(lines)

    design_rows = pred_form[~pred_form["local_instance_id"].astype(str).str.contains("Checkpoint", case=False, na=False)].copy()
    design_rows["x1_drug_mg"] = design_rows["drug_feed_amount_text_value_text"].map(parse_mass_mg)
    design_rows["x2_polymer_mg"] = design_rows["plga_mass_mg_value_text"].map(parse_mass_mg)
    design_rows["x3_surfactant_pct"] = design_rows["surfactant_concentration_text_value_text"].map(parse_percent)
    if design_rows[["x1_drug_mg", "x2_polymer_mg", "x3_surfactant_pct"]].isna().any(axis=None):
        bad_ids = design_rows.loc[
            design_rows[["x1_drug_mg", "x2_polymer_mg", "x3_surfactant_pct"]].isna().any(axis=1), "local_instance_id"
        ].tolist()
        raise RuntimeError(f"WFDTQ4VX design rows missing coordinate fields: {bad_ids}")

    checkpoint_df = pred_form[pred_form["local_instance_id"].astype(str).str.contains("Checkpoint", case=False, na=False)].copy()
    checkpoint_map: Dict[int, int] = {}
    for idx, value in checkpoint_df["local_instance_id"].items():
        m = re.search(r"(\d+)$", str(value))
        if m:
            checkpoint_map[int(m.group(1))] = idx
    for row in checkpoint_rows:
        batch_no = row["batch_no"]
        if batch_no not in checkpoint_map:
            raise RuntimeError(f"WFDTQ4VX checkpoint batch {batch_no} not found in predicted rows.")
        pred_idx = checkpoint_map[batch_no]
        x1_coded, _ = parse_coded_cell(row["x1_raw"])
        x2_coded, _ = parse_coded_cell(row["x2_raw"])
        x3_coded, _ = parse_coded_cell(row["x3_raw"])
        x1_pct = interpolate_from_coded(table1_levels["X1"], x1_coded)
        x2_pct = interpolate_from_coded(table1_levels["X2"], x2_coded)
        x3_pct = interpolate_from_coded(table1_levels["X3"], x3_coded)
        checkpoint_df.loc[pred_idx, "x1_drug_mg"] = percent_wv_to_mg(x1_pct, organic_phase_volume_ml)
        checkpoint_df.loc[pred_idx, "x2_polymer_mg"] = percent_wv_to_mg(x2_pct, organic_phase_volume_ml)
        checkpoint_df.loc[pred_idx, "x3_surfactant_pct"] = x3_pct

    all_rows = pd.concat([design_rows, checkpoint_df], ignore_index=True)
    all_rows["coordinate_signature"] = all_rows.apply(
        lambda r: signature_string(r["x1_drug_mg"], r["x2_polymer_mg"], r["x3_surfactant_pct"]),
        axis=1,
    )
    reconciled_count = int(all_rows["coordinate_signature"].nunique())
    reason = (
        "WFDTQ4VX DoE coordinate reconciliation applied: design rows keyed by "
        "drug mg + polymer mg + surfactant percent; checkpoint rows mapped from "
        "Table 7 coded levels through Table 1 factor levels; predicted/observed values ignored for identity."
    )
    return reconciled_count, reason


def maybe_reconcile_formulation_count(pred_form: pd.DataFrame, manifest_row: pd.Series) -> Dict[str, Any]:
    raw_count = int(len(pred_form))
    doi = normalize_doi(manifest_row.get("doi", ""))
    out: Dict[str, Any] = {
        "raw_count_before_reconciliation": raw_count,
        "reconciled_count_after_reconciliation": raw_count,
        "reconciliation_applied": "no",
        "reconciliation_reason": "",
    }
    if doi != WFDTQ4VX_DOI:
        return out
    reconciled_count, reason = reconcile_wfdtq4vx_coordinate_count(pred_form=pred_form, manifest_row=manifest_row)
    out["reconciled_count_after_reconciliation"] = int(reconciled_count)
    out["reconciliation_applied"] = "yes"
    out["reconciliation_reason"] = reason
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate formulation-instance pilot outputs on the fixed 3-paper DEV15 subset."
    )
    p.add_argument("--pilot-tsv", required=True)
    p.add_argument(
        "--pilot-manifest",
        default="data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv",
    )
    p.add_argument(
        "--gt-xlsx",
        default="data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx",
    )
    p.add_argument(
        "--summary-md",
        default="docs/methods/formulation_instance_pilot3_eval_2026-03-10.md",
    )
    p.add_argument(
        "--out-dir",
        default="data/cleaned/labels/manual/formulation_instance_pilot3_eval_2026-03-10",
    )
    return p.parse_args()


def load_gt(gt_xlsx: Path, target_dois: set[str]) -> pd.DataFrame:
    gt = pd.read_excel(gt_xlsx, sheet_name="review_formulations").fillna("")
    gt["doi_norm"] = gt["doi"].map(normalize_doi)
    gt = gt[gt["doi_norm"].isin(target_dois)].copy()
    gt["is_gt_formulation"] = gt["formulation_exists_gt"].astype(str).str.strip().str.lower().eq("yes")
    return gt


def build_per_doi_summary(pred: pd.DataFrame, gt: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, m in manifest.iterrows():
        doi = normalize_doi(m["doi"])
        pred_d = pred[pred["doi_norm"] == doi].copy()
        gt_d = gt[gt["doi_norm"] == doi].copy()

        pred_form = pred_d[pred_d["instance_kind"].isin(FORMULATION_INSTANCE_KINDS)].copy()
        pred_non = pred_d[pred_d["instance_kind"] == NON_FORMULATION_KIND].copy()
        pred_unclear = pred_d[pred_d["instance_kind"] == "unclear"].copy()
        gt_form = gt_d[gt_d["is_gt_formulation"]].copy()
        gt_non = gt_d[~gt_d["is_gt_formulation"]].copy()
        reconciliation = maybe_reconcile_formulation_count(pred_form=pred_form, manifest_row=m)
        pred_form_count = int(reconciliation["reconciled_count_after_reconciliation"])

        has_parent_variant = pred_form[
            pred_form["instance_kind"].eq("variant_formulation")
            & pred_form["parent_instance_id"].astype(str).str.strip().ne("")
        ]

        over = pred_form_count > len(gt_form)
        under = pred_form_count < len(gt_form)
        if not over and not under and (len(gt_non) == 0 or len(pred_non) >= len(gt_non)):
            boundary_status = "preserved"
        elif abs(pred_form_count - len(gt_form)) <= 1:
            boundary_status = "mixed"
        else:
            boundary_status = "broken"

        if len(gt_non) == 0:
            non_form_status = "not_applicable"
        elif len(pred_non) >= len(gt_non):
            non_form_status = "yes"
        elif len(pred_non) > 0:
            non_form_status = "partial"
        else:
            non_form_status = "no"

        if len(pred_form) <= 1:
            inheritance_status = "not_observed"
        elif len(has_parent_variant) > 0:
            inheritance_status = "yes"
        else:
            inheritance_status = "no"

        rows.append(
            {
                "doi_norm": doi,
                "paper_key": m["key"],
                "pilot_reason": m.get("pilot_reason", ""),
                "gt_formulation_rows": int(len(gt_form)),
                "gt_candidate_non_formulation_rows": int(len(gt_non)),
                "pred_formulation_rows": pred_form_count,
                "raw_count_before_reconciliation": int(reconciliation["raw_count_before_reconciliation"]),
                "reconciled_count_after_reconciliation": pred_form_count,
                "reconciliation_applied": reconciliation["reconciliation_applied"],
                "reconciliation_reason": reconciliation["reconciliation_reason"],
                "pred_candidate_non_formulation_rows": int(len(pred_non)),
                "pred_unclear_rows": int(len(pred_unclear)),
                "pred_new_formulation_rows": int(pred_form["instance_kind"].eq("new_formulation").sum()),
                "pred_variant_formulation_rows": int(pred_form["instance_kind"].eq("variant_formulation").sum()),
                "pred_variant_with_parent_rows": int(len(has_parent_variant)),
                "over_segmentation": "yes" if over else "no",
                "under_segmentation": "yes" if under else "no",
                "boundary_status": boundary_status,
                "non_formulation_suppressed": non_form_status,
                "inheritance_variant_separated": inheritance_status,
            }
        )
    return pd.DataFrame(rows).sort_values("doi_norm").reset_index(drop=True)


def build_predicted_instance_view(pred: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "doi_norm",
        "key",
        "local_instance_id",
        "raw_formulation_label",
        "instance_kind",
        "parent_instance_id",
        "change_role",
        "change_descriptions",
        "instance_context_tags",
        "change_context_tags",
        "formulation_role",
        "instance_confidence",
        "supporting_evidence_refs",
        "evidence_section",
        "evidence_span_text",
    ]
    out = pred.copy()
    out["counts_as_formulation_row"] = out["instance_kind"].isin(FORMULATION_INSTANCE_KINDS)
    out["change_descriptions_list"] = out["change_descriptions"].map(parse_json_list)
    out["instance_context_tags_list"] = out["instance_context_tags"].map(parse_json_list)
    out["change_context_tags_list"] = out["change_context_tags"].map(parse_json_list)
    out["short_change_note"] = out["change_descriptions_list"].map(lambda xs: " | ".join(xs[:3]))
    out["short_instance_tags"] = out["instance_context_tags_list"].map(lambda xs: ",".join(xs))
    out["short_change_tags"] = out["change_context_tags_list"].map(lambda xs: ",".join(xs))
    out = out[keep_cols + ["counts_as_formulation_row", "short_change_note", "short_instance_tags", "short_change_tags"]]
    out = out.merge(
        manifest[["key", "pilot_reason"]].rename(columns={"key": "key"}),
        on="key",
        how="left",
    )
    return out.sort_values(["doi_norm", "local_instance_id"]).reset_index(drop=True)


def build_terminal_summary(per_doi: pd.DataFrame, pilot_tsv: Path, out_dir: Path, summary_md: Path) -> Dict[str, Any]:
    workable = (
        per_doi["boundary_status"].eq("preserved").sum() >= 2
        and per_doi["non_formulation_suppressed"].isin(["yes", "not_applicable"]).sum() >= 2
    )
    return {
        "pilot_tsv": str(pilot_tsv.resolve()),
        "summary_md": str(summary_md.resolve()),
        "out_dir": str(out_dir.resolve()),
        "fixed_3paper_set": per_doi[["paper_key", "doi_norm"]].to_dict(orient="records"),
        "per_doi": per_doi.to_dict(orient="records"),
        "compressed_enum_design_workable": "yes" if workable else "mixed",
    }


def main() -> None:
    args = parse_args()
    pilot_tsv = Path(args.pilot_tsv)
    pilot_manifest = Path(args.pilot_manifest)
    gt_xlsx = Path(args.gt_xlsx)
    summary_md = Path(args.summary_md)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pilot_tsv, sep="\t", dtype=str).fillna("")
    pred["doi_norm"] = pred["doi"].map(normalize_doi)
    pred["instance_kind"] = pred["instance_kind"].astype(str).str.strip().str.lower().replace("", "unclear")
    pred["parent_instance_id"] = pred["parent_instance_id"].astype(str)

    manifest = pd.read_csv(pilot_manifest, sep="\t", dtype=str).fillna("")
    manifest["doi_norm"] = manifest["doi"].map(normalize_doi)
    target_dois = set(manifest["doi_norm"].tolist())
    pred = pred[pred["doi_norm"].isin(target_dois)].copy()

    gt = load_gt(gt_xlsx, target_dois)
    per_doi = build_per_doi_summary(pred=pred, gt=gt, manifest=manifest)
    predicted_instances = build_predicted_instance_view(pred=pred, manifest=manifest)

    per_doi_path = out_dir / "per_doi_formulation_instance_summary.tsv"
    predicted_path = out_dir / "predicted_instance_rows.tsv"
    per_doi.to_csv(per_doi_path, sep="\t", index=False)
    predicted_instances.to_csv(predicted_path, sep="\t", index=False)

    lines: List[str] = []
    lines.append("# Formulation-Instance Pilot Evaluation (2026-03-10)")
    lines.append("")
    lines.append("## Fixed 3-paper set reused")
    for _, r in manifest.sort_values("doi_norm").iterrows():
        lines.append(f"- `{normalize_doi(r['doi'])}` | key `{r['key']}` | {r.get('pilot_reason', '')}")
    lines.append("")
    lines.append("## Per-paper result summary")
    for _, r in per_doi.iterrows():
        reconciliation_note = ""
        if str(r.get("reconciliation_applied", "")).strip().lower() == "yes":
            reconciliation_note = (
                f", reconciliation={int(r['raw_count_before_reconciliation'])}->{int(r['reconciled_count_after_reconciliation'])}"
            )
        lines.append(
            "- "
            f"`{r['doi_norm']}`: GT={int(r['gt_formulation_rows'])} formulation rows, "
            f"pred={int(r['pred_formulation_rows'])}; "
            f"over-seg={r['over_segmentation']}, under-seg={r['under_segmentation']}, "
            f"boundary={r['boundary_status']}, non-form suppression={r['non_formulation_suppressed']}, "
            f"inheritance separation={r['inheritance_variant_separated']}{reconciliation_note}."
        )
    lines.append("")
    applied = per_doi[per_doi["reconciliation_applied"].eq("yes")].copy()
    if not applied.empty:
        lines.append("## Stage4 reconciliation adjustments")
        for _, r in applied.iterrows():
            lines.append(
                f"- `{r['doi_norm']}`: raw count {int(r['raw_count_before_reconciliation'])} -> "
                f"reconciled count {int(r['reconciled_count_after_reconciliation'])}. "
                f"{r['reconciliation_reason']}"
            )
        lines.append("")
    lines.append("## Engineering readout")
    total_gt = int(per_doi["gt_formulation_rows"].sum())
    total_pred = int(per_doi["pred_formulation_rows"].sum())
    total_non_gt = int(per_doi["gt_candidate_non_formulation_rows"].sum())
    total_non_pred = int(per_doi["pred_candidate_non_formulation_rows"].sum())
    lines.append(f"- Predicted formulation rows: {total_pred} vs GT {total_gt}.")
    lines.append(f"- Predicted candidate_non_formulation rows: {total_non_pred} vs GT non-formulation rows {total_non_gt}.")
    workable = build_terminal_summary(per_doi, pilot_tsv, out_dir, summary_md)["compressed_enum_design_workable"]
    lines.append(f"- Compressed enum design appears workable: **{workable}**.")
    if per_doi["under_segmentation"].eq("yes").any():
        next_bottleneck = "instance boundary enumeration in dense table blocks"
    elif per_doi["non_formulation_suppressed"].isin(["no", "partial"]).any():
        next_bottleneck = "non-synthesis variant suppression and post-processing/test-condition classification"
    else:
        next_bottleneck = "parent-link and synthesis-change attribution quality for variant rows"
    lines.append(f"- Next bottleneck: {next_bottleneck}.")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(build_terminal_summary(per_doi, pilot_tsv, out_dir, summary_md), indent=2))


if __name__ == "__main__":
    main()
