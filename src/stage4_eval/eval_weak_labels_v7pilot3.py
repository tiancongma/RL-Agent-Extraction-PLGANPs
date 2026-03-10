#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


FORMULATION_INSTANCE_KINDS = {"new_formulation", "variant_formulation"}
NON_FORMULATION_KIND = "candidate_non_formulation"


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

        has_parent_variant = pred_form[
            pred_form["instance_kind"].eq("variant_formulation")
            & pred_form["parent_instance_id"].astype(str).str.strip().ne("")
        ]

        over = len(pred_form) > len(gt_form)
        under = len(pred_form) < len(gt_form)
        if not over and not under and (len(gt_non) == 0 or len(pred_non) >= len(gt_non)):
            boundary_status = "preserved"
        elif abs(len(pred_form) - len(gt_form)) <= 1:
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
                "pred_formulation_rows": int(len(pred_form)),
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
        lines.append(
            "- "
            f"`{r['doi_norm']}`: GT={int(r['gt_formulation_rows'])} formulation rows, "
            f"pred={int(r['pred_formulation_rows'])}; "
            f"over-seg={r['over_segmentation']}, under-seg={r['under_segmentation']}, "
            f"boundary={r['boundary_status']}, non-form suppression={r['non_formulation_suppressed']}, "
            f"inheritance separation={r['inheritance_variant_separated']}."
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
