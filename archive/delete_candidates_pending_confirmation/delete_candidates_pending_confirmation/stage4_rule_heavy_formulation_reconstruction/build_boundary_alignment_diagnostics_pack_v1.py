#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_RUN_ID = "run_20260227_1016_a8d884b_goren2025_step1dev_v1"
DEFAULT_HUMAN_CSV = "data/benchmark/goren_2025/NP_dataset_formulations.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build boundary-alignment diagnostics pack for frozen DEV (15 DOIs)."
    )
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--human-csv", default=DEFAULT_HUMAN_CSV)
    p.add_argument("--out-subdir", default="step1_dev/boundary_alignment_diagnostics_v1")
    return p.parse_args()


def norm_doi(value: object) -> str:
    s = "" if value is None else str(value).strip().lower()
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in {"", "nan", "none", "null", "na", "n/a", "-", "--"}:
        return ""
    return s


def parse_first_float(value: object) -> Optional[float]:
    s = clean_text(value).replace(",", "")
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_ratio_scalar(value: object) -> Optional[float]:
    s = clean_text(value).replace(",", "")
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[:/]\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        if abs(b) < 1e-12:
            return None
        return a / b
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s)
    return None


def normalize_ratio_token(value: object) -> str:
    s = clean_text(value)
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9:/.%+\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_laga(value: object) -> str:
    s = clean_text(value)
    if not s:
        return ""
    scalar = parse_ratio_scalar(s)
    if scalar is None:
        return normalize_ratio_token(s)
    if scalar <= 0:
        return ""
    frac = Fraction(scalar).limit_denominator(1000)
    return f"{frac.numerator}:{frac.denominator}"


def parse_mw_kda(value: object) -> Optional[float]:
    s = clean_text(value)
    if not s:
        return None
    s_nocomma = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s_nocomma)
    if not m:
        return None
    v = float(m.group(0))
    if "mda" in s_nocomma:
        return v * 1000.0
    if "kda" in s_nocomma or "kg/mol" in s_nocomma:
        return v
    if re.search(r"(?<!k)da\b", s_nocomma) or "g/mol" in s_nocomma:
        return v / 1000.0
    if v > 1000.0:
        return v / 1000.0
    return v


@dataclass
class Row:
    doi: str
    ee: Optional[float]
    dp_raw: str
    dp_num: Optional[float]
    laga_norm: str
    mw_kda: Optional[float]
    loading_proxy: Optional[float]


def to_rows_human(df: pd.DataFrame) -> List[Row]:
    out: List[Row] = []
    for _, r in df.iterrows():
        out.append(
            Row(
                doi=str(r.get("doi_norm", "")),
                ee=parse_first_float(r.get("EE")),
                dp_raw=clean_text(r.get("drug/polymer")),
                dp_num=parse_ratio_scalar(r.get("drug/polymer")),
                laga_norm=normalize_laga(r.get("LA/GA")),
                mw_kda=parse_mw_kda(r.get("polymer_MW")),
                loading_proxy=parse_first_float(r.get("LC")),
            )
        )
    return out


def to_rows_extracted(df: pd.DataFrame) -> List[Row]:
    out: List[Row] = []
    for _, r in df.iterrows():
        out.append(
            Row(
                doi=str(r.get("doi_norm", "")),
                ee=parse_first_float(r.get("encapsulation_efficiency_percent")),
                dp_raw=clean_text(r.get("drug_feed_amount_text")),
                dp_num=parse_ratio_scalar(r.get("drug_feed_amount_text")),
                laga_norm=normalize_laga(r.get("la_ga_ratio")),
                mw_kda=parse_mw_kda(r.get("plga_mw_kDa")),
                loading_proxy=parse_first_float(r.get("loading_content_percent")),
            )
        )
    return out


def match_l0(h: Row, e: Row) -> bool:
    return h.ee is not None and e.ee is not None and abs(h.ee - e.ee) <= 5.0


def match_l1(h: Row, e: Row) -> bool:
    if not match_l0(h, e):
        return False
    if h.dp_num is not None and e.dp_num is not None:
        denom = max(abs(h.dp_num), abs(e.dp_num), 1e-12)
        return abs(h.dp_num - e.dp_num) / denom <= 0.10
    return h.dp_raw != "" and e.dp_raw != "" and h.dp_raw == e.dp_raw


def match_l2(h: Row, e: Row) -> bool:
    return match_l1(h, e) and h.laga_norm != "" and h.laga_norm == e.laga_norm


def match_l3(h: Row, e: Row) -> bool:
    if not match_l2(h, e):
        return False
    if h.mw_kda is None or e.mw_kda is None:
        return False
    denom = max(abs(h.mw_kda), abs(e.mw_kda), 1e-12)
    return abs(h.mw_kda - e.mw_kda) / denom <= 0.10


def max_bipartite_matching(adj: List[List[int]], right_size: int) -> int:
    match_r = [-1] * right_size

    def dfs(u: int, seen: List[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    result = 0
    for u in range(len(adj)):
        seen = [False] * right_size
        if dfs(u, seen):
            result += 1
    return result


def matching_counts_by_doi(
    h_rows: List[Row],
    e_rows: List[Row],
    predicate,
) -> Tuple[int, Dict[str, int]]:
    by_doi_h: Dict[str, List[Row]] = {}
    by_doi_e: Dict[str, List[Row]] = {}
    for r in h_rows:
        by_doi_h.setdefault(r.doi, []).append(r)
    for r in e_rows:
        by_doi_e.setdefault(r.doi, []).append(r)

    total = 0
    per_doi: Dict[str, int] = {}
    for doi in sorted(set(by_doi_h) | set(by_doi_e)):
        hs = by_doi_h.get(doi, [])
        es = by_doi_e.get(doi, [])
        if not hs or not es:
            per_doi[doi] = 0
            continue
        adj: List[List[int]] = []
        for h in hs:
            nbrs = [j for j, e in enumerate(es) if predicate(h, e)]
            adj.append(nbrs)
        m = max_bipartite_matching(adj, len(es))
        per_doi[doi] = m
        total += m
    return total, per_doi


def availability_rate(rows: Sequence[Row], field: str) -> float:
    if not rows:
        return 0.0
    if field == "EE":
        n = sum(1 for r in rows if r.ee is not None)
    elif field == "PLGA_MW":
        n = sum(1 for r in rows if r.mw_kda is not None)
    elif field == "LA_GA":
        n = sum(1 for r in rows if r.laga_norm != "")
    elif field == "drug_polymer":
        n = sum(1 for r in rows if (r.dp_num is not None or r.dp_raw != ""))
    elif field == "drug_polymer_or_loading_proxy":
        n = sum(1 for r in rows if (r.dp_num is not None or r.dp_raw != "" or r.loading_proxy is not None))
    else:
        raise ValueError(field)
    return n / len(rows)


def pair_joint_availability(
    h_rows: List[Row],
    e_rows: List[Row],
    field: str,
) -> float:
    by_doi_h: Dict[str, List[Row]] = {}
    by_doi_e: Dict[str, List[Row]] = {}
    for r in h_rows:
        by_doi_h.setdefault(r.doi, []).append(r)
    for r in e_rows:
        by_doi_e.setdefault(r.doi, []).append(r)

    joint = 0
    total = 0
    for doi in sorted(set(by_doi_h) | set(by_doi_e)):
        hs = by_doi_h.get(doi, [])
        es = by_doi_e.get(doi, [])
        if not hs or not es:
            continue
        for h in hs:
            for e in es:
                total += 1
                if field == "EE":
                    ok = h.ee is not None and e.ee is not None
                elif field == "PLGA_MW":
                    ok = h.mw_kda is not None and e.mw_kda is not None
                elif field == "LA_GA":
                    ok = h.laga_norm != "" and e.laga_norm != ""
                elif field == "drug_polymer":
                    ok = (h.dp_num is not None or h.dp_raw != "") and (e.dp_num is not None or e.dp_raw != "")
                elif field == "drug_polymer_or_loading_proxy":
                    h_ok = h.dp_num is not None or h.dp_raw != "" or h.loading_proxy is not None
                    e_ok = e.dp_num is not None or e.dp_raw != "" or e.loading_proxy is not None
                    ok = h_ok and e_ok
                else:
                    raise ValueError(field)
                if ok:
                    joint += 1
    return (joint / total) if total else 0.0


def doi_level_joint_availability(
    h_rows: List[Row],
    e_rows: List[Row],
    field: str,
) -> float:
    by_doi_h: Dict[str, List[Row]] = {}
    by_doi_e: Dict[str, List[Row]] = {}
    for r in h_rows:
        by_doi_h.setdefault(r.doi, []).append(r)
    for r in e_rows:
        by_doi_e.setdefault(r.doi, []).append(r)
    dois = sorted(set(by_doi_h) | set(by_doi_e))
    if not dois:
        return 0.0

    def has_value(rows: List[Row], f: str) -> bool:
        if f == "EE":
            return any(r.ee is not None for r in rows)
        if f == "PLGA_MW":
            return any(r.mw_kda is not None for r in rows)
        if f == "LA_GA":
            return any(r.laga_norm != "" for r in rows)
        if f == "drug_polymer":
            return any(r.dp_num is not None or r.dp_raw != "" for r in rows)
        if f == "drug_polymer_or_loading_proxy":
            return any(r.dp_num is not None or r.dp_raw != "" or r.loading_proxy is not None for r in rows)
        raise ValueError(f)

    n_joint = 0
    for doi in dois:
        if has_value(by_doi_h.get(doi, []), field) and has_value(by_doi_e.get(doi, []), field):
            n_joint += 1
    return n_joint / len(dois)


def core_signature(row: Row) -> str:
    mw = "" if row.mw_kda is None else f"{row.mw_kda:.6g}"
    laga = row.laga_norm
    if row.dp_num is not None:
        dp = f"{row.dp_num:.6g}"
    else:
        dp = normalize_ratio_token(row.dp_raw)
    dp_part = f"|dp={dp}" if dp else ""
    return f"mw={mw}|laga={laga}{dp_part}"


def build_report(
    out_md: Path,
    run_id: str,
    n_doi: int,
    h_count: int,
    e_count: int,
    availability_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    per_doi_df: pd.DataFrame,
    top5_dup_df: pd.DataFrame,
) -> None:
    top5_ratio = per_doi_df.sort_values(["E_H_ratio", "doi_norm"], ascending=[False, True]).head(5)
    top5_gap = per_doi_df.sort_values(["H_minus_M_L0", "doi_norm"], ascending=[False, True]).head(5)
    m_l0 = int(ladder_df.loc[ladder_df["level"] == "L0", "M"].iloc[0]) if (ladder_df["level"] == "L0").any() else 0
    m_l3 = int(ladder_df.loc[ladder_df["level"] == "L3", "M"].iloc[0]) if (ladder_df["level"] == "L3").any() else 0
    strictness_drop = (m_l0 - m_l3) / max(m_l0, 1)
    mw_missing = float(
        availability_df.loc[availability_df["field"] == "PLGA_MW", "extracted_missing_rate"].iloc[0]
    ) if (availability_df["field"] == "PLGA_MW").any() else 0.0
    top_dup = float(top5_dup_df["duplication_factor"].max()) if not top5_dup_df.empty else 0.0

    lines: List[str] = []
    lines.append("# Boundary-Alignment Diagnostics Pack (Frozen DEV 15 DOIs)")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- n_DEV_DOIs: `{n_doi}`")
    lines.append(f"- human_formulations (H): `{h_count}`")
    lines.append(f"- extracted_formulations (E): `{e_count}`")
    lines.append("- extracted source: `confidence_tiers__formulation_level.tsv` joined to `weak_labels__gemini.tsv` via `(key, formulation_id)`")
    lines.append("")
    lines.append("## 1) Field Availability Audit")
    lines.append("")
    lines.append("| field | human_total | human_missing | human_missing_rate | extracted_total | extracted_missing | extracted_missing_rate | joint_availability_rate_by_doi |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in availability_df.itertuples(index=False):
        lines.append(
            f"| {r.field} | {int(r.human_total)} | {int(r.human_missing)} | {float(r.human_missing_rate):.4f} | {int(r.extracted_total)} | {int(r.extracted_missing)} | {float(r.extracted_missing_rate):.4f} | {float(r.joint_availability_rate_by_doi):.4f} |"
        )
    lines.append("")
    lines.append("## 2) Relaxed Matching Ladder")
    lines.append("")
    lines.append("Trend note: added constraints reduced M from L0 to L3 in this run.")
    lines.append("")
    lines.append("| level | definition | M | precision | recall |")
    lines.append("|---|---|---:|---:|---:|")
    for r in ladder_df.itertuples(index=False):
        lines.append(f"| {r.level} | {r.definition} | {int(r.M)} | {float(r.precision):.4f} | {float(r.recall):.4f} |")
    lines.append("")
    lines.append("## 3) Per-DOI Mismatch Summary")
    lines.append("")
    lines.append("Top 5 DOIs by E/H ratio:")
    lines.append("")
    lines.append("| doi_norm | H_count | E_count | E_H_ratio | M_L0 | M_L3 | H_minus_M_L0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in top5_ratio.itertuples(index=False):
        lines.append(
            f"| {r.doi_norm} | {int(r.H_count)} | {int(r.E_count)} | {float(r.E_H_ratio):.4f} | {int(r.M_L0)} | {int(r.M_L3)} | {int(r.H_minus_M_L0)} |"
        )
    lines.append("")
    lines.append("Top 5 DOIs by (H - M_L0):")
    lines.append("")
    lines.append("| doi_norm | H_count | E_count | E_H_ratio | M_L0 | M_L3 | H_minus_M_L0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in top5_gap.itertuples(index=False):
        lines.append(
            f"| {r.doi_norm} | {int(r.H_count)} | {int(r.E_count)} | {float(r.E_H_ratio):.4f} | {int(r.M_L0)} | {int(r.M_L3)} | {int(r.H_minus_M_L0)} |"
        )
    lines.append("")
    lines.append("## 4) Over-Segmentation Analysis (Extracted)")
    lines.append("")
    lines.append("| doi_norm | total_extracted_instances | unique_core_signatures | duplication_factor |")
    lines.append("|---|---:|---:|---:|")
    for r in top5_dup_df.itertuples(index=False):
        lines.append(
            f"| {r.doi_norm} | {int(r.total_extracted_instances)} | {int(r.unique_core_signatures)} | {float(r.duplication_factor):.4f} |"
        )
    lines.append("")
    lines.append("## 5) Primary Mismatch Drivers")
    lines.append("")
    lines.append(f"- signature strictness: `{'high' if strictness_drop >= 0.8 else 'moderate'}` (L0->L3 drop = `{strictness_drop:.4f}`)")
    lines.append(f"- over-segmentation: `{'high' if top_dup >= 3.0 else 'moderate'}` (max duplication factor = `{top_dup:.4f}`)")
    lines.append(f"- missing-field noise: `{'high' if mw_missing >= 0.5 else 'moderate'}` (extracted PLGA_MW missing rate = `{mw_missing:.4f}`)")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `field_availability_audit.tsv`")
    lines.append("- `matching_ladder_summary.tsv`")
    lines.append("- `per_doi_counts.tsv`")
    lines.append("- `extracted_core_signature_clusters_top.tsv`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path.cwd()

    run_dir = repo / "data" / "results" / args.run_id
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    human_path = repo / args.human_csv
    conf_path = run_dir / "confidence_tiers__formulation_level.tsv"
    weak_path = run_dir / "step1_dev" / "weak_labels__gemini.tsv"
    manifest_path = run_dir / "step1_dev" / "dev_manifest_v1.tsv"

    human = pd.read_csv(human_path, dtype=str)
    conf = pd.read_csv(conf_path, sep="\t", dtype=str)
    weak = pd.read_csv(weak_path, sep="\t", dtype=str)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)

    manifest["doi_norm"] = manifest["doi"].map(norm_doi)
    dev_dois = sorted(set(manifest["doi_norm"]))
    key_to_doi = dict(zip(manifest["key"].astype(str), manifest["doi_norm"]))

    human["doi_norm"] = human["reference"].map(norm_doi)
    human_dev = human[human["doi_norm"].isin(dev_dois)].copy()

    needed_cols = [
        "key",
        "formulation_id",
        "la_ga_ratio",
        "plga_mw_kDa",
        "drug_feed_amount_text",
        "encapsulation_efficiency_percent",
        "loading_content_percent",
    ]
    extracted = conf[["key", "formulation_id"]].merge(
        weak[needed_cols],
        on=["key", "formulation_id"],
        how="left",
    )
    extracted["doi_norm"] = extracted["key"].astype(str).map(key_to_doi).fillna("")
    extracted_dev = extracted[extracted["doi_norm"].isin(dev_dois)].copy()

    h_rows = to_rows_human(human_dev)
    e_rows = to_rows_extracted(extracted_dev)
    h_count = len(h_rows)
    e_count = len(e_rows)

    fields = ["EE", "PLGA_MW", "LA_GA", "drug_polymer", "drug_polymer_or_loading_proxy"]
    availability_rows: List[Dict[str, object]] = []
    for f in fields:
        h_total = len(h_rows)
        e_total = len(e_rows)
        h_av = availability_rate(h_rows, f)
        e_av = availability_rate(e_rows, f)
        joint = doi_level_joint_availability(h_rows, e_rows, f)
        availability_rows.append(
            {
                "field": f,
                "human_total": h_total,
                "human_missing": int(round((1.0 - h_av) * h_total)),
                "human_missing_rate": 1.0 - h_av,
                "extracted_total": e_total,
                "extracted_missing": int(round((1.0 - e_av) * e_total)),
                "extracted_missing_rate": 1.0 - e_av,
                "joint_availability_rate_by_doi": joint,
            }
        )
    availability_df = pd.DataFrame(availability_rows)

    levels = [
        ("L0", "DOI + EE (+/-5 pp)", match_l0),
        (
            "L1",
            "L0 + drug/polymer (numeric rel tol 10%, else exact normalized token)",
            match_l1,
        ),
        ("L2", "L1 + LA:GA (normalized ratio)", match_l2),
        ("L3", "L2 + PLGA MW (Da<->kDa normalized, rel tol 10%)", match_l3),
    ]

    ladder_rows: List[Dict[str, object]] = []
    per_level_doi: Dict[str, Dict[str, int]] = {}
    for level, definition, pred in levels:
        m_total, m_per_doi = matching_counts_by_doi(h_rows, e_rows, pred)
        per_level_doi[level] = m_per_doi
        ladder_rows.append(
            {
                "level": level,
                "definition": definition,
                "H": h_count,
                "E": e_count,
                "M": m_total,
                "precision": (m_total / e_count) if e_count else math.nan,
                "recall": (m_total / h_count) if h_count else math.nan,
            }
        )
    ladder_df = pd.DataFrame(ladder_rows)

    # Per-DOI counts
    h_doi_counts = human_dev.groupby("doi_norm").size().to_dict()
    e_doi_counts = extracted_dev.groupby("doi_norm").size().to_dict()
    per_doi_rows: List[Dict[str, object]] = []
    for doi in dev_dois:
        h_n = int(h_doi_counts.get(doi, 0))
        e_n = int(e_doi_counts.get(doi, 0))
        m_l0 = int(per_level_doi["L0"].get(doi, 0))
        m_l3 = int(per_level_doi["L3"].get(doi, 0))
        per_doi_rows.append(
            {
                "doi_norm": doi,
                "H_count": h_n,
                "E_count": e_n,
                "E_H_ratio": (e_n / h_n) if h_n else math.nan,
                "M_L0": m_l0,
                "M_L3": m_l3,
                "H_minus_M_L0": h_n - m_l0,
            }
        )
    per_doi_df = pd.DataFrame(per_doi_rows).sort_values("doi_norm")

    # Over-segmentation clustering
    extracted_dev = extracted_dev.copy()
    extracted_dev["mw_kda_num"] = extracted_dev["plga_mw_kDa"].map(parse_mw_kda)
    extracted_dev["mw_norm_kda"] = extracted_dev["mw_kda_num"].map(
        lambda x: "" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.6g}"
    )
    extracted_dev["laga_norm"] = extracted_dev["la_ga_ratio"].map(normalize_laga)
    extracted_dev["dp_num"] = extracted_dev["drug_feed_amount_text"].map(parse_ratio_scalar)
    extracted_dev["dp_norm"] = extracted_dev.apply(
        lambda r: f"{r['dp_num']:.6g}" if pd.notna(r["dp_num"]) else normalize_ratio_token(r["drug_feed_amount_text"]),
        axis=1,
    )
    extracted_dev["core_signature"] = extracted_dev.apply(
        lambda r: f"mw={r['mw_norm_kda']}|laga={r['laga_norm']}"
        + (f"|dp={r['dp_norm']}" if str(r["dp_norm"]).strip() else ""),
        axis=1,
    )

    doi_totals = extracted_dev.groupby("doi_norm").size().rename("total_extracted_in_doi")
    doi_unique = extracted_dev.groupby("doi_norm")["core_signature"].nunique().rename("unique_core_signatures_in_doi")
    cluster = (
        extracted_dev.groupby(["doi_norm", "core_signature"], as_index=False)
        .size()
        .rename(columns={"size": "cluster_size"})
        .merge(doi_totals, on="doi_norm", how="left")
        .merge(doi_unique, on="doi_norm", how="left")
    )
    cluster["duplication_factor_in_doi"] = (
        cluster["total_extracted_in_doi"] / cluster["unique_core_signatures_in_doi"]
    )
    cluster["share_within_doi"] = cluster["cluster_size"] / cluster["total_extracted_in_doi"]
    cluster = cluster.sort_values(
        ["duplication_factor_in_doi", "cluster_size", "doi_norm", "core_signature"],
        ascending=[False, False, True, True],
    )
    per_doi_cluster = (
        cluster.groupby("doi_norm", as_index=False)
        .agg(
            total_extracted_instances=("total_extracted_in_doi", "max"),
            unique_core_signatures=("unique_core_signatures_in_doi", "max"),
            duplication_factor=("duplication_factor_in_doi", "max"),
        )
        .sort_values(["duplication_factor", "doi_norm"], ascending=[False, True])
    )
    cluster_top = per_doi_cluster.head(5).copy()

    # Write outputs
    availability_path = out_dir / "field_availability_audit.tsv"
    ladder_path = out_dir / "matching_ladder_summary.tsv"
    per_doi_path = out_dir / "per_doi_counts.tsv"
    cluster_path = out_dir / "extracted_core_signature_clusters_top.tsv"
    report_path = out_dir / "boundary_alignment_diagnostics_report.md"

    availability_df.to_csv(availability_path, sep="\t", index=False)
    ladder_df.to_csv(ladder_path, sep="\t", index=False)
    per_doi_df.to_csv(per_doi_path, sep="\t", index=False)
    cluster_top.to_csv(cluster_path, sep="\t", index=False)
    build_report(
        out_md=report_path,
        run_id=args.run_id,
        n_doi=len(dev_dois),
        h_count=h_count,
        e_count=e_count,
        availability_df=availability_df,
        ladder_df=ladder_df,
        per_doi_df=per_doi_df,
        top5_dup_df=cluster_top,
    )

    print(f"out_dir={out_dir}")
    print(f"H={h_count}")
    print(f"E={e_count}")
    for row in ladder_rows:
        print(
            f"{row['level']}: M={row['M']} precision={row['precision']:.6f} recall={row['recall']:.6f}"
        )


if __name__ == "__main__":
    main()
