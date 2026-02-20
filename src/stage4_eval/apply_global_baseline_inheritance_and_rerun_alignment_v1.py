#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


GLOBAL_FIELDS = [
    "solvent",
    "surfactant_name",
    "surfactant_concentration",
    "polymer_MW",
    "LA/GA",
    "aqueous/organic",
    "pH",
]


def norm_doi(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    if not s:
        return ""
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def clean_val(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def is_empty(v: object) -> bool:
    if pd.isna(v):
        return True
    s = str(v).strip().lower()
    return s in {"", "na", "n/a", "nan", "none", "null"}


def find_global_window(text: str) -> Tuple[Optional[int], Optional[int], str, str]:
    cues = [
        "All formulations",
        "All nanoparticles",
        "Nanoparticles were prepared",
        "were prepared by",
        "were prepared using",
        "The organic phase",
        "The aqueous phase",
    ]
    low = text.lower()
    for cue in cues:
        m = re.search(re.escape(cue.lower()), low)
        if m:
            c = m.start()
            return max(0, c - 1200), min(len(text), c + 1200), "cue_window", cue

    # Fallback: Methods/Materials section start + 2500 chars
    sec = re.search(
        r"(?im)^\s*(materials and methods|materials\s*&\s*methods|methods|methodology|materials)\b",
        text,
    )
    if sec:
        s = sec.start()
        return s, min(len(text), s + 2500), "methods_fallback", sec.group(1)
    return None, None, "none", ""


def extract_candidates(patterns: List[re.Pattern], text: str) -> List[str]:
    out: List[str] = []
    for p in patterns:
        for m in p.finditer(text):
            g = m.group(1) if m.groups() else m.group(0)
            v = str(g).strip()
            if v:
                out.append(v)
    return out


def pick_single_or_conflict(cands: List[str]) -> Tuple[str, bool]:
    normed = []
    for c in cands:
        cn = re.sub(r"\s+", " ", c.strip())
        if cn:
            normed.append(cn)
    uniq = sorted(set(normed), key=lambda x: x.lower())
    if len(uniq) == 0:
        return "", False
    if len(uniq) == 1:
        return uniq[0], False
    return "", True


def extract_global_fields(window_text: str) -> Tuple[Dict[str, str], List[str]]:
    # conservative regexes
    solvent_patterns = [
        re.compile(
            r"\b(dichloromethane|methylene chloride|chloroform|acetone|ethyl acetate|acetonitrile|dmf|dms[o0])\b",
            re.IGNORECASE,
        )
    ]
    surfactant_patterns = [
        re.compile(r"\b(PVA|poloxamer\s*407|pluronic\s*f[-\s]?68|tween\s*80|span\s*80|sodium cholate)\b", re.IGNORECASE)
    ]
    surf_conc_patterns = [
        re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:\(w/v\)|w/v)?", re.IGNORECASE)
    ]
    polymer_mw_patterns = [
        re.compile(r"\b(?:PLGA|poly\(d,l-lactide-co-glycolide\)).{0,30}?(\d+(?:\.\d+)?)\s*(?:kda|kDa|Da)\b", re.IGNORECASE),
        re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:kda|kDa)\b", re.IGNORECASE),
    ]
    laga_patterns = [
        re.compile(r"\b(\d{1,2}\s*[:/]\s*\d{1,2})\b")
    ]
    aq_org_patterns = [
        re.compile(r"\baqueous\s*/\s*organic\b.{0,20}?(\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"\b(\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?)\b")
    ]
    ph_patterns = [
        re.compile(r"\bpH\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
    ]

    values: Dict[str, str] = {}
    conflicts: List[str] = []

    m = [x.lower() for x in extract_candidates(solvent_patterns, window_text)]
    v, cf = pick_single_or_conflict(m)
    values["solvent"] = v
    if cf:
        conflicts.append("solvent")

    m = [x for x in extract_candidates(surfactant_patterns, window_text)]
    v, cf = pick_single_or_conflict(m)
    values["surfactant_name"] = v
    if cf:
        conflicts.append("surfactant_name")

    m = extract_candidates(surf_conc_patterns, window_text)
    v, cf = pick_single_or_conflict(m)
    values["surfactant_concentration"] = v
    if cf:
        conflicts.append("surfactant_concentration")

    m = extract_candidates(polymer_mw_patterns, window_text)
    v, cf = pick_single_or_conflict(m)
    values["polymer_MW"] = v
    if cf:
        conflicts.append("polymer_MW")

    m = extract_candidates(laga_patterns, window_text)
    v, cf = pick_single_or_conflict(m)
    values["LA/GA"] = v
    if cf:
        conflicts.append("LA/GA")

    m = extract_candidates(aq_org_patterns, window_text)
    v, cf = pick_single_or_conflict(m)
    values["aqueous/organic"] = v
    if cf:
        conflicts.append("aqueous/organic")

    m = extract_candidates(ph_patterns, window_text)
    v, cf = pick_single_or_conflict(m)
    values["pH"] = v
    if cf:
        conflicts.append("pH")

    return values, conflicts


def run_alignment(curated: pd.DataFrame, extracted: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    curated_fields = [
        "small_molecule_name",
        "polymer_MW",
        "LA/GA",
        "surfactant_name",
        "solvent",
        "drug/polymer",
        "surfactant_concentration",
        "aqueous/organic",
        "pH",
    ]
    ex_field_map: Dict[str, str] = {
        "small_molecule_name": "sig__drug_name",
        "polymer_MW": "sig__polymer_MW",
        "LA/GA": "sig__LA/GA",
        "surfactant_name": "sig__surfactant_name",
        "solvent": "sig__solvent",
        "drug/polymer": "sig__drug/polymer",
        "surfactant_concentration": "sig__surfactant_concentration",
        "aqueous/organic": "sig__aqueous/organic",
        "pH": "sig__pH",
    }

    cu = curated.copy()
    ex = extracted.copy()
    cu["doi_norm"] = cu["doi_norm"].map(norm_doi)
    ex["doi_norm"] = ex["doi_norm"].map(norm_doi)

    cu["ee_num"] = pd.to_numeric(cu["EE"], errors="coerce")
    ex["ee_num"] = pd.to_numeric(ex["group_mean_EE"], errors="coerce")

    for cf in curated_fields:
        cu[f"_k_{cf}"] = cu[cf].map(clean_val) if cf in cu.columns else ""
    for cf, ef in ex_field_map.items():
        ex[f"_k_{cf}"] = ex[ef].map(clean_val) if ef in ex.columns else ""

    align_rows: List[Dict[str, object]] = []
    matched_extracted_keys: set[Tuple[str, str]] = set()
    per_doi_counts: Dict[str, Dict[str, int]] = {}

    for doi, cu_d in cu.groupby("doi_norm", dropna=False):
        doi = str(doi)
        ex_d = ex[ex["doi_norm"] == doi].copy()
        total = 0
        matched = 0
        for c_row in cu_d.itertuples(index=False):
            c = c_row._asdict()
            c_ee = c.get("ee_num")
            total += 1

            best_sig = ""
            best_diff = float("nan")
            is_matched = False

            if pd.notna(c_ee) and len(ex_d) > 0:
                cand = ex_d.copy()
                for field in curated_fields:
                    cv = c.get(f"_k_{field}", "")
                    if cv:
                        cand = cand[cand[f"_k_{field}"] == cv]
                if len(cand) > 0:
                    cand = cand[cand["ee_num"].notna()].copy()
                    if len(cand) > 0:
                        cand["abs_diff"] = (cand["ee_num"] - float(c_ee)).abs()
                        cand = cand[cand["abs_diff"] <= 5.0]
                        if len(cand) > 0:
                            best = cand.sort_values("abs_diff", ascending=True).iloc[0]
                            best_sig = str(best.get("formulation_signature", ""))
                            best_diff = float(best["abs_diff"])
                            is_matched = True
                            matched_extracted_keys.add((doi, best_sig))

            if is_matched:
                matched += 1
            align_rows.append(
                {
                    "doi_norm": doi,
                    "curated_signature": " | ".join(f"{f}={c.get(f'_k_{f}', '')}" for f in curated_fields),
                    "matched": bool(is_matched),
                    "matched_extracted_signature": best_sig,
                    "ee_abs_diff": best_diff,
                }
            )

        per_doi_counts[doi] = {"matched": matched, "total": total}

    alignment = pd.DataFrame(align_rows)
    matched_curated = int(alignment["matched"].astype(bool).sum())
    total_curated = int(len(alignment))
    total_extracted = int(len(ex))
    matched_extracted = int(len(matched_extracted_keys))

    recall = (matched_curated / total_curated) if total_curated else float("nan")
    precision = (matched_extracted / total_extracted) if total_extracted else float("nan")

    per_doi_rows = []
    for doi in sorted(per_doi_counts.keys()):
        m = per_doi_counts[doi]["matched"]
        t = per_doi_counts[doi]["total"]
        per_doi_rows.append(
            {
                "doi_norm": doi,
                "matched_curated_formulations": m,
                "total_curated_formulations": t,
                "per_DOI_recall": (m / t) if t else float("nan"),
            }
        )
    per_doi_df = pd.DataFrame(per_doi_rows)

    summary = pd.DataFrame(
        [
            {"metric": "matched_curated_formulations", "value": matched_curated},
            {"metric": "total_curated_formulations", "value": total_curated},
            {"metric": "matched_extracted_formulations", "value": matched_extracted},
            {"metric": "total_extracted_formulations", "value": total_extracted},
            {"metric": "formulation_recall", "value": recall},
            {"metric": "formulation_precision", "value": precision},
        ]
    )
    return alignment, summary, per_doi_df, recall, precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply global baseline inheritance and rerun alignment (evaluation-only).")
    parser.add_argument(
        "--extracted-formulation-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/extracted_formulation_level_v1.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--text-root",
        default="data/cleaned/content_goren_2025/text",
    )
    parser.add_argument(
        "--key2txt",
        default="data/cleaned/index/key2txt_goren_2025.tsv",
    )
    parser.add_argument(
        "--sample-jsonl",
        default="data/cleaned/samples/sample_goren18.jsonl",
    )
    parser.add_argument(
        "--baseline-summary-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/formulation_alignment_summary.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/global_inherit_v1",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    ex_p = (cwd / args.extracted_formulation_tsv).resolve()
    cu_p = (cwd / args.curated_tsv).resolve()
    text_root = (cwd / args.text_root).resolve()
    key2txt_p = (cwd / args.key2txt).resolve()
    sample_p = (cwd / args.sample_jsonl).resolve()
    baseline_summary_p = (cwd / args.baseline_summary_tsv).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(ex_p, sep="\t", dtype=str)
    cu = pd.read_csv(cu_p, sep="\t", dtype=str)
    baseline_summary = pd.read_csv(baseline_summary_p, sep="\t", dtype=str)

    # key<->doi maps
    key2doi: Dict[str, str] = {}
    doi2keys: Dict[str, List[str]] = {}
    with sample_p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = str(obj.get("key", "")).strip()
            d = norm_doi(obj.get("doi", ""))
            if not k or not d:
                continue
            key2doi[k] = d
            doi2keys.setdefault(d, []).append(k)

    # key -> text path (headerless key2txt)
    key2txt_df = pd.read_csv(key2txt_p, sep="\t", dtype=str, header=None)
    key2txt_map = {str(r[0]).strip(): str(r[1]).strip() for _, r in key2txt_df.iterrows() if len(r) >= 2}

    # Build per-DOI global baseline
    doi_list = sorted(set(ex["doi_norm"].map(norm_doi).dropna().tolist()) | set(doi2keys.keys()))
    baseline_rows: List[Dict[str, object]] = []
    baseline_map: Dict[str, Dict[str, str]] = {}
    coverage_rows: List[Dict[str, object]] = []
    cue_hits = 0
    fallback_hits = 0
    no_window = 0

    for doi in doi_list:
        keys = doi2keys.get(doi, [])
        text_path = None
        for k in keys:
            rel = key2txt_map.get(k, "")
            if not rel:
                continue
            p = (cwd / rel).resolve()
            if p.exists():
                text_path = p
                break
            # fallback to text_root/<key>.pdf.txt
            p2 = text_root / f"{k}.pdf.txt"
            if p2.exists():
                text_path = p2
                break

        row: Dict[str, object] = {
            "doi_norm": doi,
            "cue_phrase": "",
            "window_strategy": "",
            "window_start": "",
            "window_end": "",
            "conflict_fields": "",
            "text_path": str(text_path) if text_path else "",
        }
        for f in GLOBAL_FIELDS:
            row[f] = ""

        if text_path is None:
            row["window_strategy"] = "missing_text"
            no_window += 1
            baseline_rows.append(row)
            baseline_map[doi] = {f: "" for f in GLOBAL_FIELDS}
            continue

        txt = text_path.read_text(encoding="utf-8", errors="replace")
        s, e, strategy, cue = find_global_window(txt)
        row["window_strategy"] = strategy
        row["cue_phrase"] = cue
        if strategy == "cue_window":
            cue_hits += 1
        elif strategy == "methods_fallback":
            fallback_hits += 1
        else:
            no_window += 1

        if s is None or e is None:
            baseline_rows.append(row)
            baseline_map[doi] = {f: "" for f in GLOBAL_FIELDS}
            continue

        row["window_start"] = int(s)
        row["window_end"] = int(e)
        win = txt[s:e]
        vals, conflicts = extract_global_fields(win)
        for f in GLOBAL_FIELDS:
            row[f] = vals.get(f, "")
        row["conflict_fields"] = ";".join(conflicts)
        baseline_rows.append(row)
        baseline_map[doi] = vals

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_out = out_dir / "global_baseline_per_doi.tsv"
    baseline_df.to_csv(baseline_out, sep="\t", index=False)

    # Apply inheritance to extracted formulation table
    ex2 = ex.copy()
    if "doi_norm" not in ex2.columns:
        raise RuntimeError("Extracted formulation table must contain doi_norm.")
    ex2["doi_norm"] = ex2["doi_norm"].map(norm_doi)

    target_col_map = {
        "solvent": "sig__solvent",
        "surfactant_name": "sig__surfactant_name",
        "surfactant_concentration": "sig__surfactant_concentration",
        "polymer_MW": "sig__polymer_MW",
        "LA/GA": "sig__LA/GA",
        "aqueous/organic": "sig__aqueous/organic",
        "pH": "sig__pH",
    }
    for logical, col in target_col_map.items():
        if col not in ex2.columns:
            ex2[col] = ""
        prov_col = f"{logical}__inherited_from"
        ex2[prov_col] = "none"

    cov: Dict[str, Dict[str, int]] = {}
    for idx, r in ex2.iterrows():
        doi = r["doi_norm"]
        if doi not in cov:
            cov[doi] = {"rows_changed": 0, "fields_filled": 0}
            for f in GLOBAL_FIELDS:
                cov[doi][f] = 0
        row_changed = False
        base_vals = baseline_map.get(doi, {})
        for logical, col in target_col_map.items():
            base_v = str(base_vals.get(logical, "") or "").strip()
            if not base_v:
                continue
            if is_empty(r.get(col, "")):
                ex2.at[idx, col] = base_v
                ex2.at[idx, f"{logical}__inherited_from"] = "global"
                cov[doi]["fields_filled"] += 1
                cov[doi][logical] += 1
                row_changed = True
        if row_changed:
            cov[doi]["rows_changed"] += 1

    # Recompute formulation signature from sig__ columns
    sig_order = [
        "drug_name",
        "polymer_MW",
        "LA/GA",
        "solvent",
        "surfactant_name",
        "drug/polymer",
        "surfactant_concentration",
        "aqueous/organic",
        "pH",
    ]
    for logical in sig_order:
        col = f"sig__{logical}"
        if col not in ex2.columns:
            ex2[col] = ""
    ex2["formulation_signature"] = ex2.apply(
        lambda rr: " | ".join(f"{f}={str(rr.get(f'sig__{f}', '')).strip().lower()}" for f in sig_order),
        axis=1,
    )

    ex2_out = out_dir / "extracted_formulation_level__global_inherit_v1.tsv"
    ex2.to_csv(ex2_out, sep="\t", index=False)

    cov_rows = []
    for doi in sorted(cov.keys()):
        row = {"doi_norm": doi, "rows_changed": cov[doi]["rows_changed"], "fields_filled": cov[doi]["fields_filled"]}
        for f in GLOBAL_FIELDS:
            row[f"{f}_filled"] = cov[doi][f]
        cov_rows.append(row)
    cov_df = pd.DataFrame(cov_rows)
    cov_out = out_dir / "global_inherit_coverage.tsv"
    cov_df.to_csv(cov_out, sep="\t", index=False)

    # rerun alignment with same rules
    alignment_df, summary_df, per_doi_df, new_recall, new_precision = run_alignment(cu, ex2)
    align_out = out_dir / "formulation_alignment__global_inherit_v1.tsv"
    summary_out = out_dir / "formulation_alignment_summary__global_inherit_v1.tsv"
    per_doi_out = out_dir / "per_doi_recall__global_inherit_v1.tsv"
    alignment_df.to_csv(align_out, sep="\t", index=False)
    summary_df.to_csv(summary_out, sep="\t", index=False)
    per_doi_df.to_csv(per_doi_out, sep="\t", index=False)

    # build log
    log = {
        "inputs": {
            "extracted_formulation_tsv": str(ex_p),
            "curated_tsv": str(cu_p),
            "text_root": str(text_root),
            "key2txt_tsv": str(key2txt_p),
            "sample_jsonl": str(sample_p),
            "baseline_summary_tsv": str(baseline_summary_p),
        },
        "counts": {
            "doi_total_considered": len(doi_list),
            "cue_window_hits": cue_hits,
            "methods_fallback_hits": fallback_hits,
            "no_window_or_missing_text": no_window,
            "rows_extracted_input": int(len(ex)),
            "rows_extracted_output": int(len(ex2)),
            "rows_changed_total": int(cov_df["rows_changed"].sum()) if len(cov_df) else 0,
            "fields_filled_total": int(cov_df["fields_filled"].sum()) if len(cov_df) else 0,
        },
        "outputs": {
            "extracted_formulation_level__global_inherit_v1_tsv": str(ex2_out),
            "global_baseline_per_doi_tsv": str(baseline_out),
            "global_inherit_coverage_tsv": str(cov_out),
            "formulation_alignment__global_inherit_v1_tsv": str(align_out),
            "formulation_alignment_summary__global_inherit_v1_tsv": str(summary_out),
            "per_doi_recall__global_inherit_v1_tsv": str(per_doi_out),
        },
    }
    log_out = out_dir / "inheritance_build_log.json"
    log_out.write_text(json.dumps(log, indent=2), encoding="utf-8")

    # Baseline metrics from existing summary
    base_map = dict(zip(baseline_summary["metric"], baseline_summary["value"]))
    baseline_recall = float(base_map.get("formulation_recall", "nan"))
    baseline_precision = float(base_map.get("formulation_precision", "nan"))
    delta_recall = new_recall - baseline_recall
    delta_precision = new_precision - baseline_precision

    print(f"baseline formulation_recall={baseline_recall}")
    print(f"baseline formulation_precision={baseline_precision}")
    print(f"formulation_recall_global={new_recall}")
    print(f"formulation_precision_global={new_precision}")
    print(f"delta_recall={delta_recall}")
    print(f"delta_precision={delta_precision}")


if __name__ == "__main__":
    main()
