#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd


TRACER_WHITELIST = {
    "coumarin-6",
    "coumarin 6",
    "rhodamine-123",
    "rhodamine 123",
    "rhodamine",
    "nile red",
    "dii",
    "dir",
    "fitc",
}


def norm_doi(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    if not s:
        return ""
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def clean_for_cmp(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_drug_tokens(raw: object) -> Tuple[List[str], List[str]]:
    s = "" if raw is None else str(raw)
    s = s.strip()
    if not s:
        return [], []
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r"\||,|;|\band\b", s, flags=re.I)
    primary: Set[str] = set()
    tracer: Set[str] = set()
    tracer_norm = {clean_for_cmp(x) for x in TRACER_WHITELIST}
    for p in parts:
        tok = clean_for_cmp(p)
        if not tok:
            continue
        if tok in tracer_norm:
            tracer.add(tok)
        else:
            primary.add(tok)
    return sorted(primary), sorted(tracer)


def join_set(vals: Sequence[str]) -> str:
    return " | ".join(sorted(set([v for v in vals if v])))


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def pick_best_baseline_summary(cwd: Path) -> Tuple[str, Path]:
    cands = [
        ("v2_corefields", cwd / "data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v2_corefields/formulation_alignment_summary.tsv"),
        ("v1", cwd / "data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/formulation_alignment_summary.tsv"),
    ]
    for name, p in cands:
        if p.exists():
            return name, p
    raise FileNotFoundError("No baseline formulation_alignment_summary.tsv found (v2_corefields/v1).")


def summary_to_metrics(summary_tsv: Path) -> Tuple[float, float]:
    df = pd.read_csv(summary_tsv, sep="\t", dtype=str)
    m = dict(zip(df["metric"], df["value"]))
    return float(m["formulation_recall"]), float(m["formulation_precision"])


def build_alignment(
    curated: pd.DataFrame,
    extracted_grouped: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
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
    ex = extracted_grouped.copy()
    cu["doi_norm"] = cu["doi_norm"].map(norm_doi)
    ex["doi_norm"] = ex["doi_norm"].map(norm_doi)

    cu["ee_num"] = pd.to_numeric(cu["EE"], errors="coerce")
    ex["ee_num"] = pd.to_numeric(ex["group_mean_EE"], errors="coerce")

    for cf in curated_fields:
        cu[f"_k_{cf}"] = cu[cf].map(clean_for_cmp) if cf in cu.columns else ""
    # canonicalize curated drug and remove tracer tokens
    cu["_k_small_molecule_name"] = cu["small_molecule_name"].map(lambda x: join_set(canonicalize_drug_tokens(x)[0]))

    for cf, ef in ex_field_map.items():
        ex[f"_k_{cf}"] = ex[ef].map(clean_for_cmp) if ef in ex.columns else ""

    align_rows: List[Dict[str, object]] = []
    matched_extracted_keys: Set[Tuple[str, str]] = set()
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

    per_rows = []
    for doi in sorted(per_doi_counts.keys()):
        m = per_doi_counts[doi]["matched"]
        t = per_doi_counts[doi]["total"]
        per_rows.append(
            {
                "doi_norm": doi,
                "matched_curated_formulations": m,
                "total_curated_formulations": t,
                "per_DOI_recall": (m / t) if t else float("nan"),
            }
        )
    per_doi_df = pd.DataFrame(per_rows)
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
    parser = argparse.ArgumentParser(description="Run alignment v3 with surfactant schema + drug-name normalization.")
    parser.add_argument(
        "--extracted-tsv",
        default="data/results/dev18_surfactant_schema_v1/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--sample-jsonl",
        default="data/cleaned/samples/sample_goren18.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v3_surfactant_drugnorm",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    ex_p = (cwd / args.extracted_tsv).resolve()
    cu_p = (cwd / args.curated_tsv).resolve()
    sample_p = (cwd / args.sample_jsonl).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(ex_p, sep="\t", dtype=str).fillna("")
    cu = pd.read_csv(cu_p, sep="\t", dtype=str).fillna("")

    # key->doi
    key2doi: Dict[str, str] = {}
    with sample_p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = str(obj.get("key", "")).strip()
            d = norm_doi(obj.get("doi", ""))
            if k and d:
                key2doi[k] = d

    if "doi_norm" not in ex.columns:
        key_col = "key" if "key" in ex.columns else "zotero_key" if "zotero_key" in ex.columns else None
        if key_col is None:
            raise RuntimeError("Extracted TSV missing doi_norm and key for mapping.")
        ex["doi_norm"] = ex[key_col].map(lambda k: key2doi.get(str(k), ""))
    else:
        ex["doi_norm"] = ex["doi_norm"].map(norm_doi)

    if "doi_norm" not in cu.columns:
        if "reference" in cu.columns:
            cu["doi_norm"] = cu["reference"].map(norm_doi)
        else:
            raise RuntimeError("Curated TSV missing doi_norm/reference.")
    else:
        cu["doi_norm"] = cu["doi_norm"].map(norm_doi)

    # explode extracted rows by canonical primary drug tokens; tracers separated
    exploded_rows: List[Dict[str, Any]] = []
    for _, r in ex.iterrows():
        primary, tracer = canonicalize_drug_tokens(r.get("drug_name", ""))
        prim_tokens = primary if primary else [""]
        tracer_text = join_set(tracer)
        for p in prim_tokens:
            rec = dict(r)
            rec["drug_name_canonical"] = p
            rec["tracer_name"] = tracer_text
            rec["doi_norm"] = norm_doi(rec.get("doi_norm", ""))
            exploded_rows.append(rec)
    ex_exp = pd.DataFrame(exploded_rows)

    # build signature fields
    ex_exp["sig__drug_name"] = ex_exp["drug_name_canonical"].map(clean_for_cmp)
    ex_exp["sig__polymer_MW"] = ex_exp["plga_mw_kDa"].map(clean_for_cmp) if "plga_mw_kDa" in ex_exp.columns else ""
    ex_exp["sig__LA/GA"] = ex_exp["la_ga_ratio"].map(clean_for_cmp) if "la_ga_ratio" in ex_exp.columns else ""
    ex_exp["sig__solvent"] = ex_exp["organic_solvent"].map(clean_for_cmp) if "organic_solvent" in ex_exp.columns else ""
    ex_exp["sig__surfactant_name"] = ex_exp["surfactant_name"].map(clean_for_cmp) if "surfactant_name" in ex_exp.columns else ""
    # keep optional/empty (no new derived logic)
    ex_exp["sig__drug/polymer"] = ""
    ex_exp["sig__surfactant_concentration"] = ex_exp["surfactant_concentration_text"].map(clean_for_cmp) if "surfactant_concentration_text" in ex_exp.columns else ""
    ex_exp["sig__aqueous/organic"] = ""
    ex_exp["sig__pH"] = ""

    sig_fields = [
        "sig__drug_name",
        "sig__polymer_MW",
        "sig__LA/GA",
        "sig__solvent",
        "sig__surfactant_name",
        "sig__drug/polymer",
        "sig__surfactant_concentration",
        "sig__aqueous/organic",
        "sig__pH",
    ]
    ex_exp["formulation_signature"] = ex_exp.apply(
        lambda rr: " | ".join(f"{f}={rr.get(f, '')}" for f in sig_fields),
        axis=1,
    )
    ex_exp["ee_num"] = pd.to_numeric(ex_exp["encapsulation_efficiency_percent"], errors="coerce")

    # formulation-level grouping
    grouped = (
        ex_exp.groupby(["doi_norm", "formulation_signature"] + sig_fields, dropna=False)
        .agg(
            group_size=("formulation_signature", "size"),
            n_unique_ee=("ee_num", lambda s: int(pd.Series(s).dropna().nunique())),
            unique_ee_values=("ee_num", lambda s: ";".join(str(v) for v in sorted(pd.Series(s).dropna().unique()))),
            group_mean_EE=("ee_num", "mean"),
            tracer_name=("tracer_name", lambda s: join_set([x for x in s if str(x).strip()])),
        )
        .reset_index()
    )
    v3_formulation_out = out_dir / "extracted_formulation_level_v3.tsv"
    grouped.to_csv(v3_formulation_out, sep="\t", index=False)

    # run alignment (same matching rule as baseline)
    alignment_df, summary_df, per_doi_df, new_recall, new_precision = build_alignment(cu, grouped)
    alignment_out = out_dir / "formulation_alignment_v3.tsv"
    summary_out = out_dir / "formulation_alignment_summary_v3.tsv"
    per_doi_out = out_dir / "per_doi_recall_v3.tsv"
    alignment_df.to_csv(alignment_out, sep="\t", index=False)
    summary_df.to_csv(summary_out, sep="\t", index=False)
    per_doi_df.to_csv(per_doi_out, sep="\t", index=False)

    # per_doi failure type v3 + action cards (minimal, consistent with v1 rules)
    # build per-doi ee/row stats from v3 grouped
    cu_ee = cu.copy()
    cu_ee["curated_ee_num"] = pd.to_numeric(cu_ee["EE"], errors="coerce")
    cur_stats = cu_ee.groupby("doi_norm", dropna=False).agg(curated_rows=("doi_norm", "size"), curated_ee_mean=("curated_ee_num", "mean")).reset_index()
    ex_stats = grouped.groupby("doi_norm", dropna=False).agg(extracted_rows=("doi_norm", "size"), extracted_ee_mean=("group_mean_EE", "mean")).reset_index()
    diag = per_doi_df.rename(
        columns={
            "matched_curated_formulations": "per_doi_matched_curated",
            "total_curated_formulations": "per_doi_total_curated",
            "per_DOI_recall": "per_doi_recall",
        }
    ).merge(cur_stats, on="doi_norm", how="left").merge(ex_stats, on="doi_norm", how="left")
    diag["row_ratio"] = diag["extracted_rows"] / diag["curated_rows"]
    diag["ee_mean_abs_diff"] = (diag["curated_ee_mean"] - diag["extracted_ee_mean"]).abs()
    diag["doi_level_stable"] = diag["ee_mean_abs_diff"] <= 5

    def _classify(r: pd.Series) -> str:
        c = (not bool(r["doi_level_stable"])) or (pd.notna(r["ee_mean_abs_diff"]) and r["ee_mean_abs_diff"] > 10)
        a = pd.notna(r["curated_rows"]) and pd.notna(r["extracted_rows"]) and (r["extracted_rows"] < 0.6 * r["curated_rows"])
        b = (
            pd.notna(r["curated_rows"])
            and pd.notna(r["extracted_rows"])
            and (r["extracted_rows"] >= 0.6 * r["curated_rows"])
            and pd.notna(r["per_doi_recall"])
            and (r["per_doi_recall"] < 0.3)
        )
        if c:
            return "C_contamination_or_wrong_EE"
        if a:
            return "A_under_enumeration"
        if b:
            return "B_mismatch_similar_rows"
        return "None"

    diag["failure_type"] = diag.apply(_classify, axis=1)
    fail_out = out_dir / "per_doi_failure_type_v3.tsv"
    diag[
        [
            "doi_norm",
            "per_doi_recall",
            "curated_rows",
            "extracted_rows",
            "row_ratio",
            "ee_mean_abs_diff",
            "doi_level_stable",
            "failure_type",
        ]
    ].to_csv(fail_out, sep="\t", index=False)

    queue = diag.sort_values(["per_doi_recall", "extracted_rows"], ascending=[True, False]).head(10).copy()
    queue["missing_top3"] = ""
    queue["conflict_top3"] = ""
    queue["never_match_top3"] = ""
    cards_out = out_dir / "low_recall_action_cards_v3.tsv"
    queue[["doi_norm", "failure_type", "missing_top3", "conflict_top3", "never_match_top3"]].to_csv(cards_out, sep="\t", index=False)

    # drug_name_status with new normalization
    # curated set (remove tracer from canonical set)
    rows = []
    for doi in sorted(set(cu["doi_norm"].tolist()) | set(grouped["doi_norm"].tolist())):
        cvals: Set[str] = set()
        for x in cu.loc[cu["doi_norm"] == doi, "small_molecule_name"].tolist():
            primary, _ = canonicalize_drug_tokens(x)
            cvals.update(primary)
        evals = set([clean_for_cmp(x) for x in grouped.loc[grouped["doi_norm"] == doi, "sig__drug_name"].tolist() if clean_for_cmp(x)])
        ov = cvals & evals
        if len(cvals) == 0:
            status = "missing_curated"
        elif len(evals) == 0:
            status = "missing_extracted"
        elif cvals == evals:
            status = "exact_match"
        elif len(ov) > 0:
            status = "overlap_partial"
        else:
            status = "no_overlap"
        rows.append({"doi_norm": doi, "curated_norm_set": join_set(sorted(cvals)), "extracted_norm_set": join_set(sorted(evals)), "overlap_norm_set": join_set(sorted(ov)), "drug_name_status": status})
    drug_status_df = pd.DataFrame(rows)
    status_counts = drug_status_df["drug_name_status"].value_counts().to_dict()

    # baseline delta
    baseline_name, baseline_summary_path = pick_best_baseline_summary(cwd)
    base_recall, base_precision = summary_to_metrics(baseline_summary_path)
    d_recall = new_recall - base_recall
    d_precision = new_precision - base_precision

    # build log
    log = {
        "inputs": {
            "extracted_tsv": str(ex_p),
            "curated_tsv": str(cu_p),
            "sample_jsonl": str(sample_p),
            "baseline_summary": str(baseline_summary_path),
        },
        "normalization_rules": {
            "drug_name": {
                "lowercase": True,
                "strip_parentheses_content": True,
                "split_separators": ["|", ",", ";", "and"],
                "tracer_whitelist": sorted(list(TRACER_WHITELIST)),
                "tracers_excluded_from_primary_set": True,
            }
        },
        "row_counts": {
            "extracted_input_rows": int(len(ex)),
            "extracted_exploded_rows": int(len(ex_exp)),
            "extracted_grouped_rows_v3": int(len(grouped)),
            "curated_rows": int(len(cu)),
            "alignment_rows_v3": int(len(alignment_df)),
        },
        "status_counts": status_counts,
        "outputs": {
            "extracted_formulation_level_v3_tsv": str(v3_formulation_out),
            "formulation_alignment_v3_tsv": str(alignment_out),
            "formulation_alignment_summary_v3_tsv": str(summary_out),
            "per_doi_recall_v3_tsv": str(per_doi_out),
            "per_doi_failure_type_v3_tsv": str(fail_out),
            "low_recall_action_cards_v3_tsv": str(cards_out),
        },
    }
    (out_dir / "build_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"new formulation_recall={new_recall}")
    print(f"new formulation_precision={new_precision}")
    print(f"baseline_used={baseline_name}")
    print(f"delta_recall={d_recall}")
    print(f"delta_precision={d_precision}")
    print("drug_name_status_counts=" + json.dumps(status_counts, ensure_ascii=False, sort_keys=True))

    for doi in [
        "10.1016/j.ejpb.2004.09.002",
        "10.2147/ijn.s130908",
        "10.2147/ijn.s77498",
    ]:
        sub = drug_status_df[drug_status_df["doi_norm"] == doi]
        if len(sub) == 0:
            print(f"doi_diag {doi}: MISSING")
            continue
        r = sub.iloc[0]
        print(
            f"doi_diag {doi}: status={r['drug_name_status']} "
            f"curated=[{r['curated_norm_set']}] extracted=[{r['extracted_norm_set']}] overlap=[{r['overlap_norm_set']}]"
        )


if __name__ == "__main__":
    main()
