#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

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


def run_alignment(curated: pd.DataFrame, extracted_grouped: pd.DataFrame) -> Tuple[float, float, int]:
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
    cu["_k_small_molecule_name"] = cu["small_molecule_name"].map(lambda x: " | ".join(canonicalize_drug_tokens(x)[0]))

    for cf, ef in ex_field_map.items():
        ex[f"_k_{cf}"] = ex[ef].map(clean_for_cmp) if ef in ex.columns else ""

    matched_extracted_keys: Set[Tuple[str, str]] = set()
    matched_curated = 0
    total_curated = len(cu)

    for doi, cu_d in cu.groupby("doi_norm", dropna=False):
        ex_d = ex[ex["doi_norm"] == doi].copy()
        for c_row in cu_d.itertuples(index=False):
            c = c_row._asdict()
            c_ee = c.get("ee_num")
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
                            matched_extracted_keys.add((str(doi), str(best.get("formulation_signature", ""))))
                            is_matched = True
            if is_matched:
                matched_curated += 1

    total_extracted = len(ex)
    matched_extracted = len(matched_extracted_keys)
    recall = (matched_curated / total_curated) if total_curated else float("nan")
    precision = (matched_extracted / total_extracted) if total_extracted else float("nan")
    return recall, precision, total_extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision recovery experiment v1 (evaluation-only).")
    parser.add_argument(
        "--extracted-v3",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v3_surfactant_drugnorm/extracted_formulation_level_v3.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/precision_recovery_v1",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    ex_p = (cwd / args.extracted_v3).resolve()
    cu_p = (cwd / args.curated_tsv).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(ex_p, sep="\t", dtype=str).fillna("")
    cu = pd.read_csv(cu_p, sep="\t", dtype=str).fillna("")

    # Step 1: row completeness
    core_map = {
        "polymer_MW": "sig__polymer_MW" if "sig__polymer_MW" in ex.columns else "polymer_MW",
        "LA/GA": "sig__LA/GA" if "sig__LA/GA" in ex.columns else "la_ga_ratio",
        "solvent": "sig__solvent" if "sig__solvent" in ex.columns else "organic_solvent",
        "surfactant_name": "sig__surfactant_name" if "sig__surfactant_name" in ex.columns else "surfactant_name",
        "surfactant_concentration": "sig__surfactant_concentration" if "sig__surfactant_concentration" in ex.columns else "surfactant_concentration_text",
        "drug_to_polymer_ratio_w_w": "sig__drug/polymer" if "sig__drug/polymer" in ex.columns else "drug_to_polymer_ratio_w_w",
    }

    present_lists = []
    n_present = []
    for _, r in ex.iterrows():
        pres = []
        for logical, col in core_map.items():
            val = r.get(col, "") if col in ex.columns else ""
            if clean_for_cmp(val):
                pres.append(logical)
        present_lists.append(" | ".join(pres))
        n_present.append(len(pres))
    ex["core_present_list"] = present_lists
    ex["n_core_present"] = n_present

    dist = ex["n_core_present"].value_counts().sort_index()
    dist_map = {int(k): int(v) for k, v in dist.to_dict().items()}
    ex["n_core_present_count"] = ex["n_core_present"].map(dist_map)
    ex["n_core_present_fraction"] = ex["n_core_present"].map(lambda x: dist_map[x] / len(ex) if len(ex) else 0.0)
    diag_out = out_dir / "diagnostics_row_completeness.tsv"
    ex.to_csv(diag_out, sep="\t", index=False)

    # Step 2: sweep filters
    sweep_rows: List[Dict[str, object]] = []
    for label, thr in [("A_n_core_ge_1", 1), ("B_n_core_ge_2", 2), ("C_n_core_ge_3", 3)]:
        sub = ex[ex["n_core_present"] >= thr].copy()
        rec, prec, n_ex = run_alignment(cu, sub)
        sweep_rows.append({"filter": label, "extracted_rows": n_ex, "recall": rec, "precision": prec})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_out = out_dir / "precision_recovery_sweep.tsv"
    sweep_df.to_csv(sweep_out, sep="\t", index=False)

    # choose best tradeoff: highest precision with recall >= max(0.95*baseline_recall, baseline_recall-0.02)
    baseline_recall, baseline_precision, _ = run_alignment(cu, ex)
    recall_floor = max(0.0, max(0.95 * baseline_recall, baseline_recall - 0.02))
    eligible = sweep_df[sweep_df["recall"] >= recall_floor].copy()
    if len(eligible) > 0:
        best = eligible.sort_values(["precision", "recall", "extracted_rows"], ascending=[False, False, True]).iloc[0]
    else:
        best = sweep_df.sort_values(["precision", "recall", "extracted_rows"], ascending=[False, False, True]).iloc[0]
    best_filter = str(best["filter"])
    best_thr = {"A_n_core_ge_1": 1, "B_n_core_ge_2": 2, "C_n_core_ge_3": 3}[best_filter]

    # Step 3: lightweight consolidation on best filter
    best_set = ex[ex["n_core_present"] >= best_thr].copy()
    sig_cols = [
        "sig__drug_name",
        "sig__polymer_MW",
        "sig__LA/GA",
        "sig__solvent",
        "sig__surfactant_name",
        "sig__surfactant_concentration",
    ]
    for c in sig_cols:
        if c not in best_set.columns:
            best_set[c] = ""
    best_set["consolidation_signature"] = best_set.apply(
        lambda rr: " | ".join(f"{c}={clean_for_cmp(rr.get(c, ''))}" for c in sig_cols),
        axis=1,
    )
    agg_dict: Dict[str, Any] = {
        "group_mean_EE": "first",
        "n_unique_ee": "first",
        "unique_ee_values": "first",
        "tracer_name": lambda s: " | ".join(sorted(set([str(x).strip() for x in s if str(x).strip()]))),
        "group_size": "sum" if "group_size" in best_set.columns else "size",
    }
    if "evidence_span_text" in best_set.columns:
        agg_dict["evidence_span_text"] = lambda s: " || ".join([str(x) for x in s if str(x).strip()])
    for c in sig_cols + ["formulation_signature", "doi_norm"]:
        agg_dict[c] = "first"

    consolidated = (
        best_set.groupby(["doi_norm", "consolidation_signature"], dropna=False)
        .agg(agg_dict)
        .reset_index(drop=True)
    )
    cons_rec, cons_prec, cons_n = run_alignment(cu, consolidated)
    sweep_df2 = pd.concat(
        [
            sweep_df,
            pd.DataFrame(
                [
                    {
                        "filter": f"{best_filter}_plus_consolidation",
                        "extracted_rows": cons_n,
                        "recall": cons_rec,
                        "precision": cons_prec,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    sweep_df2.to_csv(sweep_out, sep="\t", index=False)

    consolidated_out = out_dir / "extracted_best_filter_consolidated.tsv"
    consolidated.to_csv(consolidated_out, sep="\t", index=False)

    log = {
        "inputs": {"extracted_v3": str(ex_p), "curated_tsv": str(cu_p)},
        "core_fields": core_map,
        "baseline": {"recall": baseline_recall, "precision": baseline_precision, "rows": int(len(ex))},
        "recall_floor": recall_floor,
        "best_filter": {"name": best_filter, "threshold": best_thr},
        "distribution_n_core_present": dist_map,
        "outputs": {
            "diagnostics_row_completeness_tsv": str(diag_out),
            "precision_recovery_sweep_tsv": str(sweep_out),
            "extracted_best_filter_consolidated_tsv": str(consolidated_out),
        },
    }
    (out_dir / "build_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("precision_recovery_sweep:")
    print(sweep_df2.to_csv(sep="\t", index=False).strip())
    print(
        f"chosen_best_tradeoff={best_filter} "
        f"baseline_recall={baseline_recall} baseline_precision={baseline_precision} "
        f"best_recall={best['recall']} best_precision={best['precision']} "
        f"consolidated_recall={cons_rec} consolidated_precision={cons_prec}"
    )


if __name__ == "__main__":
    main()
