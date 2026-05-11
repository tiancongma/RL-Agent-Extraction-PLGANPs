#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


CANONICAL_FIELDS = [
    "small_molecule_name",
    "polymer_MW",
    "LA/GA",
    "solvent",
    "surfactant_name",
    "drug/polymer",
    "surfactant_concentration",
    "aqueous/organic",
    "pH",
    "EE",
]


def clean(v: object) -> str:
    if pd.isna(v):
        return ""
    return "".join(str(v).strip().lower().split())


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def pick_col(cols: List[str], cands: List[str]) -> str:
    lower = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return ""


def parse_curated_signature(sig: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    parts = str(sig).split("|")
    for p in parts:
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = clean(v.strip())
    return out


def top_fields_join(df: pd.DataFrame, value_col: str, n: int = 3) -> str:
    if df.empty:
        return ""
    return ",".join(df.sort_values(value_col, ascending=False)["field"].head(n).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-DOI failure profile diagnostics v1.")
    parser.add_argument(
        "--extracted-formulation-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/extracted_formulation_level_v1.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--alignment-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/formulation_alignment.tsv",
    )
    parser.add_argument(
        "--per-doi-diagnostic-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/diagnostics_v1/per_doi_diagnostic_v1.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/diagnostics_v2_failure_profile",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    p_ex = (cwd / args.extracted_formulation_tsv).resolve()
    p_cu = (cwd / args.curated_tsv).resolve()
    p_al = (cwd / args.alignment_tsv).resolve()
    p_diag = (cwd / args.per_doi_diagnostic_tsv).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(p_ex, sep="\t", dtype=str).fillna("")
    cu = pd.read_csv(p_cu, sep="\t", dtype=str).fillna("")
    al = pd.read_csv(p_al, sep="\t", dtype=str).fillna("")
    diag = pd.read_csv(p_diag, sep="\t", dtype=str).fillna("")

    # canonical mapping for extracted
    ex_cols = list(ex.columns)
    ex_map = {
        "small_molecule_name": pick_col(ex_cols, ["sig__drug_name", "drug_name"]),
        "polymer_MW": pick_col(ex_cols, ["sig__polymer_MW", "plga_mw_kDa", "polymer_MW"]),
        "LA/GA": pick_col(ex_cols, ["sig__LA/GA", "la_ga_ratio", "LA/GA"]),
        "solvent": pick_col(ex_cols, ["sig__solvent", "organic_solvent", "solvent"]),
        "surfactant_name": pick_col(ex_cols, ["sig__surfactant_name", "surfactant_name"]),
        "drug/polymer": pick_col(ex_cols, ["sig__drug/polymer", "drug_to_polymer_ratio", "drug/polymer"]),
        "surfactant_concentration": pick_col(ex_cols, ["sig__surfactant_concentration", "pva_conc_percent", "surfactant_concentration"]),
        "aqueous/organic": pick_col(ex_cols, ["sig__aqueous/organic", "phase_ratio", "aqueous/organic"]),
        "pH": pick_col(ex_cols, ["sig__pH", "pH"]),
        "EE": pick_col(ex_cols, ["group_mean_EE", "encapsulation_efficiency_percent", "EE"]),
    }

    # Step2 missingness
    miss_rows: List[Dict[str, object]] = []
    for doi, g in ex.groupby("doi_norm", dropna=False):
        n = len(g)
        for f in CANONICAL_FIELDS:
            c = ex_map.get(f, "")
            if not c:
                missing_rate = 1.0
            else:
                miss_n = int(g[c].map(clean).eq("").sum())
                missing_rate = (miss_n / n) if n else 1.0
            miss_rows.append({"doi_norm": doi, "field": f, "missing_rate": missing_rate, "n_rows": n})
    miss_df = pd.DataFrame(miss_rows)
    miss_out = out_dir / "per_doi_extracted_missingness.tsv"
    miss_df.to_csv(miss_out, sep="\t", index=False)

    # Step3 conflicts
    conf_rows: List[Dict[str, object]] = []
    for doi, g in ex.groupby("doi_norm", dropna=False):
        for f in CANONICAL_FIELDS:
            c = ex_map.get(f, "")
            if not c:
                nuniq = 0
            else:
                vals = [clean(v) for v in g[c].tolist()]
                vals = [v for v in vals if v]
                nuniq = len(set(vals))
            conf_rows.append(
                {
                    "doi_norm": doi,
                    "field": f,
                    "n_unique_nonempty_values": nuniq,
                    "conflict_flag": bool(nuniq >= 2),
                }
            )
    conf_df = pd.DataFrame(conf_rows)
    conf_out = out_dir / "per_doi_extracted_conflicts.tsv"
    conf_df.to_csv(conf_out, sep="\t", index=False)

    # Step4 alignment failure field decomposition using unmatched from formulation_alignment.tsv
    al["matched_bool"] = to_bool_series(al["matched"])
    fail_rows: List[Dict[str, object]] = []
    for doi, al_d in al.groupby("doi_norm", dropna=False):
        unmatched = al_d[~al_d["matched_bool"]]
        n_unmatched = len(unmatched)
        ex_d = ex[ex["doi_norm"] == doi]
        # precompute extracted value sets per field
        ex_field_vals: Dict[str, set[str]] = {}
        for f in CANONICAL_FIELDS:
            c = ex_map.get(f, "")
            if not c:
                ex_field_vals[f] = set()
            else:
                vals = [clean(v) for v in ex_d[c].tolist()]
                ex_field_vals[f] = set(v for v in vals if v)

        never_match_counts = {f: 0 for f in CANONICAL_FIELDS}
        for _, r in unmatched.iterrows():
            sig_map = parse_curated_signature(r.get("curated_signature", ""))
            for f in CANONICAL_FIELDS:
                cv = sig_map.get(f, "")
                if not cv:
                    continue
                if cv not in ex_field_vals.get(f, set()):
                    never_match_counts[f] += 1

        for f in CANONICAL_FIELDS:
            frac = (never_match_counts[f] / n_unmatched) if n_unmatched else 0.0
            fail_rows.append(
                {
                    "doi_norm": doi,
                    "field": f,
                    "frac_unmatched_where_field_never_matches": frac,
                    "n_unmatched_curated": n_unmatched,
                }
            )
    fail_df = pd.DataFrame(fail_rows)
    fail_out = out_dir / "per_doi_alignment_failure_fields.tsv"
    fail_df.to_csv(fail_out, sep="\t", index=False)

    # Step5 failure type
    for c in ["per_doi_recall", "curated_rows", "extracted_rows", "row_ratio", "ee_mean_abs_diff"]:
        diag[c] = pd.to_numeric(diag[c], errors="coerce")
    diag["doi_level_stable"] = to_bool_series(diag["doi_level_stable"])

    def classify(r: pd.Series) -> str:
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

    diag["failure_type"] = diag.apply(classify, axis=1)
    fail_type_cols = [
        "doi_norm",
        "per_doi_recall",
        "curated_rows",
        "extracted_rows",
        "row_ratio",
        "ee_mean_abs_diff",
        "doi_level_stable",
        "failure_type",
    ]
    fail_type_out = out_dir / "per_doi_failure_type.tsv"
    diag[fail_type_cols].to_csv(fail_type_out, sep="\t", index=False)

    # Step6 action cards top10 lowest recall
    top10 = diag.sort_values(["per_doi_recall", "extracted_rows"], ascending=[True, False]).head(10).copy()
    action_rows: List[Dict[str, str]] = []
    for _, r in top10.iterrows():
        doi = r["doi_norm"]
        miss_d = miss_df[miss_df["doi_norm"] == doi]
        conf_d = conf_df[(conf_df["doi_norm"] == doi) & (conf_df["conflict_flag"] == True)]
        fail_d = fail_df[fail_df["doi_norm"] == doi]

        missing_top3 = ",".join(miss_d.sort_values("missing_rate", ascending=False)["field"].head(3).tolist())
        conflict_top3 = ",".join(conf_d.sort_values("n_unique_nonempty_values", ascending=False)["field"].head(3).tolist())
        never_top3 = ",".join(
            fail_d.sort_values("frac_unmatched_where_field_never_matches", ascending=False)["field"].head(3).tolist()
        )

        action_rows.append(
            {
                "doi_norm": doi,
                "failure_type": r["failure_type"],
                "missing_top3": missing_top3,
                "conflict_top3": conflict_top3,
                "never_match_top3": never_top3,
            }
        )
    action_df = pd.DataFrame(action_rows)
    action_out = out_dir / "low_recall_action_cards.tsv"
    action_df.to_csv(action_out, sep="\t", index=False)

    # build log
    log = {
        "inputs": {
            "extracted_formulation_level_v1_tsv": str(p_ex),
            "curated_overlap_tsv": str(p_cu),
            "formulation_alignment_tsv": str(p_al),
            "per_doi_diagnostic_v1_tsv": str(p_diag),
        },
        "outputs": {
            "per_doi_extracted_missingness_tsv": str(miss_out),
            "per_doi_extracted_conflicts_tsv": str(conf_out),
            "per_doi_alignment_failure_fields_tsv": str(fail_out),
            "per_doi_failure_type_tsv": str(fail_type_out),
            "low_recall_action_cards_tsv": str(action_out),
        },
        "row_counts": {
            "extracted_rows": int(len(ex)),
            "curated_rows": int(len(cu)),
            "alignment_rows": int(len(al)),
            "diagnostic_rows": int(len(diag)),
            "missingness_rows": int(len(miss_df)),
            "conflicts_rows": int(len(conf_df)),
            "failure_field_rows": int(len(fail_df)),
            "failure_type_rows": int(len(diag)),
            "action_card_rows": int(len(action_df)),
        },
    }
    log_out = out_dir / "diagnostics_build_log.json"
    log_out.write_text(json.dumps(log, indent=2), encoding="utf-8")

    # print counts A/B/C
    counts = diag["failure_type"].value_counts().to_dict()
    print(f"A_under_enumeration={int(counts.get('A_under_enumeration', 0))}")
    print(f"B_mismatch_similar_rows={int(counts.get('B_mismatch_similar_rows', 0))}")
    print(f"C_contamination_or_wrong_EE={int(counts.get('C_contamination_or_wrong_EE', 0))}")


if __name__ == "__main__":
    main()
