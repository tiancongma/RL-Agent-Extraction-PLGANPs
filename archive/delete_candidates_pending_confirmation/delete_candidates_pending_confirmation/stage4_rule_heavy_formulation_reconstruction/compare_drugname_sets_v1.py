#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


SALT_TOKENS = {"hydrochloride", "hcl", "sodium", "acetate"}


def norm_doi(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    if not s:
        return ""
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def clean_raw_name(v: object) -> str:
    s = "" if v is None else str(v)
    return re.sub(r"\s+", " ", s.strip())


def normalize_drug_name(v: str) -> str:
    s = v.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    toks = s.split(" ")
    while toks and toks[-1] in SALT_TOKENS:
        toks.pop()
    s2 = " ".join(toks).strip()
    return s2 if s2 else s


def set_to_str(vals: Set[str]) -> str:
    return " | ".join(sorted(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare curated vs extracted drug-name sets per DOI.")
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--extracted-tsv",
        default="data/results/dev18_surfactant_schema_v1/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--baseline-extracted-tsv",
        default="data/results/dev18_corefields_prompt_v1/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--sample-jsonl",
        default="data/cleaned/samples/sample_goren18.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/diagnostics_v3_drugname",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    curated_p = (cwd / args.curated_tsv).resolve()
    extracted_p = (cwd / args.extracted_tsv).resolve()
    baseline_p = (cwd / args.baseline_extracted_tsv).resolve()
    sample_p = (cwd / args.sample_jsonl).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curated = pd.read_csv(curated_p, sep="\t", dtype=str).fillna("")
    extracted = pd.read_csv(extracted_p, sep="\t", dtype=str).fillna("")

    # map key->doi for extracted fallback
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

    # Curated DOI + drug
    if "doi_norm" in curated.columns:
        curated["doi_norm"] = curated["doi_norm"].map(norm_doi)
    elif "reference" in curated.columns:
        curated["doi_norm"] = curated["reference"].map(norm_doi)
    else:
        raise RuntimeError("Curated TSV missing doi_norm/reference.")
    if "small_molecule_name" not in curated.columns:
        raise RuntimeError("Curated TSV missing small_molecule_name.")
    curated["drug_raw"] = curated["small_molecule_name"].map(clean_raw_name)

    # Extracted DOI + drug
    if "doi_norm" in extracted.columns:
        extracted["doi_norm"] = extracted["doi_norm"].map(norm_doi)
    else:
        key_col = "key" if "key" in extracted.columns else "zotero_key" if "zotero_key" in extracted.columns else None
        if key_col is None:
            raise RuntimeError("Extracted TSV missing doi_norm and key mapping columns.")
        extracted["doi_norm"] = extracted[key_col].map(lambda k: key2doi.get(str(k), ""))
    if "drug_name" not in extracted.columns:
        raise RuntimeError("Extracted TSV missing drug_name.")
    extracted["drug_raw"] = extracted["drug_name"].map(clean_raw_name)

    all_doi = sorted(set(curated["doi_norm"].tolist()) | set(extracted["doi_norm"].tolist()))
    rows: List[Dict[str, object]] = []
    for doi in all_doi:
        c_raw = set(x for x in curated.loc[curated["doi_norm"] == doi, "drug_raw"].tolist() if x)
        e_raw = set(x for x in extracted.loc[extracted["doi_norm"] == doi, "drug_raw"].tolist() if x)
        c_norm = set(normalize_drug_name(x) for x in c_raw if normalize_drug_name(x))
        e_norm = set(normalize_drug_name(x) for x in e_raw if normalize_drug_name(x))
        c_only = c_norm - e_norm
        e_only = e_norm - c_norm
        ov = c_norm & e_norm
        c_n = len(c_norm)
        e_n = len(e_norm)
        ov_n = len(ov)

        if c_n == 0:
            status = "missing_curated"
        elif e_n == 0:
            status = "missing_extracted"
        elif c_norm == e_norm and c_n > 0:
            status = "exact_match"
        elif ov_n > 0:
            status = "overlap_partial"
        else:
            status = "no_overlap"

        rows.append(
            {
                "doi_norm": doi,
                "curated_drugs_raw": set_to_str(c_raw),
                "extracted_drugs_raw": set_to_str(e_raw),
                "curated_drugs_norm": set_to_str(c_norm),
                "extracted_drugs_norm": set_to_str(e_norm),
                "curated_only_norm": set_to_str(c_only),
                "extracted_only_norm": set_to_str(e_only),
                "overlap_norm": set_to_str(ov),
                "curated_n": c_n,
                "extracted_n": e_n,
                "overlap_n": ov_n,
                "drug_name_status": status,
            }
        )

    out_df = pd.DataFrame(rows)
    per_doi_out = out_dir / "per_doi_drugname_sets.tsv"
    out_df.to_csv(per_doi_out, sep="\t", index=False)

    problem_df = out_df[out_df["drug_name_status"].isin(["no_overlap", "missing_extracted", "overlap_partial"])].copy()
    problem_df = problem_df.sort_values(["drug_name_status", "curated_n"], ascending=[True, False])
    problem_out = out_dir / "per_doi_drugname_problem_cases.tsv"
    problem_df.to_csv(problem_out, sep="\t", index=False)

    status_counts = out_df["drug_name_status"].value_counts().to_dict()
    print("drug_name_status_counts=" + json.dumps(status_counts, ensure_ascii=False, sort_keys=True))

    log = {
        "inputs": {
            "curated_tsv": str(curated_p),
            "extracted_tsv": str(extracted_p),
            "baseline_extracted_tsv_optional": str(baseline_p),
            "sample_jsonl": str(sample_p),
        },
        "columns_used": {
            "curated_doi": "doi_norm" if "doi_norm" in curated.columns else "reference",
            "curated_drug": "small_molecule_name",
            "extracted_doi": "doi_norm" if "doi_norm" in extracted.columns else "key->sample_jsonl",
            "extracted_drug": "drug_name",
        },
        "row_counts": {
            "curated_rows": int(len(curated)),
            "extracted_rows": int(len(extracted)),
            "per_doi_rows": int(len(out_df)),
            "problem_rows": int(len(problem_df)),
        },
        "status_counts": status_counts,
        "outputs": {
            "per_doi_drugname_sets_tsv": str(per_doi_out),
            "per_doi_drugname_problem_cases_tsv": str(problem_out),
        },
    }
    log_out = out_dir / "drugname_build_log.json"
    log_out.write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
