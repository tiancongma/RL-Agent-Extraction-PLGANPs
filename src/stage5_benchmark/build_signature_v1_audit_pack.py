#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402
from src.utils.run_id import is_valid_run_id, validate_artifact_subdir  # noqa: E402


CORE_FIELDS = [
    "drug_name_canon",
    "la_ga_ratio_canon",
    "polymer_mw_kda_canon_or_iv",
    "organic_solvent_canon",
    "surfactant_name_canon",
    "feed_anchor_canon",
]
MISSING_FLAGS = [
    "missing_drug",
    "missing_polymer_identity",
    "missing_solvent",
    "missing_surfactant",
    "missing_feed_anchor",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build signature_v1 audit pack Excel for typical formulation grouping errors."
    )
    ap.add_argument("--run-id", default="", help="Required deterministic run_id from preflight.")
    ap.add_argument("--out-subdir", default="", help="Required subdirectory under data/results/<run_id>/ for this run variant.")
    ap.add_argument("--instance-assignment-tsv", default="", help="Default run-scoped instance_assignment_v1.tsv")
    ap.add_argument("--formulation-core-tsv", default="", help="Default run-scoped formulation_core_v1.tsv")
    ap.add_argument("--signature-trace-tsv", default="", help="Default run-scoped signature_trace_v1.tsv")
    ap.add_argument("--build-log-json", default="", help="Default run-scoped build_log.json")
    ap.add_argument("--manifest", default="", help="Optional manifest/source metadata file for DOI/title mapping.")
    ap.add_argument("--raw-extracted-tsv", default="", help="Optional raw extracted TSV (weak_labels__*.tsv).")
    ap.add_argument("--derived-values-tsv", default="", help="Optional derived_values.tsv")
    ap.add_argument("--projection-trace-tsv", default="", help="Optional projection_trace.tsv")
    ap.add_argument("--n-per-bucket", type=int, default=15)
    ap.add_argument("--top-k-cores", type=int, default=10, help="Top K largest strict-merge cores considered for bucket B.")
    return ap.parse_args()


def parse_json_obj(s: Any) -> dict[str, Any]:
    try:
        t = str(s).strip()
        return json.loads(t) if t else {}
    except Exception:
        return {}


def short_text(v: Any, limit: int = 200) -> str:
    s = "" if v is None else str(v)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _sanitize_out_subdir(s: str) -> str:
    try:
        return validate_artifact_subdir(s, param_name="--out-subdir")
    except ValueError as exc:
        raise ValueError(
            "ERROR: --out-subdir is required when reusing a run_id and must be a functional artifact path under data/results/<run_id>/ without repeating a nested run_id or timestamp/hash token."
        ) from exc


def resolve_input_paths(run_id: str, out_subdir: str, args: argparse.Namespace) -> dict[str, Path]:
    run_base = paths.DATA_RESULTS_DIR / run_id / out_subdir
    base = run_base / "formulation_core_signature_v1"
    bench_base = run_base / "benchmark_goren_2025"

    def pick(user_val: str, default_path: Path) -> Path:
        return Path(user_val).resolve() if str(user_val).strip() else default_path

    raw_default_candidates = sorted(run_base.glob("weak_labels__*.tsv"))
    raw_default = raw_default_candidates[0] if raw_default_candidates else (run_base / "weak_labels__gemini.tsv")

    return {
        "instance_assignment": pick(args.instance_assignment_tsv, base / "instance_assignment_v1.tsv"),
        "formulation_core": pick(args.formulation_core_tsv, base / "formulation_core_v1.tsv"),
        "signature_trace": pick(args.signature_trace_tsv, base / "signature_trace_v1.tsv"),
        "build_log": pick(args.build_log_json, base / "build_log.json"),
        "raw_extracted": pick(args.raw_extracted_tsv, raw_default),
        "derived_values": pick(args.derived_values_tsv, bench_base / "derived_values.tsv"),
        "projection_trace": pick(args.projection_trace_tsv, bench_base / "projection_trace.tsv"),
        "out_xlsx": base / "audit_pack__signature_v1.xlsx",
    }


def load_manifest_map(manifest_arg: str, run_id: str) -> pd.DataFrame:
    candidates: list[Path] = []
    if manifest_arg.strip():
        candidates.append(Path(manifest_arg).resolve())
    candidates += [
        paths.DATA_RESULTS_DIR / run_id / "manifest_goren_2025.tsv",
        paths.DATA_CLEANED_INDEX_DIR / "manifest_goren_2025.tsv",
        paths.DATA_CLEANED_SAMPLES_DIR / "sample_goren18.tsv",
        paths.DATA_CLEANED_SAMPLES_DIR / "sample_goren18.jsonl",
    ]

    for p in candidates:
        if not p.exists():
            continue
        if p.suffix.lower() == ".jsonl":
            df = pd.read_json(p, lines=True, dtype=False).fillna("")
        else:
            df = pd.read_csv(p, sep="\t", dtype=str).fillna("")

        key_col = "key" if "key" in df.columns else "zotero_key" if "zotero_key" in df.columns else ""
        doi_col = "doi" if "doi" in df.columns else "reference_normalized_doi" if "reference_normalized_doi" in df.columns else ""
        title_col = "title" if "title" in df.columns else "paper_title" if "paper_title" in df.columns else ""
        if key_col and doi_col:
            out = pd.DataFrame(
                {
                    "doc_key": df[key_col].astype(str),
                    "doi": df[doi_col].astype(str),
                    "title": df[title_col].astype(str) if title_col else "",
                }
            )
            return out.drop_duplicates(subset=["doc_key"])
    return pd.DataFrame(columns=["doc_key", "doi", "title"])


def parse_provenance_field(refs: list[str]) -> tuple[str, str]:
    if not refs:
        return "", ""
    first = refs[0]
    parts = str(first).split("|")
    if len(parts) >= 3:
        block = parts[0]
        span = f"{parts[1]}-{parts[2]}"
        return block, span
    return "", ""


def best_effort_derived(group_key: str, derived: pd.DataFrame, projection: pd.DataFrame) -> dict[str, str]:
    if not derived.empty:
        d = derived[derived["group_key"] == group_key]
        if not d.empty:
            row = d.iloc[0]
            return {
                "derived_field_name": str(row.get("field_name", "")),
                "derived_value": str(row.get("value", "")),
                "rule_id": str(row.get("rule_id", "")),
                "derived_from": str(row.get("derived_from", "")),
                "trace_pointer": str(row.get("trace_pointer", "")),
            }
    if not projection.empty:
        p = projection[projection["group_key"] == group_key]
        if not p.empty:
            row = p.iloc[0]
            return {
                "derived_field_name": str(row.get("curated_column", "")),
                "derived_value": str(row.get("projected_value", "")),
                "rule_id": str(row.get("rule_id", "")),
                "derived_from": str(row.get("derived_from", "")),
                "trace_pointer": str(row.get("trace_pointer", "")),
            }
    return {
        "derived_field_name": "",
        "derived_value": "",
        "rule_id": "",
        "derived_from": "",
        "trace_pointer": "",
    }


def choose_buckets(df: pd.DataFrame, n: int, top_k_cores: int) -> dict[str, pd.DataFrame]:
    # Bucket A
    a = df[(df["gate_used"] == "C") & (df["missing_polymer_identity"] == True)].copy().head(n)

    # Bucket B
    b_candidates = df[df["gate_used"] == "B"].copy()
    core_sizes = (
        b_candidates.groupby("formulation_core_id", dropna=False)
        .size()
        .reset_index(name="core_size")
        .sort_values(["core_size", "formulation_core_id"], ascending=[False, True])
    )
    top_core_ids = core_sizes.head(top_k_cores)["formulation_core_id"].tolist()
    b = b_candidates[b_candidates["formulation_core_id"].isin(top_core_ids)].copy()
    b = b.sort_values(["formulation_core_id", "instance_id"]).head(n)

    # Bucket C
    c = df[(df["gate_used"] == "A") & (df["gate_a_anchor"].astype(str).str.strip() != "")].copy().head(n)

    return {"A_unresolved_missing_polymer": a, "B_strict_top_cores": b, "C_anchor_merged": c}


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise ValueError(
            "ERROR: --run-id is required. Generate/reuse a run_id via: python -m src.utils.run_preflight ..."
        )
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid --run-id (must match required regex): {run_id}")
    out_subdir = _sanitize_out_subdir(args.out_subdir)
    io = resolve_input_paths(run_id, out_subdir, args)

    required = [io["instance_assignment"], io["formulation_core"], io["signature_trace"], io["build_log"]]
    missing_required = [str(p) for p in required if not p.exists()]
    if missing_required:
        raise FileNotFoundError(f"Missing required inputs: {missing_required}")

    assignment = pd.read_csv(io["instance_assignment"], sep="\t", dtype=str).fillna("")
    core = pd.read_csv(io["formulation_core"], sep="\t", dtype=str).fillna("")
    trace = pd.read_csv(io["signature_trace"], sep="\t", dtype=str).fillna("")
    raw = pd.read_csv(io["raw_extracted"], sep="\t", dtype=str).fillna("") if io["raw_extracted"].exists() else pd.DataFrame()
    derived = pd.read_csv(io["derived_values"], sep="\t", dtype=str).fillna("") if io["derived_values"].exists() else pd.DataFrame()
    projection = pd.read_csv(io["projection_trace"], sep="\t", dtype=str).fillna("") if io["projection_trace"].exists() else pd.DataFrame()
    manifest_map = load_manifest_map(args.manifest, run_id)

    # Parse JSON columns
    assignment["_critical"] = assignment["critical_missing_json"].map(parse_json_obj)
    assignment["_canon"] = assignment["canonical_components_json"].map(parse_json_obj)
    core["_quality"] = core["signature_quality"].map(parse_json_obj)
    core["_prov"] = core["provenance_map_json"].map(parse_json_obj)

    for f in MISSING_FLAGS:
        assignment[f] = assignment["_critical"].map(lambda d: bool(d.get(f, False)))

    assignment["group_key"] = assignment["doc_key"].astype(str) + "::" + assignment["formulation_id"].astype(str)

    # Raw rows keyed by key+formulation_id
    if not raw.empty:
        raw["group_key"] = raw["key"].astype(str) + "::" + raw["formulation_id"].astype(str)
        raw = raw.drop_duplicates(subset=["group_key"], keep="first")

    merged = assignment.merge(
        core.drop(columns=["_quality", "_prov"], errors="ignore"),
        on=["formulation_core_id", "signature_hash", "signature_string", "merge_risk_level"],
        how="left",
        suffixes=("", "_core"),
    )
    merged = merged.merge(
        trace[["instance_id", "evidence_ref"]].drop_duplicates(subset=["instance_id"]),
        on="instance_id",
        how="left",
        suffixes=("", "_trace"),
    )
    if not raw.empty:
        merged = merged.merge(raw, on="group_key", how="left", suffixes=("", "_raw"))
    if not manifest_map.empty:
        merged = merged.merge(manifest_map, on="doc_key", how="left")
    else:
        merged["doi"] = ""
        merged["title"] = ""

    buckets = choose_buckets(merged, n=args.n_per_bucket, top_k_cores=args.top_k_cores)

    rows: list[dict[str, Any]] = []
    for bucket_name, bdf in buckets.items():
        for _, r in bdf.iterrows():
            canon = parse_json_obj(r.get("canonical_components_json", ""))
            crit = parse_json_obj(r.get("critical_missing_json", ""))
            q = parse_json_obj(r.get("signature_quality", ""))
            prov = parse_json_obj(r.get("provenance_map_json", ""))
            group_key = str(r.get("group_key", ""))
            dpack = best_effort_derived(group_key, derived, projection)

            why_no_merge = ""
            if str(r.get("gate_used", "")) == "C":
                if bool(crit.get("missing_polymer_identity", False)):
                    why_no_merge = "core_incomplete_no_anchor_missing_polymer_identity"
                else:
                    why_no_merge = "core_incomplete_or_anchor_missing"

            row = {
                "bucket": bucket_name,
                "zotero_key": r.get("doc_key", ""),
                "doi": r.get("doi", ""),
                "title": r.get("title", ""),
                "formulation_core_id": r.get("formulation_core_id", ""),
                "gate_used": r.get("gate_used", ""),
                "merge_reason": r.get("merge_reason", ""),
                "why_no_merge": why_no_merge,
                "signature_hash": r.get("signature_hash", ""),
                "signature_string": r.get("signature_string", ""),
                "signature_quality": json.dumps(q, ensure_ascii=False, sort_keys=True) if q else r.get("signature_quality", ""),
                "merge_risk_level": r.get("merge_risk_level", ""),
                # Raw extracted
                "drug_name_raw": r.get("drug_name", ""),
                "la_ga_ratio_raw": r.get("la_ga_ratio", ""),
                "mw_raw": r.get("plga_mw_kDa", ""),
                "polymer_code_raw": r.get("vendor_product_code", ""),
                "solvent_raw": r.get("organic_solvent", ""),
                "surfactant_raw": r.get("surfactant_name", ""),
                "feed_raw": r.get("drug_feed_amount_text", ""),
                "EE_raw": r.get("encapsulation_efficiency_percent", ""),
                "size_raw": r.get("size_nm", ""),
                "pdi_raw": r.get("pdi", ""),
                # Canonical
                "drug_name_canon": canon.get("drug_name_canon", ""),
                "la_ga_ratio_canon": canon.get("la_ga_ratio_canon", ""),
                "mw_kda_canon_or_iv_canon": canon.get("polymer_mw_kda_canon_or_iv", ""),
                "solvent_canon": canon.get("organic_solvent_canon", ""),
                "surfactant_canon": canon.get("surfactant_name_canon", ""),
                "feed_anchor_canon": canon.get("feed_anchor_canon", ""),
                # Missing flags
                "missing_drug": bool(crit.get("missing_drug", False)),
                "missing_polymer_identity": bool(crit.get("missing_polymer_identity", False)),
                "missing_solvent": bool(crit.get("missing_solvent", False)),
                "missing_surfactant": bool(crit.get("missing_surfactant", False)),
                "missing_feed_anchor": bool(crit.get("missing_feed_anchor", False)),
                # Instance evidence
                "instance_evidence_id": r.get("evidence_ref", ""),
                "instance_evidence_span_text_200": short_text(r.get("evidence_span_text", ""), 200),
                # Optional derived/projection
                **dpack,
            }

            # Per-core-field provenance references
            for field in CORE_FIELDS:
                refs = prov.get(field, [])
                refs = refs if isinstance(refs, list) else []
                row[f"{field}__evidence_refs"] = "|".join([str(x) for x in refs])
                block_id, span_id = parse_provenance_field([str(x) for x in refs])
                row[f"{field}__evidence_block_id"] = block_id
                row[f"{field}__evidence_span_id"] = span_id
            rows.append(row)

    audit_df = pd.DataFrame(rows)
    if audit_df.empty:
        audit_df = pd.DataFrame(
            columns=[
                "bucket", "zotero_key", "doi", "title", "formulation_core_id", "gate_used", "merge_reason",
                "why_no_merge", "signature_hash", "signature_string", "signature_quality", "merge_risk_level",
            ]
        )

    # Summary sheet
    bucket_counts = (
        audit_df["bucket"].value_counts().rename_axis("bucket").reset_index(name="n_rows")
        if not audit_df.empty else pd.DataFrame(columns=["bucket", "n_rows"])
    )
    summary_rows: list[dict[str, Any]] = []
    for bucket_name, bdf in audit_df.groupby("bucket", dropna=False):
        n = len(bdf)
        for f in MISSING_FLAGS:
            rate = float(bdf[f].astype(bool).mean() * 100.0) if n else 0.0
            summary_rows.append({"bucket": bucket_name, "metric": f"{f}_rate_percent", "value": round(rate, 2)})
        summary_rows.append({"bucket": bucket_name, "metric": "n_rows", "value": n})
    summary_df = pd.DataFrame(summary_rows)

    top_missing = pd.DataFrame(
        [{"flag": f, "count": int(audit_df[f].astype(bool).sum())} for f in MISSING_FLAGS]
    ).sort_values(["count", "flag"], ascending=[False, True])

    io["out_xlsx"].parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(io["out_xlsx"], engine="openpyxl") as writer:
        audit_df.to_excel(writer, sheet_name="audit_cases", index=False)
        bucket_counts.to_excel(writer, sheet_name="summary", index=False, startrow=0)
        summary_df.to_excel(writer, sheet_name="summary", index=False, startrow=len(bucket_counts) + 3)
        top_missing.to_excel(writer, sheet_name="summary", index=False, startrow=len(bucket_counts) + len(summary_df) + 6)

    # Console outputs required
    print(f"excel_path={io['out_xlsx']}")

    bucket_missing_rates: dict[str, dict[str, float]] = {}
    for bucket_name, bdf in audit_df.groupby("bucket", dropna=False):
        bucket_missing_rates[bucket_name] = {
            f: round(float(bdf[f].astype(bool).mean() * 100.0), 2) if len(bdf) else 0.0
            for f in MISSING_FLAGS
        }
    print(f"bucket_counts={bucket_counts.to_dict(orient='records') if not bucket_counts.empty else []}")
    print(f"bucket_missing_flag_rates_percent={bucket_missing_rates}")

    for bucket_name in ["A_unresolved_missing_polymer", "B_strict_top_cores", "C_anchor_merged"]:
        bdf = audit_df[audit_df["bucket"] == bucket_name].copy().head(5)
        preview_cols = [
            "zotero_key", "doi", "gate_used", "merge_reason", "signature_string",
            "missing_drug", "missing_polymer_identity", "missing_solvent", "missing_surfactant", "missing_feed_anchor",
        ]
        print(f"\n[{bucket_name}] preview_top5")
        if bdf.empty:
            print("(empty)")
        else:
            print(bdf[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
