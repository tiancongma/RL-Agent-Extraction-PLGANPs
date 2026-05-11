#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare alignment eval between schema_v2 and schema_v3 with shared curated overlap subset."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--curated-template",
        default="data/benchmark/goren_2025/NP_dataset_formulations.csv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument("--modes", default="strict,relaxed,canonicalized")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/alignment_eval_v3_v1",
    )
    return parser.parse_args()


def norm_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
    except ValueError:
        return s
    if f.is_integer():
        return str(int(f))
    return f"{f:.6g}"


def project_core_to_curated(
    *,
    core_tsv: Path,
    curated_template: Path,
    projected_out: Path,
    trace_out: Path,
) -> None:
    core = pd.read_csv(core_tsv, sep="\t", dtype=str).fillna("")
    curated_cols = pd.read_csv(curated_template, nrows=0).columns.tolist()

    projected_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []
    for _, r in core.iterrows():
        row = {c: "" for c in curated_cols}
        mapped: list[str] = []
        missing: list[str] = []

        small_molecule_name = str(r.get("drug_name_normalized", "")) or str(r.get("drug_name", ""))
        solvent = str(r.get("organic_solvent_normalized", "")) or str(r.get("organic_solvent", ""))
        mw_lo = norm_num(r.get("polymer_mw_lower_kDa", ""))
        mw_hi = norm_num(r.get("polymer_mw_upper_kDa", ""))
        mw_single = norm_num(r.get("polymer_mw_kDa", ""))
        if mw_lo and mw_hi:
            polymer_mw = mw_lo if mw_lo == mw_hi else f"{mw_lo}-{mw_hi}"
        elif mw_single:
            polymer_mw = mw_single
        else:
            polymer_mw = mw_lo or mw_hi

        mapping = {
            "reference": str(r.get("reference_normalized_doi", "")),
            "small_molecule_name": small_molecule_name,
            "solvent": solvent,
            "LA/GA": norm_num(r.get("la_ga_ratio", "")),
            "drug/polymer": norm_num(r.get("drug_to_polymer_mass_ratio", "")),
            "polymer_MW": polymer_mw,
            "surfactant_concentration": norm_num(r.get("pva_conc_percent", "")),
            "pH": norm_num(r.get("aqueous_phase_pH", "")),
        }

        for c in curated_cols:
            if c in mapping:
                row[c] = mapping[c]
                if str(mapping[c]).strip():
                    mapped.append(c)
                else:
                    missing.append(c)
            else:
                missing.append(c)

        projected_rows.append(row)
        trace_rows.append(
            {
                "formulation_core_id": str(r.get("formulation_core_id", "")),
                "reference_normalized_doi": str(r.get("reference_normalized_doi", "")),
                "core_signature": str(r.get("core_signature", "")),
                "mapped_fields": ",".join(mapped),
                "missing_fields": ",".join(missing),
            }
        )

    projected_df = pd.DataFrame(projected_rows, columns=curated_cols)
    trace_df = pd.DataFrame(trace_rows)
    projected_out.parent.mkdir(parents=True, exist_ok=True)
    projected_df.to_csv(projected_out, sep="\t", index=False)
    trace_df.to_csv(trace_out, sep="\t", index=False)


def load_metrics_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def per_doi_metrics(
    *,
    projected_tsv: Path,
    curated_tsv: Path,
    alignment_rows_tsv: Path,
    schema_name: str,
) -> pd.DataFrame:
    projected = pd.read_csv(projected_tsv, sep="\t", dtype=str).fillna("")
    curated = pd.read_csv(curated_tsv, sep="\t", dtype=str).fillna("")
    rows = pd.read_csv(alignment_rows_tsv, sep="\t", dtype=str).fillna("")

    def doi_col(df: pd.DataFrame) -> pd.Series:
        if "doi_norm" in df.columns:
            return df["doi_norm"].astype(str).str.strip().str.lower()
        if "reference" in df.columns:
            return df["reference"].astype(str).str.strip().str.lower()
        return pd.Series([""] * len(df))

    projected["doi_eval"] = doi_col(projected)
    curated["doi_eval"] = doi_col(curated)
    rows["doi_norm"] = rows["doi_norm"].astype(str).str.strip().str.lower()

    out_rows: list[dict[str, Any]] = []
    for mode in sorted(set(rows["mode"].tolist())):
        rm = rows[rows["mode"] == mode].copy()
        for doi in sorted(set(curated["doi_eval"].tolist()) | set(projected["doi_eval"].tolist())):
            ctot = int((curated["doi_eval"] == doi).sum())
            ptot = int((projected["doi_eval"] == doi).sum())
            rm_d = rm[rm["doi_norm"] == doi]
            mcur = int(
                rm_d[(rm_d["matched"].astype(str).str.lower() == "true") & (rm_d["curated_row_id"].astype(str).str.strip() != "")][
                    "curated_row_id"
                ]
                .nunique()
            )
            mproj = int(
                rm_d[(rm_d["matched"].astype(str).str.lower() == "true") & (rm_d["projected_row_id"].astype(str).str.strip() != "")][
                    "projected_row_id"
                ]
                .nunique()
            )
            recall = (mcur / ctot) if ctot else 0.0
            precision = (mproj / ptot) if ptot else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            out_rows.append(
                {
                    "schema": schema_name,
                    "mode": mode,
                    "doi": doi,
                    "total_curated_rows": ctot,
                    "total_projected_rows": ptot,
                    "matched_curated_rows": mcur,
                    "matched_projected_rows": mproj,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                }
            )
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    modes = str(args.modes)
    curated_template = Path(args.curated_template)
    curated_tsv = Path(args.curated_tsv)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"data/results/{run_id}/benchmark_goren_2025/alignment_eval_v3_v1"
    )

    p_core_v2 = Path(f"data/results/{run_id}/benchmark_goren_2025/schema_v2/formulation_core.tsv")
    p_core_v3 = Path(f"data/results/{run_id}/benchmark_goren_2025/schema_v3/formulation_core.tsv")
    p_doe_diag = Path(f"data/results/{run_id}/benchmark_goren_2025/derivation_v1/doe_decode_diagnostics.tsv")

    required = [curated_template, curated_tsv, p_core_v2, p_core_v3, p_doe_diag]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path("src/stage5_benchmark/run_alignment_eval_v1.py")
    if not eval_script.exists():
        raise FileNotFoundError(f"Required reusable script missing: {eval_script}")

    schema_cfg = {
        "schema_v2": p_core_v2,
        "schema_v3": p_core_v3,
    }
    schema_metrics: dict[str, Any] = {}
    per_schema_paths: dict[str, dict[str, Path]] = {}

    for schema_name, core_path in schema_cfg.items():
        schema_dir = out_dir / schema_name
        schema_dir.mkdir(parents=True, exist_ok=True)
        projected_tsv = schema_dir / f"projected_to_curated__{schema_name}.tsv"
        projection_trace = schema_dir / f"projection_trace__{schema_name}.tsv"
        project_core_to_curated(
            core_tsv=core_path,
            curated_template=curated_template,
            projected_out=projected_tsv,
            trace_out=projection_trace,
        )

        cmd = [
            sys.executable,
            str(eval_script),
            "--run-id",
            run_id,
            "--projected-tsv",
            str(projected_tsv),
            "--curated-tsv",
            str(curated_tsv),
            "--modes",
            modes,
            "--out-dir",
            str(schema_dir),
        ]
        subprocess.run(cmd, check=True)

        # Keep schema-specific copies in stable filenames.
        for name in ["alignment_rows.tsv", "metrics_summary.json", "failure_types.tsv"]:
            src = schema_dir / name
            dst = out_dir / f"{schema_name}__{name}"
            if src.exists():
                shutil.copyfile(src, dst)

        metrics = load_metrics_json(schema_dir / "metrics_summary.json")
        schema_metrics[schema_name] = metrics
        per_schema_paths[schema_name] = {
            "projected": projected_tsv,
            "alignment_rows": schema_dir / "alignment_rows.tsv",
            "metrics": schema_dir / "metrics_summary.json",
        }

    # Overall comparison table.
    summary_rows: list[dict[str, Any]] = []
    for schema_name, payload in schema_metrics.items():
        by_mode = payload.get("metrics_by_mode", {})
        for mode, mm in by_mode.items():
            rec = float(mm.get("recall", 0.0))
            prec = float(mm.get("precision", 0.0))
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            summary_rows.append(
                {
                    "schema": schema_name,
                    "mode": mode,
                    "total_curated_rows": int(mm.get("total_curated_rows", 0)),
                    "total_projected_rows": int(mm.get("total_projected_rows", 0)),
                    "matched_curated_rows": int(mm.get("matched_curated_rows", 0)),
                    "matched_projected_rows": int(mm.get("matched_projected_rows", 0)),
                    "recall": rec,
                    "precision": prec,
                    "f1": f1,
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(["mode", "schema"]).reset_index(drop=True)
    summary_tsv = out_dir / "metrics_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)

    # Per-DOI breakdown.
    per_doi_df = pd.concat(
        [
            per_doi_metrics(
                projected_tsv=per_schema_paths["schema_v2"]["projected"],
                curated_tsv=curated_tsv,
                alignment_rows_tsv=per_schema_paths["schema_v2"]["alignment_rows"],
                schema_name="schema_v2",
            ),
            per_doi_metrics(
                projected_tsv=per_schema_paths["schema_v3"]["projected"],
                curated_tsv=curated_tsv,
                alignment_rows_tsv=per_schema_paths["schema_v3"]["alignment_rows"],
                schema_name="schema_v3",
            ),
        ],
        ignore_index=True,
    ).sort_values(["doi", "mode", "schema"]).reset_index(drop=True)
    per_doi_path = out_dir / "per_doi_metrics.tsv"
    per_doi_df.to_csv(per_doi_path, sep="\t", index=False)

    doe_diag = pd.read_csv(p_doe_diag, sep="\t", dtype=str).fillna("")
    doe_enabled_dois = sorted(
        set(
            doe_diag[doe_diag["status"].astype(str).str.lower().isin(["ok", "partial"])]["doi"]
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
    )
    per_doi_doe_df = per_doi_df[per_doi_df["doi"].isin(doe_enabled_dois)].copy()
    per_doi_doe_path = out_dir / "per_doi_metrics_doe_enabled.tsv"
    per_doi_doe_df.to_csv(per_doi_doe_path, sep="\t", index=False)

    # Schema v2 vs v3 comparison table overall + DOE DOIs.
    cmp_overall = summary_df.pivot_table(
        index=["mode"],
        columns="schema",
        values=["recall", "precision", "f1"],
        aggfunc="first",
    )
    cmp_overall.columns = [f"{m}_{s}" for m, s in cmp_overall.columns]
    cmp_overall = cmp_overall.reset_index()
    cmp_overall["scope"] = "overall"

    cmp_doi = per_doi_doe_df.pivot_table(
        index=["doi", "mode"],
        columns="schema",
        values=["recall", "precision", "f1"],
        aggfunc="first",
    )
    if not cmp_doi.empty:
        cmp_doi.columns = [f"{m}_{s}" for m, s in cmp_doi.columns]
        cmp_doi = cmp_doi.reset_index()
        cmp_doi["scope"] = "doe_enabled_doi"
        cmp_df = pd.concat([cmp_overall, cmp_doi], ignore_index=True, sort=False)
    else:
        cmp_df = cmp_overall.copy()
    cmp_path = out_dir / "schema_v2_vs_v3_metrics_comparison.tsv"
    cmp_df.to_csv(cmp_path, sep="\t", index=False)

    metrics_json = {
        "run_id": run_id,
        "curated_template": str(curated_template),
        "curated_tsv": str(curated_tsv),
        "modes": [m.strip() for m in modes.split(",") if m.strip()],
        "doe_enabled_dois": doe_enabled_dois,
        "schema_metrics": schema_metrics,
        "outputs": {
            "metrics_summary_tsv": str(summary_tsv),
            "per_doi_metrics_tsv": str(per_doi_path),
            "per_doi_metrics_doe_enabled_tsv": str(per_doi_doe_path),
            "comparison_tsv": str(cmp_path),
            "schema_v2_alignment_rows_tsv": str(out_dir / "schema_v2__alignment_rows.tsv"),
            "schema_v3_alignment_rows_tsv": str(out_dir / "schema_v3__alignment_rows.tsv"),
            "schema_v2_metrics_json": str(out_dir / "schema_v2__metrics_summary.json"),
            "schema_v3_metrics_json": str(out_dir / "schema_v3__metrics_summary.json"),
        },
    }
    metrics_json_path = out_dir / "metrics_summary.json"
    metrics_json_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print(f"output_dir={out_dir}")
    print(f"output_metrics_summary_tsv={summary_tsv}")
    print(f"output_metrics_summary_json={metrics_json_path}")
    print(f"output_per_doi_metrics={per_doi_path}")
    print(f"output_per_doi_metrics_doe={per_doi_doe_path}")
    print(f"output_comparison={cmp_path}")
    focus = per_doi_df[(per_doi_df["doi"] == "10.1002/jps.24101") & (per_doi_df["mode"] == "strict")]
    if not focus.empty:
        print("focus_10.1002_jps_24101_strict=" + focus.to_json(orient="records"))


if __name__ == "__main__":
    main()
