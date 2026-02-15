# src/stage4_eval/rule_demo_pva_vs_size.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use centralized paths (no hard-coded roots)
from src.utils.paths import PROJECT_ROOT


REQUIRED_COLUMNS = [
    "pva_conc_percent_main",
    "size_nm_main",
]


DEFAULT_STRATA_COLS = ["emul_type_main", "emul_method_main", "organic_solvent_main"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo (descriptive only): PVA concentration vs particle size with 3-way stratified tables."
    )
    p.add_argument("--in-tsv", required=True, help="Path to formulations_consensus_weak.tsv")
    p.add_argument("--out-dir", required=True, help="Output directory (will be created if missing).")

    p.add_argument(
        "--pva-bins",
        default="1,2",
        help='Comma-separated thresholds in percent, e.g. "1,2" -> <=1, (1,2], >2',
    )

    p.add_argument(
        "--strata-cols",
        default=",".join(DEFAULT_STRATA_COLS),
        help="Comma-separated column names used for stratification (3-way recommended).",
    )

    p.add_argument(
        "--min-n-per-cell",
        type=int,
        default=2,
        help="Mark cells as sparse if total n (across PVA bins) is less than this threshold.",
    )

    p.add_argument(
        "--top-k-cells",
        type=int,
        default=25,
        help="Keep a Top-K table of the largest 3-way strata cells by total n (for easier demo).",
    )

    p.add_argument(
        "--plot-format",
        choices=["pdf", "png", "both", "none"],
        default="pdf",
        help="Plot output format (overall PVA-bin boxplot only).",
    )

    p.add_argument(
        "--title",
        default="PVA concentration vs particle size (descriptive, stratified)",
        help="Plot title.",
    )

    p.add_argument("--verbose", action="store_true", help="Print diagnostics.")
    return p.parse_args()


def _parse_bins(s: str) -> List[float]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) < 1:
        raise ValueError("pva-bins must have at least one threshold, e.g. '1,2'")
    vals: List[float] = []
    for x in parts:
        try:
            vals.append(float(x))
        except ValueError as e:
            raise ValueError(f"Invalid pva-bins value: {x}") from e
    vals = sorted(vals)
    if len(set(vals)) != len(vals):
        raise ValueError(f"Duplicate thresholds in pva-bins: {vals}")
    return vals


def _make_bin_labels(thresholds: List[float]) -> Tuple[List[float], List[str]]:
    # bins: (-inf, t1], (t1, t2], ..., (tk, inf)
    edges = [-np.inf] + thresholds + [np.inf]
    labels: List[str] = []
    if len(thresholds) == 1:
        t1 = thresholds[0]
        labels = [f"≤{t1:g}%", f">{t1:g}%"]
    else:
        labels.append(f"≤{thresholds[0]:g}%")
        for i in range(len(thresholds) - 1):
            a = thresholds[i]
            b = thresholds[i + 1]
            labels.append(f"({a:g}%, {b:g}%]")
        labels.append(f">{thresholds[-1]:g}%")
    return edges, labels


def _coerce_numeric(series: pd.Series) -> pd.Series:
    # Conservative coercion: keep digits, dot, minus. Strip units like "%", "nm", etc.
    s = series.astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _normalize_cat(series: pd.Series) -> pd.Series:
    # Normalize categorical strings for stable grouping (lowercase, collapse whitespace).
    # Keep empty as NA-like.
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _describe_sizes(x: np.ndarray) -> Dict[str, Any]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _build_overall_bin_table(df_valid: pd.DataFrame, pva_labels: List[str], thresholds: List[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lab in pva_labels:
        x = df_valid.loc[df_valid["pva_bin"] == lab, "size_nm"].to_numpy(dtype=float)
        desc = _describe_sizes(x)
        rows.append(
            {
                "scope": "overall",
                "pva_bin": lab,
                "pva_thresholds_percent": ",".join(f"{t:g}" for t in thresholds),
                **desc,
            }
        )
    return pd.DataFrame(rows)


def _build_stratified_table(
    df_valid: pd.DataFrame,
    strata_cols: List[str],
    pva_labels: List[str],
    thresholds: List[float],
    min_n_per_cell: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Group by strata (3-way), then within each cell, summarize by PVA bin
    grouped = df_valid.groupby(strata_cols, dropna=False)

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        # total n in this strata cell (regardless of PVA bin)
        total_n = int(g.shape[0])
        is_sparse = total_n < int(min_n_per_cell)

        strata_kv = {col: (val if (val is not np.nan) else np.nan) for col, val in zip(strata_cols, key)}
        strata_kv["cell_n_total"] = total_n
        strata_kv["is_sparse"] = bool(is_sparse)

        for lab in pva_labels:
            gg = g.loc[g["pva_bin"] == lab]
            x = gg["size_nm"].to_numpy(dtype=float)
            desc = _describe_sizes(x)
            rows.append(
                {
                    **strata_kv,
                    "pva_bin": lab,
                    "pva_thresholds_percent": ",".join(f"{t:g}" for t in thresholds),
                    **desc,
                }
            )

    out = pd.DataFrame(rows)

    # Order columns nicely
    col_order = (
        strata_cols
        + ["cell_n_total", "is_sparse", "pva_bin", "pva_thresholds_percent", "n", "mean", "median", "q25", "q75", "min", "max"]
    )
    for c in col_order:
        if c not in out.columns:
            out[c] = np.nan
    return out[col_order]


def _build_pairwise_tables(
    df_valid: pd.DataFrame,
    strata_cols: List[str],
    pva_labels: List[str],
    thresholds: List[float],
    min_n_per_cell: int,
) -> pd.DataFrame:
    # Build 2-way stratified summaries for each pair among strata_cols.
    if len(strata_cols) < 2:
        return pd.DataFrame()

    all_rows: List[pd.DataFrame] = []
    for i in range(len(strata_cols)):
        for j in range(i + 1, len(strata_cols)):
            pair = [strata_cols[i], strata_cols[j]]
            t = _build_stratified_table(
                df_valid=df_valid,
                strata_cols=pair,
                pva_labels=pva_labels,
                thresholds=thresholds,
                min_n_per_cell=min_n_per_cell,
            )
            t.insert(0, "strata_pair", f"{pair[0]}__{pair[1]}")
            all_rows.append(t)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def _build_counts_table(df_valid: pd.DataFrame, strata_cols: List[str]) -> pd.DataFrame:
    # A counts-only table helps you justify sparsity and show coverage.
    g = (
        df_valid.groupby(strata_cols, dropna=False)
        .size()
        .reset_index(name="cell_n_total")
        .sort_values("cell_n_total", ascending=False)
        .reset_index(drop=True)
    )
    return g


def main() -> int:
    args = parse_args()

    in_path = (PROJECT_ROOT / args.in_tsv).resolve() if not Path(args.in_tsv).is_absolute() else Path(args.in_tsv)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_bins(args.pva_bins)
    edges, pva_labels = _make_bin_labels(thresholds)

    strata_cols = [c.strip() for c in args.strata_cols.split(",") if c.strip()]
    if len(strata_cols) != 3:
        # You asked for 3-way stratification; enforce it to keep the demo consistent.
        raise ValueError(f"--strata-cols must contain exactly 3 columns (got {len(strata_cols)}): {strata_cols}")

    df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}. Available columns: {list(df.columns)}")

    for c in strata_cols:
        if c not in df.columns:
            raise KeyError(f"Missing strata column: {c}. Available columns: {list(df.columns)}")

    # Numeric coercion
    df["pva_pct"] = _coerce_numeric(df["pva_conc_percent_main"])
    df["size_nm"] = _coerce_numeric(df["size_nm_main"])

    # Normalize categorical strata columns (lowercase, stable whitespace)
    for c in strata_cols:
        df[c] = _normalize_cat(df[c])

    before = len(df)
    df_valid = df.loc[df["pva_pct"].notna() & df["size_nm"].notna()].copy()
    after = len(df_valid)

    # Bin PVA
    df_valid["pva_bin"] = pd.cut(
        df_valid["pva_pct"],
        bins=edges,
        labels=pva_labels,
        include_lowest=True,
        right=True,
    )

    if args.verbose:
        print(f"[info] input rows = {before}")
        print(f"[info] valid rows (pva & size numeric) = {after}")
        print("[info] overall PVA-bin counts:")
        print(df_valid["pva_bin"].value_counts(dropna=False).sort_index())
        print("[info] strata coverage (top 10 cells):")
        print(_build_counts_table(df_valid, strata_cols).head(10).to_string(index=False))

    # 1) Overall by PVA bin
    overall_bins = _build_overall_bin_table(df_valid, pva_labels, thresholds)
    (out_dir / "pva_vs_size__overall_bins.tsv").write_text(
        overall_bins.to_csv(sep="\t", index=False),
        encoding="utf-8",
    )

    # 2) Counts table (3-way cells)
    counts_3way = _build_counts_table(df_valid, strata_cols)
    counts_3way.to_csv(out_dir / "pva_vs_size__counts.tsv", sep="\t", index=False)

    # 3) Full 3-way stratified table (can be large)
    strat_3way = _build_stratified_table(
        df_valid=df_valid,
        strata_cols=strata_cols,
        pva_labels=pva_labels,
        thresholds=thresholds,
        min_n_per_cell=int(args.min_n_per_cell),
    )
    strat_3way.to_csv(out_dir / "pva_vs_size__stratified__3way.tsv", sep="\t", index=False)

    # 4) Pairwise stratified table
    strat_pairwise = _build_pairwise_tables(
        df_valid=df_valid,
        strata_cols=strata_cols,
        pva_labels=pva_labels,
        thresholds=thresholds,
        min_n_per_cell=int(args.min_n_per_cell),
    )
    strat_pairwise.to_csv(out_dir / "pva_vs_size__stratified__2way_pairwise.tsv", sep="\t", index=False)

    # 5) Top-K largest 3-way cells (for demo)
    top_cells = counts_3way.head(int(args.top_k_cells)).copy()
    # Merge the top-cells list back to stratified table for a compact demo view
    # Use a left merge on the three strata cols.
    demo_top = strat_3way.merge(
        top_cells[strata_cols],
        on=strata_cols,
        how="inner",
    )
    demo_top.to_csv(out_dir / "pva_vs_size__stratified__3way__topK.tsv", sep="\t", index=False)

    # Plot (overall only), descriptive
    if args.plot_format != "none":
        data_by_bin = []
        for lab in pva_labels:
            g = df_valid.loc[df_valid["pva_bin"] == lab, "size_nm"].astype(float).dropna().to_numpy()
            data_by_bin.append(g)

        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.boxplot(
            data_by_bin,
            labels=pva_labels,
            showfliers=True,
            widths=0.6,
        )
        ax.set_title(args.title)
        ax.set_xlabel("PVA concentration bin")
        ax.set_ylabel("Particle size (nm)")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

        # n annotations
        y_max = np.nanmax([np.nanmax(x) if len(x) > 0 else np.nan for x in data_by_bin])
        if not np.isnan(y_max):
            for i, x in enumerate(data_by_bin, start=1):
                ax.text(i, y_max, f"n={len(x)}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()

        out_pdf = out_dir / "pva_vs_size__plot.pdf"
        out_png = out_dir / "pva_vs_size__plot.png"
        if args.plot_format in ["pdf", "both"]:
            fig.savefig(out_pdf)
        if args.plot_format in ["png", "both"]:
            fig.savefig(out_png, dpi=300)
        plt.close(fig)

    if args.verbose:
        print(f"[OK] wrote: {out_dir / 'pva_vs_size__overall_bins.tsv'}")
        print(f"[OK] wrote: {out_dir / 'pva_vs_size__counts.tsv'}")
        print(f"[OK] wrote: {out_dir / 'pva_vs_size__stratified__3way.tsv'}")
        print(f"[OK] wrote: {out_dir / 'pva_vs_size__stratified__3way__topK.tsv'}")
        print(f"[OK] wrote: {out_dir / 'pva_vs_size__stratified__2way_pairwise.tsv'}")
        if args.plot_format != "none":
            if args.plot_format in ["pdf", "both"]:
                print(f"[OK] wrote: {out_dir / 'pva_vs_size__plot.pdf'}")
            if args.plot_format in ["png", "both"]:
                print(f"[OK] wrote: {out_dir / 'pva_vs_size__plot.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
