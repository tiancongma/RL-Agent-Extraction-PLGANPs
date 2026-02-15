from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROJECT_ROOT


REQUIRED_COLUMNS = ["pva_conc_percent_main", "size_nm_main"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo (descriptive only): PVA vs size with 2-way or 3-way stratified tables (demo-only script)."
    )
    p.add_argument("--in-tsv", required=True, help="Input TSV (e.g., formulations_consensus_weak__canon_demo.tsv)")
    p.add_argument("--out-dir", required=True, help="Output directory (will be created if missing).")

    p.add_argument("--pva-bins", default="1,2", help='Comma-separated thresholds, e.g. "1,2" -> <=1, (1,2], >2')
    p.add_argument(
        "--strata-cols",
        default="emul_type_canon_demo,emul_method_canon_demo",
        help="Comma-separated column names used for stratification (2 or 3 columns).",
    )
    p.add_argument("--min-n-per-cell", type=int, default=3, help="Mark cell as sparse if total n < this.")
    p.add_argument("--top-k-cells", type=int, default=20, help="Top-K strata cells by total n.")
    p.add_argument("--plot-format", choices=["pdf", "png", "both", "none"], default="none")
    p.add_argument("--title", default="PVA concentration vs particle size (descriptive, stratified)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


def _parse_bins(s: str) -> List[float]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) < 1:
        raise ValueError("pva-bins must have at least one threshold, e.g. '1,2'")
    vals = sorted(float(x) for x in parts)
    if len(set(vals)) != len(vals):
        raise ValueError(f"Duplicate thresholds in pva-bins: {vals}")
    return vals


def _make_bin_labels(thresholds: List[float]) -> tuple[list[float], list[str]]:
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
    s = series.astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _describe_sizes(x: np.ndarray) -> Dict[str, Any]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _counts_table(df_valid: pd.DataFrame, strata_cols: List[str]) -> pd.DataFrame:
    return (
        df_valid.groupby(strata_cols, dropna=False)
        .size()
        .reset_index(name="cell_n_total")
        .sort_values("cell_n_total", ascending=False)
        .reset_index(drop=True)
    )


def _strat_table(
    df_valid: pd.DataFrame,
    strata_cols: List[str],
    pva_labels: List[str],
    thresholds: List[float],
    min_n_per_cell: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = df_valid.groupby(strata_cols, dropna=False)

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        total_n = int(g.shape[0])
        is_sparse = total_n < int(min_n_per_cell)

        strata_kv = {col: val for col, val in zip(strata_cols, key)}
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
    col_order = (
        strata_cols
        + ["cell_n_total", "is_sparse", "pva_bin", "pva_thresholds_percent", "n", "mean", "median", "q25", "q75", "min", "max"]
    )
    return out[col_order]


def main() -> int:
    args = parse_args()

    in_path = _resolve(args.in_tsv)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_bins(args.pva_bins)
    edges, pva_labels = _make_bin_labels(thresholds)

    strata_cols = [c.strip() for c in args.strata_cols.split(",") if c.strip()]
    if len(strata_cols) not in (2, 3):
        raise ValueError(f"--strata-cols must contain 2 or 3 columns (got {len(strata_cols)}): {strata_cols}")

    df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
    for c in strata_cols:
        if c not in df.columns:
            raise KeyError(f"Missing strata column: {c}")

    df["pva_pct"] = _coerce_numeric(df["pva_conc_percent_main"])
    df["size_nm"] = _coerce_numeric(df["size_nm_main"])

    df_valid = df.loc[df["pva_pct"].notna() & df["size_nm"].notna()].copy()
    df_valid["pva_bin"] = pd.cut(df_valid["pva_pct"], bins=edges, labels=pva_labels, include_lowest=True, right=True)

    if args.verbose:
        print(f"[info] input rows={len(df)}")
        print(f"[info] valid rows (pva & size numeric)={len(df_valid)}")
        print("[info] pva_bin counts:")
        print(df_valid["pva_bin"].value_counts(dropna=False).sort_index())

    counts = _counts_table(df_valid, strata_cols)
    counts_out = out_dir / "pva_vs_size__counts__2way.tsv"
    counts.to_csv(counts_out, sep="\t", index=False)

    strat = _strat_table(df_valid, strata_cols, pva_labels, thresholds, int(args.min_n_per_cell))
    strat_out = out_dir / "pva_vs_size__stratified__2way.tsv"
    strat.to_csv(strat_out, sep="\t", index=False)

    # Top-K for quick inspection
    top_cells = counts.head(int(args.top_k_cells))[strata_cols]
    strat_top = strat.merge(top_cells, on=strata_cols, how="inner")
    strat_top_out = out_dir / "pva_vs_size__stratified__2way__topK.tsv"
    strat_top.to_csv(strat_top_out, sep="\t", index=False)

    if args.verbose:
        print(f"[OK] wrote: {counts_out}")
        print(f"[OK] wrote: {strat_out}")
        print(f"[OK] wrote: {strat_top_out}")

    # Optional plot (overall only)
    if args.plot_format != "none":
        data_by_bin = []
        for lab in pva_labels:
            g = df_valid.loc[df_valid["pva_bin"] == lab, "size_nm"].astype(float).dropna().to_numpy()
            data_by_bin.append(g)

        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.boxplot(data_by_bin, labels=pva_labels, showfliers=True, widths=0.6)
        ax.set_title(args.title)
        ax.set_xlabel("PVA concentration bin")
        ax.set_ylabel("Particle size (nm)")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
        fig.tight_layout()

        if args.plot_format in ("pdf", "both"):
            fig.savefig(out_dir / "pva_vs_size__plot__2way.pdf")
        if args.plot_format in ("png", "both"):
            fig.savefig(out_dir / "pva_vs_size__plot__2way.png", dpi=300)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
