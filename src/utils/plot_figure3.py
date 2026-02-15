# src/utils/plot_figure3.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from figure_style import apply_figure_style, stylize_axes_left_bottom_only

# Use centralized paths (no hard-coded roots)
from src.utils.paths import PROJECT_ROOT


ALLOWED_DECISIONS = ["accept_model1", "accept_model2", "override", "unclear"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot GT decision distribution (Figure 3) from gt_field_decisions TSV."
    )
    p.add_argument(
        "--gt-tsv",
        required=True,
        help="Path to gt_field_decisions__run_<run_id>.tsv (or __GT.tsv).",
    )
    p.add_argument(
        "--out-png",
        required=True,
        help="Output PNG path (e.g., figures/fig3_gt_decision_distribution.png).",
    )
    p.add_argument(
        "--out-tsv",
        default="",
        help="Optional output TSV path for the decision summary table.",
    )
    p.add_argument(
        "--out-pdf",
        default="",
        help="Output PDF path (e.g., figures/fig3_gt_decision_distribution.pdf).",
    )
    p.add_argument(
        "--exclude-fields",
        default="note,notes,comment,comments,free_text,free-text",
        help="Comma-separated field_name values to exclude (case-insensitive).",
    )
    p.add_argument(
        "--decision-col",
        default="gt_decision",
        help="Column name for GT decisions (default: gt_decision).",
    )
    p.add_argument(
        "--field-col",
        default="field_name",
        help="Column name for field names (default: field_name).",
    )
    p.add_argument(
    "--out-xlsx",
    default="",
    help="Optional Excel (.xlsx) output with two columns (gt_decision, percent).",
    )

    p.add_argument(
        "--title",
        default="GT decision distribution (conflict-only, field-level)",
        help="Plot title.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gt_path = (PROJECT_ROOT / args.gt_tsv).resolve() if not Path(args.gt_tsv).is_absolute() else Path(args.gt_tsv)
    out_png = (PROJECT_ROOT / args.out_png).resolve() if not Path(args.out_png).is_absolute() else Path(args.out_png)
    out_tsv = None
    out_xlsx = None
    if args.out_tsv.strip():
        out_tsv = (PROJECT_ROOT / args.out_tsv).resolve() if not Path(args.out_tsv).is_absolute() else Path(args.out_tsv)
        
    if args.out_xlsx.strip():
        out_xlsx = (PROJECT_ROOT / args.out_xlsx).resolve() if not Path(args.out_xlsx).is_absolute() else Path(args.out_xlsx)

    if not gt_path.exists():
        raise FileNotFoundError(f"GT TSV not found: {gt_path}")

    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    if args.decision_col not in df.columns:
        raise ValueError(f"Missing decision column '{args.decision_col}' in {gt_path.name}")
    if args.field_col not in df.columns:
        raise ValueError(f"Missing field column '{args.field_col}' in {gt_path.name}")

    # Normalize
    df[args.decision_col] = df[args.decision_col].astype(str).str.strip().str.lower()
    df[args.field_col] = df[args.field_col].astype(str).str.strip()

    # Exclude note-like fields (case-insensitive match on field_name)
    exclude = {s.strip().lower() for s in args.exclude_fields.split(",") if s.strip()}
    mask_excl = df[args.field_col].str.lower().isin(exclude)
    df_use = df.loc[~mask_excl].copy()

    # Keep only allowed decisions (in case of blanks or typos)
    df_use = df_use[df_use[args.decision_col].isin(ALLOWED_DECISIONS)].copy()

    if df_use.empty:
        raise ValueError(
            "No rows left after filtering. Check --exclude-fields and decision column values."
        )

    counts = df_use[args.decision_col].value_counts(dropna=False).to_dict()
    total = sum(counts.get(k, 0) for k in ALLOWED_DECISIONS)

    rows = []
    for k in ALLOWED_DECISIONS:
        c = int(counts.get(k, 0))
        pct = (c / total) * 100.0 if total > 0 else 0.0
        rows.append({"gt_decision": k, "count": c, "percent": pct})

    summary = pd.DataFrame(rows)

    # Apply global figure style BEFORE creating the figure/axes
    apply_figure_style()

    # Plot (bar chart)
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = plt.gca()

    x = summary["gt_decision"].tolist()
    y = summary["percent"].tolist()
    bar_width = 0.55
    lw = plt.rcParams.get("axes.linewidth", 1.0)
    bars = ax.bar(x, y, width=bar_width, linewidth=lw)


    ax.set_ylabel("Percent (%)")
    ax.set_xlabel("GT decision")
    ax.set_title(args.title)

    # Annotate bars with percent
    for b, pct in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Apply axes style AFTER plotting (hide top/right spines, disable top/right ticks)
    stylize_axes_left_bottom_only(ax)

    # Tight layout and save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png.with_suffix(".pdf"))
    plt.savefig(out_png.with_suffix(".png"), dpi=300)
    plt.close(fig)

    # Optional TSV output
    if out_tsv is not None:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_tsv, sep="\t", index=False)

    # Optional Excel output (two columns only)
    if out_xlsx is not None:
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        summary[["gt_decision", "percent"]].to_excel(out_xlsx, index=False)


    # Console print (useful in terminal)
    excluded_n = int(mask_excl.sum())
    print(f"[OK] Read: {gt_path}")
    print(f"[OK] Rows total: {len(df)} | excluded_by_field: {excluded_n} | used: {len(df_use)}")
    print("[OK] Summary:")
    print(summary.to_string(index=False))
    print(f"[OK] Wrote PNG: {out_png}")
    if out_tsv is not None:
        print(f"[OK] Wrote TSV: {out_tsv}")
    if out_xlsx is not None:
        print(f"[OK] Wrote XLSX: {out_xlsx}")


if __name__ == "__main__":
    main()
