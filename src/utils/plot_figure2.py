import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from figure_style import apply_figure_style, stylize_axes_left_bottom_only


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def apply_style():
    # Global style lock for paper figures
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,   # embed TrueType fonts
        "ps.fonttype": 42,
    })


def plot_bar_counts(ax, df: pd.DataFrame, cat_col: str, count_col: str, title: str, xlabel: str):
    # Keep category order as in file unless you want to enforce one.
    cats = df[cat_col].astype(str).tolist()
    counts = df[count_col].astype(float).tolist()

    ax.bar(cats, counts, width=0.4)
    ax.set_title(title)
    # ax.set_xlabel(xlabel)
    ax.set_ylabel("# papers")

    # rotate if needed
    if any(len(c) > 10 for c in cats):
        ax.tick_params(axis="x", rotation=20)


def plot_box_with_points(ax, lengths: np.ndarray, title: str):
    ax.boxplot(
        lengths,
        vert=True,
        widths=0.4,
        showfliers=False,
    )

    # jittered points overlay (deterministic)
    rng = np.random.default_rng(42)
    x = 1 + rng.uniform(-0.08, 0.08, size=len(lengths))
    ax.scatter(x, lengths, s=18)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Text length (characters)")
    ax.set_xticks([1])
    ax.set_xticklabels(["Audit sample (N=20)"])


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 2 (three-panel) as a single PDF.")
    parser.add_argument(
        "--panelA-tsv",
        required=True,
        help="TSV with columns: category + count (e.g., panelA_source_type__sample.tsv OR panelA_source_category__retrieval_pool.tsv)"
    )
    parser.add_argument(
        "--panelB-tsv",
        required=True,
        help="TSV with columns including text_length (e.g., panelB_text_length.tsv)"
    )
    parser.add_argument(
        "--panelC-tsv",
        required=True,
        help="TSV with columns: parse_quality + count (e.g., panelC_parse_quality.tsv)"
    )
    parser.add_argument(
        "--out-pdf",
        required=True,
        help="Output PDF path, e.g., data/cleaned/figures/figure2/Figure2.pdf"
    )
    parser.add_argument(
        "--width-in",
        type=float,
        default=7.0,
        help="Figure width in inches (double-column typical ~7.0)"
    )
    parser.add_argument(
        "--height-in",
        type=float,
        default=2.6,
        help="Figure height in inches"
    )
    parser.add_argument(
        "--out-xlsx",
        default=None,
        help="Optional Excel output path, e.g., data/cleaned/figures/figure2/Figure2_raw_tables.xlsx"
    )
    args = parser.parse_args()

    # Keep existing behavior (apply_style), but also support the centralized style module if present.
    # We do NOT remove apply_style() to avoid changing other parts of the script.
    apply_style()

    # Load data
    a = read_tsv(args.panelA_tsv)
    b = read_tsv(args.panelB_tsv)
    c = read_tsv(args.panelC_tsv)

    # Infer column names for panel A
    # Accept either (source_type,count) or (source_category,count)
    if "count" not in a.columns:
        raise ValueError(f"Panel A TSV must have a 'count' column. Got columns: {list(a.columns)}")
    a_cat_col = "source_type" if "source_type" in a.columns else ("source_category" if "source_category" in a.columns else None)
    if a_cat_col is None:
        raise ValueError(f"Panel A TSV must have 'source_type' or 'source_category'. Got columns: {list(a.columns)}")

    # Panel B: text_length column
    if "text_length" not in b.columns:
        raise ValueError(f"Panel B TSV must have 'text_length'. Got columns: {list(b.columns)}")
    lengths = pd.to_numeric(b["text_length"], errors="coerce").dropna().to_numpy()
    if len(lengths) == 0:
        raise ValueError("Panel B has no valid numeric text_length values after parsing.")

    # Panel C columns
    if "count" not in c.columns:
        raise ValueError(f"Panel C TSV must have 'count'. Got columns: {list(c.columns)}")
    c_cat_col = "parse_quality" if "parse_quality" in c.columns else None
    if c_cat_col is None:
        raise ValueError(f"Panel C TSV must have 'parse_quality'. Got columns: {list(c.columns)}")

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(args.width_in, args.height_in))

    plot_bar_counts(
        axes[0], a, cat_col=a_cat_col, count_col="count",
        title="Source modality", xlabel="Source"
    )
    plot_box_with_points(
        axes[1], lengths=lengths, title="Document length"
    )
    plot_bar_counts(
        axes[2], c, cat_col=c_cat_col, count_col="count",
        title="Parse quality", xlabel="Quality"
    )

    # Apply spine/tick styling (prefer centralized helper if available)
    for ax in axes:
        try:
            stylize_axes_left_bottom_only(ax)
        except Exception:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(top=False, right=False)

    fig.tight_layout(w_pad=1.2)

    out_dir = os.path.dirname(args.out_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Wrote:", args.out_pdf)

    # Optional: write an Excel workbook containing the raw tables for collaborators
    if args.out_xlsx:
        out_xlsx_dir = os.path.dirname(args.out_xlsx)
        if out_xlsx_dir:
            os.makedirs(out_xlsx_dir, exist_ok=True)

        # Prepare consistent tables for export
        panelA = a[[a_cat_col, "count"]].copy()
        panelA.columns = ["category", "count"]

        panelB = b.copy()
        if "key" in panelB.columns and "source_type" in panelB.columns:
            panelB = panelB[["key", "source_type", "text_length"]].copy()
        else:
            panelB = panelB[[col for col in ["key", "source_type", "text_length"] if col in panelB.columns]].copy()

        panelC = c[[c_cat_col, "count"]].copy()
        panelC.columns = ["parse_quality", "count"]

        with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
            panelA.to_excel(writer, sheet_name="panelA_bar_counts", index=False)
            panelB.to_excel(writer, sheet_name="panelB_text_length", index=False)
            panelC.to_excel(writer, sheet_name="panelC_parse_quality", index=False)

        print("[OK] Wrote:", args.out_xlsx)


if __name__ == "__main__":
    main()
