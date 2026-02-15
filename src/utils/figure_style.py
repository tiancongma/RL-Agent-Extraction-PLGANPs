import matplotlib as mpl


def apply_figure_style(
    *,
    font_family: str = "Arial",
    base_font_size: int = 9,
    title_size: int = 10,
    label_size: int = 9,
    tick_size: int = 8,
    legend_size: int = 8,
    axes_linewidth: float = 1.0,
    line_width: float = 1.0,
) -> None:
    """
    Global matplotlib style for paper figures (Windows/PPT friendly).
    Call once near the start of each plotting script.
    """
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": axes_linewidth,
            "lines.linewidth": line_width,
            "patch.linewidth": axes_linewidth,
            # Embed TrueType fonts in PDF for PPT compatibility
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def stylize_axes_left_bottom_only(ax) -> None:
    """
    Keep only left and bottom spines, disable top/right ticks.
    Call after plotting (or after axes creation).
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
