"""
Shared Tufte-style matplotlib helpers for the lead-crime figures.

Applies a few of Edward Tufte's principles: maximise the data-ink ratio
(no top/right spines, light frame, minimal grid), keep colour purposeful,
and let the data dominate. Import and call set_tufte_rcparams() once at the
top of a figure script, then tufte_ax(ax) on each axis.
"""
import matplotlib as mpl

_GREY = "#555555"


def set_tufte_rcparams():
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": _GREY,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.color": _GREY,
        "ytick.color": _GREY,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def tufte_ax(ax):
    """Strip top/right spines and lighten the remaining frame and ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GREY)
    ax.tick_params(colors=_GREY, labelcolor="black")
    return ax
