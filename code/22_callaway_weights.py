#!/usr/bin/env python3
"""
22_callaway_weights.py — Visualise the implicit TWFE weights (D_c - D̄)²
========================================================================
Following Callaway, Goodman-Bacon & Sant'Anna (2024), the TWFE estimator
with continuous treatment can be written as a weighted average of cell-
specific effects, where each cell's identifying weight is proportional
to (D_c - D̄)², with D_c = 1/(d_c + 1) the treatment intensity and D̄
the cross-cell mean.

This script produces a transparency figure showing the empirical
distribution of those weights across cells in each country, and the
share of identification weight contributed by cells in each distance ring.

Output
------
  output/figures/fig_callaway_weights.pdf (copied to Overleaf figures/)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

CODE_DIR = Path(__file__).resolve().parent
ROOT     = CODE_DIR.parent
PANELS   = ROOT / "output" / "panels"
FIGS     = ROOT / "output" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

try:
    from config import OVERLEAF_FIGS as OVERLEAF_FIGS
except ImportError:
    OVERLEAF_FIGS = None

COUNTRIES = [
    ("KE", "Kenya",   "panel_kenya.parquet",   "#006970"),
    ("GH", "Ghana",   "panel_ghana.parquet",   "#FFB52C"),
    ("SN", "Senegal", "panel_senegal.parquet", "#85A5AD"),
]

# Distance ring breakpoints (km) for the share-of-weight panel
# The sample is every cell within 50 km of ANY TSIP site, so cells near only a non-lead
# site can be more than 50 km from a battery site; they carry the rest of the weight.
RING_EDGES = [0, 5, 10, 20, 50, np.inf]
RING_LABELS = ["0–5 km", "5–10 km", "10–20 km", "20–50 km", ">50 km"]


def load_cells(pfile):
    """One row per cell: cell_id, distance_km, D = 1/(d+1)."""
    p = pd.read_parquet(PANELS / pfile)
    cells = (p[["cell_id", "inv_dist_battery"]]
             .drop_duplicates("cell_id")
             .reset_index(drop=True))
    cells["D"]    = cells["inv_dist_battery"]
    cells["d_km"] = 1.0 / cells["D"] - 1.0
    return cells


def add_weights(cells):
    Dbar = cells["D"].mean()
    cells = cells.copy()
    cells["w"]  = (cells["D"] - Dbar) ** 2
    cells["Dbar"] = Dbar
    return cells


def ring_share(cells):
    """Share of total (D - D̄)² weight in each distance ring."""
    bins = pd.cut(cells["d_km"], bins=RING_EDGES, labels=RING_LABELS,
                  include_lowest=True)
    tot = cells["w"].sum()
    share = (cells.assign(ring=bins)
                  .groupby("ring", observed=True)["w"].sum() / tot)
    # Reindex to ensure all rings appear
    share = share.reindex(RING_LABELS, fill_value=0.0)
    return share


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2),
                              gridspec_kw={"height_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor("white")

    for j, (code, name, pfile, color) in enumerate(COUNTRIES):
        cells = add_weights(load_cells(pfile))
        Dbar  = cells["Dbar"].iloc[0]

        # ── Top row: scatter of weight against distance ─────────────────────
        ax = axes[0, j]
        ax.scatter(cells["d_km"], cells["w"],
                   s=10, alpha=0.45, color=color, edgecolor="none")
        # Mark D̄ as a vertical reference (in km space)
        d_at_Dbar = 1.0 / Dbar - 1.0
        if d_at_Dbar <= 50:
            ax.axvline(d_at_Dbar, color="#444", linestyle="--", linewidth=1)
        # Direct text label (a legend on an axvline is dropped by the house-style hook)
        ax.text(0.97, 0.95, f"$\\bar{{D}}$ = {Dbar:.3f}, i.e. $\\bar{{d}}$ = {d_at_Dbar:.0f} km"
                + ("" if d_at_Dbar <= 50 else "\n(beyond the axis)"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
        ax.set_xlabel("Distance to nearest battery site (km)", fontsize=10)
        if j == 0:
            ax.set_ylabel(r"Implicit weight $(D_c - \bar{D})^2$", fontsize=10)
        ax.set_xlim(0, 50)          # cells beyond 50 km are in the bottom-row ">50 km" bar
        ax.set_title(f"{name}  (n={len(cells):,} cells)", fontsize=11,
                     fontweight="bold", color="#0B4C5B")
        ax.grid(alpha=0.25)

        # ── Bottom row: share of identification weight by ring ──────────────
        ax2 = axes[1, j]
        share = ring_share(cells)
        bars = ax2.bar(range(len(share)), share.values * 100,
                       color=color, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, share.values):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                     f"{v*100:.0f}%", ha="center", fontsize=9,
                     fontweight="bold", color="#0B4C5B")
        ax2.set_xticks(range(len(share)))
        ax2.set_xticklabels(share.index, fontsize=9)
        ax2.set_ylabel("Share of total weight (%)", fontsize=10)
        ax2.set_ylim(0, max(85, share.values.max() * 100 * 1.25))
        ax2.set_xlabel("Distance ring", fontsize=10)
        ax2.grid(axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)

    fig.suptitle("Empirical TWFE identifying weights $(D_c - \\bar{D})^2$ by cell",
                 fontsize=13, fontweight="bold", color="#0B4C5B")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIGS / "fig_callaway_weights.pdf"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    if OVERLEAF_FIGS:
        shutil.copy(out, Path(OVERLEAF_FIGS) / out.name)
    print(f"✓  {out.name}")
    plt.close()


if __name__ == "__main__":
    main()
