"""
62_published_effects.py  --  Appendix Figure A10(b): published estimates of the effect of living
near a lead-acid battery site, by distance.

Top panel, test scores (SD):
  Kenya      Ipapa (2026), July 2026 working paper, Tables 2-3: schools within 2-10 km of a plant
             versus schools 10-20 km away.
  Indonesia  Berkhout et al. (2025), ADB Economics Working Paper 774, Table 6 col 2: numeracy,
             0-3 and 3-6 km bands versus beyond 6 km.
  Mexico     Litzow et al. (2026), May 2026 working paper, Figure A2 panel B (math, distance
             rings out to 10 miles versus the 25-30 mile ring), read from the PDF's vector graphics.
Bottom panel, terminated pregnancies (percentage points):
  Bangladesh Kundu, Mahzab and Plambeck (2026), SSRN 6901718, Figure 4: women within 1-15 km of a
             lead-emitting battery site versus women farther away, digitised from the image.
Each estimate is plotted at the outer distance of its treated area, with a 95% CI.

Inputs:  data/published_distance_effects.csv (values and their location in each paper)
Outputs: output/figures/fig_published_effects.pdf (copied to Overleaf figures/)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGURES, save_to_overleaf
from tufte_style import set_tufte_rcparams, tufte_ax

set_tufte_rcparams()
d = pd.read_csv(DATA / "published_distance_effects.csv")
COL = {"Kenya": "#006970", "Indonesia": "#E08A00", "Mexico": "#1f5fa8", "Bangladesh": "#B23A48"}
SHIFT = {"Kenya": 0.0, "Indonesia": -0.3, "Mexico": 0.0, "Bangladesh": 0.0}   # Indonesia sits beside Kenya at 6 km
LABEL_AT = {"Kenya": "max", "Indonesia": "min", "Mexico": "max", "Bangladesh": "max"}
YLAB = {"test_scores": "Effect on test scores (SD)",
        "pregnancy": "Effect on terminated pregnancies (percentage points)"}

fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.0), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.35))
for ax, panel in zip(axes, ["test_scores", "pregnancy"]):
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    for c, g in d[d.panel == panel].groupby("country", sort=False):
        x = g.distance_km + SHIFT[c]
        ax.errorbar(x, g.estimate, yerr=1.96 * g.se, fmt="o", ms=4, color=COL[c], ecolor=COL[c],
                    elinewidth=1.2, capsize=2.5)
        pt = g.loc[g.distance_km.idxmax() if LABEL_AT[c] == "max" else g.distance_km.idxmin()]
        ax.annotate(c, xy=(pt.distance_km + SHIFT[c], pt.estimate), xytext=(6, 0),
                    textcoords="offset points", color=COL[c], fontsize=9, va="center")
    ax.text(0, 1.04, YLAB[panel], transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    tufte_ax(ax)
axes[1].set_xlim(0, 19.5)
axes[1].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
axes[1].set_xlabel("Distance from the site (km)")
out = FIGURES / "fig_published_effects.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_published_effects.pdf", str(out), subfolder="figures")
print("saved fig_published_effects.pdf;", d.groupby("country").size().to_dict())
