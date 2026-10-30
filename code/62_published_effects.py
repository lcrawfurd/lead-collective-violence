"""
62_published_effects.py  --  Appendix Figure A10(b): published estimates of the effect of living
near a battery recycling site on children's test scores, by distance.

Kenya: Ipapa (2026), July 2026 working paper, Tables 2-3 (schools within 2-10 km of a plant
versus schools 10-20 km away). Indonesia: Berkhout et al. (2025), ADB Economics Working Paper 774,
Table 6 col 2 (numeracy, 0-3 and 3-6 km bands versus beyond 6 km). Mexico: Litzow et al. (2026),
May 2026 working paper, Table 1 panel A col 4 (math, schools within 5 miles = 8.05 km versus
5-30 miles). Each estimate is plotted at the outer distance of its treated area, with a 95% CI.

Inputs:  data/published_distance_effects.csv (values and table locations from each paper)
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
COL = {"Kenya": "#006970", "Indonesia": "#E08A00", "Mexico": "#1f5fa8"}
SHIFT = {"Kenya": 0.0, "Indonesia": -0.3, "Mexico": 0.3}    # small offsets so points at the same distance do not overlap
LABEL_AT = {"Kenya": "max", "Indonesia": "min", "Mexico": "max"}  # which point carries the country label

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
for c, g in d.groupby("country", sort=False):
    x = g.distance_km + SHIFT[c]
    ax.errorbar(x, g.estimate, yerr=1.96 * g.se, fmt="o", ms=4, color=COL[c], ecolor=COL[c],
                elinewidth=1.2, capsize=2.5)
    # Direct label beside the country's last point
    last = g.loc[g.distance_km.idxmax() if LABEL_AT[c] == "max" else g.distance_km.idxmin()]
    ax.annotate(c, xy=(last.distance_km + SHIFT[c], last.estimate), xytext=(6, 0),
                textcoords="offset points", color=COL[c], fontsize=9, va="center")
ax.set_xlim(0, 11.8)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xlabel("Distance from the site (km)")
ax.text(0, 1.03, "Effect on test scores (SD)", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
tufte_ax(ax)
out = FIGURES / "fig_published_effects.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_published_effects.pdf", str(out), subfolder="figures")
print("saved fig_published_effects.pdf;", len(d), "estimates")
