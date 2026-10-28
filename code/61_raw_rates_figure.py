"""
61_raw_rates_figure.py  --  Paper figure: the raw data behind the difference-in-differences.

Share of cells with any ACLED riot in each year, Kenya, Ghana and Senegal pooled, for cells
within 5 km of a battery recycling site, cells within 5 km of a non-lead TSIP site (and not
a battery site), and all other cells. No regression; the paper's estimates are differences
in these lines before and after 2007, net of country-year effects.

Inputs:  output/panels/panel_{kenya,ghana,senegal}.parquet (01_build_panels.py)
Outputs: output/figures/fig_raw_rates.pdf (copied to Overleaf figures/)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES, PANELS, POST_YR, save_to_overleaf
from tufte_style import set_tufte_rcparams, tufte_ax

NEAR5 = 1.0 / 6.0                 # 1/(d+1) > 1/6  <=>  d < 5 km
set_tufte_rcparams()

p = pd.concat([pd.read_parquet(PANELS / f"panel_{c}.parquet").assign(country=c)
               for c in ["kenya", "ghana", "senegal"]], ignore_index=True)
p["cid"] = p.country + "_" + p.cell_id.astype(str)
bat, nl = p.inv_dist_battery > NEAR5, p.inv_dist_nonlead > NEAR5
p["grp"] = np.where(bat, "bat", np.where(nl, "nl", "oth"))
raw = 100 * (p.assign(y=(p.n_riots > 0).astype(float)).groupby(["grp", "year"]).y.mean().unstack(0))
ncell = p.drop_duplicates("cid").grp.value_counts()

GROUPS = [("bat", f"Within 5 km of a battery site ({ncell['bat']} cells)", "#006970"),
          ("nl", f"Within 5 km of another toxic site ({ncell['nl']} cells)", "#E08A00"),
          ("oth", f"All other cells ({ncell['oth']:,})", "#9AA5A8")]

fig, ax = plt.subplots(figsize=(7.5, 3.9))
for g, _, col in GROUPS:
    ax.plot(raw.index, raw[g], "o-", ms=2.5, lw=1.5, color=col)
last = raw.index.max()
ends = sorted([(raw.loc[last, g], lab, col) for g, lab, col in GROUPS])
gap = raw.max().max() * 0.07               # minimum vertical spacing between end labels
ypos, prev = [], -np.inf
for y, *_ in ends:
    prev = max(y, prev + gap)
    ypos.append(prev)
for (y, lab, col), yl in zip(ends, ypos):
    ax.annotate(lab, xy=(last, y), xytext=(last + 1.0, yl), color=col, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.5) if abs(yl - y) > 0.3 else None)
ax.axvline(POST_YR - 0.5, color="black", ls="--", lw=0.9, alpha=0.6)
ax.set_xlim(raw.index.min() - 0.5, last + 0.5)
ax.set_ylim(0, max(raw.max().max(), max(ypos)) * 1.05)
ax.set_xticks([2000, 2005, 2010, 2015, 2020])
ax.text(0, 1.03, "Cells with any riot in the year (%)", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10)
tufte_ax(ax)
out = FIGURES / "fig_raw_rates.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_raw_rates.pdf", str(out), subfolder="figures")
print("saved fig_raw_rates.pdf;", dict(ncell))
