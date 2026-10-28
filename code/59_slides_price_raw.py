"""
59_slides_price_raw.py  --  Slide figure: the real world lead price (left) next to
the raw riot rates behind the difference-in-differences (right).

Left:  real lead price, US$ per tonne in 2010 prices, 1997-2024 (as 49).
Right: share of cells with any riot in the year, pooled Kenya, Ghana and Senegal,
       for cells within 5 km of a battery site, cells within 5 km of a non-lead TSIP
       site (and not a battery site), and all other cells (as panel L of 47).

Inputs:  data/wb_lead_food_prices.csv, output/panels/panel_{kenya,ghana,senegal}.parquet (01).
Outputs: output/figures/fig_slides_price_raw.pdf (copied to Overleaf figures/).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGURES, PANELS, POST_YR, save_to_overleaf
from tufte_style import set_tufte_rcparams, tufte_ax

NEAR5 = 1.0 / 6.0                 # 1/(d+1) > 1/6  <=>  d < 5 km
set_tufte_rcparams()

px = pd.read_csv(DATA / "wb_lead_food_prices.csv")
px = px[(px.year >= 1997) & (px.year <= 2024)]

# Kenya, Ghana and Senegal panels from 01_build_panels.py, stacked with country-prefixed cell ids
p = pd.concat([pd.read_parquet(PANELS / f"panel_{c}.parquet").assign(cell_id=lambda d, c=c: c + "_" + d.cell_id.astype(str))
               for c in ("kenya", "ghana", "senegal")], ignore_index=True)
bat = p.inv_dist_battery > NEAR5
nl = p.inv_dist_nonlead > NEAR5
p["grp"] = np.where(bat, "bat", np.where(nl, "nl", "oth"))
raw = 100 * (p.assign(y=(p.n_riots > 0).astype(float))
             .groupby(["grp", "year"]).y.mean().unstack(0))
ncell = p.drop_duplicates("cell_id").grp.value_counts()
if ncell["bat"] != 30:
    print(f"note: {ncell['bat']} battery cells")

GROUPS = [("bat", f"Near a battery site ({ncell['bat']} cells)", "#006970"),
          ("nl", f"Near another toxic site ({ncell['nl']} cells)", "#E08A00"),
          ("oth", f"Other cells ({ncell['oth']:,})", "#9AA5A8")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9),
                             gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.25))

# Left: lead price
a1.plot(px.year, px.lead_price, color="#1f5fa8", lw=2)
a1.set_ylim(0, px.lead_price.max() * 1.08)
a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
a1.text(0, 1.03, "Real lead price, US$ per tonne (2010 prices)", transform=a1.transAxes,
        ha="left", va="bottom", fontsize=10)

# Right: raw riot rates, labelled at the line ends
last = raw.index.max()
for g, lab, col in GROUPS:
    a2.plot(raw.index, raw[g], "o-", ms=2.5, lw=1.5, color=col)
ends = sorted([(raw.loc[last, g], g, lab, col) for g, lab, col in GROUPS])
ymax = raw.max().max() * 1.05
gap = ymax * 0.07                                   # minimum vertical spacing of labels
ypos, prev = [], -np.inf
for y, *_ in ends:
    prev = max(y, prev + gap)
    ypos.append(prev)
for (y, g, lab, col), yl in zip(ends, ypos):
    a2.annotate(lab, xy=(last, y), xytext=(last + 1.2, yl), color=col, fontsize=8.5,
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.5) if abs(yl - y) > 0.3 else None)
a2.set_ylim(0, max(ymax, max(ypos) * 1.05))
a2.set_xlim(raw.index.min() - 0.5, last + 0.5)
a2.text(0, 1.03, "Cells with any riot in the year (%)", transform=a2.transAxes,
        ha="left", va="bottom", fontsize=10)

for ax in (a1, a2):
    ax.axvline(POST_YR - 0.5, color="black", ls="--", lw=0.9, alpha=0.6)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    tufte_ax(ax)

out = FIGURES / "fig_slides_price_raw.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_slides_price_raw.pdf", str(out), subfolder="figures")
print("saved fig_slides_price_raw.pdf;", dict(ncell))
