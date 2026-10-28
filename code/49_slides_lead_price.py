"""
49_slides_lead_price.py  --  Slide figure: the real world lead price, 1997-2024.

Same series and years as panel (a) of fig_prices (38_price_placebo.py), lead only,
styled to sit next to motorisation_combined on the "Before v. after" slide.

Outputs: output/figures/fig_slides_lead_price.pdf (copied to Overleaf figures/).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGURES, POST_YR, save_to_overleaf

px = pd.read_csv(DATA / "wb_lead_food_prices.csv")
px = px[(px.year >= 1997) & (px.year <= 2024)]
peak = px.loc[px.lead_price.idxmax()]

fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.plot(px.year, px.lead_price, color="#1f5fa8", lw=2)
ax.axvline(POST_YR, color="black", ls="--", lw=1.0, alpha=0.7)
ax.annotate(f"{int(peak.year)} peak", xy=(peak.year, peak.lead_price),
            xytext=(peak.year + 1.5, peak.lead_price), fontsize=8, color="#1f5fa8",
            va="center", arrowprops=dict(arrowstyle="-", color="#1f5fa8", lw=0.7))
ax.set_ylim(0, px.lead_price.max() * 1.08)
ax.set_xticks([2000, 2005, 2010, 2015, 2020])
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlabel("Year", fontsize=10)
ax.set_title("Real lead price, US$ per tonne (2010 prices)", loc="left", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_slides_lead_price.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_slides_lead_price.pdf", str(out), subfolder="figures")
print(f"peak {int(peak.year)}: {peak.lead_price:,.0f}; saved fig_slides_lead_price.pdf")
