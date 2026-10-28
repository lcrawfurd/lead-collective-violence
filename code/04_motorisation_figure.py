"""
04_motorisation_figure.py  --  Figure: motorisation in Kenya, Ghana and Senegal around 2007.

Three panels: (a) share of households owning a motorcycle, from DHS StatCompiler
(indicator HC_TRNS_H_SCT); (b) motor vehicles per 1,000 people, from the World Bank WDI
(indicator IS.VEH.NVEH.P3, retrieved 2026-07-03); (c) the real world lead price, US$ per
tonne in 2010 prices, from the World Bank Pink Sheet (as in 49). The first two series are
small published tables typed in below so the figure does not depend on a live download.

Inputs:  data/wb_lead_food_prices.csv (panel c); panels (a) and (b) are listed in this file.
Outputs: output/figures/motorisation_combined.pdf (copied to Overleaf figures/).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from config import DATA, FIGURES, END_YR, save_to_overleaf

# ---------------------------------------------------------------------------
# World Bank motor vehicles per 1,000 people (IS.VEH.NVEH.P3)
# Hardcoded from the World Bank WDI API (indicator IS.VEH.NVEH.P3), retrieved
# 2026-07-03, so the figure is deterministic and does not depend on a live
# network fetch. Re-fetch with wbdata if these need refreshing.
# ---------------------------------------------------------------------------
wb_series = {
    'Kenya':   [(2003, 17.0), (2004, 17.0), (2005, 18.0), (2006, 18.7),
                (2007, 20.0), (2008, 21.0), (2009, 23.0), (2010, 23.9)],
    'Ghana':   [(2003, 21.0), (2005, 16.0), (2006, 18.0), (2007, 34.0),
                (2008, 28.0), (2009, 30.0)],
    'Senegal': [(2005, 20.0), (2006, 20.0), (2007, 20.0), (2008, 22.0)],
}
wb_available = True

# ---------------------------------------------------------------------------
# DHS data: % households possessing a motorcycle
# Source: DHS StatCompiler, HC_TRNS_H_SCT
# ---------------------------------------------------------------------------
moto = {
    'Kenya':    [(1998, 0.9), (2003, 0.7), (2008, 2.1), (2014, 7.3), (2015, 9.5), (2020, 11.6), (2022, 13.5)],
    'Ghana':    [(1993, 1.1), (1998, 1.1), (2003, 2.1), (2008, 4.5), (2014, 8.8), (2016, 11.0), (2019, 13.4), (2022, 18.2)],
    'Senegal':  [(1993, 2.8), (1997, 3.5), (2005, 5.9), (2008, 6.7), (2010, 8.1), (2012, 11.1), (2014, 10.3), (2018, 12.5), (2020, 12.9)],
}

# ---------------------------------------------------------------------------
# Combined two-panel figure (fleet measures only; cumulative OICA sales dropped 2026-10-05
# because a running total of sales rises by construction even when annual sales fall)
# ---------------------------------------------------------------------------
colors = {
    'Kenya':    '#1f77b4',
    'Ghana':    '#ff7f0e',
    'Senegal':  '#2ca02c',
}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

panel_labels = ['(a)', '(b)', '(c)']

# --- Panel (a): DHS motorcycle ownership ---
ax = axes[0]
for country, series in moto.items():
    years = [y for y, v in series]
    vals  = [v for y, v in series]
    ax.plot(years, vals, marker='o', markersize=4, linewidth=1.6,
            color=colors[country], label=country)
ax.axvline(x=2007, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
ax.text(2007.5, ax.get_ylim()[1] * 0.97, '2007', fontsize=8, va='top', color='black', alpha=0.7)
ax.set_xlabel('Survey year', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.set_title(panel_labels[0] + ' Households owning a motorcycle', loc='left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel (b): WB motor vehicles per 1,000 ---
ax = axes[1]
if wb_available and wb_series:
    for country, series in wb_series.items():
        years = [y for y, v in series]
        vals  = [v for y, v in series]
        ax.plot(years, vals, marker='o', markersize=4, linewidth=1.6,
                color=colors[country], label=country)
ax.axvline(x=2007, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
ymax = ax.get_ylim()[1]
ax.text(2007.3, ymax * 0.97, '2007', fontsize=8, va='top', color='black', alpha=0.7)
ax.set_xlabel('Year', fontsize=10)
ax.set_title(panel_labels[1] + ' Motor vehicles per 1,000 people', loc='left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel (c): real world lead price (World Bank Pink Sheet), sample years ---
ax = axes[2]
px = pd.read_csv(DATA / 'wb_lead_food_prices.csv')
px = px[(px.year >= 1997) & (px.year <= END_YR)]   # sample years
ax.plot(px.year, px.lead_price, color='black', linewidth=1.6)
ax.set_ylim(0, px.lead_price.max() * 1.08)
ax.axvline(x=2007, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
ax.text(2007.5, ax.get_ylim()[1] * 0.97, '2007', fontsize=8, va='top', color='black', alpha=0.7)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Year', fontsize=10)
ax.set_title(panel_labels[2] + ' Real lead price, US$ per tonne', loc='left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Shared legend below all panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

outpath = str(FIGURES / 'motorisation_combined.pdf')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
save_to_overleaf('motorisation_combined.pdf', outpath, subfolder='figures')
print('Saved combined figure.')
