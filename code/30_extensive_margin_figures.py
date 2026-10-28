"""
30_extensive_margin_figures.py
Distance-ring figure on the extensive margin (1[any event in cell-year]): mutually
exclusive 0-5, 5-10, 10-15, 15-20 km rings x Post in one regression per country and
outcome (reference: cells beyond 20 km), cell + year FE, SEs clustered on 1-degree blocks.

Inputs:  output/panels/panel_{kenya,ghana,senegal}.parquet (01_build_panels.py)
Outputs: output/figures/fig_binary_distance_extensive.pdf (+ Overleaf figures/)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyfixest as pf

from config import PANELS, FIGURES, OVERLEAF, POST_YR
from utils import stars, add_block, coef_sep, tcrit
from tufte_style import set_tufte_rcparams, tufte_ax
set_tufte_rcparams()

print("Loading panels...")
PAN = {
    "Kenya":   pd.read_parquet(str(PANELS / "panel_kenya.parquet")),
    "Ghana":   pd.read_parquet(str(PANELS / "panel_ghana.parquet")),
    "Senegal": pd.read_parquet(str(PANELS / "panel_senegal.parquet")),
}
for _c, p in PAN.items():
    p["post"] = (p["year"] >= POST_YR).astype(int)
    add_block(p, _c)          # 1-degree blocks: the cluster for every reported SE

OUTCOMES = [("n_riots", "Riots"), ("n_protests", "Protests"),
            ("n_org_violence", "Organised violence")]


# ── Distance rings (extensive margin), joint mutually-exclusive bands ─────────
print("Distance rings (extensive margin)...")
BANDS = [(0, 5), (5, 10), (10, 15), (15, 20)]

def ring_regression(panel, ycount):
    d = panel.copy()
    d["y"] = (d[ycount] > 0).astype(float)
    # distance from inv_dist: d_km = 1/inv - 1
    d["d_km"] = np.where(d["inv_dist_battery"] > 0, 1.0/d["inv_dist_battery"] - 1, np.inf)
    band_vars = []
    for lo, hi in BANDS:
        v = f"ring_{lo}_{hi}"
        in_band = ((d["d_km"] > lo) & (d["d_km"] <= hi)).astype(int)
        d[v] = in_band * d["post"]
        band_vars.append((lo, hi, v))
    formula = "y ~ " + " + ".join(v for _,_,v in band_vars) + " | cell_id + year"
    m = pf.feols(formula, data=d, vcov={"CRV1": "block"})
    out = []
    for i, (lo, hi, v) in enumerate(band_vars):
        b, se, p = coef_sep(m, v)
        n_cells = int(((d["d_km"] > lo) & (d["d_km"] <= hi)).groupby(d["cell_id"]).any().sum())
        out.append(dict(lo=lo, hi=hi, coef=b, se=se, p=p, stars=stars(p), n=n_cells, crit=tcrit(m)))
    return out

ring = {c: {lab: ring_regression(PAN[c], yc) for yc, lab in OUTCOMES} for c in PAN}

# Figure. Use capped error bars (not filled bands) and a fixed, legible y-range:
# a few ring x outcome cells have very wide SEs (sparse rings),
# whose 95% bands run to implausible values (>1 in probability units). Filled
# bands let one such cell auto-scale the axis and crush the actual estimates to
# an invisible flat line. We fix the y-axis to the range the point estimates and
# their credible CIs occupy; error bars that would exceed the frame are clipped,
# and a caption note flags that the table reports all coefficients and SEs.
YLO, YHI = -0.15, 0.55
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
mids = [(lo+hi)/2 for lo,hi in BANDS]
xt = [f"{lo}-{hi}" for lo,hi in BANDS]
cols = {"Riots":"tab:blue","Protests":"tab:orange","Organised violence":"tab:red"}
offset = {"Riots":-0.35, "Protests":0.0, "Organised violence":0.35}  # dodge x for legibility
for ax, c in zip(axes, PAN):
    for _, lab in OUTCOMES:
        r = ring[c][lab]
        x = [m + offset[lab] for m in mids]
        coef = np.array([d["coef"] for d in r])
        se = np.array([d["se"] for d in r])
        cr = np.array([d["crit"] for d in r])
        # clip drawn CI to the frame so degenerate SEs don't dominate
        lo = np.clip(coef - cr*se, YLO, YHI)
        hi = np.clip(coef + cr*se, YLO, YHI)
        ax.errorbar(x, coef, yerr=[coef-lo, hi-coef], fmt="o", ms=4,
                    color=cols[lab], ecolor=cols[lab], elinewidth=1.0,
                    capsize=2, label=lab)
    ax.axhline(0, color="#999999", lw=0.7, ls="--")
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("Distance ring (km)")
    ax.set_ylabel(r"$\Delta$ Pr(any event), ref: $>$20 km")
    ax.set_title(c); ax.set_xticks(mids); ax.set_xticklabels(xt, fontsize=8)
    tufte_ax(ax)
    if c == list(PAN)[0]:
        ax.legend()
plt.suptitle("Marginal Effect by Distance Ring (Extensive Margin, Joint Regression)", fontsize=11)
plt.tight_layout()
for dest in [str(FIGURES), str(OVERLEAF / "figures") if OVERLEAF else None]:
    if dest:
        plt.savefig(f"{dest}/fig_binary_distance_extensive.pdf", bbox_inches="tight")
plt.close()
print("  Saved: fig_binary_distance_extensive.pdf")

print("\nDone.")
