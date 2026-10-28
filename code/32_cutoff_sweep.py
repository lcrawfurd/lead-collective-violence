#!/usr/bin/env python3
"""
32_cutoff_sweep.py — Treatment-timing sweep (extensive margin)
==============================================================================
Generalises the discrete 2005/2009 timing robustness to a
continuous sweep of assumed onset years for ALL three countries plus pooled.
For each cutoff T in 2004..2011, re-estimate the paper's main design with onset T:

  1[any riot]_ct = α_c + δ_t + β · B_c · 1[t >= T] + γ · N_c · 1[t >= T] + ε_ct

with B_c, N_c = 1[within 5 km of a battery / non-lead site] (pooled uses country×year
FEs), and plot β̂ with 95% CIs (block-clustered SEs, t(G-1) critical value) against T.

Outputs
-------
  output/figures/fig_cutoff_sweep.pdf      (+ Overleaf figures/)
  output/panels/cutoff_sweep.csv            (read by 11_generate_values.py)
"""

import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt

from config import PANELS, FIGURES, save_to_overleaf
from utils import add_block, coef_sep, tcrit


try:
    from tufte_style import set_tufte_rcparams, tufte_ax
    set_tufte_rcparams()
    _HAS_TUFTE = True
except Exception:
    _HAS_TUFTE = False
    def tufte_ax(ax):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        return ax

CUTOFFS = list(range(2004, 2012))   # 2004..2011
NEAR5 = 1.0 / 6.0                   # 1/(d+1) > 1/6  <=>  d < 5 km
COLORS = {"Kenya": "#006970", "Ghana": "#FFB52C", "Senegal": "#85A5AD", "Pooled": "#C8102E"}

print("Loading panels...")
PAN = {
    "Kenya":   pd.read_parquet(str(PANELS / "panel_kenya.parquet")),
    "Ghana":   pd.read_parquet(str(PANELS / "panel_ghana.parquet")),
    "Senegal": pd.read_parquet(str(PANELS / "panel_senegal.parquet")),
}

for _c, _p in PAN.items():
    add_block(_p, _c)         # 1-degree blocks (country-specific)

# Pooled panel with cross-country cell ids and country×year groups
_parts = []
for c, p in PAN.items():
    pp = p.copy()
    pp["country"]     = c
    pp["cell_id_str"] = c[:2] + "_" + pp["cell_id"].astype(str)
    add_block(pp, c)       # 1-degree blocks: cluster for every reported SE
    _parts.append(pp)
POOLED = pd.concat(_parts, ignore_index=True)


def coef_at(panel, cutoff, pooled=False):
    """Main design with onset year `cutoff` instead of 2007: 1[any riot] on
    bt = 1[within 5 km of a battery site] x 1[year >= cutoff] and nt = the same for
    non-lead sites, one regression; returns the battery coefficient (block-clustered)."""
    d = panel.copy()
    d["y"]  = (d["n_riots"] > 0).astype(float)
    on = (d["year"] >= cutoff).astype(float)
    d["bt"] = (d["inv_dist_battery"] > NEAR5).astype(float) * on
    d["nt"] = (d["inv_dist_nonlead"] > NEAR5).astype(float) * on
    if pooled:
        d["cty_year"] = d["country"] + "_" + d["year"].astype(str)
        m = pf.feols("y ~ bt + nt | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    else:
        m = pf.feols("y ~ bt + nt | cell_id + year", data=d, vcov={"CRV1": "block"})
    b, se, p = coef_sep(m, "bt")
    return b, se, p, tcrit(m)


rows = []
for label in ["Kenya", "Ghana", "Senegal", "Pooled"]:
    pooled = (label == "Pooled")
    panel = POOLED if pooled else PAN[label]
    for T in CUTOFFS:
        b, se, p, cr = coef_at(panel, T, pooled)
        rows.append({"country": label, "cutoff": T, "coef": b, "se": se, "p": p, "crit": cr})
        print(f"  {label:8s} T={T}: {b:+.3f} ({se:.3f}) p={p:.3f}")

df = pd.DataFrame(rows)
df.to_csv(str(PANELS / "cutoff_sweep.csv"), index=False)
print(f"Saved: {PANELS / 'cutoff_sweep.csv'}")

# ── Small-multiples figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
fig.patch.set_facecolor("white")
for ax, label in zip(axes.flat, ["Kenya", "Ghana", "Senegal", "Pooled"]):
    g = df[df.country == label].sort_values("cutoff")
    ylo = g.coef - g.crit * g.se
    yhi = g.coef + g.crit * g.se
    col = COLORS[label]
    ax.fill_between(g.cutoff, ylo, yhi, color=col, alpha=0.18, linewidth=0)
    ax.plot(g.cutoff, g.coef, marker="o", markersize=4, linewidth=2, color=col)
    ax.axhline(0, color="#666", linewidth=0.7)
    ax.axvline(2007, color="#444", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=0.2)
    tufte_ax(ax)
for ax in axes[1, :]:
    ax.set_xlabel("Assumed treatment onset year", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("Battery riot coefficient", fontsize=10)
plt.tight_layout()
out = FIGURES / "fig_cutoff_sweep.pdf"
plt.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
print(f"✓  {out.name}")
plt.close()
save_to_overleaf("fig_cutoff_sweep.pdf", str(FIGURES / "fig_cutoff_sweep.pdf"), subfolder="figures")
print("Done.")
