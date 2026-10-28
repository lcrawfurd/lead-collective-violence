"""
51_slides_rings_pooled.py  --  Slide figure: effect by distance ring, three countries pooled.

Pooled version of fig_binary_distance_extensive (30_extensive_margin_figures.py), which
estimates each country separately. Same rings (0-5, 5-10, 10-15, 15-20 km from a battery
site, each x Post, reference > 20 km), outcome 1[any event], cell FE; country x year FE
in place of year FE, as in the headline estimates. SEs clustered on 1-degree spatial
blocks; 95% bars use the t(G-1) critical value. Riots and organised violence only.

Also the paper's version: battery ring minus non-lead ring (lead-specific gap by distance).

Outputs: output/figures/fig_slides_rings_pooled.pdf, fig_rings_gap_pooled.pdf
         (copied to Overleaf figures/).
"""
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PANELS, FIGURES, POST_YR, save_to_overleaf
from utils import add_block, tcrit

COUNTRIES = ["Kenya", "Ghana", "Senegal"]
BANDS = [(0, 5), (5, 10), (10, 15), (15, 20)]
parts = []
for cn in COUNTRIES:
    p = pd.read_parquet(PANELS / f"panel_{cn.lower()}.parquet")
    p["cell_id_str"] = cn[:2] + "_" + p.cell_id.astype(str)
    add_block(p, cn)       # 1-degree blocks: cluster for every reported SE
    p["cty_year"] = cn + "_" + p.year.astype(str)
    parts.append(p)
P = pd.concat(parts, ignore_index=True)
P["post"] = (P.year >= POST_YR).astype(int)
P["d_km"] = 1.0 / P.inv_dist_battery - 1
VARS = []
for lo, hi in BANDS:
    v = f"ring_{lo}_{hi}"
    P[v] = ((P.d_km > lo) & (P.d_km <= hi)).astype(int) * P.post
    VARS.append(v)

res = {}
for ycol, lab in [("n_riots", "Riots"), ("n_org_violence", "Organised violence")]:
    d = P.copy()
    d["y"] = (d[ycol] > 0).astype(float)
    m = pf.feols("y ~ " + " + ".join(VARS) + " | cell_id_str + cty_year", data=d,
                 vcov={"CRV1": "block"})
    CR = tcrit(m)
    rows = [(float(m.coef()[v]), float(m.se()[v])) for v in VARS]
    res[lab] = np.array(rows)
    print(lab, "  ".join(f"{b:.3f} ({s:.3f})" for b, s in rows))

COL = {"Riots": "#B3261E", "Organised violence": "#555555"}
OFF = {"Riots": -0.1, "Organised violence": 0.1}
x = np.arange(len(BANDS))
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for lab, r in res.items():
    ax.errorbar(x + OFF[lab], r[:, 0], yerr=CR * r[:, 1], fmt="o", ms=6, color=COL[lab],
                elinewidth=1.4, capsize=3)
    ylab = r[0, 0] if lab == "Riots" else r[0, 0] + CR * r[0, 1] + 0.02
    ax.text(x[0] + OFF[lab] + 0.12, ylab, lab, color=COL[lab], fontsize=10, va="center")
ax.axhline(0, color="#999999", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS])
ax.set_xlabel("Distance from a battery site (km)")
ax.set_ylabel("Change in Pr(any event) after 2007, vs cells > 20 km")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_slides_rings_pooled.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_slides_rings_pooled.pdf", str(out), subfolder="figures")
print("saved fig_slides_rings_pooled.pdf")


# ── Paper figure: battery ring minus non-lead ring (lead-specific gap by distance) ──
# Same regression with mutually exclusive non-lead rings (0-5, 5-10, 10-15, 15-20 km from
# the nearest non-lead TSIP site, each x Post) added alongside the battery rings:
#   1[any event]_ct = a_c + d_jt + sum_k b_k Bring_k,c x Post_t + sum_k g_k Nring_k,c x Post_t + e
# Plotted: b_k - g_k per ring (lincom), 95% bars from the block-clustered SE x t(G-1).
# The slides keep fig_slides_rings_pooled.pdf (battery rings only) above.
from utils import lincom
P["d_nl_km"] = 1.0 / P.inv_dist_nonlead - 1
NVARS = []
for lo, hi in BANDS:
    v = f"nring_{lo}_{hi}"
    P[v] = ((P.d_nl_km > lo) & (P.d_nl_km <= hi)).astype(int) * P.post
    NVARS.append(v)
gap_res = {}
for ycol, lab in [("n_riots", "Riots"), ("n_org_violence", "Organised violence")]:
    d = P.copy()
    d["y"] = (d[ycol] > 0).astype(float)
    m = pf.feols("y ~ " + " + ".join(VARS + NVARS) + " | cell_id_str + cty_year", data=d,
                 vcov={"CRV1": "block"})
    CRg = tcrit(m)
    g = [lincom(m, {b: 1, n: -1}) for b, n in zip(VARS, NVARS)]
    gap_res[lab] = np.array([(e, s_) for e, s_, _ in g])
    print(lab, "gap by ring:", "  ".join(f"{e:.3f} ({s_:.3f})" for e, s_, _ in g))

fig, ax = plt.subplots(figsize=(6.4, 3.9))
for lab, r in gap_res.items():
    ax.errorbar(x + OFF[lab], r[:, 0], yerr=CRg * r[:, 1], fmt="o", ms=6, color=COL[lab],
                elinewidth=1.4, capsize=3)
    ylab = r[0, 0] if lab == "Riots" else r[0, 0] + CRg * r[0, 1] + 0.02
    ax.text(x[0] + OFF[lab] + 0.12, ylab, lab, color=COL[lab], fontsize=10, va="center")
ax.axhline(0, color="#999999", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS])
ax.set_xlabel("Distance from the site (km)")
ax.set_ylabel("Battery ring minus non-lead ring, change in Pr(any event) after 2007")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_rings_gap_pooled.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_rings_gap_pooled.pdf", str(out), subfolder="figures")
print("saved fig_rings_gap_pooled.pdf")
