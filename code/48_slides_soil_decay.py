"""
48_slides_soil_decay.py  --  Slide figure: predicted soil lead by distance to a battery site.

Redraws the fitted distance-decay curve from the Beyond Hotspots replication
(ULAB Model/Replication/code/05_tsip_distance_decay.do) in house style, with a zero
line instead of the 0-5000 m average. Same data (output/tsipclean.dta, written by that
do-file) and same model: ln(lead) = alpha + beta*ln(dist) + site FE, clustered by site,
alpha recovered as mean(lnlead) - beta*mean(lndist), Duan smearing on retransformation.
Checks the area-weighted ring means against the published macros (soilinner, soilouter)
and stops if they differ.

Inputs:  data/beyond_hotspots/tsipclean.dta, data/beyond_hotspots/estimates.txt.
Outputs: output/figures/ulab_soil_decay.pdf (copied to Overleaf figures/; the slides use it).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGURES, save_to_overleaf

ULAB = DATA / "beyond_hotspots"      # inputs copied from the Beyond Hotspots replication package
d = pd.read_stata(ULAB / "tsipclean.dta", convert_categoricals=False)
d = d.dropna(subset=["lnlead", "lndist", "site_id"])

m = pf.feols("lnlead ~ lndist | site_id", data=d, vcov={"CRV1": "site_id"})
beta = float(m.coef()["lndist"])
alpha = d.lnlead.mean() - beta * d.lndist.mean()
smear = float(np.exp(m.resid()).mean())
k = np.exp(alpha) * smear


def ring_mean(r_in, r_out):
    """Area-weighted mean of k*r^beta over an annulus, as in 05_tsip_distance_decay.do."""
    return 2 * k / ((beta + 2) * (r_out**2 - r_in**2)) * (r_out**(beta + 2) - r_in**(beta + 2))


inner, outer = ring_mean(0.1, 165), ring_mean(165, 5000)
pub = (ULAB / "estimates.txt").read_text()
pub_in = float(re.search(r"\\soilinner\}\{\$?([\d.]+)", pub).group(1))
pub_out = float(re.search(r"\\soilouter\}\{\$?([\d.]+)", pub).group(1))
print(f"beta={beta:.4f}  N={m._N}  ring means {inner:.0f} / {outer:.0f} "
      f"(published {pub_in:.0f} / {pub_out:.0f})")
if round(inner) != pub_in or round(outer) != pub_out:
    raise SystemExit("Ring means do not match the published estimates; not writing figure.")

x = np.linspace(0.1, 5000, 2000)
y = np.exp(alpha + beta * np.log(x)) * smear
keep = y <= 11000

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(x[keep], y[keep], color="#1f5fa8", lw=2)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_ylim(-300, 10500)
ax.set_xlim(-50, 5000)
ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
ax.set_yticklabels([f"{v:,}" for v in [0, 2000, 4000, 6000, 8000, 10000]])
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.set_xticklabels([f"{v:,}" for v in [0, 1000, 2000, 3000, 4000, 5000]])
ax.set_xlabel("Distance to site (m)")
ax.set_ylabel("Predicted mean soil lead (mg/kg)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = FIGURES / "ulab_soil_decay.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("ulab_soil_decay.pdf", str(out), subfolder="figures")
print("saved ulab_soil_decay.pdf")
