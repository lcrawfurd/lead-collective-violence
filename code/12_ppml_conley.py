"""
12_ppml_conley.py
SE comparison on the main design (any event; 5 km battery and non-lead x Post):
battery coefficient and lead-specific gap under cell-clustered, 1-degree block,
Conley 50 km and Conley 100 km SEs.

Inputs:  output/panels/panel_{kenya,ghana,senegal}.parquet (01_build_panels.py)
Outputs: output/tables/tab_ppml_conley.tex  SE comparison (+ Overleaf)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings
warnings.filterwarnings("ignore")

from utils import stars, conley_vcov, add_block, coef_sep, lincom
from config import TABLES, PANELS, OVERLEAF

PANEL = str(PANELS)
TABS  = str(TABLES)

# ── Load panels ──────────────────────────────────────────────────────────────

print("Loading panels...")
ke = pd.read_parquet(f"{PANEL}/panel_kenya.parquet")
gh = pd.read_parquet(f"{PANEL}/panel_ghana.parquet")
sn = pd.read_parquet(f"{PANEL}/panel_senegal.parquet")
countries = {"Kenya": ke, "Ghana": gh, "Senegal": sn}
for _cn, _p in countries.items():
    add_block(_p, _cn)        # 1-degree spatial blocks

COUNTRY_ORDER = ["Kenya", "Ghana", "Senegal", "Pooled"]

# Build pooled panel
pooled_parts = []
for cn, df in countries.items():
    p = df.copy(); p["country"] = cn; pooled_parts.append(p)
pooled = pd.concat(pooled_parts, ignore_index=True)
pooled["cell_id_str"] = pooled["country"] + "_" + pooled["cell_id"].astype(str)
pooled["post"] = (pooled["year"] >= 2007).astype(int)
pooled["cty_year"]  = pooled["country"] + "_" + pooled["year"].astype(str)

# ── Helper: main design with four SE estimators ───────────────────────────────
# Main design (as 55_binary_main_table.py): y = 1[any event], bt = 1[within 5 km of a
# battery site] x Post, nt = the same for non-lead sites, one regression; cell + year FE
# (country) or cell + country x year FE (pooled). Reports the battery coefficient and
# the lead-specific gap bt - nt with SEs from: cell-clustered CRV1, 1-degree block CRV1
# (the paper's SEs; stars from these, t(G-1)), Conley 50 km and Conley 100 km.
NEAR5 = 1.0 / 6.0
W_BAT, W_GAP = {"bt": 1.0}, {"bt": 1.0, "nt": -1.0}


def _se(V, names, w):
    v = np.zeros(len(names))
    for t, x in w.items():
        v[names.index(t)] = x
    var = float(v @ V @ v)
    return float(np.sqrt(var)) if var > 0 else np.nan


def run_se_comparison(panel, ycount, pooled):
    d = panel.copy()
    d["post"] = (d["year"] >= 2007).astype(int)
    d["y"] = (d[ycount] > 0).astype(float)
    d["bt"] = (d["inv_dist_battery"] > NEAR5).astype(float) * d["post"]
    d["nt"] = (d["inv_dist_nonlead"] > NEAR5).astype(float) * d["post"]
    fe = "cell_id_str + cty_year" if pooled else "cell_id_str + year"
    mb = pf.feols(f"y ~ bt + nt | {fe}", data=d, vcov={"CRV1": "block"})
    mc = pf.feols(f"y ~ bt + nt | {fe}", data=d, vcov={"CRV1": "cell_id_str"})
    names = list(mb.coef().index)
    out = {"bat": coef_sep(mb, "bt"), "gap": lincom(mb, W_GAP)}     # (est, block SE, p)
    for key, w in (("bat", W_BAT), ("gap", W_GAP)):
        out[key + "_cell"] = _se(np.asarray(mc._vcov), names, w)
        for km in (50, 100):
            V, _ = conley_vcov(mb, d, "cell_id_str", "cell_lat", "cell_lon", km)
            out[f"{key}_c{km}"] = _se(V, names, w)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: SE comparison on the main design (battery coefficient and gap)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== SE Comparison Table ===")
RES = {}
for out_name, ycount in [("Riots", "n_riots"), ("Organised violence", "n_org_violence")]:
    for cname in COUNTRY_ORDER:
        if cname == "Pooled":
            panel = pooled
        else:
            panel = countries[cname].copy()
            panel["cell_id_str"] = panel["cell_id"].astype(str)
        r = run_se_comparison(panel, ycount, cname == "Pooled")
        RES[(out_name, cname)] = r
        print(f"  {cname:8s} {out_name:20s} bat={r['bat'][0]:+.3f} [cell {r['bat_cell']:.3f} blk {r['bat'][1]:.3f} "
              f"c50 {r['bat_c50']:.3f} c100 {r['bat_c100']:.3f}]  gap={r['gap'][0]:+.3f} [cell {r['gap_cell']:.3f} "
              f"blk {r['gap'][1]:.3f} c50 {r['gap_c50']:.3f} c100 {r['gap_c100']:.3f}]")


def fmt(b, p):
    return f"{'$-$' if b < 0 else ''}{abs(b):.3f}{stars(p)}"


def fmtse(v):
    # A Conley SE can fail to compute (non-positive variance at this cutoff).
    return "---\\tnote{a}" if (v is None or np.isnan(v)) else f"({v:.3f})"


tex = [r"\begin{table}[htbp]",
       r"\caption{Standard errors under different clustering choices}",
       r"\label{tab:ppml_conley}", r"\begin{threeparttable}", r"\small",
       r"\begin{tabular}{llccccc}",
       r"\toprule",
       r" & & & \multicolumn{4}{c}{Standard error} \\",
       r"\cmidrule(lr){4-7}",
       r" & & Coef. & 1\textdegree{} block & Cell & Conley 50\,km & Conley 100\,km \\",
       r"\midrule"]
for out_name in ["Riots", "Organised violence"]:
    tex.append(f"\\textit{{{out_name}}} \\\\[2pt]")
    for cname in COUNTRY_ORDER:
        r = RES[(out_name, cname)]
        for k, lab in [("bat", "Battery"), ("gap", "Gap")]:
            tex.append(f"{'\\quad ' + cname if k == 'bat' else ''} & {lab}"
                       f" & {fmt(r[k][0], r[k][2])} & {fmtse(r[k][1])} & {fmtse(r[k + '_cell'])}"
                       f" & {fmtse(r[k + '_c50'])} & {fmtse(r[k + '_c100'])} \\\\")
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
tex += [r"\end{tabular}", r"\begin{tablenotes}", r"\setstretch{1}\footnotesize",
        # Table note lives in main.tex (\setTabNote{tab_ppml_conley}{...}).
        r"\item \TabNote{tab_ppml_conley}",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
tex_str = "\n".join(tex)
for path in [f"{TABS}/tab_ppml_conley.tex"] + ([f"{OVERLEAF}/tables/tab_ppml_conley.tex"] if OVERLEAF else []):
    with open(path, "w") as f: f.write(tex_str)
print("Saved: tab_ppml_conley.tex")

print("\nDone.")
