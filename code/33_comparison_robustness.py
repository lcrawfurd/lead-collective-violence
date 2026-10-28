#!/usr/bin/env python3
"""
33_comparison_robustness.py
===========================
Robustness of the lead-specific gap to (a) a non-mining comparison group and
(b) an urban-cell restriction. Both address the concern that the non-lead
comparison is disproportionately rural artisanal mining, so the riot-vs-
organised-violence contrast could reflect urban/rural composition rather than
lead.

Re-estimates the binary escalating differences (1[<5 km] x Post 2007) on the
extensive margin, exactly as in 23_vehicle_dose_main.py, under three columns:
  (1) Baseline        - full non-lead comparison (as in the main results)
  (2) Non-mining      - non-lead comparison excluding artisanal-mining / ore sites
  (3) Urban cells     - baseline comparison, restricted to above-median-population cells

Inputs
------
  output/panels/panel_{kenya,ghana,senegal}.parquet (01), output/panels/urban_controls.parquet (19),
  data/tsip_sites_cleaned.csv

Outputs
-------
  tables/comparison_values.tex           (in-text macros; + Overleaf)
"""
import numpy as np
import pandas as pd
import pyfixest as pf

from utils import add_block, lincom
from config import PANELS as PANELS_DIR, TABLES, TSIP_PATH, save_to_overleaf

POST_YR = 2007
THRESH_5KM = 1.0 / 6.0        # 1/(d+1) > 1/6  <=>  d < 5 km
COUNTRIES = [("KE", "Kenya", "panel_kenya.parquet"),
             ("GH", "Ghana", "panel_ghana.parquet"),
             ("SN", "Senegal", "panel_senegal.parquet")]


def load_tsip():
    p = str(TSIP_PATH)
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.read_excel(p, engine="openpyxl")
    return df


def haversine_matrix(clats, clons, slats, slons):
    """Great-circle distance (km) from each cell to each site -> (ncell, nsite)."""
    R = 6371.0
    clat = np.radians(clats)[:, None]; clon = np.radians(clons)[:, None]
    slat = np.radians(slats)[None, :]; slon = np.radians(slons)[None, :]
    dlat = slat - clat; dlon = slon - clon
    a = np.sin(dlat / 2) ** 2 + np.cos(clat) * np.cos(slat) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ddd_gap_bin(d, ycount, nlcol):
    """Binary lead-specific gap (battery - non-lead), joint regression, block-clustered SE."""
    d = d.copy()
    d["y"]  = (d[ycount] > 0).astype(float)
    d["bt"] = d["bat5"] * d["post"]
    d["nt"] = d[nlcol]  * d["post"]
    m = pf.feols("y ~ bt + nt | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    gap, se, p = lincom(m, {"bt": 1, "nt": -1})
    return gap, se, p, int(m._N)


def dddd_het_bin(d, nlcol):
    """Binary quadruple difference: riot gap minus organised-violence gap."""
    parts = []
    for typ, ycol in [("riot", "n_riots"), ("org", "n_org_violence")]:
        s = d[["cell_id_str", "block", "year", "country", "cty_year", "bat5", nlcol, "post"]].copy()
        s["y"] = (d[ycol] > 0).astype(float)
        s["type"] = typ
        parts.append(s)
    st = pd.concat(parts, ignore_index=True)
    st["is_riot"] = (st["type"] == "riot").astype(int)
    st["bt"] = st["bat5"] * st["post"]
    st["nt"] = st[nlcol]  * st["post"]
    st["bt_r"] = st["bt"] * st["is_riot"]
    st["nt_r"] = st["nt"] * st["is_riot"]
    st["cell_type"]   = st["cell_id_str"] + "_" + st["type"]
    st["ctyyeartype"] = st["cty_year"] + "_" + st["type"]
    m = pf.feols("y ~ bt + nt + bt_r + nt_r | cell_type + ctyyeartype",
                 data=st, vcov={"CRV1": "block"})     # both types' rows for a cell share its block
    diff, se, p = lincom(m, {"bt_r": 1, "nt_r": -1})
    return diff, se, p


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── Build TSIP site sets ─────────────────────────────────────────────────────
tsip = load_tsip()
tsip = tsip[tsip["country"].isin(["Kenya", "Ghana", "Senegal"])].copy()
batt = tsip["SiteIndustry"].astype(str).str.contains("Battery", case=False, na=False)
lead = tsip["KeyPollutant"].astype(str).str.contains("Lead", case=False, na=False)
nl_all = tsip[~batt & ~lead].copy()                         # true non-lead comparison
mining = nl_all["SiteIndustry"].astype(str).str.contains("Min|Ore|Gold", case=False, na=False)
nl_nonmining = nl_all[~mining].copy()
N_NL, N_MINING = len(nl_all), int(mining.sum())
MINING_SHARE = round(100 * mining.mean())
print(f"Non-lead comparison sites: {N_NL} | mining: {N_MINING} ({MINING_SHARE}%) | "
      f"non-mining: {len(nl_nonmining)}")

# ── Urban covariate (baseline population per cell) ───────────────────────────
uc = pd.read_parquet(PANELS_DIR / "urban_controls.parquet")      # 19_urban_controls.py
base_pop = (uc[uc.year < POST_YR].groupby("cell_id")["log_pop_cell"].mean()
            .rename("base_pop").reset_index())

# ── Build pooled panel with bat5 / nl5 (all) / nl5_nm (non-mining) ───────────
parts = []
for code, cname, pfile in COUNTRIES:
    p = pd.read_parquet(PANELS_DIR / pfile).copy()
    p["country"] = cname
    p["cell_id_str"] = code + "_" + p["cell_id"].astype(str)
    add_block(p, code)       # 1-degree blocks: cluster for every reported SE
    p["cty_year"] = cname + "_" + p["year"].astype(str)
    p["post"] = (p["year"] >= POST_YR).astype(int)
    p["bat5"] = (p["inv_dist_battery"] > THRESH_5KM).astype(float)
    p["nl5"]  = (p["inv_dist_nonlead"]  > THRESH_5KM).astype(float)
    # recompute distance to nearest NON-MINING non-lead site for this country
    sub = nl_nonmining[nl_nonmining["country"] == cname]
    cells = p[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates("cell_id")
    if len(sub) > 0:
        dmat = haversine_matrix(cells["cell_lat"].values, cells["cell_lon"].values,
                                sub["latitude"].values, sub["longitude"].values)
        cells["inv_dist_nl_nm"] = 1.0 / (dmat.min(axis=1) + 1.0)
    else:
        cells["inv_dist_nl_nm"] = 0.0
    p = p.merge(cells[["cell_id", "inv_dist_nl_nm"]], on="cell_id", how="left")
    p["nl5_nm"] = (p["inv_dist_nl_nm"] > THRESH_5KM).astype(float)
    p = p.merge(base_pop, on="cell_id", how="left")
    parts.append(p)
pooled = pd.concat(parts, ignore_index=True)
pooled["base_pop"] = pooled["base_pop"].fillna(pooled["base_pop"].median())

# Urban cells = above-median baseline population (pooled)
med = pooled.groupby("cell_id")["base_pop"].first().median()
pooled["urban"] = (pooled["base_pop"] > med).astype(int)
urban = pooled[pooled["urban"] == 1].copy()

# ── Estimate the three columns ───────────────────────────────────────────────
def col(d, nlcol, label, dddd=True):
    g, gse, gp, gN = ddd_gap_bin(d, "n_riots", nlcol)
    out = dict(g=g, gse=gse, gp=gp, N=gN)
    if dddd:
        out["dd"], out["dse"], out["dp"] = dddd_het_bin(d, nlcol)
        print(f"[{label:14s}] gap={g:.3f}{stars(gp)} (se {gse:.3f})  "
              f"DDDD={out['dd']:.3f}{stars(out['dp'])} (se {out['dse']:.3f})  N={gN}")
    else:
        print(f"[{label:14s}] gap={g:.3f}{stars(gp)} (se {gse:.3f})  N={gN}")
    return out

print("\nEstimating...")
base = col(pooled, "nl5",    "Baseline", dddd=False)   # only its cell-year count is quoted
nonm = col(pooled, "nl5_nm", "Non-mining")
urb  = col(urban,  "nl5",    "Urban cells")

# ── Write in-text macros ─────────────────────────────────────────────────────
def mac(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"
macros = [
    mac("nlMiningShare", str(MINING_SHARE)),
    mac("nlNonMiningN", str(len(nl_nonmining))),
    mac("gapNonMin", f"{nonm['g']:.3f}"),
    mac("gapUrban", f"{urb['g']:.3f}"),
    mac("ddddNonMin", f"{nonm['dd']:.3f}"),
    mac("ddddUrban", f"{urb['dd']:.3f}"),
    mac("cmpNBase", f"{base['N']:,}"), mac("cmpNNonMin", f"{nonm['N']:,}"), mac("cmpNUrban", f"{urb['N']:,}"),
]
mout = TABLES / "comparison_values.tex"
mout.write_text("\n".join(macros) + "\n")
save_to_overleaf("comparison_values.tex", str(mout), subfolder="tables")
print(f"Saved: {mout}")
print("\nDone.")
