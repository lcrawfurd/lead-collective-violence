#!/usr/bin/env python3
"""
57_comparison_table.py -- Robustness of the main design in one table (tab:comparison)
==============================================================================
Every row re-estimates the paper's main design (as in 55_binary_main_table.py) on the
pooled three-country panel, with one change per row:

  outcome   1[any riot in cell-year]
  treatment bt = 1[cell within 5 km of a battery site] x 1[year >= 2007]
            nt = 1[cell within 5 km of a non-lead TSIP site] x 1[year >= 2007]
  FE        cell + country x year
  SE        CRV1 clustered on 1-degree spatial blocks; stars from t(G-1);
            wild cluster restricted bootstrap p (utils.wild_p) for the gap and contrast

Columns: DD = battery coefficient from the joint regression; DDD = lead-specific gap
(bt - nt); conflict-type contrast (DDDD) = riot gap minus organised-violence gap from
the stacked regression (cell x type and country-year x type FE, as fit_dddd in 55);
treated cells = cells within 5 km of a battery site in the estimation sample.

Rows
  (1) Baseline
  (2) Non-lead comparison without mining and ore-processing sites   (as 33)
  (4) Cells with above-median pre-2007 population                    (as 33)
  (5) City x year FE for Nairobi, Mombasa, Accra and Dakar, cells within 30 km
  (6) Kenya from 2005 only (Kenyan years before 2005 dropped)
  (7) Coarsened exact matching, (8) propensity-score matching
      (as the archived 20_cell_matching.py, _archive/offcuts_2026-10-22/)
      Matching is done within country on pre-2007 covariates (riots, protests,
      night lights, population, location), treated = within 5 km of a battery site.
      The matched samples pool treated cells with their matched controls, so they have
      no non-lead comparison: DD only (y ~ bt | cell + country x year; CEM weighted).

Inputs
  output/panels/panel_*.parquet (01), output/panels/urban_controls.parquet (19),
  data/tsip_sites_cleaned.csv

Outputs
  output/tables/tab_comparison.tex             (+ Overleaf)
  output/tables/comparison_table_values.tex    (macros, prefix cmp; + Overleaf)
"""
import numpy as np
import pandas as pd
import pyfixest as pf

from config import PANELS, TABLES, TSIP_PATH, POST_YR, save_to_overleaf
from utils import add_block, coef_sep, lincom, wild_p, stars

NEAR5 = 1.0 / 6.0                      # 1/(d+1) > 1/6  <=>  d < 5 km
COUNTRIES = [("Kenya", "KE"), ("Ghana", "GH"), ("Senegal", "SN")]
PRE_YEARS = list(range(1997, POST_YR))
K_MATCH, SEED = 3, 20260410            # as 20_cell_matching.py
MINING_PAT = "Min|Ore|Gold"            # as 33
CITY_CENTERS = {                       # as 26 panel D
    "Kenya":   [("Nairobi", -1.286, 36.817), ("Mombasa", -4.043, 39.668)],
    "Ghana":   [("Accra",    5.556, -0.197)],
    "Senegal": [("Dakar",   14.716, -17.467)],
}
CITY_RADIUS_KM = 30


def hav(clats, clons, slats, slons):
    R = 6371.0
    clat = np.radians(clats)[:, None]; clon = np.radians(clons)[:, None]
    slat = np.radians(slats)[None, :]; slon = np.radians(slons)[None, :]
    a = (np.sin((slat - clat) / 2) ** 2
         + np.cos(clat) * np.cos(slat) * np.sin((slon - clon) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Panels ───────────────────────────────────────────────────────────────────
tsip = pd.read_csv(TSIP_PATH)
is_bat = tsip.SiteIndustry.astype(str).str.contains("Battery", case=False, na=False)
is_lead = tsip.KeyPollutant.astype(str).str.contains("Lead", case=False, na=False)
NL = tsip[~is_bat & ~is_lead].copy()
nl_mask = {
    "nl5_nm": ~NL.SiteIndustry.astype(str).str.contains(MINING_PAT, case=False, na=False),
}

uc = pd.read_parquet(PANELS / "urban_controls.parquet")      # 19_urban_controls.py
base_pop = (uc[uc.year < POST_YR].groupby("cell_id")["log_pop_cell"].mean()
            .rename("base_pop").reset_index())

parts = []
for cn, code in COUNTRIES:
    p = pd.read_parquet(PANELS / f"panel_{cn.lower()}.parquet").copy()
    p["country"] = cn
    p["cell_id_str"] = code + "_" + p.cell_id.astype(str)
    p["cty_year"] = cn + "_" + p.year.astype(str)
    add_block(p, cn)
    p["post"] = (p.year >= POST_YR).astype(int)
    p["bat5"] = (p.inv_dist_battery > NEAR5).astype(float)
    p["nl5"] = (p.inv_dist_nonlead > NEAR5).astype(float)
    p["bt"] = p.bat5 * p.post
    p["nt"] = p.nl5 * p.post
    cells = p[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates("cell_id").copy()
    # Check the recomputed full non-lead distance reproduces the panel's nl5
    sub_all = NL[NL.country == cn]
    d_all = hav(cells.cell_lat.values, cells.cell_lon.values,
                sub_all.latitude.values, sub_all.longitude.values).min(1)
    chk = cells.assign(nl5_chk=(1 / (d_all + 1) > NEAR5).astype(float)).merge(
        p[["cell_id", "nl5"]].drop_duplicates("cell_id"), on="cell_id")
    print(f"{cn}: recomputed non-lead 5 km indicator matches panel in "
          f"{100 * (chk.nl5_chk == chk.nl5).mean():.1f}% of cells")
    for col, mask in nl_mask.items():
        sub = NL[(NL.country == cn) & mask]
        if len(sub):
            dmin = hav(cells.cell_lat.values, cells.cell_lon.values,
                       sub.latitude.values, sub.longitude.values).min(1)
            cells[col] = (1 / (dmin + 1) > NEAR5).astype(float)
        else:
            cells[col] = 0.0
    # City bands
    cells["city_id"] = "none"
    for name, la, lo in CITY_CENTERS[cn]:
        dc = hav(cells.cell_lat.values, cells.cell_lon.values, np.array([la]), np.array([lo]))[:, 0]
        cells.loc[dc <= CITY_RADIUS_KM, "city_id"] = name + "_" + cn
    p = p.merge(cells[["cell_id", "nl5_nm", "city_id"]], on="cell_id", how="left")
    p["nt_nm"] = p.nl5_nm * p.post
    p["city_year"] = p.city_id + "_" + p.year.astype(str)
    p = p.merge(base_pop, on="cell_id", how="left")
    parts.append(p)
P = pd.concat(parts, ignore_index=True)
P["base_pop"] = P["base_pop"].fillna(P["base_pop"].median())
med = P.groupby("cell_id_str")["base_pop"].first().median()      # as 33
P["urban"] = (P["base_pop"] > med).astype(int)
N_NL = len(NL)
print(f"Non-lead sites: {N_NL}; without mining {int(nl_mask['nl5_nm'].sum())}")


# ── Estimators ───────────────────────────────────────────────────────────────
def fit_row(d, nt="nt", extra_fe=None):
    """Joint DD/DDD and stacked DDDD on the main design. extra_fe: an added FE column."""
    d = d.copy()
    d["y"] = (d.n_riots > 0).astype(float)
    fe = ["cell_id_str", "cty_year"] + ([extra_fe] if extra_fe else [])
    m = pf.feols(f"y ~ bt + {nt} | " + " + ".join(fe), data=d, vcov={"CRV1": "block"})
    dd = coef_sep(m, "bt")
    gap = lincom(m, {"bt": 1, nt: -1})
    gap_w = wild_p(d, "y", ["bt", nt], {"bt": 1, nt: -1}, fe)
    # Stacked riots and organised violence, FE interacted with type
    sp = []
    for typ, ycol in [("riot", "n_riots"), ("org", "n_org_violence")]:
        q = d[fe + ["block", "bt", nt]].copy()
        q["y"] = (d[ycol] > 0).astype(float)
        q["riot"] = int(typ == "riot")
        for f in fe:
            q[f + "_t"] = q[f].astype(str) + "_" + typ
        sp.append(q)
    st = pd.concat(sp, ignore_index=True)
    st["bt_riot"] = st.bt * st.riot
    st["nt_riot"] = st[nt] * st.riot
    X = ["bt", nt, "bt_riot", "nt_riot"]
    fe_t = [f + "_t" for f in fe]
    m4 = pf.feols("y ~ " + " + ".join(X) + " | " + " + ".join(fe_t), data=st, vcov={"CRV1": "block"})
    w4 = {"bt_riot": 1, "nt_riot": -1}
    dddd = lincom(m4, w4)
    dddd_w = wild_p(st, "y", X, w4, fe_t)
    return dict(dd=dd, gap=gap, gap_w=gap_w, dddd=dddd, dddd_w=dddd_w,
                ntreat=int(d.loc[d.bat5 == 1, "cell_id_str"].nunique()), N=int(m._N))


def fit_dd_only(d, weights=None):
    d = d.copy()
    d["y"] = (d.n_riots > 0).astype(float)
    m = pf.feols("y ~ bt | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"},
                 weights=weights)
    return dict(dd=coef_sep(m, "bt"), gap=None, gap_w=None, dddd=None, dddd_w=None,
                ntreat=int(d.loc[d.bat5 == 1, "cell_id_str"].nunique()), N=int(m._N))


# ── Matching (as 20_cell_matching.py, within country) ────────────────────────
COV = ["pre_log_riots", "pre_log_protests", "pre_log_ntl", "pre_log_pop", "cell_lat", "cell_lon"]


def baseline(p):
    pre = (p[p.year.isin(PRE_YEARS)].groupby("cell_id")
           .agg(pre_log_riots=("log_riots", "mean"), pre_log_protests=("log_protests", "mean"))
           .reset_index())
    ntl = (uc[uc.year.isin(PRE_YEARS)].groupby("cell_id")
           .agg(pre_log_ntl=("log_ntl", "mean")).reset_index())
    pop = uc[uc.year == 2000][["cell_id", "log_pop_cell"]].rename(columns={"log_pop_cell": "pre_log_pop"})
    b = (p[["cell_id", "cell_lat", "cell_lon", "bat5"]].drop_duplicates("cell_id")
         .rename(columns={"bat5": "treated"})
         .merge(pre, on="cell_id", how="left").merge(ntl, on="cell_id", how="left")
         .merge(pop, on="cell_id", how="left"))
    b[COV] = b[COV].fillna(0)
    b["treated"] = b.treated.astype(int)
    return b


def psm(b):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(b[COV].values)
    ps = LogisticRegression(max_iter=1000, random_state=SEED).fit(X, b.treated).predict_proba(X)[:, 1]
    t, c = ps[b.treated == 1].reshape(-1, 1), ps[b.treated == 0].reshape(-1, 1)
    ids = b.loc[b.treated == 0, "cell_id"].values
    _, idx = NearestNeighbors(n_neighbors=K_MATCH).fit(c).kneighbors(t)
    return set(b.loc[b.treated == 1, "cell_id"]) | set(ids[idx.ravel()])


def cem(b, n_bins=3):
    df = b.copy()
    for c in COV:
        try:
            df[c + "_bin"] = pd.qcut(df[c], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            df[c + "_bin"] = 0
        df[c + "_bin"] = df[c + "_bin"].fillna(-1).astype(int)
    df["stratum"] = df[[c + "_bin" for c in COV]].astype(str).agg("_".join, axis=1)
    g = df.groupby("stratum").treated.agg(["sum", "count"])
    keep = g[(g["sum"] > 0) & (g["count"] > g["sum"])].index
    df = df[df.stratum.isin(keep)].copy()
    nt_, nc_ = (df.treated == 1).sum(), (df.treated == 0).sum()
    df["cem_w"] = 1.0
    for s_ in keep:
        ins = df.stratum == s_
        mt, mc = ((ins) & (df.treated == 1)).sum(), ((ins) & (df.treated == 0)).sum()
        df.loc[ins & (df.treated == 0), "cem_w"] = (nc_ / nt_) * (mt / mc)
    return df[["cell_id", "cem_w"]]


psm_parts, cem_parts = [], []
for cn, _ in COUNTRIES:
    pc = P[P.country == cn]
    b = baseline(pc)
    psm_parts.append(pc[pc.cell_id.isin(psm(b))])
    cem_parts.append(pc.merge(cem(b), on="cell_id", how="inner"))
P_psm = pd.concat(psm_parts, ignore_index=True)
P_cem = pd.concat(cem_parts, ignore_index=True)


# ── Rows ─────────────────────────────────────────────────────────────────────
ROWS = [
    ("Base",    "(1) Baseline",                                        lambda: fit_row(P)),
    ("NonMin",  "(2) No mining or ore-processing sites in comparison", lambda: fit_row(P, nt="nt_nm")),
    ("Urban",   "(3) Above-median pre-2007 population cells",          lambda: fit_row(P[P.urban == 1])),
    ("CityFE",  r"(4) City $\times$ year FE (4 cities, 30\,km)",       lambda: fit_row(P, extra_fe="city_year")),
    ("KeLate",  "(5) Kenya from 2005 only",                            lambda: fit_row(P[(P.country != "Kenya") | (P.year >= 2005)])),
    ("CEM",     "(6) Coarsened exact matching",                        lambda: fit_dd_only(P_cem, weights="cem_w")),
    ("PSM",     "(7) Propensity-score matching (1:3)",                 lambda: fit_dd_only(P_psm)),
]
R = {}
for key, lab, f in ROWS:
    R[key] = f()
    r = R[key]
    msg = f"{key:8s} DD {r['dd'][0]:.3f} ({r['dd'][1]:.3f})"
    if r["gap"]:
        msg += (f"  gap {r['gap'][0]:.3f} ({r['gap'][1]:.3f}) wild {r['gap_w']:.3f}"
                f"  DDDD {r['dddd'][0]:.3f} ({r['dddd'][1]:.3f}) wild {r['dddd_w']:.3f}")
    print(msg + f"  treated cells {r['ntreat']}  N {r['N']:,}")


# ── Table ────────────────────────────────────────────────────────────────────
def c(t):
    return f"{t[0]:.3f}{stars(t[2])}" if t else ""


def s(t):
    return f"({t[1]:.3f})" if t else ""


def w(v):
    return f"[{v:.3f}]" if v is not None else ""


L = [r"\begin{table}[H]",
     r"\caption{Robustness of the lead-specific riot effect (pooled, main design)}",
     r"\label{tab:comparison}", r"\begin{threeparttable}", r"\footnotesize",
     r"\begin{tabular}{>{\raggedright\arraybackslash}p{5.2cm}cccc}", r"\toprule",
     r" & (1) DD & (2) DDD & (3) Conflict-type & Treated \\",
     r" & Battery & Lead-specific gap & contrast & cells \\", r"\midrule"]
for key, lab, _ in ROWS:
    r = R[key]
    L.append(f"{lab} & {c(r['dd'])} & {c(r['gap'])} & {c(r['dddd'])} & {r['ntreat']} \\\\")
    L.append(f" & {s(r['dd'])} & {s(r['gap'])} & {s(r['dddd'])} & \\\\")
    if r["gap"]:
        L.append(f" & & {w(r['gap_w'])} & {w(r['dddd_w'])} & \\\\[2pt]")
    else:
        L[-1] = L[-1] + "[2pt]"
L += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[flushleft]", r"\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_comparison}{...}).
      r"\item \TabNote{tab_comparison}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_comparison.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
save_to_overleaf("tab_comparison.tex", str(TABLES / "tab_comparison.tex"), subfolder="tables")

# ── Macros (prefix cmp) ──────────────────────────────────────────────────────
M = {}
for key, _, _ in ROWS:
    r = R[key]
    for part in ("dd", "gap", "dddd"):
        t = r[part]
        if t is None:
            continue
        nm = {"dd": "DD", "gap": "Gap", "dddd": "DDDD"}[part]
        M[f"cmp{nm}{key}"] = f"{t[0]:.3f}"
        M[f"cmp{nm}{key}SE"] = f"{t[1]:.3f}"
        M[f"cmp{nm}{key}Stars"] = stars(t[2])
        if part != "dd":
            M[f"cmp{nm}{key}WildP"] = f"{r[part + '_w']:.3f}"
    M[f"cmpNTreat{key}"] = f"{r['ntreat']}"
# Write only the macros the paper quotes; a missing one stops the build.
KEEP = ["cmpDDBase", "cmpGapBase", "cmpGapNonMin", "cmpDDDDNonMin", "cmpGapUrban", "cmpDDDDUrban",
        "cmpGapCityFE", "cmpGapKeLate", "cmpDDCEM", "cmpDDPSM"]
missing = [k for k in KEEP if k not in M]
if missing:
    raise SystemExit(f"57: macros not computed: {missing}")
M = {k: M[k] for k in KEEP}
lines = ["% Generated by 57_comparison_table.py -- do not edit by hand"]
for k, v in M.items():
    v = "$-$" + v[1:] if v.startswith("-") else v
    lines.append(r"\newcommand{\%s}{%s}" % (k, v))
(TABLES / "comparison_table_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
save_to_overleaf("comparison_table_values.tex", str(TABLES / "comparison_table_values.tex"),
                 subfolder="tables")
print(f"Saved tab_comparison.tex and comparison_table_values.tex ({len(M)} macros)")
