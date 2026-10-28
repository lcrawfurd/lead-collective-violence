#!/usr/bin/env python3
"""
47_slides_binary.py — Binary-treatment versions of the slide exhibits
==============================================================================
The seminar slides (slides_york_ehpe.tex) state the design and the headline table
with the binary treatment 1[d_c < 5 km] x Post(2007), from 23_vehicle_dose_main.py.
Several other slide exhibits came from the continuous 1/(d+1) specification, so
the slides mixed two scales. This script re-estimates those exhibits with the
binary treatment, on the extensive margin (1[any event in cell-year]). It writes
slide-only outputs and leaves the paper's figures and macros alone.

Each block copies the specification of the script it replaces, with only the
treatment changed:
  A. DD event study by country, cell + year FE                  (as 30)
     + joint pre-trend Wald test, + Kenya export for Honest DiD (as 30/31)
  B. DDD event study, pooled, battery and non-lead event-time terms in one
     regression, cell + country x year FE                       (as 03, joint)
  C. Lead-specific gap by conflict type, pooled                 (as 23)
  D. Mob-justice gap from the riot trigger coding               (as 36)
  E. Leave-one-out: country, nearest site, treated cell         (as 37)
  F. 1-degree macro-cell x year FE                              (as 35)
  G. Randomisation inference, 1,000 relabellings by country     (as 10)
  H. Onset-year sweep 2004-2011, by country and pooled          (as 32)
  I. Food-price placebo, within-period prices                   (as 38)
Every reported SE is CRV1 clustered on 1-degree spatial blocks (utils.add_block); t-tests
use t(G-1), joint tests F(q, G-1), CIs the t(G-1) critical value. A2 adds the linear
pre-trend test (slope on the pre-period sample) with a wild cluster bootstrap p.
  J. Honest DiD for Kenya riots (calls 31_honest_did_extensive.R)

Outputs
-------
  output/tables/slides_binary_values.tex   (macros, prefix \\sb...; + Overleaf)
  output/tables/tab_pretrends.tex          (appendix pre-trend table; + Overleaf)
  output/figures/fig_slides_{dd_event,ddd_event,onset,ri,raw_rates,conflict_type}.pdf (+ Overleaf)
  output/figures/fig_ddd_gap_event.pdf, fig_honest_did_gap.pdf  (+ Overleaf)
  output/figures/fig_slides_honest_{rm,sd}.pdf                  (+ Overleaf)
  output/panels/slides_bin/                 (Honest DiD inputs and results)

Inputs: output/panels/panel_*.parquet (01), trigger_classified.csv (36),
        botec_extensive.csv (29); data/tsip_sites_cleaned.csv, data/wb_lead_food_prices.csv,
        data/ACLED_global.csv
"""
import shutil
import subprocess
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from config import (PANELS, TABLES, FIGURES, DATA, TSIP_PATH, ACLED_GLOBAL, POST_YR, OVERLEAF_FIGS,
                    save_to_overleaf)
from utils import (stars, add_block, coef_sep, lincom, wald_F, tcrit, wild_p,
                   haversine_km)
from tufte_style import set_tufte_rcparams, tufte_ax
set_tufte_rcparams()
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})

THRESH_5KM = 1.0 / 6.0          # 1/(d+1) > 1/6  <=>  d < 5 km (as 23)
BASE_YEAR = POST_YR - 1
N_PERMS, SEED = 1000, 20260410
COUNTRIES = ["Kenya", "Ghana", "Senegal"]
CCOL = {"Kenya": "#006970", "Ghana": "#E08A00", "Senegal": "#5B7F88", "Pooled": "#C8102E"}
OUTCOMES = [("n_riots", "Riots"), ("n_protests", "Protests"),
            ("n_org_violence", "Organised violence")]
HD_DIR = PANELS / "slides_bin"
HD_DIR.mkdir(parents=True, exist_ok=True)

M = {}   # macro name -> string


def diff_se(m, a, b):
    """coef[a] - coef[b], its block-clustered SE, and the t(G-1) p-value."""
    return lincom(m, {a: 1, b: -1})


def save_fig(fig, name):
    fig.savefig(str(FIGURES / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    save_to_overleaf(f"{name}.pdf", str(FIGURES / f"{name}.pdf"), subfolder="figures")
    print(f"  Saved: {name}.pdf")


# ── Panels ───────────────────────────────────────────────────────────────────
PAN = {}
for cn in COUNTRIES:
    p = pd.read_parquet(str(PANELS / f"panel_{cn.lower()}.parquet")).copy()
    p["post"] = (p.year >= POST_YR).astype(int)
    p["country"] = cn
    p["cell_id_str"] = cn[:2] + "_" + p.cell_id.astype(str)
    p["cty_year"] = cn + "_" + p.year.astype(str)
    p["bat5"] = (p.inv_dist_battery > THRESH_5KM).astype(float)
    p["nl5"] = (p.inv_dist_nonlead > THRESH_5KM).astype(float)
    p["bt"] = p.bat5 * p.post
    p["nt"] = p.nl5 * p.post
    p["n_battles"] = (p.n_org_violence - p.n_vac_attack - p.n_vac_abduction).clip(lower=0)
    add_block(p, cn)
    PAN[cn] = p
POOLED = pd.concat(PAN.values(), ignore_index=True)
print(f"Pooled: {POOLED.cell_id_str.nunique():,} cells, {len(POOLED):,} cell-years, "
      f"{POOLED.loc[POOLED.bat5 == 1, 'cell_id_str'].nunique()} treated cells")


def gap_pooled(d, ycol, fe="cell_id_str + cty_year"):
    """Binary lead-specific gap (battery minus non-lead), joint regression, block-clustered SE."""
    d = d.copy()
    d["y"] = (d[ycol] > 0).astype(float)
    m = pf.feols(f"y ~ bt + nt | {fe}", data=d, vcov={"CRV1": "block"})
    return diff_se(m, "bt", "nt") + (int(m._N),)


# ── A. DD event study by country ─────────────────────────────────────────────
print("\nA. DD event study by country (binary, block-clustered)...")


def event_dd(panel, ycount):
    d = panel.copy()
    d["y"] = (d[ycount] > 0).astype(float)
    yrs = [y for y in sorted(d.year.unique()) if y != BASE_YEAR]
    for yr in yrs:
        d[f"t_{yr}"] = d.bat5 * (d.year == yr).astype(int)
    m = pf.feols("y ~ " + " + ".join(f"t_{yr}" for yr in yrs) + " | cell_id + year",
                 data=d, vcov={"CRV1": "block"})
    nm = list(m.coef().index)
    idx = [nm.index(f"t_{yr}") for yr in yrs]
    b = np.array([float(m.coef()[f"t_{yr}"]) for yr in yrs])
    s = np.array([float(m.se()[f"t_{yr}"]) for yr in yrs])   # block-clustered
    cr = tcrit(m)
    es = pd.DataFrame(dict(year=yrs, coef=b, lo=b - cr * s, hi=b + cr * s))
    es = pd.concat([es, pd.DataFrame([dict(year=BASE_YEAR, coef=0, lo=0, hi=0)])])
    pre = [i for i, yr in enumerate(yrs) if yr < POST_YR]
    # Joint 9-year pre-trend test (appendix): F(q, G-1), block-clustered
    _, _, p_pre = wald_F(m, [f"t_{yrs[i]}" for i in pre])
    V = np.array(m._vcov)[np.ix_(idx, idx)]                  # Honest DiD: block-clustered
    t_max = float(np.nanmax(np.abs(b[pre] / np.where(s[pre] > 0, s[pre], np.nan))))
    return es.sort_values("year"), (p_pre, t_max), yrs, b, V


ES, PRE = {}, {}
for yc, lab in OUTCOMES:
    for cn in COUNTRIES:
        es, pp, yrs, b, V = event_dd(PAN[cn], yc)
        ES[(lab, cn)] = es
        PRE[(lab, cn)] = pp
        if yc == "n_riots":
            print(f"  {cn:8s} riots: joint pre-trend p={pp[0]:.3f} (F, block-clustered); "
                  f"largest pre-period |t| {pp[1]:.2f}")
            if cn == "Kenya":
                pd.DataFrame({"year": yrs, "betahat": b}).to_csv(
                    str(HD_DIR / "honest_ke_betahat.csv"), index=False)
                pd.DataFrame(V).to_csv(str(HD_DIR / "honest_ke_sigma.csv"), index=False)
                (HD_DIR / "honest_ke_meta.txt").write_text(
                    f"n_pre={sum(y < POST_YR for y in yrs)}\n"
                    f"n_post={sum(y >= POST_YR for y in yrs)}\n")

fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharex=True, sharey=True)
for i, (ax, (yc, lab)) in enumerate(zip(axes, OUTCOMES)):
    for cn in COUNTRIES:
        es = ES[(lab, cn)]
        ax.plot(es.year, es.coef, "o-", ms=2.5, lw=1.3, color=CCOL[cn], label=cn)
        ax.fill_between(es.year, es.lo, es.hi, alpha=0.10, color=CCOL[cn], lw=0)
    ax.axhline(0, color="#999999", lw=0.7, ls="--")
    ax.axvline(POST_YR - 0.5, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_title(lab)
    ax.set_xticks([2000, 2010, 2020])
    if i == 0:
        ax.set_ylabel("Change in Pr(any event)")
        ax.legend(frameon=False)
    tufte_ax(ax)
fig.tight_layout()
save_fig(fig, "fig_slides_dd_event")

M["sbPreKeMaxT"] = f"{PRE[('Riots', 'Kenya')][1]:.1f}"     # largest pre-period |t|, Kenya riots


# ── A2. Linear pre-trend test (main pre-trend test) + appendix tab:pretrends ─
# Pre-period sample (1997-2006): 1[any event] on bat5 x (year - 2006), cell + year FE
# (country) or cell + country-year FE (pooled); block-clustered SE, t(G-1) p, and a wild
# cluster restricted bootstrap p (Rademacher, 9,999 draws, seed 20261014).
# The appendix table adds the joint F-test (F(q, G-1), block-clustered) of the nine
# pre-period coefficients from the event study above (pooled: same event study with
# cell + country-year FE).
print("\nA2. Linear pre-trend test (pre-2007 sample)...")


def pooled_pre_F(ycount):
    d = POOLED.copy()
    d["y"] = (d[ycount] > 0).astype(float)
    yrs = [y for y in sorted(d.year.unique()) if y != BASE_YEAR]
    for yr in yrs:
        d[f"t_{yr}"] = d.bat5 * (d.year == yr).astype(int)
    m = pf.feols("y ~ " + " + ".join(f"t_{yr}" for yr in yrs) + " | cell_id_str + cty_year",
                 data=d, vcov={"CRV1": "block"})
    return wald_F(m, [f"t_{yr}" for yr in yrs if yr < POST_YR])[2]


PT = {}       # (outcome label, place) -> (slope, se, p, wild p, joint F p)
for yc, olab in OUTCOMES:
    for lab, k in [("Kenya", "Ke"), ("Ghana", "Gh"), ("Senegal", "Sn"), ("Pooled", "Pool")]:
        d = (POOLED if lab == "Pooled" else PAN[lab])
        d = d[d.year < POST_YR].copy()
        d["y"] = (d[yc] > 0).astype(float)
        d["pre_slope"] = d.bat5 * (d.year - BASE_YEAR)
        fe = ["cell_id_str", "cty_year"] if lab == "Pooled" else ["cell_id_str", "year"]
        m = pf.feols("y ~ pre_slope | " + " + ".join(fe), data=d, vcov={"CRV1": "block"})
        b, s, p = coef_sep(m, "pre_slope")
        wp = wild_p(d, "y", ["pre_slope"], {"pre_slope": 1}, fe)
        pF = pooled_pre_F(yc) if lab == "Pooled" else PRE[(olab, lab)][0]
        PT[(olab, lab)] = (b, s, p, wp, pF)
        if yc == "n_riots":
            M[f"sbPreSlope{k}"] = f"{b:.4f}"
            if k == "Ke":
                M["sbPreSlopeKeSE"] = f"{s:.4f}"
            M[f"sbPreSlope{k}WildP"] = f"{wp:.2f}"
            M[f"sbNTreatBlk{k}"] = f"{d.loc[d.bat5 == 1, 'block'].nunique()}"
            if k == "Pool":
                M["sbNBlkPool"] = f"{d.block.nunique()}"
        print(f"  {olab:18s} {lab:8s} slope={b:+.4f} (SE {s:.4f}) p={p:.3f} wild p={wp:.3f} "
              f"joint F p={pF:.3f}")


def _num(x, nd):
    return f"$-${abs(x):.{nd}f}" if x < 0 else f"{x:.{nd}f}"


T = [r"\begin{table}[H]", r"\caption{Pre-Trend Tests (Battery Sites, Main Design, 1997--2006)}",
     r"\label{tab:pretrends}", r"\begin{threeparttable}", r"\footnotesize",
     r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{lccccccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{Riots} & \multicolumn{3}{c}{Protests} & \multicolumn{3}{c}{Organised violence} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
     r" & Slope & Wild $p$ & Joint $p$ & Slope & Wild $p$ & Joint $p$ & Slope & Wild $p$ & Joint $p$ \\",
     r"\midrule"]
for lab in ["Kenya", "Ghana", "Senegal", "Pooled"]:
    row1, row2 = [lab], [""]
    for _, olab in OUTCOMES:
        b, s, p, wp, pF = PT[(olab, lab)]
        row1 += [_num(b, 4) + stars(p), f"[{wp:.2f}]", f"{pF:.3f}"]
        row2 += [f"({s:.4f})", "", ""]
    T.append(" & ".join(row1) + r" \\")
    T.append(" & ".join(row2) + r" \\[3pt]")
T += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[flushleft]", r"\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_pretrends}{...}).
      r"\item \TabNote{tab_pretrends}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_pretrends.tex").write_text("\n".join(T) + "\n", encoding="utf-8")
save_to_overleaf("tab_pretrends.tex", str(TABLES / "tab_pretrends.tex"), subfolder="tables")
print("  Saved: tab_pretrends.tex")


# ── B. DDD event study, pooled ──────────────────────────────────────────────
print("\nB. DDD event study, pooled (binary, block-clustered)...")


def event_ddd(ycount):
    d = POOLED.copy()
    d["y"] = (d[ycount] > 0).astype(float)
    yrs = [y for y in sorted(d.year.unique()) if y != BASE_YEAR]
    for yr in yrs:
        d[f"b_{yr}"] = d.bat5 * (d.year == yr).astype(int)
        d[f"n_{yr}"] = d.nl5 * (d.year == yr).astype(int)
    rhs = " + ".join([f"b_{yr}" for yr in yrs] + [f"n_{yr}" for yr in yrs])
    m = pf.feols(f"y ~ {rhs} | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    cr = tcrit(m)
    out = {}
    for pre, lab in [("b", "bat"), ("n", "nl")]:
        b = np.array([float(m.coef()[f"{pre}_{yr}"]) for yr in yrs])
        s = np.array([float(m.se()[f"{pre}_{yr}"]) for yr in yrs])
        es = pd.DataFrame(dict(year=yrs, coef=b, lo=b - cr * s, hi=b + cr * s))
        es = pd.concat([es, pd.DataFrame([dict(year=BASE_YEAR, coef=0, lo=0, hi=0)])])
        out[lab] = es.sort_values("year")
    g = [lincom(m, {f"b_{yr}": 1, f"n_{yr}": -1})[:2] for yr in yrs]
    gb = np.array([x[0] for x in g]); gs = np.array([x[1] for x in g])
    es = pd.DataFrame(dict(year=yrs, coef=gb, lo=gb - cr * gs, hi=gb + cr * gs))
    es = pd.concat([es, pd.DataFrame([dict(year=BASE_YEAR, coef=0, lo=0, hi=0)])])
    out["gap"] = es.sort_values("year")
    if ycount == "n_riots":
        # Honest DiD input for the pooled lead-specific gap: gap_t = b_t - n_t by year,
        # with its block-clustered vcov L V L' (L picks b_t and subtracts n_t).
        names = list(m.coef().index)
        Lm = np.zeros((len(yrs), len(names)))
        for k, yr in enumerate(yrs):
            Lm[k, names.index(f"b_{yr}")] = 1
            Lm[k, names.index(f"n_{yr}")] = -1
        Vg = Lm @ np.asarray(m._vcov) @ Lm.T
        pd.DataFrame({"year": yrs, "betahat": gb}).to_csv(
            str(HD_DIR / "honest_gap_betahat.csv"), index=False)
        pd.DataFrame(Vg).to_csv(str(HD_DIR / "honest_gap_sigma.csv"), index=False)
        (HD_DIR / "honest_gap_meta.txt").write_text(
            f"n_pre={sum(y < POST_YR for y in yrs)}\n"
            f"n_post={sum(y >= POST_YR for y in yrs)}\n")
    return out


DDD_ES = {yc: event_ddd(yc) for yc, _ in OUTCOMES}
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharex=True, sharey=True)
for i, (ax, (yc, lab)) in enumerate(zip(axes, OUTCOMES)):
    es = DDD_ES[yc]
    for key, col, lbl in [("bat", "#006970", "Battery sites"),
                          ("nl", "#E08A00", "Other toxic sites")]:
        ax.plot(es[key].year, es[key].coef, "o-", ms=2.5, lw=1.3, color=col, label=lbl)
        ax.fill_between(es[key].year, es[key].lo, es[key].hi, alpha=0.12, color=col, lw=0)
    ax.axhline(0, color="#999999", lw=0.7, ls="--")
    ax.axvline(POST_YR - 0.5, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_title(lab)
    ax.set_xticks([2000, 2010, 2020])
    if i == 0:
        ax.set_ylabel("Change in Pr(any event)")
        ax.legend(frameon=False)
    tufte_ax(ax)
fig.tight_layout()
save_fig(fig, "fig_slides_ddd_event")

# Paper version: the lead-specific gap itself (battery minus other toxic sites), year by year
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharex=True, sharey=True)
for i, (ax, (yc, lab)) in enumerate(zip(axes, OUTCOMES)):
    es = DDD_ES[yc]["gap"]
    ax.plot(es.year, es.coef, "o-", ms=2.5, lw=1.3, color="#006970")
    ax.fill_between(es.year, es.lo, es.hi, alpha=0.15, color="#006970", lw=0)
    ax.axhline(0, color="#999999", lw=0.7, ls="--")
    ax.axvline(POST_YR - 0.5, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_title(lab)
    ax.set_xticks([2000, 2010, 2020])
    if i == 0:
        ax.set_ylabel("Battery minus other toxic sites, Pr(any event)")
    tufte_ax(ax)
fig.tight_layout()
save_fig(fig, "fig_ddd_gap_event")


# ── C. Gap by conflict type ─────────────────────────────────────────────────
print("\nC. Lead-specific gap by conflict type (pooled)...")
for yc, key in [("n_riots", "Riot"), ("n_peaceful_protest", "Peace"),
                ("n_org_violence", "Org"), ("n_battles", "Battles")]:
    g, s, p, n = gap_pooled(POOLED, yc)
    M[f"sbGap{key}"] = f"{g:.3f}"
    print(f"  {key:8s} gap={g:.3f}{stars(p)} ({s:.3f})  N={n:,}")

# DDDD: riot gap minus organised-violence gap in one stacked regression (as 23)
parts = []
for typ, yc in [("riot", "n_riots"), ("org", "n_org_violence")]:
    d = POOLED[["cell_id_str", "block", "cty_year", "bt", "nt"]].copy()
    d["y"] = (POOLED[yc] > 0).astype(float)
    d["type"] = typ
    parts.append(d)
st = pd.concat(parts, ignore_index=True)
st["is_riot"] = (st.type == "riot").astype(int)
st["bt_riot"] = st.bt * st.is_riot
st["nt_riot"] = st.nt * st.is_riot
st["cell_type"] = st.cell_id_str + "_" + st.type
st["cyt"] = st.cty_year + "_" + st.type
m = pf.feols("y ~ bt + nt + bt_riot + nt_riot | cell_type + cyt", data=st,
             vcov={"CRV1": "block"})        # both types' rows for a cell share its block
g4, s4, p4 = diff_se(m, "bt_riot", "nt_riot")
M["sbDDDD"], M["sbDDDDSE"] = f"{g4:.3f}", f"{s4:.3f}"
print(f"  DDDD riots minus organised = {g4:.3f}{stars(p4)} ({s4:.3f})")


# ── D. Mob-justice gap ──────────────────────────────────────────────────────
print("\nD. Mob-justice gap (as 36)...")
ev = pd.read_csv(str(PANELS / "trigger_classified.csv"))
for cat in ["political", "services"]:
    ct = (ev[ev.trigger == cat].groupby(["country", "cell_id", "year"]).size()
          .rename("n_cat").reset_index())
    Pc = POOLED.merge(ct, on=["country", "cell_id", "year"], how="left")
    Pc["n_cat"] = Pc.n_cat.fillna(0)
    g, s_, p, _ = gap_pooled(Pc, "n_cat")
    M[f"sbTrig{cat.capitalize()}Gap"] = f"{g:.3f}"
    print(f"  {cat:10s} gap={g:.3f}{stars(p)} ({s_:.3f})")
mj = (ev[ev.trigger == "mob_justice"].groupby(["country", "cell_id", "year"]).size()
      .rename("n_mj").reset_index())
P_mj = POOLED.merge(mj, on=["country", "cell_id", "year"], how="left")
P_mj["n_mj"] = P_mj.n_mj.fillna(0)
print(f"  matched {int(P_mj.n_mj.sum()):,} of {len(ev[ev.trigger == 'mob_justice']):,} mob-justice riots to panel cells")
assert len(P_mj) == len(POOLED)
g, s, p, _ = gap_pooled(P_mj, "n_mj")
M["sbMobGap"], M["sbMobGapSE"] = f"{g:.3f}", f"{s:.3f}"
print(f"  mob justice gap={g:.3f}{stars(p)} ({s:.3f})")


# ── E. Leave-one-out ────────────────────────────────────────────────────────
print("\nE. Leave-one-out (as 37)...")
tsip = pd.read_csv(TSIP_PATH)
site_key = []
for cn in COUNTRIES:
    p = PAN[cn]
    bat = tsip[(tsip.country.str.contains(cn, case=False, na=False)) &
               (tsip.SiteIndustry.str.contains("Battery", case=False, na=False))].reset_index(drop=True)
    D = haversine_km(p.cell_lat.values[:, None], p.cell_lon.values[:, None],
                     bat.latitude.values[None, :], bat.longitude.values[None, :])
    site_key.append(cn[:2] + "_" + pd.Series(D.argmin(1)).astype(str))
POOLED["site_key"] = pd.concat(site_key, ignore_index=True).values
treated = sorted(POOLED.loc[POOLED.bat5 == 1, "cell_id_str"].unique())
rows = []
for cn in COUNTRIES:
    rows.append(("country", cn) + gap_pooled(POOLED[POOLED.country != cn], "n_riots")[:3])
for k in sorted(POOLED.site_key.unique()):
    sub = POOLED[POOLED.site_key != k]
    if sub.loc[sub.bat5 == 1, "cell_id_str"].nunique() < 5:
        continue
    rows.append(("site", k) + gap_pooled(sub, "n_riots")[:3])
for c in treated:
    rows.append(("cell", c) + gap_pooled(POOLED[POOLED.cell_id_str != c], "n_riots")[:3])
loo = pd.DataFrame(rows, columns=["level", "dropped", "gap", "se", "p"])
M["sbLooN"] = f"{len(loo)}"
M["sbLooMin"], M["sbLooMax"] = f"{loo.gap.min():.3f}", f"{loo.gap.max():.3f}"
M["sbLooNSig"] = f"{int((loo.p < 0.05).sum())}"
print(f"  {len(loo)} estimates: {loo.gap.min():.3f} to {loo.gap.max():.3f}; "
      f"significant at 5% in {(loo.p < 0.05).sum()}")


# ── F. Macro-cell x year FE ─────────────────────────────────────────────────
print("\nF. 1-degree macro-cell x year FE (as 35)...")
POOLED["macro_year"] = (np.floor(POOLED.cell_lat).astype(int).astype(str) + "_"
                        + np.floor(POOLED.cell_lon).astype(int).astype(str) + "_"
                        + POOLED.year.astype(str))
g, s, p, _ = gap_pooled(POOLED, "n_riots", fe="cell_id_str + macro_year")
M["sbMacroGap"], M["sbMacroGapSE"] = f"{g:.3f}", f"{s:.3f}"
print(f"  macro-cell x year FE gap={g:.3f}{stars(p)} ({s:.3f})")


# ── G. Randomisation inference ──────────────────────────────────────────────
print(f"\nG. Randomisation inference ({N_PERMS} relabellings, seed={SEED})...")
rng = np.random.default_rng(SEED)
RI = {}
for cn in COUNTRIES:
    p = PAN[cn].sort_values(["cell_id", "year"])
    cells = p.drop_duplicates("cell_id")[["cell_id", "cell_lat", "cell_lon"]].reset_index(drop=True)
    T = p.year.nunique()
    Y = (p.n_riots.values > 0).astype(float).reshape(len(cells), T)
    post = (np.sort(p.year.unique()) >= POST_YR).astype(float)
    Yd = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    pc = post - post.mean()

    def dd(b):  # two-way FE DD in a balanced panel: x_it = b_i * post_t
        bc = b - b.mean()
        return float(bc @ Yd @ pc) / float((bc ** 2).sum() * (pc ** 2).sum())

    sites = tsip[tsip.country == cn].reset_index(drop=True)
    is_bat = sites.SiteIndustry.str.contains("Battery", case=False, na=False).values
    D = haversine_km(cells.cell_lat.values[:, None], cells.cell_lon.values[:, None],
                     sites.latitude.values[None, :], sites.longitude.values[None, :])
    b_obs = (D[:, is_bat].min(1) < 5).astype(float)
    b_pan = p.drop_duplicates("cell_id").bat5.values
    print(f"  {cn}: own-distance treatment matches panel in {100*(b_obs == b_pan).mean():.1f}% of cells")
    obs = dd(b_pan)
    perm = np.array([dd((D[:, rng.choice(len(sites), is_bat.sum(), replace=False)].min(1) < 5)
                        .astype(float)) for _ in range(N_PERMS)])
    pv = float((perm >= obs).mean())
    RI[cn] = (obs, perm, pv, int(is_bat.sum()))
    M[f"sbRIp{cn[:2].capitalize()}"] = f"{pv:.3f}"
    print(f"  {cn}: observed DD {obs:.3f}, RI p={pv:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
for i, (ax, cn) in enumerate(zip(axes, COUNTRIES)):
    obs, perm, pv, nb = RI[cn]
    ax.hist(perm, bins=40, color="#B8C4C8", edgecolor="white", density=True)
    ax.axvline(obs, color="#C8102E", lw=1.8)
    ax.text(obs, ax.get_ylim()[1] * 1.02, "observed", color="#C8102E", fontsize=8,
            ha="center", va="bottom")
    ax.set_title(f"{cn} (p = {pv:.3f}, {nb} sites)")
    ax.set_xlabel("DD coefficient, within 5 km")
    if i == 0:
        ax.set_ylabel("Density")
    tufte_ax(ax)
fig.tight_layout()
save_fig(fig, "fig_slides_ri")


# ── H. Onset-year sweep ─────────────────────────────────────────────────────
print("\nH. Onset-year sweep (as 32)...")
rows = []
for lab in COUNTRIES + ["Pooled"]:
    d = (POOLED if lab == "Pooled" else PAN[lab]).copy()
    d["y"] = (d.n_riots > 0).astype(float)
    fe = "cell_id_str + cty_year" if lab == "Pooled" else "cell_id_str + year"
    for T in range(2004, 2012):
        d["tp"] = d.bat5 * (d.year >= T).astype(int)
        m = pf.feols(f"y ~ tp | {fe}", data=d, vcov={"CRV1": "block"})
        b, s, p = coef_sep(m, "tp")
        rows.append(dict(country=lab, cutoff=T, coef=b, se=s, p=p, crit=tcrit(m)))
sw = pd.DataFrame(rows)
tmin = (sw.coef / sw.se).min()
M["sbOnsetN"] = f"{len(sw)}"
M["sbOnsetMinT"] = f"{tmin:.1f}"
print(f"  {len(sw)} estimates; smallest t = {tmin:.2f}; significant at 5% in {(sw.p < 0.05).sum()}")

fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.8), sharex=True, sharey=True)
for ax, lab in zip(axes.flat, COUNTRIES + ["Pooled"]):
    g = sw[sw.country == lab].sort_values("cutoff")
    ax.fill_between(g.cutoff, g.coef - g.crit * g.se, g.coef + g.crit * g.se,
                    color=CCOL[lab], alpha=0.18, lw=0)
    ax.plot(g.cutoff, g.coef, "o-", ms=3, lw=1.6, color=CCOL[lab])
    ax.axhline(0, color="#666", lw=0.7)
    ax.axvline(POST_YR, color="#444", ls="--", lw=0.8, alpha=0.7)
    ax.set_title(lab)
    tufte_ax(ax)
for ax in axes[1, :]:
    ax.set_xlabel("Assumed onset year")
axes[0, 0].set_ylabel("DD coefficient, within 5 km")
fig.tight_layout()
fig.subplots_adjust(hspace=0.55)
save_fig(fig, "fig_slides_onset")


# ── I. Food-price placebo ───────────────────────────────────────────────────
print("\nI. Food-price placebo (within-period prices, as 38)...")
px = pd.read_csv(str(DATA / "wb_lead_food_prices.csv"))
px = px[(px.year >= 1997) & (px.year <= 2024)].copy()
px["post"] = (px.year >= POST_YR).astype(int)
for c in ["lead_price", "food_index"]:
    z = (px[c] - px[c].mean()) / px[c].std()
    px[c + "_r"] = z - z.groupby(px.post).transform("mean")
d = POOLED.drop(columns="post").merge(px[["year", "lead_price_r", "food_index_r"]],
                                      on="year", how="inner")
d["y"] = (d.n_riots > 0).astype(float)
for a, s_ in [("lead", "lead_price_r"), ("food", "food_index_r")]:
    d[f"bat_{a}"] = d.bat5 * d[s_]
    d[f"nl_{a}"] = d.nl5 * d[s_]
m = pf.feols("y ~ bat_lead + nl_lead + bat_food + nl_food | cell_id_str + cty_year",
             data=d, vcov={"CRV1": "block"})
g, s, p = diff_se(m, "bat_food", "nl_food")
M["sbFoodGap"], M["sbFoodSE"] = f"{g:.3f}", f"{s:.3f}"
print(f"  food x proximity gap={g:.3f}{stars(p)} ({s:.3f})")


# ── J. Honest DiD, Kenya riots ──────────────────────────────────────────────
print("\nJ. Honest DiD (R)...")
hd_fig = HD_DIR / "figs"
hd_fig.mkdir(exist_ok=True)
# Smoothness grid scaled to the binary event-study coefficients (the continuous
# version uses 0-0.1 for coefficients about seven times larger).
r = subprocess.run(["Rscript", str(PANELS.parent.parent / "code" / "31_honest_did_extensive.R"),
                    str(HD_DIR), str(hd_fig), "0.03"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:])
if r.returncode != 0:
    raise SystemExit("47: Honest DiD R step failed")
res = dict(l.split("=") for l in (HD_DIR / "honest_ke_extensive_results.txt").read_text().split())
for k_r, k_m in [("rm_lb0", "sbHonestRMlb"), ("rm_cross", "sbHonestRMcross"),
                 ("sd_lb0", "sbHonestSDlb")]:
    M[k_m] = res[k_r]
for src, dst in [("fig_honest_did_kenya.pdf", "fig_slides_honest_rm.pdf"),
                 ("fig_honest_did_kenya_sd.pdf", "fig_slides_honest_sd.pdf")]:
    shutil.copy(hd_fig / src, FIGURES / dst)
    if OVERLEAF_FIGS:
        shutil.copy(hd_fig / src, OVERLEAF_FIGS / dst)
print(f"  {res}")

# J2. Honest DiD for the pooled lead-specific riot gap (event_ddd above, riots)
r = subprocess.run(["Rscript", str(PANELS.parent.parent / "code" / "31_honest_did_extensive.R"),
                    str(HD_DIR), str(hd_fig), "0.03", "honest_gap"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:])
if r.returncode != 0:
    raise SystemExit("47: Honest DiD R step (pooled gap) failed")
res = dict(l.split("=") for l in (HD_DIR / "honest_gap_extensive_results.txt").read_text().split())
for k_r, k_m in [("rm_lb0", "sbHonestGapRMlb"), ("rm_cross", "sbHonestGapRMcross")]:
    M[k_m] = res[k_r]
shutil.copy(hd_fig / "fig_honest_did_gap.pdf", FIGURES / "fig_honest_did_gap.pdf")
if OVERLEAF_FIGS:
    shutil.copy(hd_fig / "fig_honest_did_gap.pdf", OVERLEAF_FIGS / "fig_honest_did_gap.pdf")
print(f"  pooled gap: {res}")


# ── K. Magnitude: attributable cell-years and remediation break-even ────────
print("\nK. Attributable cell-years (read from 29, binary pooled DDDD)...")
# The total is computed once, in 29_extensive_margin_main.py, so the slides and the
# paper cannot drift: pooled four-way difference x post-2007 cell-years within 5 km.
_bt = pd.read_csv(PANELS / "botec_extensive.csv").iloc[0]
bt = dict(riot=float(_bt.riot_attr), lo=float(_bt.riot_lo), hi=float(_bt.riot_hi),
          fat=float(_bt.fat_attr))
n_years = int(POOLED.year.max()) - POST_YR + 1
n_sites = int(sum(tsip[tsip.country == cn].SiteIndustry
                  .str.contains("Battery", case=False, na=False).sum() for cn in COUNTRIES))
M["sbBotecRiots"] = f"{bt['riot']:,.0f}"
M["sbBotecLo"] = f"{bt['lo']:,.0f}"
M["sbBotecHi"] = f"{bt['hi']:,.0f}"
M["sbBotecPerYear"] = f"{bt['riot'] / n_years:,.0f}"
M["sbBotecFat"] = f"{bt['fat']:,.0f}"
print(f"  total {bt['riot']:.1f} (95% CI {bt['lo']:.1f} to {bt['hi']:.1f}); "
      f"{bt['riot'] / n_years:.1f} a year; riot-death cell-years {bt['fat']:.1f}")
# Break-even value of an averted riot-death cell-year, exactly as 41
COSTS, HORIZONS = (40_300, 118_750, 430_000), (10, 20)
per_site_death = round(bt["fat"]) / n_sites / n_years
th = [c / (per_site_death * h) for c in COSTS for h in HORIZONS]
M["sbRemThreshLo"], M["sbRemThreshHi"] = f"{min(th):,.0f}", f"{max(th):,.0f}"
print(f"  break-even ${min(th):,.0f} to ${max(th):,.0f} ({n_sites} sites, {n_years} years)")
pre_near = POOLED[(POOLED.bat5 == 1) & (POOLED.year < POST_YR)]
M["sbPreNearPct"] = f"{100 * (pre_near.n_riots > 0).mean():.0f}"


# ── L. Raw riot rates by group ──────────────────────────────────────────────
print("\nL. Raw riot rates by group...")
POOLED["grp"] = np.where(POOLED.bat5 == 1, "Within 5 km of a battery site",
                np.where(POOLED.nl5 == 1, "Within 5 km of another toxic site", "Other cells, 5-50 km"))
raw = (POOLED.assign(any_riot=(POOLED.n_riots > 0).astype(float))
       .groupby(["grp", "year"]).any_riot.mean().unstack(0))
ncell = POOLED.drop_duplicates("cell_id_str").grp.value_counts()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for g_, col in [("Within 5 km of a battery site", "#006970"),
                ("Within 5 km of another toxic site", "#E08A00"),
                ("Other cells, 5-50 km", "#9AA5A8")]:
    ax.plot(raw.index, 100 * raw[g_], "o-", ms=2.5, lw=1.5, color=col,
            label=f"{g_.replace('Within 5 km of a', 'Near a').replace('Within 5 km of another', 'Near another')} ({ncell[g_]:,} cells)")
ax.axvline(POST_YR - 0.5, color="#bbbbbb", lw=0.7, ls=":")
ax.set_ylabel("Cells with any riot in the year (%)")
ax.legend(frameon=False)
tufte_ax(ax)
fig.tight_layout()
save_fig(fig, "fig_slides_raw_rates")


# ── M. Coefficient plot of the gap by conflict type ─────────────────────────
print("\nM. Coefficient plot by conflict type...")
rows = []
for yc, lab in [("n_riots", "Riots"), ("n_peaceful_protest", "Peaceful protest"),
                ("n_org_violence", "Organised violence"), ("n_battles", "  of which battles")]:
    g, s_, p, _ = gap_pooled(POOLED, yc)
    rows.append((lab, g, s_))
CR = float(scipy_stats.t.ppf(0.975, POOLED.block.nunique() - 1))   # t(G-1), pooled blocks
fig, ax = plt.subplots(figsize=(5.6, 2.6))
for i, (lab, g, s_) in enumerate(rows):
    y = len(rows) - 1 - i
    col = "#006970" if i < 2 else "#9AA5A8"
    ax.plot([g - CR * s_, g + CR * s_], [y, y], color=col, lw=2)
    ax.plot(g, y, "o", color=col, ms=6)
    ax.text(g + CR * s_ + 0.01, y, f"{g:.3f}", va="center", fontsize=9, color=col)
ax.axvline(0, color="#666", lw=0.7)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlabel("Lead-specific gap, change in Pr(any event)")
ax.tick_params(axis="y", length=0)
tufte_ax(ax)
fig.tight_layout()
save_fig(fig, "fig_slides_conflict_type")


# ── Macros ──────────────────────────────────────────────────────────────────
lines = ["% Generated by 47_slides_binary.py -- binary 1[<5 km] x Post versions of slide exhibits"]
lines += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in M.items()]
out = TABLES / "slides_binary_values.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
save_to_overleaf("slides_binary_values.tex", str(out), subfolder="tables")
print(f"\nSaved: {out.name} ({len(M)} macros)")
