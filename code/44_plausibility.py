r"""
44_plausibility.py
Numbers for Appendix D.2 (effect-size plausibility): push published lead-crime effect
sizes through a Poisson bridge to the probability that a cell-year within 5 km of a
battery site records any riot, and compare with our estimates.

  A cell-year records a riot with P = 1 - exp(-lambda). Lead multiplies lambda by a
  factor m. For elasticity estimates m = r**elasticity, with r the ring-average blood
  lead level (BLL) relative to background: r = (B + Delta) / B. For the Aizer-Currie
  row m = 1 + rho * Delta. Predicted gap = (1 - exp(-m * lambda0)) - p0.

Inputs, all read from files (nothing typed in except the published elasticities):
  Delta  ring-average BLL ELEVATION within 5 km of a battery site, the same in every row:
         the population-weighted mean of the two ring estimates in Crawfurd et al. (2025,
         Beyond Hotspots; replication estimates.txt): \bloodinner within 165 m and
         \bloodouter at 165 m-5 km, weighted by \popinner and \popouter. These are
         increments over background (that paper subtracts background soil lead).
  B      background BLL: IHME GBD 2023 mean BLL, all ages, Kenya, Ghana and Senegal (the
         file used by that paper; reported in ug/L, divided by 10). The three countries
         give the low and high r of each elasticity row.
  p0     comparison-cell riot probability: the pre-2007 probability of any riot in cells
         within 5 km of a battery site, plus the pooled non-lead post-2007 coefficient
         from the main joint regression (bmRiotPooledNl, 55_binary_main_table.py), i.e.
         what a battery cell would show after 2007 if it moved like a non-lead cell.
  Delta (Africa) the same calculation with the soil-lead distance decay re-estimated on the
         TSIP battery sites in Kenya, Ghana and Senegal only, following that paper's
         05_tsip_distance_decay.do: subtract background soil lead (41.8 mg/kg), regress log
         soil lead on log distance with site fixed effects, retransform with Duan smearing,
         take area-weighted means over 0-165 m and 165 m-5 km, convert soil to BLL with the
         AALMv3 lookup (times 2, soil being half of exposure) and weight the two rings by
         the same populations. Few African samples lie beyond 1 km, so the 1-5 km part is
         extrapolated; the global and African curves bracket the dose.
Compared with: the pooled lead-specific gap (bmRiotPooledGap) and the pooled four-way
difference, riots minus organised violence (bmDDDDPooled).

Also reads data/literature_benchmarks.csv (published riot/conflict effect sizes for Section 6).
Outputs: output/tables/plausibility_values.tex, output/tables/tab_plausibility.tex
(both copied to Overleaf tables/)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest as pf

from config import DATA, TABLES, PANELS, POST_YR, save_to_overleaf

ULAB = DATA / "beyond_hotspots"      # inputs copied from the Beyond Hotspots replication package
IHME = ULAB / "IHME_GBD_2023_LEAD_RISK_1990_2023_BLL_2023_BOTH_Y2025M11D12.CSV"
COUNTRIES = ["Kenya", "Ghana", "Senegal"]
NEAR5 = 1.0 / 6.0

# ── published effect sizes ────────────────────────────────────────────────────
ELAST_REYES  = 0.8           # violent crime w.r.t. lead (Reyes 2007)
ELAST_HIGNEY = 0.09          # publication-bias-corrected (Higney et al. 2022)
N_HIGNEY     = 542           # estimates pooled by Higney et al. (2022)
AIZER_RHO    = 0.25          # relative rise in the riot-prone pool per ug/dL (Aizer and Currie 2019 range)
NEAR_M, FAR_M = 165, 5000    # ring boundaries of the Beyond Hotspots estimates, metres


def tex_macro(text, name, src):
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{\$?([-\d.,]+)\$?\}", text)
    if not m:
        raise SystemExit(f"44: {name} missing from {src}")
    return float(m.group(1).replace(",", ""))


# ── Delta: ring-average elevation within 5 km ────────────────────────────────
est = (ULAB / "estimates.txt").read_text()
BLL_NEAR = tex_macro(est, "bloodinner", "estimates.txt")
BLL_FAR = tex_macro(est, "bloodouter", "estimates.txt")
POP_NEAR = tex_macro(est, "popinner", "estimates.txt")
POP_FAR = tex_macro(est, "popouter", "estimates.txt")
DELTA = (POP_NEAR * BLL_NEAR + POP_FAR * BLL_FAR) / (POP_NEAR + POP_FAR)

# ── Delta (Africa): distance decay re-estimated on KE/GH/SN sites ─────────────
BG_SOIL = 41.80                  # background soil lead subtracted in Beyond Hotspots (mg/kg)
alm = pd.read_excel(ULAB / "AALMv3 results.xlsx", sheet_name="main")
alm = alm[["soil mgkg", "bll (intake 3x AALM)"]].dropna()
alm.columns = ["soil", "bll"]
alm = alm.groupby("soil").bll.first().sort_index()     # first row per soil level, as 05


def soil_to_bll(soil):
    x, y = alm.index.values, alm.values
    if soil > x[-1]:                                     # linear extrapolation, as Stata ipolate, epolate
        v = y[-2] + (y[-1] - y[-2]) * (soil - x[-2]) / (x[-1] - x[-2])
    else:
        v = np.interp(soil, x, y)
    return 2 * v


def ring_blls(sites):
    s = sites.copy()
    s["lnlead"] = np.log((s.lead - BG_SOIL).clip(lower=1e-5) + 1)
    s["lndist"] = np.log(s.dist + 1)
    m = pf.feols("lnlead ~ lndist | site_id", data=s, vcov={"CRV1": "site_id"})
    b = m.coef()["lndist"]
    k = np.exp(s.lnlead.mean() - b * s.lndist.mean()) * np.mean(np.exp(m.resid()))
    ring = lambda r1, r2: 2 * k / ((b + 2) * (r2**2 - r1**2)) * (r2**(b + 2) - r1**(b + 2))
    return b, soil_to_bll(ring(0.1, NEAR_M)), soil_to_bll(ring(NEAR_M, FAR_M))


tsip = pd.read_stata(ULAB / "tsip_ulabsitesonly.dta")
b_all, near_all, far_all = ring_blls(tsip)
if abs(near_all - BLL_NEAR) > 0.1 or abs(far_all - BLL_FAR) > 0.1:   # must reproduce the paper
    raise SystemExit(f"44: global decay gives {near_all:.2f}/{far_all:.2f}, paper {BLL_NEAR}/{BLL_FAR}")
af = tsip[tsip.country.isin(COUNTRIES)]
b_af, near_af, far_af = ring_blls(af)
DELTA_AF = (POP_NEAR * near_af + POP_FAR * far_af) / (POP_NEAR + POP_FAR)
N_AF_SITES, N_AF_OBS = af.site_id.nunique(), len(af)
N_AF_FAR = int((af.dist > 1000).sum())
print(f"decay slope: all {b_all:.2f}, Africa {b_af:.2f}; Africa rings {near_af:.1f}/{far_af:.1f}; "
      f"Delta Africa {DELTA_AF:.2f}; {N_AF_SITES} sites, {N_AF_OBS} samples, {N_AF_FAR} beyond 1 km")

# ── B: background BLL, IHME GBD 2023, all ages ───────────────────────────────
ih = pd.read_csv(IHME, encoding="utf-8-sig")
ih = ih[ih.location_name.isin(COUNTRIES) & (ih.age_group_name == "All ages")
        & (ih.rei_name == "Lead exposure in blood")]
if sorted(ih.location_name) != sorted(COUNTRIES):
    raise SystemExit("44: IHME all-ages BLL missing for a study country")
BG = {r.location_name: r["mean"] / 10 for _, r in ih.iterrows()}       # ug/L -> ug/dL
R = {c: (BG[c] + DELTA) / BG[c] for c in COUNTRIES}
r_lo, r_hi = min(R.values()), max(R.values())
R_AF = {c: (BG[c] + DELTA_AF) / BG[c] for c in COUNTRIES}
ra_lo, ra_hi = min(R_AF.values()), max(R_AF.values())

# ── p0 and comparison estimates ──────────────────────────────────────────────
bm = (TABLES / "binary_main_values.tex").read_text(encoding="utf-8")
def bmv(name):
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", bm)
    if not m:
        raise SystemExit(f"44: {name} missing from binary_main_values.tex; run 55 first")
    return float(m.group(1).replace("$-$", "-"))
NL_POST = bmv("bmRiotPooledNl")
GAP = bmv("bmRiotPooledGap")
DDDD = bmv("bmDDDDPooled")
pre = []
for cn in COUNTRIES:
    p = pd.read_parquet(PANELS / f"panel_{cn.lower()}.parquet")
    pre.append(p[(p.year < POST_YR) & (p.inv_dist_battery > NEAR5)])
PRE_BAT = float((pd.concat(pre).n_riots > 0).mean())      # as 'pre' in 55 (pooled)
P0 = PRE_BAT + NL_POST
lam0 = -np.log(1 - P0)


def gap_pp(mult):
    return 100 * ((1 - np.exp(-mult * lam0)) - P0)


reyes = [gap_pp(r ** ELAST_REYES) for r in (r_lo, r_hi)]
higney = [gap_pp(r ** ELAST_HIGNEY) for r in (r_lo, r_hi)]
aizer = gap_pp(1 + AIZER_RHO * DELTA)
reyes_af = [gap_pp(r ** ELAST_REYES) for r in (ra_lo, ra_hi)]
higney_af = [gap_pp(r ** ELAST_HIGNEY) for r in (ra_lo, ra_hi)]
aizer_af = gap_pp(1 + AIZER_RHO * DELTA_AF)
print(f"Africa decay: Reyes {reyes_af[1]:.1f}-{reyes_af[0]:.1f}; Higney {higney_af[1]:.2f}-{higney_af[0]:.2f}; Aizer {aizer_af:.1f}")
print(f"Delta = {DELTA:.3f} ug/dL; background {', '.join(f'{c} {BG[c]:.2f}' for c in COUNTRIES)}; "
      f"r = {r_lo:.3f}-{r_hi:.3f}")
print(f"p0 = {PRE_BAT:.3f} + {NL_POST:.3f} = {P0:.3f}")
print(f"Reyes {reyes[0]:.1f}-{reyes[1]:.1f} pp; Higney {higney[0]:.2f}-{higney[1]:.2f} pp; "
      f"Aizer {aizer:.1f} pp; vs gap {100*GAP:.1f} pp and four-way {100*DDDD:.1f} pp")

v = ["% Generated by 44_plausibility.py -- do not edit by hand"]
def add(k, val): v.append(f"\\newcommand{{\\{k}}}{{{val}}}")
add("plBllNear", f"{BLL_NEAR:.1f}"); add("plBllFar", f"{BLL_FAR:.1f}")
add("plNearM", f"{NEAR_M}")
add("plBgLo", f"{min(BG.values()):.1f}"); add("plBgHi", f"{max(BG.values()):.1f}")
add("plElReyes", f"{ELAST_REYES:g}"); add("plElHigney", f"{ELAST_HIGNEY:g}"); add("plNHigney", f"{N_HIGNEY}")
add("plPzero", f"{P0:.2f}")
add("plReyesLo", f"{reyes[0]:.1f}"); add("plReyesHi", f"{reyes[1]:.1f}")
add("plHigneyLo", f"{higney[0]:.1f}"); add("plHigneyHi", f"{higney[1]:.1f}")
add("plAizerRho", f"{100*AIZER_RHO:.0f}")
add("plNAfSites", f"{N_AF_SITES}"); add("plNAfObs", f"{N_AF_OBS}"); add("plNAfFar", f"{N_AF_FAR}")
add("plBgSoil", f"{BG_SOIL:.1f}")
# Benchmarks from other riot studies (Section 6): published estimates, read from a sourced
# file (data/literature_benchmarks.csv gives the citation and table for each), and our gap
# as a share of the comparison-cell riot probability, on the same percent-of-mean scale.
bench = pd.read_csv(DATA / "literature_benchmarks.csv").set_index("key")
for k in ["benchMcGuirkPct", "benchHsiangPct"]:
    if k not in bench.index:
        raise SystemExit(f"44: {k} missing from data/literature_benchmarks.csv")
    add(k, f"{bench.loc[k, 'value']:.0f}")
add("gapPctBase", f"{100 * GAP / P0:.0f}")
(TABLES / "plausibility_values.tex").write_text("\n".join(v) + "\n", encoding="utf-8")
save_to_overleaf("plausibility_values.tex", str(TABLES / "plausibility_values.tex"), subfolder="tables")
print("Saved: plausibility_values.tex")

# ── Appendix table ────────────────────────────────────────────────────────────
t = [r"\begin{table}[htbp]",
     r"\caption{Effect-Size Plausibility}", r"\label{tab:plausibility}",
     r"\begin{threeparttable}", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
     r" & \multicolumn{2}{c}{Soil-lead distance decay} \\",
     r"\cmidrule(lr){2-3}",
     r" & All TSIP sites & Kenya, Ghana, Senegal \\",
     r"\midrule",
     f"Distance-decay elasticity & {b_all:.2f} & {b_af:.2f} \\\\".replace("-", "$-$"),
     f"Blood lead increase within 5\\,km ($\\mu$g/dL) & {DELTA:.1f} & {DELTA_AF:.1f} \\\\",
     f"Ratio to background, $r$ & {r_lo:.2f}--{r_hi:.2f} & {ra_lo:.2f}--{ra_hi:.2f} \\\\[3pt]",
     r"\multicolumn{3}{l}{\textit{Predicted within-5\,km gap from published effect sizes (pp)}} \\",
     f"\\citet{{reyes2007}}, $\\varepsilon={ELAST_REYES:g}$ & {min(reyes):.1f}--{max(reyes):.1f} & {min(reyes_af):.1f}--{max(reyes_af):.1f} \\\\",
     f"\\citet{{higney2022}}, $\\varepsilon={ELAST_HIGNEY:g}$ & {min(higney):.1f}--{max(higney):.1f} & {min(higney_af):.1f}--{max(higney_af):.1f} \\\\",
     f"\\citet{{aizer2019}}, individual dose-response & {aizer:.1f} & {aizer_af:.1f} \\\\",
     r"\midrule",
     r"\multicolumn{3}{l}{\textit{Estimated (pp)}} \\",
     f"Lead-specific gap, riots (Table~\\ref{{tab:main}}) & \\multicolumn{{2}}{{c}}{{{100*GAP:.1f}}} \\\\",
     f"Riots minus organised violence (Table~\\ref{{tab:main}}) & \\multicolumn{{2}}{{c}}{{{100*DDDD:.1f}}} \\\\",
     r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\setstretch{1}\footnotesize",
     # Table note lives in main.tex (\setTabNote{tab_plausibility}{...}).
     r"\item \TabNote{tab_plausibility}",
     r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_plausibility.tex").write_text("\n".join(t), encoding="utf-8")
save_to_overleaf("tab_plausibility.tex", str(TABLES / "tab_plausibility.tex"), subfolder="tables")
print("Saved: tab_plausibility.tex")
