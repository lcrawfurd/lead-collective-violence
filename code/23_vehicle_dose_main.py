#!/usr/bin/env python3
"""
23_vehicle_dose_main.py — Vehicle-dose specification (extensive margin)
==============================================================================
Re-estimates the main design with the binary post-2007 indicator replaced by a
continuous, country-specific vehicle-stock dose:

  Y_ct = α_c + δ_jt + β · B_c · V_jt + γ · N_c · V_jt + ε_ct

B_c = 1[cell within 5 km of a battery site], N_c = 1[within 5 km of a non-lead site]
(1/(d+1) > 1/6), the same indicators as the main design;
Y_ct = 1[any riot in cell-year] (extensive margin, matching the main
specification in 29_extensive_margin_main.py) and V_jt is cumulative new
vehicle registrations per 1,000 persons in country j at year t (OICA × WB
population, 2005-2018), standardised within country to mean 0 sd 1.

The table reports the escalating differences for the pooled three-country
sample (cell + country×year FEs), riots outcome:
  DD   — battery × vehicle dose, y ~ B_c V_jt alone           (β_battery)
  DDD  — lead-specific gap, battery minus non-lead          (β_bat − β_nl)
  conflict-type — conflict-type contrast, riot gap minus org-viol gap (quadruple diff)

Outputs
-------
  output/tables/tab_escalating_diff.tex       binary 1[<5km] x Post DD/DDD/DDDD table (+ Overleaf)
  output/tables/tab_vehicle_dose.tex          vehicle-dose DD/DDD/DDDD table (+ Overleaf)
  output/tables/vehicle_values.tex            \vehFirstYr, \vehLastYr (+ Overleaf)
  output/panels/vehicle_dose_summary.csv      coefs/SEs (read by 11_generate_values.py)
  output/figures/fig_vehicle_event.pdf        implied effect β·V_jt for a cell within 5 km
                                              of a battery site (β from the joint B/N regression)
"""

import sys, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import add_block, coef_sep, lincom, wild_p, tcrit


CODE_DIR   = Path(__file__).resolve().parent
ROOT       = CODE_DIR.parent
DATA       = ROOT / "data"
PANELS_DIR = ROOT / "output" / "panels"
TABLES_DIR = ROOT / "output" / "tables"
FIGS_DIR   = ROOT / "output" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from config import OVERLEAF_TABS as OVERLEAF_TABLES, OVERLEAF_FIGS as OVERLEAF_FIGURES
except ImportError:
    OVERLEAF_TABLES = OVERLEAF_FIGURES = None

COUNTRIES = [
    ("KE", "Kenya",   "panel_kenya.parquet"),
    ("GH", "Ghana",   "panel_ghana.parquet"),
    ("SN", "Senegal", "panel_senegal.parquet"),
]

# CGD palette
COLORS = {"Kenya": "#006970", "Ghana": "#FFB52C", "Senegal": "#85A5AD"}

OICA_ROW = {"Ghana": 145, "Kenya": 147, "Senegal": 156}
OICA_YEARS = list(range(2005, 2019))
POP = {
    "Ghana":   {2005:22449438,2006:23040702,2007:23642279,2008:24249946,2009:24862664,
                2010:25474995,2011:26095088,2012:26734268,2013:27386193,2014:28041592,
                2015:28696068,2016:29356742,2017:30008354,2018:30637585},
    "Kenya":   {2005:35796484,2006:36904008,2007:38036349,2008:39206351,2009:40408891,
                2010:41598567,2011:42758461,2012:43888303,2013:44986791,2014:46051440,
                2015:47088526,2016:48139687,2017:49197766,2018:50207107},
    "Senegal": {2005:11235472,2006:11500094,2007:11770344,2008:12048959,2009:12337389,
                2010:12635413,2011:12958897,2012:13310495,2013:13710849,2014:14152102,
                2015:14593331,2016:15034521,2017:15475002,2018:15914498},
}


def build_oica_cumulative():
    """country -> {year: cumulative vehicles per 1,000 people}."""
    df = pd.read_excel(DATA / "total_sales_2018.xlsx", header=None)
    out = {}
    for country, ridx in OICA_ROW.items():
        row = df.iloc[ridx]
        cumsum = 0
        out[country] = {}
        for yr in OICA_YEARS:
            cidx = 4 + (yr - 2005)
            try:
                cumsum += float(row[cidx])
            except (ValueError, TypeError):
                continue
            if yr in POP[country]:
                out[country][yr] = cumsum / POP[country][yr] * 1000
    return out


def stars_p(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# Within 5 km of a site: 1/(d+1) > 1/6  <=>  d < 5 km.
THRESH_5KM = 1.0 / 6.0
POST_YR = 2007


def build_pooled(panels, oica):
    """Stack KE/GH/SN with a within-country-standardised vehicle dose (2005-2018)."""
    parts = []
    for code, cname, _ in COUNTRIES:
        p = panels[code].copy()
        p = p[(p.year >= 2005) & (p.year <= 2018)].copy()
        p["country"]     = cname
        p["cell_id_str"] = code + "_" + p["cell_id"].astype(str)
        p["cty_year"]    = cname + "_" + p["year"].astype(str)
        add_block(p, cname)
        p["vehicles"]    = p.year.map(oica.get(cname, {}))
        p = p.dropna(subset=["vehicles"])
        vmu, vsd = p.vehicles.mean(), p.vehicles.std()
        p["veh_std"] = (p["vehicles"] - vmu) / vsd
        p["bat5"]    = (p["inv_dist_battery"] > THRESH_5KM).astype(float)
        p["nl5"]     = (p["inv_dist_nonlead"]  > THRESH_5KM).astype(float)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def build_pooled_full(panels):
    """Stack KE/GH/SN over the full panel (all years) for the binary post-2007 spec."""
    parts = []
    for code, cname, _ in COUNTRIES:
        p = panels[code].copy()
        p["country"]     = cname
        p["cell_id_str"] = code + "_" + p["cell_id"].astype(str)
        p["cty_year"]    = cname + "_" + p["year"].astype(str)
        add_block(p, cname)
        p["post"]        = (p["year"] >= POST_YR).astype(int)
        p["bat5"]        = (p["inv_dist_battery"] > THRESH_5KM).astype(float)
        p["nl5"]         = (p["inv_dist_nonlead"]  > THRESH_5KM).astype(float)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def dd_battery(pooled, ycount):
    """DD: pooled battery × vehicle dose on the extensive margin, block-clustered SE."""
    import pyfixest as pf
    d = pooled.copy()
    d["y"]     = (d[ycount] > 0).astype(float)
    d["bat_v"] = d["bat5"] * d["veh_std"]
    m = pf.feols("y ~ bat_v | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    coef, se, p = coef_sep(m, "bat_v")
    return coef, se, p, int(m._N)


def ddd_gap(pooled, ycount):
    """DDD: lead-specific gap (battery − non-lead) from a joint regression."""
    import pyfixest as pf
    d = pooled.copy()
    d["y"]     = (d[ycount] > 0).astype(float)
    d["bat_v"] = d["bat5"] * d["veh_std"]
    d["nl_v"]  = d["nl5"] * d["veh_std"]
    m = pf.feols("y ~ bat_v + nl_v | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    gap, se, p = lincom(m, {"bat_v": 1, "nl_v": -1})
    nl, nl_se, nl_p = coef_sep(m, "nl_v")     # non-lead coefficient, for context
    b, b_se, b_p = coef_sep(m, "bat_v")       # battery coefficient in the joint regression
    return gap, se, p, nl, nl_se, nl_p, b, b_se, b_p, tcrit(m)


def _stack(pooled, cols, typecol_map):
    """Stack riots and organised violence as two outcome types (rows keep `block`)."""
    parts = []
    for typ, ycol in typecol_map:
        d = pooled[["cell_id_str", "block", "year", "country", "cty_year"] + cols].copy()
        d["y"]    = (pooled[ycol] > 0).astype(float)
        d["type"] = typ
        parts.append(d)
    st = pd.concat(parts, ignore_index=True)
    st["is_riot"]     = (st["type"] == "riot").astype(int)
    st["cell_type"]   = st["cell_id_str"] + "_" + st["type"]
    st["ctyyeartype"] = st["cty_year"] + "_" + st["type"]
    return st


def dddd_het(pooled):
    """DDDD: quadruple difference — riot gap minus organised-violence gap.
    Stack riots and organised violence; interact both treatments with 1[riot]."""
    import pyfixest as pf
    st = _stack(pooled, ["bat5", "nl5", "veh_std"],
                [("riot", "n_riots"), ("org", "n_org_violence")])
    st["bat_v"]       = st["bat5"] * st["veh_std"]
    st["nl_v"]        = st["nl5"] * st["veh_std"]
    st["bat_v_riot"]  = st["bat_v"] * st["is_riot"]
    st["nl_v_riot"]   = st["nl_v"]  * st["is_riot"]
    # Clustered on the cell's block: both types' rows for a cell fall in the same cluster.
    m = pf.feols("y ~ bat_v + nl_v + bat_v_riot + nl_v_riot | cell_type + ctyyeartype",
                 data=st, vcov={"CRV1": "block"})
    return lincom(m, {"bat_v_riot": 1, "nl_v_riot": -1})


# ── Binary 1[<5km] × Post(2007) versions of DD / DDD / conflict-type ──────────────────
# Each also returns a wild cluster restricted bootstrap p (Rademacher, 9,999 draws,
# seed 20261014, 1-degree blocks).
def dd_battery_bin(pooled_full, ycount):
    """DD: pooled 1[<5km battery] × Post on the extensive margin, block-clustered SE."""
    import pyfixest as pf
    d = pooled_full.copy()
    d["y"]  = (d[ycount] > 0).astype(float)
    d["bt"] = d["bat5"] * d["post"]
    m = pf.feols("y ~ bt | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    coef, se, p = coef_sep(m, "bt")
    wp = wild_p(d, "y", ["bt"], {"bt": 1}, ["cell_id_str", "cty_year"])
    return coef, se, p, int(m._N), wp


def ddd_gap_bin(pooled_full, ycount):
    """DDD: binary lead-specific gap (battery − non-lead, both 1[<5km]×Post)."""
    import pyfixest as pf
    d = pooled_full.copy()
    d["y"]  = (d[ycount] > 0).astype(float)
    d["bt"] = d["bat5"] * d["post"]
    d["nt"] = d["nl5"]  * d["post"]
    m = pf.feols("y ~ bt + nt | cell_id_str + cty_year", data=d, vcov={"CRV1": "block"})
    gap, se, p = lincom(m, {"bt": 1, "nt": -1})
    wp = wild_p(d, "y", ["bt", "nt"], {"bt": 1, "nt": -1}, ["cell_id_str", "cty_year"])
    return gap, se, p, wp


def dddd_het_bin(pooled_full):
    """DDDD: binary quadruple difference — riot gap minus organised-violence gap."""
    import pyfixest as pf
    st = _stack(pooled_full, ["bat5", "nl5", "post"],
                [("riot", "n_riots"), ("org", "n_org_violence")])
    st["bt"]          = st["bat5"] * st["post"]
    st["nt"]          = st["nl5"]  * st["post"]
    st["bt_riot"]     = st["bt"] * st["is_riot"]
    st["nt_riot"]     = st["nt"] * st["is_riot"]
    X = ["bt", "nt", "bt_riot", "nt_riot"]
    m = pf.feols("y ~ " + " + ".join(X) + " | cell_type + ctyyeartype",
                 data=st, vcov={"CRV1": "block"})
    diff, se, p = lincom(m, {"bt_riot": 1, "nt_riot": -1})
    wp = wild_p(st, "y", X, {"bt_riot": 1, "nt_riot": -1}, ["cell_type", "ctyyeartype"])
    return diff, se, p, wp


# ── Implied effect over time figure (extensive margin) ───────────────────────
def fig_implied_effect_over_time(pooled, b, b_se, crit):
    """Implied effect b̂ × V_jt^std by country and year: the change in Pr(any riot) in a
    cell within 5 km of a battery site (B_c = 1, N_c = 0) relative to a cell with
    B_c = N_c = 0, at country j's standardised vehicle stock in year t. b̂ is the battery
    coefficient from the joint regression with N_c × V_jt (cell + country-year FE,
    block-clustered SE); the band is b̂ ± t(G-1) × SE, scaled by V_jt^std."""
    v = (pooled.groupby(["country", "year"]).veh_std.first().reset_index())
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for _, cname, _ in COUNTRIES:
        g = v[v.country == cname].sort_values("year")
        x, vs = g.year.values, g.veh_std.values
        lo, hi = (b - crit * b_se) * vs, (b + crit * b_se) * vs
        ax.fill_between(x, np.minimum(lo, hi), np.maximum(lo, hi), color=COLORS[cname],
                        alpha=0.12, lw=0)
        ax.plot(x, b * vs, marker="o", markersize=3.5, linewidth=1.8,
                color=COLORS[cname], label=cname)
    ax.axhline(0, color="#666", linewidth=0.7)
    ax.axvline(2007, color="#444", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Implied change in Pr(any riot), cell within 5 km of a battery site")
    ax.legend(frameon=False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIGS_DIR / "fig_vehicle_event.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    if OVERLEAF_FIGURES:
        shutil.copy(out, Path(OVERLEAF_FIGURES) / out.name)
    print(f"✓  {out.name}")
    plt.close(fig)


# ── LaTeX tables: escalating differences (binary, main text) + vehicle (appendix)
def _save(fname, lines):
    out = TABLES_DIR / fname
    out.write_text("\n".join(lines))
    print(f"✓  {fname}")
    if OVERLEAF_TABLES:
        shutil.copy(out, Path(OVERLEAF_TABLES) / out.name)


def write_main_binary(bdd, bddd, bdddd, n_bin, n_treat_cells=None, n_cells=None):
    """Main-text table: binary 1[<5km] x Post escalating differences."""
    def c(coef, p): return f"{coef:.3f}{stars_p(p)}"
    def s(se):      return f"({se:.3f})"
    lines = [
        r"\begin{table}[H]",
        r"\caption{The Lead-Specific Riot Effect in Three Differences (Pooled, Extensive Margin)}",
        r"\label{tab:escalating}",
        r"\begin{threeparttable}",
        r"\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & (1) DD & (2) DDD & (3) Conflict-type \\",
        r" & Battery gradient & Lead-specific gap & Conflict-type contrast \\",
        r"\midrule",
        r"$\mathbf{1}[<5\,\text{km}] \times \text{Post (2007)}$ & "
        + f"{c(bdd[0], bdd[4])} & {c(bddd[0], bddd[3])} & {c(bdddd[0], bdddd[3])}" + r" \\",  # stars from the wild bootstrap p
        r" & " + f"{s(bdd[1])} & {s(bddd[1])} & {s(bdddd[1])}" + r" \\",
        r"Wild bootstrap $p$ & " + f"[{bdd[4]:.3f}] & [{bddd[3]:.3f}] & [{bdddd[3]:.3f}]" + r" \\[3pt]",
        r"Difference across & time, distance & $+$ site type & $+$ conflict type \\",
        r"Removes & --- & industrial proximity & conflict-neutral confounds \\",
        r"\midrule",
        r"Treated cells ($<$5\,km) / all cells & \multicolumn{3}{c}{"
        + (f"{n_treat_cells:,} / {n_cells:,}" if n_treat_cells is not None else "---") + r"} \\",
        r"Cell-years & \multicolumn{3}{c}{" + f"{n_bin:,}" + r" (pooled KE $+$ GH $+$ SN)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\setstretch{1}\footnotesize",
        # Table note lives in main.tex (\setTabNote{tab_escalating_diff}{...}).
        r"\item \TabNote{tab_escalating_diff}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    _save("tab_escalating_diff.tex", lines)


def write_appendix_vehicle(dd, ddd, dddd, n_veh):
    """Appendix table: continuous vehicle-dose escalating differences."""
    def c(coef, p): return f"{coef:.3f}{stars_p(p)}"
    def s(se):      return f"({se:.3f})"
    lines = [
        r"\begin{table}[H]",
        r"\caption{Escalating Differences under a Continuous Vehicle-Dose Treatment (Pooled, Extensive Margin)}",
        r"\label{tab:vehicle_dose}",
        r"\begin{threeparttable}",
        r"\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & (1) DD & (2) DDD & (3) Conflict-type \\",
        r" & Battery gradient & Lead-specific gap & Conflict-type contrast \\",
        r"\midrule",
        r"$\mathbf{1}[<5\,\text{km}] \times \text{Vehicle dose}$ & "
        + f"{c(dd[0], dd[2])} & {c(ddd[0], ddd[2])} & {c(dddd[0], dddd[2])}" + r" \\",
        r" & " + f"{s(dd[1])} & {s(ddd[1])} & {s(dddd[1])}" + r" \\[3pt]",
        r"Difference across & time, distance & $+$ site type & $+$ conflict type \\",
        r"\midrule",
        r"Cell-years & \multicolumn{3}{c}{" + f"{n_veh:,}" + r" (pooled KE $+$ GH $+$ SN, 2005--2018)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\setstretch{1}\footnotesize",
        # Table note lives in main.tex (\setTabNote{tab_vehicle_dose}{...}).
        r"\item \TabNote{tab_vehicle_dose}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    _save("tab_vehicle_dose.tex", lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading panels...")
    panels = {code: pd.read_parquet(PANELS_DIR / pfile)
              for code, _, pfile in COUNTRIES}

    print("Building OICA cumulative series...")
    oica = build_oica_cumulative()
    for cname, s in oica.items():
        print(f"  {cname}: {len(s)} years, {s[2005]:.1f} → {s[2018]:.1f} per 1,000")

    # In-text macros: first and last year of the vehicle series
    _v = ["% Generated by 23_vehicle_dose_main.py -- do not edit by hand",
          r"\newcommand{\vehFirstYr}{" + f"{min(OICA_YEARS)}" + "}",
          r"\newcommand{\vehLastYr}{" + f"{max(OICA_YEARS)}" + "}"]
    _out = TABLES_DIR / "vehicle_values.tex"
    _out.write_text("\n".join(_v) + "\n", encoding="utf-8")
    if OVERLEAF_TABLES:
        shutil.copy(_out, Path(OVERLEAF_TABLES) / _out.name)

    print("\nBuilding pooled panels...")
    pooled      = build_pooled(panels, oica)     # 2005-2018, vehicle dose
    pooled_full = build_pooled_full(panels)       # full panel, binary post-2007

    print("\nBinary 1[<5km] x Post(2007), extensive margin:")
    bdd   = dd_battery_bin(pooled_full, "n_riots")   # coef, se, p, n
    bddd  = ddd_gap_bin(pooled_full, "n_riots")      # gap, se, p
    bdddd = dddd_het_bin(pooled_full)                 # diff, se, p
    n_bin = bdd[3]
    n_treat_cells = int(pooled_full.loc[pooled_full["bat5"] == 1, "cell_id_str"].nunique())
    n_all_cells   = int(pooled_full["cell_id_str"].nunique())
    print(f"  DD   (battery)            = {bdd[0]:+.3f} ({bdd[1]:.3f}){stars_p(bdd[2])}")
    print(f"  DDD  (lead-specific gap)  = {bddd[0]:+.3f} ({bddd[1]:.3f}){stars_p(bddd[2])}")
    print(f"  conflict-type (conflict-type)      = {bdddd[0]:+.3f} ({bdddd[1]:.3f}){stars_p(bdddd[2])}")
    print(f"  wild bootstrap p: DD {bdd[4]:.4f}  DDD {bddd[3]:.4f}  DDDD {bdddd[3]:.4f}")

    print("\nVehicle dose, extensive margin:")
    dd   = dd_battery(pooled, "n_riots")        # coef, se, p, n
    ddd  = ddd_gap(pooled, "n_riots")           # gap, se, p, nl, nl_se, nl_p, b, b_se, b_p, crit
    dddd = dddd_het(pooled)                      # diff, se, p
    n_pooled = dd[3]
    print(f"  DD   (battery)            = {dd[0]:+.3f} ({dd[1]:.3f}){stars_p(dd[2])}")
    print(f"  DDD  (lead-specific gap)  = {ddd[0]:+.3f} ({ddd[1]:.3f}){stars_p(ddd[2])}  "
          f"[battery {ddd[6]:+.3f} ({ddd[7]:.3f}){stars_p(ddd[8])}; non-lead {ddd[3]:+.3f} ({ddd[4]:.3f}){stars_p(ddd[5])}]")
    print(f"  conflict-type (conflict-type)      = {dddd[0]:+.3f} ({dddd[1]:.3f}){stars_p(dddd[2])}")

    # CSV for 11_generate_values.py
    pd.DataFrame([{
        "dd_coef": dd[0],
        "ddd_gap": ddd[0],  "ddd_se": ddd[1],
        "nl_coef": ddd[3],
        "dddd_diff": dddd[0], "dddd_se": dddd[1],
        "bin_dd_coef": bdd[0],   "bin_dd_se": bdd[1],
        "bin_ddd_gap": bddd[0],  "bin_ddd_se": bddd[1],
        "bin_dddd_diff": bdddd[0], "bin_dddd_se": bdddd[1],
    }]).to_csv(PANELS_DIR / "vehicle_dose_summary.csv", index=False)
    print(f"✓  vehicle_dose_summary.csv")

    print("\nWriting tables...")
    write_main_binary(bdd, bddd, bdddd, n_bin, n_treat_cells, n_all_cells)       # main text
    write_appendix_vehicle(dd, ddd, dddd, n_pooled)  # appendix

    print("\nWriting implied-effect figure (probability units)...")
    fig_implied_effect_over_time(pooled, ddd[6], ddd[7], ddd[9])

    print("\nDone.")
