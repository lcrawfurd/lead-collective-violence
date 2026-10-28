"""
36_trigger_coding.py

Separates the two mechanisms that both predict the paper's headline signature
(riots up near battery sites, organised violence flat):

  IMPULSE CONTROL  lead impairs inhibitory control, so spontaneous violence rises
                   regardless of what the riot is about.
  GRIEVANCE        a visibly poisoning site generates local anger, so riots and
                   protests about the site (or about pollution) rise.

The conflict-type contrast in the paper cannot tell these apart. The riot TRIGGER can.
We classify each ACLED riot from its notes field and re-estimate the lead-specific gap
separately by trigger category.

Discriminating prediction:
  - If the gap is concentrated in pollution/contamination-triggered riots -> grievance.
  - If the gap holds for MOB JUSTICE riots (lynchings of suspected thieves, witchcraft
    accusations, spontaneous assaults on individuals), that is impulse control: anger at
    the recycling site cannot explain why neighbours beat a suspected phone thief.

Classification is deterministic keyword matching on ACLED `notes`, in a fixed priority
order, with pollution FIRST so that any mention of contamination counts as grievance --
biasing the test AGAINST the impulse-control reading. Counts and sample notes per
category are printed so the coding can be audited by hand.

Inputs:
  data/acled_kenya.csv, data/acled_ghana.csv, data/ACLED_global.csv (Senegal),
  data/tsip_sites_cleaned.csv, output/panels/panel_*.parquet (01),
  output/panels/urban_controls.parquet (19)

Outputs (tables also copied to Overleaf):
  output/tables/tab_trigger.tex, tab_trigger_comp.tex, tab_trigger_robust.tex
  output/tables/trigger_values.tex
  output/panels/trigger_classified.csv   (event-level, for manual audit; read by 47)
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf

from config import (TSIP_PATH, ACLED_KE, ACLED_GH, ACLED_GLOBAL, PANELS, TABLES,
                    GRID, POST_YR, END_YR, save_to_overleaf)
from utils import stars, add_block, lincom


# ── Trigger categories, in priority order ────────────────────────────────────
# Pollution first: any contamination mention is scored as grievance, which makes the
# impulse-control reading harder to support, not easier.
CATEGORIES = [
    # Narrowed after hand-auditing: bare "poisoning" caught mob-justice events
    # ("lynched a man for poisoning their cows") and "quarry" caught fee disputes.
    # Both inflated the grievance count, which is the count that has to be large for
    # the grievance story to work -- so the narrower list is the honest one.
    ("pollution", [
        "pollut", "contaminat", "toxic waste", "lead poison", "hazardous waste",
        "smelter", "smelting", "fumes", "effluent", "industrial waste", "e-waste",
        "scrap yard", "chemical spill", "air quality", "water contamin",
        "dumping of waste", "environmental damage",
    ]),
    ("mob_justice", [
        "mob ", "a mob", "lynch", "vigilante", "thief", "thieves", "stole", "stealing",
        "theft", "robber", "burglar", "witchcraft", "witch", "suspected criminal",
        "beaten to death", "stoned to death", "burnt to death", "cattle rustler",
        "suspected of stealing", "break into", "beak into",
    ]),
    ("political", [
        "election", "polling", "voter", "ballot", "candidate", "party", "supporter",
        "governor", "member of parliament", "opposition", "politic", "president",
        "impeach", "primaries", "nomination", "mca", "senator",
    ]),
    ("policing", [
        "police brutality", "extrajudicial", "in custody", "shot by police",
        "killed by police", "death of a student", "court", "verdict",
    ]),
    ("economic", [
        "cost of living", "fuel price", "price increase", "increased cost", "tax",
        "levy", "wage", "salary", "unpaid", "strike", "retrench", "unemploy",
        "economic hardship", "subsid", "fare",
    ]),
    ("services", [
        "water shortage", "electricity", "power outage", "road", "school", "hospital",
        "clinic", "flood", "garbage", "refuse collection", "eviction", "demoli",
        "housing", "land dispute",
    ]),
]

def classify(note):
    n = str(note).lower()
    if n in ("nan", ""):
        return "unclassified"
    for name, terms in CATEGORIES:
        if any(t in n for t in terms):
            return name
    return "other"


def hav(lat1, lon1, lat2, lon2):
    R, p = 6371.0, np.pi / 180
    dlat, dlon = (lat2 - lat1) * p, (lon2 - lon1) * p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_riots(cn, path):
    cols = ["year", "event_type", "latitude", "longitude", "notes"]
    if path and os.path.exists(path):
        d = pd.read_csv(path, low_memory=False)
    else:
        d = pd.read_csv(ACLED_GLOBAL, low_memory=False)
        d = d[d["country"] == cn]
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    d = d.dropna(subset=["latitude", "longitude", "year"]).copy()
    d = d[d["event_type"] == "Riots"].copy()
    d["year"] = d["year"].astype(int)
    d = d[d["year"] <= END_YR]
    d["cell_lat"] = (np.floor(d.latitude / GRID) * GRID + GRID/2).round(4)
    d["cell_lon"] = (np.floor(d.longitude / GRID) * GRID + GRID/2).round(4)
    d["cell_id"] = d.cell_lat.astype(str) + "_" + d.cell_lon.astype(str)
    d["country"] = cn
    return d[cols + ["cell_id", "country"]]


COUNTRIES = [("Kenya", ACLED_KE), ("Ghana", ACLED_GH), ("Senegal", None)]

print("Classifying riot triggers from ACLED notes...\n")
ev = pd.concat([load_riots(cn, p) for cn, p in COUNTRIES], ignore_index=True)
ev["trigger"] = ev["notes"].apply(classify)

CATS = [c for c, _ in CATEGORIES] + ["other", "unclassified"]
print("Riot events by trigger category:")
tot = len(ev)
for c in CATS:
    n = (ev.trigger == c).sum()
    print(f"  {c:14s} {n:5d}  ({100*n/tot:4.1f}%)")
print(f"  {'TOTAL':14s} {tot:5d}")

print("\nSample notes per category (audit these):")
for c in CATS:
    sub = ev[ev.trigger == c]
    if not len(sub):
        continue
    s = str(sub.sample(1, random_state=3).iloc[0].notes)[:150]
    print(f"  [{c}] {s}")

ev.to_csv(str(PANELS / "trigger_classified.csv"), index=False)

# ── Attach category counts to the paper's own cell-year panels ───────────────
PAN = {c: pd.read_parquet(str(PANELS / f"panel_{c.lower()}.parquet")) for c, _ in COUNTRIES}
parts = []
for cn, _ in COUNTRIES:
    p = PAN[cn].copy()
    p["post"] = (p.year >= POST_YR).astype(int)
    p["country"] = cn
    p["cell_id_str"] = cn[:2] + "_" + p.cell_id.astype(str)
    add_block(p, cn)       # 1-degree blocks: cluster for every reported SE
    p["cty_year"] = cn + "_" + p.year.astype(str)
    e = ev[ev.country == cn]
    for c in CATS:
        g = (e[e.trigger == c].groupby(["cell_id", "year"]).size()
             .rename(f"n_{c}").reset_index())
        p = p.merge(g, on=["cell_id", "year"], how="left")
        p[f"n_{c}"] = p[f"n_{c}"].fillna(0)
    parts.append(p)
POOLED = pd.concat(parts, ignore_index=True)


def lpm_gap(d, ycol, nl5=None):
    """Joint extensive-margin LPM on the main binary design (cell within 5 km of a
    battery / non-lead site, x 2007 onwards); gap = b_battery - b_nonlead,
    block-clustered SE. 1/(d+1) > 1/6 <=> d < 5 km. nl5: optional column holding an
    alternative 0/1 non-lead indicator (e.g. without mining sites)."""
    d = d.copy()
    d["y"] = (d[ycol] > 0).astype(float)
    d["bat_post"] = (d.inv_dist_battery > 1.0 / 6.0).astype(int) * d.post
    nl = d[nl5] if nl5 else (d.inv_dist_nonlead > 1.0 / 6.0).astype(int)
    d["nl_post"] = nl * d.post
    m = pf.feols("y ~ bat_post + nl_post | cell_id_str + cty_year",
                 data=d, vcov={"CRV1": "block"})
    return lincom(m, {"bat_post": 1, "nl_post": -1})


print("\n" + "="*76)
print("LEAD-SPECIFIC GAP BY RIOT TRIGGER (pooled, extensive margin)")
print("="*76)
print(f"  {'trigger':14s} {'events':>7s} {'cell-yrs>0':>11s} {'gap':>9s} {'se':>8s}")
res = {}
for c in CATS + ["all"]:
    ycol = "n_riots" if c == "all" else f"n_{c}"
    nev = len(ev) if c == "all" else int((ev.trigger == c).sum())
    npos = int((POOLED[ycol] > 0).sum())
    if npos < 30:
        print(f"  {c:14s} {nev:7d} {npos:11d}   too few positive cell-years")
        continue
    g, s, pv = lpm_gap(POOLED, ycol)
    res[c] = dict(gap=g, se=s, p=pv, n_events=nev, n_pos=npos)
    print(f"  {c:14s} {nev:7d} {npos:11d} {g:9.3f}{stars(pv):<3s} ({s:.3f})")

print(f"\nSaved: {PANELS/'trigger_classified.csv'}")

# ── Table + in-text macros ───────────────────────────────────────────────────
LBL = {"pollution": "Pollution / contamination", "mob_justice": "Mob justice (individuals)",
       "political": "Political / electoral", "policing": "Policing and courts",
       "economic": "Prices, wages, taxes", "services": "Services / infrastructure",
       "other": "Other / unspecified", "all": "All riots"}
ORDER = ["pollution", "mob_justice", "political", "services", "economic", "policing", "other", "all"]

t = [r"\begin{table}[htbp]",
     r"\caption{Lead-Specific Gap by Riot Trigger}", r"\label{tab:trigger}",
     r"\begin{threeparttable}", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
     r"Riot trigger & Events & Share (\%) & Lead-specific gap \\", r"\midrule"]
for c in ORDER:
    nev = len(ev) if c == "all" else int((ev.trigger == c).sum())
    sh = 100 * nev / len(ev)
    if c == "all":
        t.append(r"\midrule")
    if c in res:
        r_ = res[c]
        cell = f"{r_['gap']:.3f}{stars(r_['p'])}"
        se = f"({r_['se']:.3f})"
    else:
        cell, se = "---\\tnote{a}", ""
    t.append(f"{LBL[c]} & {nev:,} & {sh:.1f} & {cell} \\\\")
    if se:
        t.append(f" & & & {se} \\\\[2pt]")
t += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_trigger}{...}).
      r"\item \TabNote{tab_trigger}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_trigger.tex").write_text("\n".join(t), encoding="utf-8")
save_to_overleaf("tab_trigger.tex", str(TABLES / "tab_trigger.tex"), subfolder="tables")

npoll = int((ev.trigger == "pollution").sum())
# In-text macros, collected here and written once at the end of the script
V = ["% auto-generated by 36_trigger_coding.py",
     r"\newcommand{\trigPollutionN}{" + f"{npoll}" + "}",
     r"\newcommand{\trigPollutionShare}{" + f"{100*npoll/len(ev):.1f}" + "}",
     r"\newcommand{\trigMobShare}{" + f"{100*(ev.trigger=='mob_justice').mean():.0f}" + "}",
     r"\newcommand{\trigPolShare}{" + f"{100*(ev.trigger=='political').mean():.0f}" + "}",
     r"\newcommand{\trigServShare}{" + f"{100*(ev.trigger=='services').mean():.0f}" + "}",
     r"\newcommand{\trigMobGap}{" + f"{res['mob_justice']['gap']:.3f}" + "}",
     r"\newcommand{\trigPoliticalGap}{" + f"{res['political']['gap']:.3f}" + "}",
     r"\newcommand{\trigNRiots}{" + f"{len(ev):,}" + "}"]
print("\nSaved: tab_trigger.tex")

# ── Composition test: are near-site riots ABOUT the sites? ───────────────────
# The gap-by-category test above asks whether the LEVEL of each riot type rises.
# This asks whether the MIX changes. If the sites generate riots about themselves,
# riots near battery sites should be disproportionately environmental. The
# design-consistent comparison is battery-near vs NON-LEAD-near, since both are
# contaminated industrial locations and urbanicity is then broadly held fixed
# (comparing near vs far instead mostly recovers the urban/rural gradient).
from scipy.stats import fisher_exact, chi2_contingency

tsip_all = pd.read_csv(TSIP_PATH)
dparts = []
for cn, _ in COUNTRIES:
    c = tsip_all[tsip_all.country.str.contains(cn, case=False, na=False)]
    bat = c[c.SiteIndustry.str.contains("Battery", case=False, na=False)]
    nl = c[~c.KeyPollutant.str.contains("Lead", case=False, na=False)]
    e = ev[ev.country == cn].copy()
    # ev lost lat/lon on the round-trip through csv only if columns were dropped; re-read guard
    e["d_bat"] = hav(e.latitude.values[:, None], e.longitude.values[:, None],
                     bat.latitude.values[None, :], bat.longitude.values[None, :]).min(1)
    e["d_nl"] = hav(e.latitude.values[:, None], e.longitude.values[:, None],
                    nl.latitude.values[None, :], nl.longitude.values[None, :]).min(1)
    dparts.append(e)
ED = pd.concat(dparts, ignore_index=True)

bn = ED[ED.d_bat <= 5]                          # near a battery site
nn = ED[(ED.d_nl <= 5) & (ED.d_bat > 5)]        # near a non-lead site only
print("\n" + "="*76)
print("RIOT TRIGGER COMPOSITION: within 5km of a battery site vs a non-lead site")
print("="*76)
print(f"  {'trigger':16s} {'battery':>9s} {'non-lead':>10s} {'diff':>9s}")
CATS_C = [c for c, _ in CATEGORIES] + ["other"]
for c in CATS_C:
    sb, sn = 100*(bn.trigger == c).mean(), 100*(nn.trigger == c).mean()
    print(f"  {c:16s} {sb:8.1f}% {sn:9.1f}% {sb-sn:+8.1f}pp")
print(f"  {'N riots':16s} {len(bn):9,} {len(nn):10,}")

npb, npn = int((bn.trigger == "pollution").sum()), int((nn.trigger == "pollution").sum())
_, p_poll = fisher_exact([[npb, len(bn)-npb], [npn, len(nn)-npn]])
sub = ED[(ED.d_bat <= 5) | ((ED.d_nl <= 5) & (ED.d_bat > 5))]
_, p_comp, _, _ = chi2_contingency(pd.crosstab(sub.d_bat <= 5, sub.trigger))
print(f"\n  pollution share: battery {100*npb/len(bn):.2f}% vs non-lead {100*npn/len(nn):.2f}%"
      f"  (Fisher p={p_poll:.3f})")
print(f"  chi2 equal-composition test across all categories: p={p_comp:.4f}")
print(f"  UPPER BOUND on grievance: {npb} of {len(bn):,} near-site riots ({100*npb/len(bn):.2f}%) "
      f"are pollution-triggered.")

V += [r"\newcommand{\trigPollNearShare}{" + f"{100*npb/len(bn):.2f}" + "}",
      r"\newcommand{\trigPollNlShare}{" + f"{100*npn/len(nn):.2f}" + "}",
      r"\newcommand{\trigChiP}{" + ("$<$0.001" if p_comp < 0.001 else f"{p_comp:.3f}") + "}",
      r"\newcommand{\trigPolShareDiff}{"
      + f"{100*((bn.trigger=='political').mean() - (nn.trigger=='political').mean()):.0f}" + "}"]

# ── Theft of batteries or scrap metal ────────────────────────────────────────
# A battery recycling site creates something worth stealing. If theft of batteries or
# scrap drove the mob-justice result, riot notes would say so. Count riots whose notes
# mention both a theft word and a battery/scrap/lead-acid object. Keyword matching can
# miss thefts described in other words, so this is a count of explicit mentions only.
THEFT = r"\b(?:stole|stolen|steal\w*|theft|thie\w*)\b"
OBJ = r"\b(?:batter(?:y|ies)|scrap|lead[- ]acid|lead plates?)\b"
nt_ = ED.notes.fillna("")
is_theft = nt_.str.contains(THEFT, case=False) & nt_.str.contains(OBJ, case=False)
print(f"\n  battery/scrap theft mentioned: {int(is_theft.sum())} of {len(ED):,} riots; "
      f"{int((is_theft & (ED.d_bat <= 5)).sum())} within 5 km of a battery site")
for _, r in ED[is_theft].iterrows():
    print(f"    [{r.country} {r.year}, {r.d_bat:.1f} km] {str(r.notes)[:150]}")
V += [r"\newcommand{\trigTheftN}{" + f"{int(is_theft.sum())}" + "}",
      r"\newcommand{\trigTheftNearN}{" + f"{int((is_theft & (ED.d_bat <= 5)).sum())}" + "}"]

# ── Mob-justice gap by country (the cleanest discriminating category) ─────────
print("\n=== mob-justice lead-specific gap by country ===")
mj = {}
for cn in ["Kenya", "Ghana"]:          # the text quotes these two
    p = PAN[cn].copy()
    p["post"] = (p.year >= POST_YR).astype(int)
    p["cell_id_str"] = p.cell_id.astype(str)
    p["cty_year"] = p.year.astype(str)
    add_block(p, cn)
    g = (ev[(ev.country == cn) & (ev.trigger == "mob_justice")]
         .groupby(["cell_id", "year"]).size().rename("n_mj").reset_index())
    p = p.merge(g, on=["cell_id", "year"], how="left")
    p["n_mj"] = p.n_mj.fillna(0)
    npos = int((p.n_mj > 0).sum())
    if npos < 30:
        raise SystemExit(f"36: {cn} has {npos} cell-years with a mob-justice riot; too few to estimate")
    gap, se, pv = lpm_gap(p, "n_mj")
    mj[cn] = (gap, se, pv)
    print(f"  {cn:9s} n_pos={npos:4d}  gap={gap:.3f}{stars(pv)} ({se:.3f})")

for cn, code in [("Kenya", "Ke"), ("Ghana", "Gh")]:
    V.append(r"\newcommand{\trigMobGap" + code + "}{" + f"{mj[cn][0]:.3f}" + "}")

# ── Same test on PROTESTS (the paper's Panel B outcome) ──────────────────────
# Environmental grievance plausibly surfaces as protest rather than riot, so the
# riot-only test above would miss it. Senegalese lead plants (Sebikotane,
# Ndiakhate) have drawn documented community mobilisation, which makes this the
# obvious place for a grievance channel to appear if one exists.
def load_protests(cn, path):
    if path and os.path.exists(path):
        d = pd.read_csv(path, low_memory=False)
    else:
        d = pd.read_csv(ACLED_GLOBAL, low_memory=False); d = d[d.country == cn]
    d = d.dropna(subset=["latitude", "longitude", "year"])
    d = d[d.event_type == "Protests"].copy()
    c = tsip_all[tsip_all.country.str.contains(cn, case=False, na=False)]
    bat = c[c.SiteIndustry.str.contains("Battery", case=False, na=False)]
    nl = c[~c.KeyPollutant.str.contains("Lead", case=False, na=False)]
    d["d_bat"] = hav(d.latitude.values[:, None], d.longitude.values[:, None],
                     bat.latitude.values[None, :], bat.longitude.values[None, :]).min(1)
    d["d_nl"] = hav(d.latitude.values[:, None], d.longitude.values[:, None],
                    nl.latitude.values[None, :], nl.longitude.values[None, :]).min(1)
    d["country"] = cn
    return d[["country", "year", "notes", "d_bat", "d_nl"]]

PR = pd.concat([load_protests(cn, p) for cn, p in COUNTRIES], ignore_index=True)
PR["trigger"] = PR.notes.apply(classify)
pb = PR[PR.d_bat <= 5]
pn = PR[(PR.d_nl <= 5) & (PR.d_bat > 5)]
qb, qn = int((pb.trigger == "pollution").sum()), int((pn.trigger == "pollution").sum())
_, p_pr = fisher_exact([[qb, len(pb)-qb], [qn, len(pn)-qn]])
print("\n=== same test on PROTESTS ===")
print(f"  total protests: {len(PR):,}; pollution-triggered: "
      f"{int((PR.trigger=='pollution').sum())} ({100*(PR.trigger=='pollution').mean():.1f}%)")
print(f"  pollution share near battery {100*qb/len(pb):.2f}% vs near non-lead "
      f"{100*qn/len(pn):.2f}%  (Fisher p={p_pr:.3f})")

V += [r"\newcommand{\trigNProtests}{" + f"{len(PR):,}" + "}",
      r"\newcommand{\trigPollProtShare}{" + f"{100*(PR.trigger=='pollution').mean():.1f}" + "}",
      r"\newcommand{\trigPollProtNear}{" + f"{100*qb/len(pb):.2f}" + "}",
      r"\newcommand{\trigPollProtNl}{" + f"{100*qn/len(pn):.2f}" + "}",
      r"\newcommand{\trigPollProtFisherP}{" + f"{p_pr:.2f}" + "}"]

# ── Composition table (Appendix C): does the MIX of triggers differ near battery sites? ──
_near_cats = [c for c in ORDER if c != "all"]
_pc = "$<$0.001" if p_comp < 0.001 else f"{p_comp:.3f}"
t = [r"\begin{table}[htbp]",
     r"\caption{Riot Trigger Composition Near Battery and Non-Lead Sites}", r"\label{tab:trigger_comp}",
     r"\begin{threeparttable}", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
     r"Riot trigger & Battery (\%) & Non-lead (\%) & Diff.\ (pp) \\", r"\midrule"]
for c in _near_cats:
    sb, sn_ = 100 * (bn.trigger == c).mean(), 100 * (nn.trigger == c).mean()
    t.append(f"{LBL[c]} & {sb:.2f} & {sn_:.2f} & " + f"{sb - sn_:+.2f}".replace("-", "$-$") + " \\\\")
t += [r"\midrule",
      f"Riots & {len(bn):,} & {len(nn):,} & \\\\",
      f"Fisher exact $p$, pollution share & \\multicolumn{{3}}{{c}}{{{p_poll:.2f}}} \\\\",
      f"$\\chi^2$ $p$, all triggers & \\multicolumn{{3}}{{c}}{{{_pc}}} \\\\",
      r"\midrule",
      r"\textit{Protests} & & & \\",
      f"Pollution-triggered share & {100*qb/len(pb):.2f} & {100*qn/len(pn):.2f} & {100*(qb/len(pb)-qn/len(pn)):+.2f} \\\\",
      f"Protests & {len(pb):,} & {len(pn):,} & \\\\",
      f"Fisher exact $p$, pollution share & \\multicolumn{{3}}{{c}}{{{p_pr:.2f}}} \\\\",
      r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_trigger_comp}{...}).
      r"\item \TabNote{tab_trigger_comp}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_trigger_comp.tex").write_text("\n".join(t), encoding="utf-8")
save_to_overleaf("tab_trigger_comp.tex", str(TABLES / "tab_trigger_comp.tex"), subfolder="tables")
print("Saved: tab_trigger_comp.tex")


# ── Trigger-specific gaps in the comparison-group robustness samples ─────────
# Same lead-specific gap by trigger as tab_trigger, re-estimated in the two
# comparison-group checks of tab_comparison (57_comparison_table.py):
#   Non-mining: non-lead indicator recomputed without artisanal mining and
#               ore-processing sites (SiteIndustry matching "Min|Ore|Gold"), as 57 row 2.
#   Urban:      cells with above-median pre-2007 log population, as 57 row 3.
MINING_PAT = "Min|Ore|Gold"                                 # as 33 and 57
_ib = tsip_all.SiteIndustry.astype(str).str.contains("Battery", case=False, na=False)
_il = tsip_all.KeyPollutant.astype(str).str.contains("Lead", case=False, na=False)
NLS = tsip_all[~_ib & ~_il]
NLS_NM = NLS[~NLS.SiteIndustry.astype(str).str.contains(MINING_PAT, case=False, na=False)]
_cells = POOLED[["country", "cell_id", "cell_lat", "cell_lon"]].drop_duplicates(["country", "cell_id"])
_nm = []
for cn, _ in COUNTRIES:
    cc = _cells[_cells.country == cn]
    ss = NLS_NM[NLS_NM.country == cn]
    dmin = hav(cc.cell_lat.values[:, None], cc.cell_lon.values[:, None],
               ss.latitude.values[None, :], ss.longitude.values[None, :]).min(1)
    _nm.append(cc.assign(nl5_nm=(1 / (dmin + 1) > 1.0 / 6.0).astype(int)))
POOLED = POOLED.merge(pd.concat(_nm)[["country", "cell_id", "nl5_nm"]],
                      on=["country", "cell_id"], how="left")
assert POOLED.nl5_nm.notna().all()
_uc = pd.read_parquet(PANELS / "urban_controls.parquet")      # 19_urban_controls.py
_bp = (_uc[_uc.year < POST_YR].groupby("cell_id")["log_pop_cell"].mean()
       .rename("base_pop").reset_index())
POOLED = POOLED.merge(_bp, on="cell_id", how="left")
POOLED["base_pop"] = POOLED["base_pop"].fillna(POOLED["base_pop"].median())   # as 57
_med = POOLED.groupby("cell_id_str")["base_pop"].first().median()
POOLED["urban"] = (POOLED["base_pop"] > _med).astype(int)

ROB_CATS = [c for c in ORDER if c in res]
ROB_COLS = [("Base", "Baseline", lambda c: lpm_gap(POOLED, c)),
            ("NonMin", "Non-mining comparison", lambda c: lpm_gap(POOLED, c, nl5="nl5_nm")),
            ("Urban", "Urban cells", lambda c: lpm_gap(POOLED[POOLED.urban == 1], c))]
ROB = {}
print("\n=== lead-specific gap by trigger: baseline / non-mining / urban ===")
for c in ROB_CATS:
    ycol = "n_riots" if c == "all" else f"n_{c}"
    for key, _, f in ROB_COLS:
        sub = POOLED[POOLED.urban == 1] if key == "Urban" else POOLED
        ROB[(c, key)] = f(ycol) if int((sub[ycol] > 0).sum()) >= 30 else None
    print(f"  {c:12s} " + "  ".join(
        (f"{ROB[(c, k)][0]:.3f}{stars(ROB[(c, k)][2]):<3s}({ROB[(c, k)][1]:.3f})" if ROB[(c, k)] else "   ---   ")
        for k, _, _ in ROB_COLS))

t = [r"\begin{table}[htbp]",
     r"\caption{Lead-Specific Gap by Riot Trigger: Comparison-Group Robustness}",
     r"\label{tab:trigger_robust}", r"\begin{threeparttable}", r"\small",
     r"\begin{tabular}{lccc}", r"\toprule",
     r"Riot trigger & " + " & ".join(lab for _, lab, _ in ROB_COLS) + r" \\", r"\midrule"]
for c in ROB_CATS:
    if c == "all":
        t.append(r"\midrule")
    rr = [ROB[(c, k)] for k, _, _ in ROB_COLS]
    t.append(f"{LBL[c]} & " + " & ".join(f"{r_[0]:.3f}{stars(r_[2])}" if r_ else "---" for r_ in rr) + r" \\")
    t.append(" & " + " & ".join(f"({r_[1]:.3f})" if r_ else "" for r_ in rr) + r" \\[2pt]")
t += [r"\midrule",
      "Cells & " + " & ".join(f"{POOLED[POOLED.urban == 1].cell_id_str.nunique() if k == 'Urban' else POOLED.cell_id_str.nunique():,}"
                              for k, _, _ in ROB_COLS) + r" \\",
      r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_trigger_robust}{...}).
      r"\item \TabNote{tab_trigger_robust}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_trigger_robust.tex").write_text("\n".join(t), encoding="utf-8")
save_to_overleaf("tab_trigger_robust.tex", str(TABLES / "tab_trigger_robust.tex"), subfolder="tables")

(TABLES / "trigger_values.tex").write_text("\n".join(V) + "\n", encoding="utf-8")
save_to_overleaf("trigger_values.tex", str(TABLES / "trigger_values.tex"), subfolder="tables")
print("Saved: tab_trigger_robust.tex, trigger_values.tex")
