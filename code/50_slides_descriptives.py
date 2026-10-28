"""
50_slides_descriptives.py  --  Slide figures describing comparisons 3 and 4.

(3) Battery sites v. other toxic sites: what the 180 comparison sites are (main
    pollutant, industry), and the pre-2007 conflict rates of cells within 5 km of each.
(4) Riots v. organised violence: ACLED events in the panel cells by sub-event type,
    with the share of events that kill anyone.

Site definitions and event-to-cell assignment are imported from 01_build_panels.py,
so the counts match the analysis panel.

Outputs: output/figures/fig_slides_sites_desc.pdf, fig_slides_violence_desc.pdf,
         output/tables/slides_desc_values.tex (all copied to Overleaf).
"""
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CODE, PANELS, FIGURES, TABLES, TSIP_PATH, ACLED_GLOBAL, POST_YR, save_to_overleaf

_spec = importlib.util.spec_from_file_location("bp", CODE / "01_build_panels.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

NEAR5 = 1.0 / 6.0          # 1/(d+1) > 1/6  <=>  d < 5 km
M = {}

# ── Sites ────────────────────────────────────────────────────────────────────
tsip = pd.read_csv(TSIP_PATH)
names = [c["name"] for c in bp.COUNTRIES]
t = tsip[tsip.country.isin(names)]
is_bat = t.SiteIndustry.str.contains("Battery", case=False, na=False)
nl = t[~t.KeyPollutant.str.contains("Lead", case=False, na=False) & ~is_bat]

PEST = ["Pesticides", "Dieldrin", "Chlordane", "Malathion", "DDT", "Endrin"]
def pollutant(x):
    if x.startswith("Mercury"):
        return "Mercury"
    if x.startswith("Chromium"):
        return "Chromium"
    if x in PEST:
        return "Pesticides"
    if x in ("Arsenic", "Cadmium"):
        return x
    if x.startswith("PCBs"):
        return "PCBs"
    return "Other or unknown"
pol = nl.KeyPollutant.fillna("Unknown").map(pollutant).value_counts()
pol = pol.reindex([k for k in pol.index if k != "Other or unknown"] + ["Other or unknown"])

IND = {"Artisanal Mining (hand mining)": "Artisanal mining",
       "Industrial/Municipal Dump Site": "Dump sites",
       "Mining and Ore Processing": "Mining and ore processing",
       "Agriculture": "Agriculture",
       "Industrial Estate (mixed industries)": "Industrial estates",
       "Multiple Diverse Industries": "Industrial estates"}
ind = nl.SiteIndustry.map(IND).fillna("Other industries").value_counts()
ind = ind.reindex([k for k in ind.index if k != "Other industries"] + ["Other industries"])
for k, v in pol.items():
    print(f"  pollutant {k:18s} {v}")
for k, v in ind.items():
    print(f"  industry  {k:26s} {v}")

# Pre-2007 conflict rates, cells within 5 km of each type of site (as fig_slides_raw_rates)
P = pd.concat([pd.read_parquet(PANELS / f"panel_{c.lower()}.parquet") for c in names],
              ignore_index=True)
pre = P[P.year < POST_YR]
grp = np.select([pre.inv_dist_battery > NEAR5, pre.inv_dist_nonlead > NEAR5],
                ["bat", "nl"], "far")
for g, key in [("bat", "Bat"), ("nl", "Nl")]:
    d = pre[grp == g]
    M[f"sdPreRiot{key}"] = f"{100 * (d.n_riots > 0).mean():.0f}"
    M[f"sdPreOrg{key}"] = f"{100 * (d.n_org_violence > 0).mean():.0f}"
    print(f"  pre-{POST_YR} {g}: any riot {100*(d.n_riots>0).mean():.1f}%, "
          f"any organised {100*(d.n_org_violence>0).mean():.1f}%")

fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
for ax, s, title in [(a, pol, f"(a) Main pollutant, {len(nl)} other toxic sites"),
                     (b, ind, f"(b) Industry, {len(nl)} other toxic sites")]:
    s = s.iloc[::-1]
    ax.barh(s.index, s.values, color="#C0661A")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9, color="#333333")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_slides_sites_desc.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_slides_sites_desc.pdf", str(out), subfolder="figures")

# ── Violence types ───────────────────────────────────────────────────────────
ev = []
for c in bp.COUNTRIES:
    d = bp.assign_cell(bp.load_acled(c["name"], c["acled_path"], ACLED_GLOBAL,
                                     exclude_years=c.get("exclude_years")))
    cells = set(zip(P.loc[P.cell_id.notna(), "cell_lat"].round(4),
                    P.loc[P.cell_id.notna(), "cell_lon"].round(4)))
    d = d[[(la, lo) in cells for la, lo in zip(d.cell_lat, d.cell_lon)]]
    ev.append(d)
ev = pd.concat(ev, ignore_index=True)
riot = ev[ev.event_type == "Riots"]
org = ev[ev.event_type.isin(["Battles", "Violence against civilians"])]
# Check against the panel totals
assert len(riot) == int(P.n_riots.sum()), (len(riot), P.n_riots.sum())
assert len(org) == int(P.n_org_violence.sum()), (len(org), P.n_org_violence.sum())

rows = []
for lab, d in [("Riots", riot), ("Organised", org)]:
    for st, g in d.groupby("sub_event_type"):
        rows.append((lab, st, len(g), 100 * (g.fatalities > 0).mean()))
tab = pd.DataFrame(rows, columns=["group", "sub", "n", "pct_fatal"])
tab = tab[tab.n >= 20].sort_values(["group", "n"], ascending=[False, True])
print(tab.to_string(index=False))
M["sdNRiots"], M["sdNOrg"] = f"{len(riot):,}", f"{len(org):,}"
M["sdRiotFatalPct"] = f"{100 * (riot.fatalities > 0).mean():.0f}"
M["sdOrgFatalPct"] = f"{100 * (org.fatalities > 0).mean():.0f}"
M["sdYrMin"], M["sdYrMax"] = f"{int(ev.year.min())}", f"{int(ev.year.max())}"

COL = {"Riots": "#B3261E", "Organised": "#555555"}
fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
y = np.arange(len(tab))
labels = tab["sub"].str.replace("Abduction/forced disappearance", "Abduction")
a.barh(y, tab.n, color=[COL[g] for g in tab.group])
for i, v in enumerate(tab.n):
    a.text(v, i, f" {v:,}", va="center", fontsize=9, color="#333333")
a.set_yticks(y)
a.set_yticklabels(labels)
for t_, g in zip(a.get_yticklabels(), tab.group):
    t_.set_color(COL[g])
a.set_title(f"(a) Events, {M['sdYrMin']}-{M['sdYrMax']}", loc="left", fontsize=10)
b.barh(y, tab.pct_fatal, color=[COL[g] for g in tab.group])
for i, v in enumerate(tab.pct_fatal):
    b.text(v, i, f" {v:.0f}%", va="center", fontsize=9, color="#333333")
b.set_title("(b) Share of events with any death", loc="left", fontsize=10)
for ax in (a, b):
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_slides_violence_desc.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_slides_violence_desc.pdf", str(out), subfolder="figures")

lines = ["% Generated by 50_slides_descriptives.py"]
lines += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in M.items()]
(TABLES / "slides_desc_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
save_to_overleaf("slides_desc_values.tex", str(TABLES / "slides_desc_values.tex"), subfolder="tables")
print("\n".join(lines))
