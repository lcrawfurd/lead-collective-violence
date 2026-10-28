"""
29_extensive_margin_main.py
Build the EXTENSIVE-MARGIN main results table (new primary specification).

Outcome: 1[any event in cell-year] (linear probability model). Units-invariant,
full-sample, directly models the 0->1 transition that is the mechanism (cells
near battery sites becoming riot-prone). Avoids the Chen & Roth (2024) units-
dependence of log(events+1), and the FE-Poisson separation/instability shown in
28_poisson_diagnostics.py (FE Poisson drops 63-93% of always-zero cells).

For each outcome (riots, protests, organised violence) x {Kenya, Ghana, Senegal,
Pooled}:
  - Battery single-treatment LPM
  - Non-lead single-treatment LPM
  - Lead-specific GAP (battery - non-lead) from a joint LPM
SEs clustered on 1-degree blocks.

Also the attributable-riot BOTEC: the pooled binary within-5 km four-way difference
(riot gap minus organised-violence gap) x post-2007 cell-years within 5 km.

Outputs:
  output/tables/tab_main_extensive.tex   (+ Overleaf)
  output/panels/botec_extensive.csv      (read by 11_generate_values.py and 47_slides_binary.py)
"""

import pandas as pd
import pyfixest as pf

from config import PANELS, TABLES, POST_YR, save_to_overleaf
from utils import stars, add_block, coef_sep, lincom, tcrit


print("Loading panels...")
PAN = {
    "Kenya":   pd.read_parquet(str(PANELS / "panel_kenya.parquet")),
    "Ghana":   pd.read_parquet(str(PANELS / "panel_ghana.parquet")),
    "Senegal": pd.read_parquet(str(PANELS / "panel_senegal.parquet")),
}
for _c, p in PAN.items():
    p["post"] = (p["year"] >= POST_YR).astype(int)
    add_block(p, _c)          # 1-degree spatial blocks: the cluster for every reported SE

OUTCOMES = [("riots", "n_riots", "Panel A: Riots"),
            ("protests", "n_protests", "Panel B: Protests"),
            ("org", "n_org_violence", "Panel C: Organised violence (battles $+$ VAC)")]


def build_pooled(panels):
    parts = []
    for c, p in panels.items():
        pp = p.copy(); pp["country"] = c
        pp["cell_id_str"] = c[:2] + "_" + pp["cell_id"].astype(str)
        pp["cty_year"] = c + "_" + pp["year"].astype(str)
        parts.append(pp)
    return pd.concat(parts, ignore_index=True)

POOLED = build_pooled(PAN)


def lpm_single(panel, ycount, treat_col, pooled=False):
    """Extensive-margin LPM for one treatment; SE clustered on 1-degree blocks."""
    d = panel.copy()
    d["y"] = (d[ycount] > 0).astype(float)
    d["tp"] = d[treat_col] * d["post"]
    cell_col = "cell_id_str" if pooled else "cell_id"
    fe = "cell_id_str + cty_year" if pooled else "cell_id + year"
    m = pf.feols(f"y ~ tp | {fe}", data=d, vcov={"CRV1": "block"})
    return coef_sep(m, "tp")


def lpm_gap(panel, ycount, pooled=False):
    """Joint LPM: battery and non-lead together; gap = b_bat - b_nl, block-clustered SE."""
    d = panel.copy()
    d["y"] = (d[ycount] > 0).astype(float)
    d["bat_post"] = d["inv_dist_battery"] * d["post"]
    d["nl_post"]  = d["inv_dist_nonlead"] * d["post"]
    cell_col = "cell_id_str" if pooled else "cell_id"
    fe = "cell_id_str + cty_year" if pooled else "cell_id + year"
    m = pf.feols(f"y ~ bat_post + nl_post | {fe}", data=d, vcov={"CRV1": "block"})
    return lincom(m, {"bat_post": 1, "nl_post": -1})


CN = ["Kenya", "Ghana", "Senegal", "Pooled"]
results = {}  # (outcome, country) -> dict

for okey, ycount, _ in OUTCOMES:
    for c in CN:
        pooled = (c == "Pooled")
        panel = POOLED if pooled else PAN[c]
        b, sb, pb = lpm_single(panel, ycount, "inv_dist_battery", pooled)
        nlb, snl, pnl = lpm_single(panel, ycount, "inv_dist_nonlead", pooled)
        g, sg, pg = lpm_gap(panel, ycount, pooled)
        results[(okey, c)] = dict(bat=b, se_bat=sb, p_bat=pb,
                                  nl=nlb, se_nl=snl, p_nl=pnl,
                                  gap=g, se_gap=sg, p_gap=pg)
        print(f"  {okey:8s} {c:8s}: bat={b:.3f}{stars(pb)} nl={nlb:.3f}{stars(pnl)} "
              f"GAP={g:.3f}{stars(pg)}")

# Pre-treatment means (share of pre-period cell-years with any event)
premean = {}
for okey, ycount, _ in OUTCOMES:
    for c in CN:
        panel = POOLED if c == "Pooled" else PAN[c]
        pre = panel[panel.year < POST_YR]
        premean[(okey, c)] = (pre[ycount] > 0).mean()

# Cell counts
cells = {c: (POOLED["cell_id_str"].nunique() if c == "Pooled"
             else PAN[c]["cell_id"].nunique()) for c in CN}
# Treated cells: within 5 km of a battery site, i.e. 1/(d+1) >= 1/6. Reported because
# identification rests on a small number of inner-ring cells and readers should see how few.
NEAR5 = 1.0 / 6.0
treated = {}
cellyears = {}
for c in CN:
    d = POOLED if c == "Pooled" else PAN[c]
    idc = "cell_id_str" if c == "Pooled" else "cell_id"
    treated[c] = int(d.loc[d["inv_dist_battery"] >= NEAR5, idc].nunique())
    cellyears[c] = int(len(d))

# ── LaTeX table ──────────────────────────────────────────────────────────────
def c2(coef, p):
    return "---" if pd.isna(coef) else f"{coef:.3f}{stars(p)}"
def s2(se):
    return "" if pd.isna(se) else f"({se:.3f})"

lines = [
    r"\begin{landscape}",
    r"\begin{table}[p]",
    r"\caption{Continuous Proximity: Riots, Protests and Organised Violence by Inverse Distance to a Battery Site}",
    r"\label{tab:main_continuous}",
    r"\begin{threeparttable}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{lcccccccc}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Kenya} & \multicolumn{2}{c}{Ghana} & \multicolumn{2}{c}{Senegal} & \multicolumn{2}{c}{Pooled} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
    r" & Battery & Non-lead & Battery & Non-lead & Battery & Non-lead & Battery & Non-lead \\",
    r"\midrule",
]
for okey, ycount, panel_title in OUTCOMES:
    lines.append(r"\textit{" + panel_title + r"} \\[2pt]")
    # battery / non-lead row
    cells_bat_nl = []
    for c in CN:
        r = results[(okey, c)]
        cells_bat_nl.append(c2(r["bat"], r["p_bat"]))
        cells_bat_nl.append(c2(r["nl"], r["p_nl"]))
    lines.append(r"\quad $\frac{1}{d+1} \times \text{Post}$ & " + " & ".join(cells_bat_nl) + r" \\")
    se_row = []
    for c in CN:
        r = results[(okey, c)]
        se_row.append(s2(r["se_bat"])); se_row.append(s2(r["se_nl"]))
    lines.append(r" & " + " & ".join(se_row) + r" \\[2pt]")
    # gap row (spans battery col of each country pair; put gap under battery, blank under non-lead)
    gap_cells = []
    for c in CN:
        r = results[(okey, c)]
        gap_cells.append(c2(r["gap"], r["p_gap"]))
        gap_cells.append("")
    lines.append(r"\quad Lead-specific gap & " + " & ".join(gap_cells) + r" \\")
    gse = []
    for c in CN:
        r = results[(okey, c)]
        gse.append(s2(r["se_gap"])); gse.append("")
    lines.append(r" & " + " & ".join(gse) + r" \\[2pt]")
    # pre mean
    pm = [f"\\multicolumn{{2}}{{c}}{{{premean[(okey,c)]:.3f}}}" for c in CN]
    lines.append(r"\quad Pre-treatment mean & " + " & ".join(pm) + r" \\[4pt]")
lines += [
    r"\midrule",
    r"Cells & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cells[c]:,}}}" for c in CN) + r" \\",
    r"\quad of which treated ($<$5\,km) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{treated[c]:,}}}" for c in CN) + r" \\",
    r"Cell-years & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cellyears[c]:,}}}" for c in CN) + r" \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{tablenotes}[flushleft]",
    r"\setstretch{1}\footnotesize",
    # Table note lives in main.tex (\setTabNote{tab_main_extensive}{...}).
    r"\item \TabNote{tab_main_extensive}",
    r"\end{tablenotes}",
    r"\end{threeparttable}",
    r"\end{table}",
    r"\end{landscape}",
]
tex = "\n".join(lines)
path = str(TABLES / "tab_main_extensive.tex")
with open(path, "w") as f:
    f.write(tex)
save_to_overleaf("tab_main_extensive.tex", path, subfolder="tables")
print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# EXTENSIVE-MARGIN BOTEC
# Attributable riot-active cell-years = (four-way difference) x (number of
# post-2007 cell-years within 5 km of a battery site). The four-way difference
# (DDDD) is the headline estimate in Table tab:escalating: the binary within-5 km
# lead-specific gap for riots minus the same gap for organised violence, from one
# stacked pooled regression with cell x type and country-year x type FE and
# SEs clustered on 1-degree blocks. It nets out anything that raises all conflict near battery
# sites. Riot deaths use the same netting: fatal-riot gap minus the
# organised-violence gap. Linear in the gap, so the CI propagates directly.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("EXTENSIVE-MARGIN BOTEC (binary within 5 km, pooled DDDD)")
print("="*64)

POST = 2007
NEAR5 = 1.0 / 6.0          # 1/(d+1) > 1/6  <=>  d < 5 km


def dddd_gap(pooled, ycol):
    """Pooled binary within-5 km lead-specific gap for ycol minus that for organised violence."""
    p = pooled.copy()
    p["bt"] = (p["inv_dist_battery"] > NEAR5) * p["post"]
    p["nt"] = (p["inv_dist_nonlead"] > NEAR5) * p["post"]
    parts = []
    for typ, yc in [("a", ycol), ("o", "n_org_violence")]:
        d = p[["cell_id_str", "block", "cty_year", "bt", "nt"]].copy()
        d["y"] = (p[yc] > 0).astype(float)
        d["typ"] = typ
        parts.append(d)
    st = pd.concat(parts, ignore_index=True)
    r = (st["typ"] == "a").astype(int)
    st["bt_a"], st["nt_a"] = st["bt"] * r, st["nt"] * r
    st["cell_typ"] = st["cell_id_str"] + "_" + st["typ"]
    st["cyt"] = st["cty_year"] + "_" + st["typ"]
    m = pf.feols("y ~ bt + nt + bt_a + nt_a | cell_typ + cyt", data=st,
                 vcov={"CRV1": "block"})       # both types' rows for a cell share its block
    diff, se, _ = lincom(m, {"bt_a": 1, "nt_a": -1})
    return diff, se, tcrit(m)


gap, se, crit = dddd_gap(POOLED, "n_riots")      # crit = t(G-1) 97.5% quantile
gap_fat, se_fat, _ = dddd_gap(POOLED, "fat_riots")
n_treat_post = int(((POOLED["inv_dist_battery"] > NEAR5) & (POOLED["year"] >= POST)).sum())
_post_all = POOLED[POOLED["year"] >= POST]
botec = dict(riot_attr=gap * n_treat_post,
             riot_lo=(gap - crit * se) * n_treat_post,
             riot_hi=(gap + crit * se) * n_treat_post,
             riot_total_cy=int((_post_all["n_riots"] > 0).sum()),
             fat_attr=gap_fat * n_treat_post)
print(f"  riot DDDD={gap:.3f} ({se:.3f}); fatal-riot DDDD={gap_fat:.3f} ({se_fat:.3f}); "
      f"treated post cell-years={n_treat_post}")

n_post_years = int(max(p["year"].max() for p in PAN.values()) - POST + 1)

# Within-5km pre-2007 riot base rate, for interpreting the binary within-5km DD
# magnitude (1/(d+1) > 1/6  <=>  d < 5 km). These dense urban cells have a far
# higher base rate than the panel average, so a large percentage-point effect is
# in range.
_near5 = pd.concat([p[p["inv_dist_battery"] > 1.0/6.0] for p in PAN.values()])
near_base_pre = float((_near5[_near5["year"] < POST]["n_riots"] > 0).mean() * 100)
print(f"  Within-5km pre-{POST} P(any riot) = {near_base_pre:.1f}%")
botec_rows = [{
    "riot_attr": botec["riot_attr"], "riot_lo": botec["riot_lo"], "riot_hi": botec["riot_hi"],
    "riot_per_year": botec["riot_attr"]/n_post_years,
    "riot_share": 100*botec["riot_attr"]/botec["riot_total_cy"],
    "near_base_pre": near_base_pre,
    "fat_attr": botec["fat_attr"],
    "fat_dddd": gap_fat,   # fatal-riot four-way difference
}]
pd.DataFrame(botec_rows).to_csv(str(PANELS / "botec_extensive.csv"), index=False)
print(f"\n  Attributable riot-active cell-years: {botec['riot_attr']:.0f} "
      f"(95% CI {botec['riot_lo']:.0f}-{botec['riot_hi']:.0f})")
print(f"  = {botec['riot_attr']/n_post_years:.0f}/year; "
      f"{100*botec['riot_attr']/botec['riot_total_cy']:.0f}% of {botec['riot_total_cy']} riot-active cell-years")
print(f"  Attributable riot-fatality cell-years: {botec['fat_attr']:.0f}")
print(f"  Saved: {PANELS / 'botec_extensive.csv'}")
