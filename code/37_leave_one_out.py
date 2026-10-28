"""
37_leave_one_out.py

Leave-one-out robustness for the lead-specific riot gap, at three levels:

  COUNTRY        drop each country from the pooled specification (3 estimates)
  NEAREST SITE   drop every cell whose nearest battery site is site j (one per site)
  TREATED CELL   drop each individual cell within 5 km of a battery site (30 cells)

Identification rests on a small number of inner-ring cells -- 30 treated cells across
the three countries, 20 of them in Kenya -- so the natural question is whether any single
country, site, or cell is carrying the result. This answers it directly.

Inference note: every regression in the sweep, and the baseline, uses SEs clustered on
1-degree spatial blocks, as in the main tables.

Outputs (+ Overleaf):
  output/tables/tab_leave_one_out.tex
  output/tables/loo_values.tex
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import TSIP_PATH, PANELS, TABLES, POST_YR, save_to_overleaf
from utils import stars, add_block, lincom

NEAR5 = 1.0 / 6.0          # 1/(d+1) > 1/6  <=>  d < 5 km (as in 55_binary_main_table.py)
COUNTRIES = ["Kenya", "Ghana", "Senegal"]


def hav(lat1, lon1, lat2, lon2):
    R, p = 6371.0, np.pi / 180
    dlat, dlon = (lat2 - lat1) * p, (lon2 - lon1) * p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


print("Building pooled panel with nearest-site labels...")
tsip = pd.read_csv(TSIP_PATH)
parts = []
for cn in COUNTRIES:
    p = pd.read_parquet(str(PANELS / f"panel_{cn.lower()}.parquet")).copy()
    p["post"] = (p.year >= POST_YR).astype(int)
    p["country"] = cn
    p["cell_id_str"] = cn[:2] + "_" + p.cell_id.astype(str)
    add_block(p, cn)       # 1-degree blocks: cluster for every reported SE
    p["cty_year"] = cn + "_" + p.year.astype(str)
    bat = tsip[(tsip.country.str.contains(cn, case=False, na=False)) &
               (tsip.SiteIndustry.str.contains("Battery", case=False, na=False))].reset_index(drop=True)
    D = hav(p.cell_lat.values[:, None], p.cell_lon.values[:, None],
            bat.latitude.values[None, :], bat.longitude.values[None, :])
    p["site_key"] = cn[:2] + "_" + pd.Series(D.argmin(1), index=p.index).astype(str)
    parts.append(p)
POOLED = pd.concat(parts, ignore_index=True)
POOLED["y"] = (POOLED.n_riots > 0).astype(float)
# Main binary design: cell within 5 km of a battery / non-lead site x 2007 onwards
POOLED["bat_post"] = (POOLED.inv_dist_battery > NEAR5).astype(int) * POOLED.post
POOLED["nl_post"] = (POOLED.inv_dist_nonlead > NEAR5).astype(int) * POOLED.post

treated_cells = sorted(POOLED.loc[POOLED.inv_dist_battery > NEAR5, "cell_id_str"].unique())
print(f"  {POOLED.cell_id_str.nunique():,} cells, {len(treated_cells)} treated, "
      f"{POOLED.site_key.nunique()} distinct nearest-sites")


def gap(d, pooled=True):
    """Lead-specific gap for riots; block-clustered SE, t(G-1) p."""
    fe = "cell_id_str + cty_year" if pooled else "cell_id_str + year"
    m = pf.feols(f"y ~ bat_post + nl_post | {fe}", data=d, vcov={"CRV1": "block"})
    g, se, pv = lincom(m, {"bat_post": 1, "nl_post": -1})
    return g, se, pv


base_g, base_se, base_p = gap(POOLED)
print(f"\nBaseline pooled gap (block-clustered SE): {base_g:.3f}{stars(base_p)} ({base_se:.3f})\n")

rows = []

print("Leave-one-country-out:")
for cn in COUNTRIES:
    g, se, pv = gap(POOLED[POOLED.country != cn])
    rows.append(dict(level="country", dropped=cn, gap=g, se=se, p=pv))
    print(f"  drop {cn:9s} gap={g:.3f}{stars(pv):<4s} ({se:.3f})")

print("\nLeave-one-nearest-site-out (cells whose closest battery site is j):")
for k in sorted(POOLED.site_key.unique()):
    sub = POOLED[POOLED.site_key != k]
    if sub.loc[sub.inv_dist_battery > NEAR5, "cell_id_str"].nunique() < 5:
        continue
    g, se, pv = gap(sub)
    rows.append(dict(level="site", dropped=k, gap=g, se=se, p=pv))
n_site = sum(1 for r in rows if r["level"] == "site")
sg = [r["gap"] for r in rows if r["level"] == "site"]
sp = [r["p"] for r in rows if r["level"] == "site"]
print(f"  {n_site} site-drops: gap range {min(sg):.3f} to {max(sg):.3f}; "
      f"significant at 5% in {sum(1 for x in sp if x < 0.05)}/{n_site}")

print("\nLeave-one-treated-cell-out:")
for c in treated_cells:
    g, se, pv = gap(POOLED[POOLED.cell_id_str != c])
    rows.append(dict(level="cell", dropped=c, gap=g, se=se, p=pv))
cg = [r["gap"] for r in rows if r["level"] == "cell"]
cp = [r["p"] for r in rows if r["level"] == "cell"]
print(f"  {len(cg)} cell-drops: gap range {min(cg):.3f} to {max(cg):.3f}; "
      f"significant at 5% in {sum(1 for x in cp if x < 0.05)}/{len(cg)}")

df = pd.DataFrame(rows)

# ── Table ────────────────────────────────────────────────────────────────────
def blk(level, label):
    d = df[df.level == level]
    return (label, len(d), d.gap.min(), d.gap.max(),
            int((d.p < 0.05).sum()), d.loc[d.gap.idxmin(), "dropped"])

t = [r"\begin{table}[htbp]", r"\caption{Leave-One-Out: Lead-Specific Riot Gap}",
     r"\label{tab:loo}", r"\begin{threeparttable}", r"\small",
     r"\begin{tabular}{lccccc}", r"\toprule",
     r"Unit dropped & Estimates & Min gap & Max gap & Sig.\ at 5\% & Most influential \\",
     r"\midrule"]
for lvl, lab in [("country", "Country"), ("site", "Nearest battery site"),
                 ("cell", "Treated cell ($<$5\\,km)")]:
    lb, n, lo, hi, ns, worst = blk(lvl, lab)
    t.append(f"{lb} & {n} & {lo:.3f} & {hi:.3f} & {ns}/{n} & {str(worst).replace('_','\\_')} \\\\")
t += [r"\midrule",
      f"Baseline (nothing dropped) & --- & \\multicolumn{{2}}{{c}}{{{base_g:.3f}{stars(base_p)}}} & "
      f"({base_se:.3f}) & --- \\\\",
      r"\bottomrule", r"\end{tabular}",
      r"\begin{tablenotes}\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_leave_one_out}{...}).
      r"\item \TabNote{tab_leave_one_out}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_leave_one_out.tex").write_text("\n".join(t), encoding="utf-8")
save_to_overleaf("tab_leave_one_out.tex", str(TABLES / "tab_leave_one_out.tex"), subfolder="tables")

allg = df.gap
v = ["% auto-generated by 37_leave_one_out.py",
     r"\newcommand{\looN}{" + f"{len(df)}" + "}",
     r"\newcommand{\looMin}{" + f"{allg.min():.3f}" + "}",
     r"\newcommand{\looMax}{" + f"{allg.max():.3f}" + "}"]
(TABLES / "loo_values.tex").write_text("\n".join(v) + "\n", encoding="utf-8")
save_to_overleaf("loo_values.tex", str(TABLES / "loo_values.tex"), subfolder="tables")
print(f"\nAll {len(df)} drops: gap {allg.min():.3f} to {allg.max():.3f}; "
      f"{100*(df.p<0.05).mean():.0f}% significant at 5%")
print("Saved: tab_leave_one_out.tex, loo_values.tex")
