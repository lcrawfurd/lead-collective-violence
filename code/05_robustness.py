"""
05_robustness.py
Robustness to the outcome transformation: the lead-specific gap (battery minus non-lead,
binary 1[within 5 km] x Post design as in 55_binary_main_table.py) under log(y+1), counts,
asinh(y) and the extensive margin 1[y>0], for riots and organised violence.

Inputs:  output/panels/panel_{kenya,ghana,senegal}.parquet (01_build_panels.py)
Outputs: output/tables/tab_alt_estimators.tex    appendix table (+ Overleaf)
         output/tables/tab_slides_alt_rows.tex   riots-only tabular for the slides (+ Overleaf)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings
warnings.filterwarnings("ignore")

from utils import stars, add_block, lincom
from config import TABLES, PANELS, OVERLEAF, POST_YR
PANEL_DIR = str(PANELS)
TABS      = str(TABLES)

# ── Load panels ───────────────────────────────────────────────────────────────

print("Loading panels...")
ke = pd.read_parquet(f"{PANEL_DIR}/panel_kenya.parquet")
gh = pd.read_parquet(f"{PANEL_DIR}/panel_ghana.parquet")
sn = pd.read_parquet(f"{PANEL_DIR}/panel_senegal.parquet")
for _cn, _p in [("Kenya", ke), ("Ghana", gh), ("Senegal", sn)]:
    add_block(_p, _cn)        # 1-degree blocks: the cluster for every reported SE
_ke = ke.copy(); _ke["country"] = "Kenya"
_gh = gh.copy(); _gh["country"] = "Ghana"
_sn = sn.copy(); _sn["country"] = "Senegal"
pooled = pd.concat([_ke, _gh, _sn], ignore_index=True)
pooled["cell_id"] = pooled["country"] + "_" + pooled["cell_id"].astype(str)
pooled["post"] = (pooled["year"] >= 2007).astype(int)
pooled["cty_year"] = pooled["country"] + "_" + pooled["year"].astype(str)

COUNTRIES_BASE = [("Kenya", ke, 2007), ("Ghana", gh, 2007), ("Senegal", sn, 2007)]


# ── tab_alt_estimators.tex: alternative outcome transformations ───────────────
print("\n=== Alternative estimators table ===")

# Main design (as 55_binary_main_table.py): bt = 1[within 5 km of a battery site] x Post,
# nt = the same for non-lead sites, one regression; each cell is the lead-specific gap
# bt - nt under that outcome transformation. Block-clustered SE, t(G-1) stars.
NEAR5 = 1.0 / 6.0

def alt_gap(panel, ycol, pooled):
    p = panel.copy()
    p["bt"] = (p["inv_dist_battery"] > NEAR5).astype(float) * p["post"]
    p["nt"] = (p["inv_dist_nonlead"] > NEAR5).astype(float) * p["post"]
    fe = "cell_id + cty_year" if pooled else "cell_id + year"
    m = pf.feols(f"{ycol} ~ bt + nt | {fe}", data=p, vcov={"CRV1": "block"})
    return lincom(m, {"bt": 1, "nt": -1})

# Outcome pairs: (log column already in panel, raw count column, label)
ALT_OUTCOMES = [
    ("log_riots",        "n_riots",        "Riots"),
    ("log_org_violence", "n_org_violence",  "Organised violence"),
]

alt_results = {}
for cname, panel in [(c, pn) for c, pn, _ in COUNTRIES_BASE] + [("Pooled", pooled)]:
    for log_col, cnt_col, label in ALT_OUTCOMES:
        p = panel.copy()
        p["post"] = (p["year"] >= POST_YR).astype(int)
        p["y_counts"] = p[cnt_col].astype(float)
        p["y_asinh"]  = np.arcsinh(p[cnt_col].astype(float))
        p["y_ext"]    = (p[cnt_col] > 0).astype(float)
        res = tuple(alt_gap(p, y, cname == "Pooled") for y in (log_col, "y_counts", "y_asinh", "y_ext"))
        alt_results[(cname, label)] = res
        print(f"  {cname} {label} gaps: " + "  ".join(
            f"{nm}={r[0]:.3f}{stars(r[2])} ({r[1]:.3f})" for nm, r in zip(("log", "cnt", "asinh", "ext"), res)))

def fmt_alt(r):
    """Format one (coef, se, p) triple as (coef_str, se_str)."""
    c, s, p = r
    sign = "$-$" if c < 0 else ""
    star = stars(p)
    return f"{sign}{abs(c):.3f}{star}", f"({s:.3f})"

def make_alt_table(alt_results):
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Robustness: Alternative Outcome Transformations (Lead-Specific Gap)}",
        r"\label{tab:alt_estimators}",
        r"\begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" & & $\log(y+1)$ & Counts & $\text{asinh}(y)$ & $\mathbf{1}[y>0]$ \\",
        r" & & (OLS) & (OLS) & (OLS) & (main outcome) \\",
        r"\midrule",
    ]
    for label in ["Riots", "Organised violence"]:
        lines.append(r"\textit{" + label + r"} \\[2pt]")
        for cname in ["Kenya", "Ghana", "Senegal", "Pooled"]:
            key = (cname, label)
            if key not in alt_results:
                continue
            r_log, r_cnt, r_asinh, r_ext = alt_results[key]
            c1, s1 = fmt_alt(r_log);   c2, s2 = fmt_alt(r_cnt)
            c3, s3 = fmt_alt(r_asinh); c4, s4 = fmt_alt(r_ext)
            lines.append(f"\\quad {cname} & & {c1} & {c2} & {c3} & {c4} \\\\")
            lines.append(f" & & {s1} & {s2} & {s3} & {s4} \\\\[4pt]")
        if label == "Riots":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\setstretch{1}\footnotesize",
        # Table note lives in main.tex (\setTabNote{tab_alt_estimators}{...}).
        r"\item \TabNote{tab_alt_estimators}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(lines)

alt_tex = make_alt_table(alt_results)

# Slide version (riots only, extensive margin first): the whole tabular for the
# "intensive margin" backup slide, which \inputs this file.
_srow = []
for cname in ["Kenya", "Ghana", "Senegal", "Pooled"]:
    r_log, r_cnt, r_asinh, r_ext = alt_results[(cname, "Riots")]
    cells = [(r[0], r[1], stars(r[2])) for r in (r_ext, r_log, r_asinh, r_cnt)]
    _srow.append(f"    {cname} & " + " & ".join(
        (f"$-${abs(c):.3f}" if c < 0 else f"{c:.3f}") + (f"$^{{{st}}}$" if st else "")
        for c, _, st in cells) + r" \\")
    _srow.append("            & " + " & ".join(f"({se:.3f})" for _, se, _ in cells) + r" \\")
for base in [TABS] + ([f"{OVERLEAF}/tables"] if OVERLEAF else []):
    with open(f"{base}/tab_slides_alt_rows.tex", "w") as f:
        f.write("% Generated by 05_robustness.py -- table for the slides' intensive-margin backup\n"
                + "\n".join([r"  \begin{tabular}{lcccc}", r"    \toprule",
                              r"    Riot gap & $\mathbf{1}[y>0]$ & $\log(y+1)$ & $\text{asinh}(y)$ & Count \\",
                              r"    \midrule"] + _srow + [r"    \bottomrule", r"  \end{tabular}"]) + "\n")
for base in [TABS] + ([f"{OVERLEAF}/tables"] if OVERLEAF else []):
    path = f"{base}/tab_alt_estimators.tex"
    with open(path, "w") as f:
        f.write(alt_tex)
    print(f"Saved: {path}")
