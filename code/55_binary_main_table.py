"""
55_binary_main_table.py  --  Main results table on the binary treatment, by country.

Treatment: B_c = 1[within 5 km of a battery recycling site] x Post (2007 on), and
N_c = 1[within 5 km of a non-lead TSIP site] x Post, both in one regression:
    1[any event]_ct ~ B_c x Post + N_c x Post | cell FE + year FE   (each country)
                                               | cell FE + country-year FE (pooled)
Lead-specific gap = coefficient on B_c x Post minus coefficient on N_c x Post, with its
standard error from the full variance matrix. SEs are clustered on 1-degree spatial
blocks (utils.add_block); p-values and stars use t(G-1). Each gap also gets a wild
cluster restricted bootstrap p-value (Rademacher, 9,999 draws, same blocks). Outcomes: riots, protests,
organised violence (battles + violence against civilians).

Outputs: output/tables/tab_main_binary.tex, output/tables/binary_main_values.tex
(both copied to Overleaf).
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as st

from config import PANELS, TABLES, POST_YR, save_to_overleaf
from utils import add_block, coef_sep, lincom, wild_p, stars

NEAR5 = 1.0 / 6.0           # 1/(d+1) > 1/6  <=>  d < 5 km
COUNTRIES = [("Kenya", "Ke"), ("Ghana", "Gh"), ("Senegal", "Se")]
OUTCOMES = [("n_riots", "Riot", "Panel A: Riots"), ("n_protests", "Protest", "Panel B: Protests"),
            ("n_org_violence", "Org", "Panel C: Organised violence (battles $+$ VAC)")]

PAN = {}
for cn, _ in COUNTRIES:
    p = pd.read_parquet(PANELS / f"panel_{cn.lower()}.parquet")
    p["post"] = (p.year >= POST_YR).astype(int)
    p["bt"] = (p.inv_dist_battery > NEAR5).astype(int) * p.post
    p["nt"] = (p.inv_dist_nonlead > NEAR5).astype(int) * p.post
    p["cell_id_str"] = cn[:2] + "_" + p.cell_id.astype(str)
    p["cty_year"] = cn + "_" + p.year.astype(str)
    add_block(p, cn)
    PAN[cn] = p
PAN["Pooled"] = pd.concat(PAN.values(), ignore_index=True)


def fit(d, ycol, pooled):
    d = d.copy()
    d["y"] = (d[ycol] > 0).astype(float)
    cell = "cell_id_str" if pooled else "cell_id"
    fe = "cell_id_str + cty_year" if pooled else "cell_id + year"
    m = pf.feols(f"y ~ bt + nt | {fe}", data=d, vcov={"CRV1": "block"})
    out = {"Bat": coef_sep(m, "bt"), "Nl": coef_sep(m, "nt"),
           "Gap": lincom(m, {"bt": 1, "nt": -1})}                 # (coef, SE, p), block-clustered
    out["GapWildP"] = wild_p(d, "y", ["bt", "nt"], {"bt": 1, "nt": -1}, [cell, fe.split(" + ")[1]])
    out["G"] = int(d["block"].nunique())
    pre = d[(d.post == 0) & (d.inv_dist_battery > NEAR5)]
    out["pre"] = float((pre[ycol] > 0).mean())
    return out




def fit_dddd(d, pooled, dist="rademacher"):
    """Riot gap minus organised-violence gap, from one stacked regression with
    cell x type and (country-)year x type fixed effects, block-clustered."""
    cell = "cell_id_str" if pooled else "cell_id"
    yr = "cty_year" if pooled else "year"
    parts = []
    for typ, ycol in [("riot", "n_riots"), ("org", "n_org_violence")]:
        q = d[[cell, yr, "block", "bt", "nt"]].copy()
        q["y"] = (d[ycol] > 0).astype(float)
        q["riot"] = int(typ == "riot")
        q["cell_type"] = q[cell].astype(str) + "_" + typ
        q["yr_type"] = q[yr].astype(str) + "_" + typ
        parts.append(q)
    st_ = pd.concat(parts, ignore_index=True)
    st_["bt_riot"] = st_.bt * st_.riot
    st_["nt_riot"] = st_.nt * st_.riot
    X = ["bt", "nt", "bt_riot", "nt_riot"]
    m = pf.feols("y ~ " + " + ".join(X) + " | cell_type + yr_type", data=st_, vcov={"CRV1": "block"})
    w = {"bt_riot": 1, "nt_riot": -1}
    return lincom(m, w), wild_p(st_, "y", X, w, ["cell_type", "yr_type"], dist=dist)


R, M = {}, {}
for ycol, key, _ in OUTCOMES:
    for cn in ["Kenya", "Ghana", "Senegal", "Pooled"]:
        r = fit(PAN[cn], ycol, cn == "Pooled")
        R[(key, cn)] = r
        tag = dict(Kenya="Ke", Ghana="Gh", Senegal="Se", Pooled="Pooled")[cn]
        for k in ("Bat", "Nl", "Gap"):
            c, s, p = r[k]
            M[f"bm{key}{tag}{k}"] = f"{c:.3f}"
            M[f"bm{key}{tag}{k}SE"] = f"{s:.3f}"
            # Gaps have a wild bootstrap p (shown in brackets); their stars use it
            M[f"bm{key}{tag}{k}Stars"] = stars(r["GapWildP"] if k == "Gap" else p)
        M[f"bm{key}{tag}GapWildP"] = f"{r['GapWildP']:.3f}"
        print(f"{key:8s} {cn:8s} bat {r['Bat'][0]:.3f} ({r['Bat'][1]:.3f})  nl {r['Nl'][0]:.3f} "
              f"({r['Nl'][1]:.3f})  gap {r['Gap'][0]:.3f} ({r['Gap'][1]:.3f}) p={r['Gap'][2]:.3f} "
              f"wild p={r['GapWildP']:.3f}  blocks={r['G']}  pre {r['pre']:.3f}")

# Panel D: riot gap minus organised-violence gap, by country
D4 = {}
for cn in ["Kenya", "Ghana", "Senegal", "Pooled"]:
    (c, s_, p), wp = fit_dddd(PAN[cn], cn == "Pooled")
    D4[cn] = (c, s_, p, wp)
    tag = dict(Kenya="Ke", Ghana="Gh", Senegal="Se", Pooled="Pooled")[cn]
    M[f"bmDDDD{tag}"] = f"{c:.3f}"; M[f"bmDDDD{tag}SE"] = f"{s_:.3f}"
    M[f"bmDDDD{tag}Stars"] = stars(wp); M[f"bmDDDD{tag}WildP"] = f"{wp:.3f}"
    print(f"DDDD     {cn:8s} {c:.3f} ({s_:.3f}) p={p:.3f} wild p={wp:.3f}")


# Webb six-point weights as a check on the Rademacher wild p (printed only; the table
# reports Rademacher). Few treated clusters make Rademacher p coarse.
print("\nWild bootstrap p, Rademacher vs Webb (riot gap; riots minus organised violence):")
for cn in ["Kenya", "Ghana", "Senegal", "Pooled"]:
    d = PAN[cn].copy()
    d["y"] = (d["n_riots"] > 0).astype(float)
    cell = "cell_id_str" if cn == "Pooled" else "cell_id"
    yr = "cty_year" if cn == "Pooled" else "year"
    wp_webb = wild_p(d, "y", ["bt", "nt"], {"bt": 1, "nt": -1}, [cell, yr], dist="webb")
    (_, _, _), wp4_webb = fit_dddd(PAN[cn], cn == "Pooled", dist="webb")
    print(f"  {cn:8s} gap: Rademacher {R[('Riot', cn)]['GapWildP']:.4f}  Webb {wp_webb:.4f}   "
          f"DDDD: Rademacher {D4[cn][3]:.4f}  Webb {wp4_webb:.4f}   blocks {R[('Riot', cn)]['G']}")


def cellfmt(c, s, p):
    return f"{c:.3f}{stars(p)}", f"({s:.3f})"

cols = ["Kenya", "Ghana", "Senegal", "Pooled"]
L = [r"\begin{landscape}", r"\begin{table}[p]",
     r"\caption{Main results: riots, protests and organised violence within 5\,km of a battery site}",
     r"\label{tab:main}", r"\begin{threeparttable}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{0.92}",
     r"\begin{tabular}{lcccccccc}", r"\toprule",
     r" & \multicolumn{2}{c}{Kenya} & \multicolumn{2}{c}{Ghana} & \multicolumn{2}{c}{Senegal} & \multicolumn{2}{c}{Pooled} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r" & Battery & Non-lead & Battery & Non-lead & Battery & Non-lead & Battery & Non-lead \\", r"\midrule"]
for _, key, title in OUTCOMES:
    L.append(r"\textit{" + title + r"} \\")
    a = [cellfmt(*R[(key, c)][k]) for c in cols for k in ("Bat", "Nl")]
    L.append(r"\quad $\mathbf{1}[<5\,\text{km}] \times \text{Post}$ & " + " & ".join(x[0] for x in a) + r" \\")
    L.append(" & " + " & ".join(x[1] for x in a) + r" \\")
    # The gap spans both columns of its country; the s.e. and the wild bootstrap p share a line.
    g = [cellfmt(*R[(key, c)]["Gap"][:2], R[(key, c)]["GapWildP"]) for c in cols]
    L.append(r"\quad Lead-specific gap & " + " & ".join(r"\multicolumn{2}{c}{" + x[0] + "}" for x in g) + r" \\")
    L.append(" & " + " & ".join(r"\multicolumn{2}{c}{" + f"{x[1]} [{R[(key, c)]['GapWildP']:.3f}]" + "}"
                                 for x, c in zip(g, cols)) + r" \\")
    L.append(r"\quad Pre-2007 mean, battery cells & " + " & ".join(
        r"\multicolumn{2}{c}{" + f"{R[(key, c)]['pre']:.3f}" + "}" for c in cols) + r" \\")
L.append(r"\textit{Panel D: Riots minus organised violence} \\")
d4 = [cellfmt(D4[c][0], D4[c][1], D4[c][3]) for c in cols]
L.append(r"\quad Riot gap $-$ organised-violence gap & " + " & ".join(r"\multicolumn{2}{c}{" + x[0] + "}" for x in d4) + r" \\")
L.append(" & " + " & ".join(r"\multicolumn{2}{c}{" + f"{x[1]} [{D4[c][3]:.3f}]" + "}" for x, c in zip(d4, cols)) + r" \\")
n_cells = {c: PAN[c].drop_duplicates("cell_id_str").shape[0] for c in cols}
n_treat = {c: PAN[c].loc[PAN[c].inv_dist_battery > NEAR5, "cell_id_str"].nunique() for c in cols}
n_cy = {c: len(PAN[c]) for c in cols}
# Pooled cell counts for the text: cells within 5 km of a non-lead site, and of both kinds
_cells = PAN["Pooled"].drop_duplicates("cell_id_str")
_b5, _n5 = _cells.inv_dist_battery > NEAR5, _cells.inv_dist_nonlead > NEAR5
M["nNlCells"] = f"{int(_n5.sum())}"
M["nBothCells"] = f"{int((_b5 & _n5).sum())}"
print(f"pooled cells within 5 km: battery {int(_b5.sum())}, non-lead {int(_n5.sum())}, both {int((_b5 & _n5).sum())}")
L += [r"\midrule",
      "Cells & " + " & ".join(r"\multicolumn{2}{c}{" + f"{n_cells[c]:,}" + "}" for c in cols) + r" \\",
      r"\quad of which within 5\,km of a battery site & " + " & ".join(r"\multicolumn{2}{c}{" + f"{n_treat[c]}" + "}" for c in cols) + r" \\",
      "Cell-years & " + " & ".join(r"\multicolumn{2}{c}{" + f"{n_cy[c]:,}" + "}" for c in cols) + r" \\",
      r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[flushleft]", r"\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_main_binary}{...}).
      r"\item \TabNote{tab_main_binary}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", r"\end{landscape}"]
(TABLES / "tab_main_binary.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
save_to_overleaf("tab_main_binary.tex", str(TABLES / "tab_main_binary.tex"), subfolder="tables")
# Write only the macros the paper and slides quote, plus bmRiotPooledGap, which
# 44_plausibility.py reads. A missing one stops the build.
KEEP = ["bmRiotKeGap", "bmRiotKeGapWildP", "bmRiotGhGap", "bmRiotGhGapWildP",
        "bmRiotSeGap", "bmRiotSeGapWildP", "bmRiotPooledBat", "bmRiotPooledNl", "bmRiotPooledGap",
        "bmProtestPooledGap", "bmProtestPooledGapWildP", "bmOrgPooledGap", "bmOrgPooledGapWildP",
        "bmDDDDKe", "bmDDDDGh", "bmDDDDSe", "bmDDDDPooled", "bmDDDDPooledWildP",
        "nNlCells", "nBothCells"]
missing = [k for k in KEEP if k not in M]
if missing:
    raise SystemExit(f"55: macros not computed: {missing}")
lines = ["% Generated by 55_binary_main_table.py"]
for k, v in ((k, M[k]) for k in KEEP):
    v = v.replace("-", "$-$", 1) if v.startswith("-") else v
    lines.append(r"\newcommand{\%s}{%s}" % (k, v))
(TABLES / "binary_main_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
save_to_overleaf("binary_main_values.tex", str(TABLES / "binary_main_values.tex"), subfolder="tables")
print("saved tab_main_binary.tex, binary_main_values.tex")


# ── Appendix: sub-event types, pooled, same binary design ───────────────────────
SUB = [("Riots", [("n_mob_violence", "Mob violence"), ("n_violent_demo", "Violent demonstration")]),
       ("Protests", [("n_peaceful_protest", "Peaceful protest"), ("n_protest_intervention", "Protest with intervention")]),
       ("Organised violence", [("n_battles", "Battles"), ("n_vac_attack", "VAC: attack"),
                               ("n_vac_abduction", "VAC: abduction")]),
       ("Fatalities", [("fat_riots", "Any riot death")])]
PP = PAN["Pooled"].copy()
T = [r"\begin{table}[H]",
     r"\caption{Riot, protest and organised-violence sub-types within 5\,km of a battery site}",
     r"\label{tab:sub_events}", r"\begin{threeparttable}", r"\footnotesize",
     r"\begin{tabular}{lcccc}", r"\toprule",
     r" & Battery & Non-lead & Lead-specific gap & Pre-2007 mean \\", r"\midrule"]
for gi, (grp, rows) in enumerate(SUB):
    T.append(r"\textit{" + grp + r"} \\[2pt]")
    for ri, (col, lab) in enumerate(rows):
        r = fit(PP, col, True)
        b_, n_, g_ = (cellfmt(*r[k]) for k in ("Bat", "Nl", "Gap"))
        end = r" \\[4pt]" if (ri == len(rows) - 1 and gi < len(SUB) - 1) else r" \\"
        T.append(f"\\quad {lab} & {b_[0]} & {n_[0]} & {g_[0]} & {r['pre']:.3f} \\\\")
        T.append(f" & {b_[1]} & {n_[1]} & {g_[1]} & {end}")
        print(f"sub {lab:26s} bat {r['Bat'][0]:.3f} nl {r['Nl'][0]:.3f} gap {r['Gap'][0]:.3f} ({r['Gap'][1]:.3f})")
T += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}", r"\setstretch{1}\footnotesize",
      # Table note lives in main.tex (\setTabNote{tab_sub_events}{...}).
      r"\item \TabNote{tab_sub_events}",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
(TABLES / "tab_sub_events.tex").write_text("\n".join(T) + "\n", encoding="utf-8")
save_to_overleaf("tab_sub_events.tex", str(TABLES / "tab_sub_events.tex"), subfolder="tables")
print("saved tab_sub_events.tex")
