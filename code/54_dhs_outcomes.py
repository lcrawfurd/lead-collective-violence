"""
54_dhs_outcomes.py  --  DHS balance and outcomes near battery sites v. other toxic sites,
Kenya, Ghana and Senegal, before and after 2007.

Rounds: Kenya 2003 | 2008-09, 2014, 2022; Ghana 2003 | 2008, 2014, 2022;
Senegal 2005 | 2010-11, 2015, 2019. All files in data/DHS.

Each DHS cluster is assigned by distance from its (displaced) GPS point: within 5 km of a
battery recycling site ("battery"), within 5 km of another toxic site only ("other"). The
comparison is battery v. other, as in the paper's lead-specific gap:
    y ~ battery + battery x post | country-region-round FE + urban,  weights = DHS weights,
    SEs clustered by DHS cluster.
post = round after 2007. Outcomes:
  women (IR):     wealth (standardised within round), years of schooling, anaemic (v457 in 1-3; measured rounds only)
  children (KR):  anaemic 6-59 months (hw57 in 1-3), haemoglobin g/dl (hw56), died before
                  age 5 among births in the last 5 years (b5 = 0 and b7 < 60), stunted
                  (hw70 < -2 SD; not in the pre-2007 rounds, so post-only comparison)
  household (PR): attending school, ages 6-14 (hv121 > 0)
Outputs: output/panels/dhs_outcomes.csv (read by 60_balance_table.py); output/tables/dhs_values.tex
(macros used in the paper, copied to Overleaf).
"""
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
import pyfixest as pf

from config import DATA, TABLES, PANELS, TSIP_PATH, save_to_overleaf

DHS = DATA / "DHS"


def one(pattern):
    hits = glob.glob(str(pattern), recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"expected one file for {pattern}, found {hits}")
    return hits[0]


def ke(folder):
    b = glob.glob(str(DHS / folder))[0]
    return dict(ge=one(f"{b}/**/KEGE*.shp"), ir=one(f"{b}/**/KEIR*FL.DTA"),
                kr=one(f"{b}/**/KEKR*FL.DTA"), pr=one(f"{b}/**/KEPR*FL.DTA"))


def local(folder, cc):
    b = DHS / folder
    return dict(ge=one(f"{b}/**/{cc}GE*.shp"), ir=one(f"{b}/**/{cc}IR*FL.DTA"),
                kr=one(f"{b}/**/{cc}KR*FL.DTA"), pr=one(f"{b}/**/{cc}PR*FL.DTA"))


SURVEYS = {
    ("Kenya", 2003): ke("KE_2003_DHS_*"), ("Kenya", 2008): ke("KE_2008-09_DHS_*"),
    ("Kenya", 2014): ke("KE_2014_DHS_*"), ("Kenya", 2022): ke("KE_2022_DHS"),
    ("Senegal", 2005): local("SN_2005", "SN"), ("Senegal", 2010): local("SN_2010-11", "SN"),
    ("Senegal", 2015): local("SN_2015", "SN"), ("Senegal", 2019): local("SN_2019", "SN"),
    ("Ghana", 2003): local("GH_2003", "GH"),
}
for yr in (2008, 2014, 2022):
    SURVEYS[("Ghana", yr)] = local(f"GH_{yr}", "GH")


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def read(f, cols):
    it = pd.read_stata(f, iterator=True, convert_categoricals=False)
    have = [c for c in cols if c in it.variable_labels()]
    return pd.read_stata(f, columns=have, convert_categoricals=False)


tsip = pd.read_csv(TSIP_PATH)
W, K, H = [], [], []
for (cn, yr), f in SURVEYS.items():
    c = tsip[tsip.country == cn]
    is_bat = c.SiteIndustry.str.contains("Battery", case=False, na=False)
    bat, nl = c[is_bat], c[~c.KeyPollutant.str.contains("Lead", case=False, na=False) & ~is_bat]
    g = gpd.read_file(f["ge"])
    g = g[(g.LATNUM != 0) & (g.LONGNUM != 0)]
    db = hav(g.LATNUM.values[:, None], g.LONGNUM.values[:, None],
             bat.latitude.values[None, :], bat.longitude.values[None, :]).min(1)
    dn = hav(g.LATNUM.values[:, None], g.LONGNUM.values[:, None],
             nl.latitude.values[None, :], nl.longitude.values[None, :]).min(1)
    g = g.assign(group=np.select([db < 5, dn < 5], ["battery", "other"], "out"),
                 urban=(g.URBAN_RURA == "U").astype(float))
    g = g[g.group != "out"][["DHSCLUST", "group", "urban"]]
    tag = dict(country=cn, round=yr, post=int(yr > 2007))

    ir = read(f["ir"], ["v001", "v005", "v024", "v133", "v191", "v457"])
    ir["wealth_z"] = (ir.v191 - ir.v191.mean()) / ir.v191.std()
    ir["school_yrs"] = ir.v133.where(ir.v133 < 90)
    ir["anaemic_w"] = ir.v457.isin([1, 2, 3]).astype(float).where(ir.v457.isin([1, 2, 3, 4])) \
        if "v457" in ir else np.nan
    W.append(ir.rename(columns={"v001": "DHSCLUST", "v005": "w", "v024": "region"})
               .merge(g, on="DHSCLUST").assign(**tag))

    kr = read(f["kr"], ["v001", "v005", "v024", "hw1", "hw56", "hw57", "hw70", "b5", "b7"])
    kr["anaemic_c"] = np.where(kr.get("hw1", np.nan) >= 6,
                               kr.hw57.isin([1, 2, 3]).astype(float).where(kr.hw57.isin([1, 2, 3, 4])),
                               np.nan) if "hw57" in kr else np.nan
    kr["hb_c"] = (kr.hw56 / 10).where(kr.hw56 < 250) if "hw56" in kr else np.nan
    kr["died_u5"] = ((kr.b5 == 0) & (kr.b7 < 60)).astype(float)
    kr["stunted"] = (kr.hw70 < -200).astype(float).where(kr.hw70 < 9990) if "hw70" in kr else np.nan
    K.append(kr.rename(columns={"v001": "DHSCLUST", "v005": "w", "v024": "region"})
               .merge(g, on="DHSCLUST").assign(**tag))

    pr = read(f["pr"], ["hv001", "hv005", "hv024", "hv105", "hv121"])
    pr = pr[(pr.hv105 >= 6) & (pr.hv105 <= 14)].copy()
    pr["attends"] = (pr.hv121 > 0).astype(float).where(pr.hv121.isin([0, 1, 2]))
    H.append(pr.rename(columns={"hv001": "DHSCLUST", "hv005": "w", "hv024": "region"})
               .merge(g, on="DHSCLUST").assign(**tag))
    print(f"{cn} {yr}: clusters battery {int((g.group == 'battery').sum())}, "
          f"other {int((g.group == 'other').sum())}")

OUT = [("women", W, ["wealth_z", "school_yrs", "anaemic_w"]),
       ("children", K, ["anaemic_c", "hb_c", "died_u5", "stunted"]),
       ("household", H, ["attends"])]
res = []
for unit, frames, ys in OUT:
    d = pd.concat(frames, ignore_index=True)
    d["w"] = d.w / 1e6
    d["bat"] = (d.group == "battery").astype(int)
    d["bat_post"] = d.bat * d.post
    d["fe"] = d.country + "_" + d["round"].astype(str) + "_" + d.region.astype(str)
    d["clus"] = d.country + "_" + d["round"].astype(str) + "_" + d.DHSCLUST.astype(str)
    for y in ys:
        s = d.dropna(subset=[y])
        rounds = s.groupby(["country", "round"]).size()
        has_pre = (s.post == 0).any()
        for scope, sub in [("all", s)]:          # the paper reports pooled estimates only
            if sub.empty or sub.bat.sum() == 0:
                continue
            pre_ok = (sub.post == 0).any() and (sub.post == 1).any()
            fml = f"{y} ~ bat + bat_post + urban | fe" if pre_ok else f"{y} ~ bat + urban | fe"
            m = pf.feols(fml, data=sub, weights="w", vcov={"CRV1": "clus"})
            term = "bat_post" if pre_ok else "bat"
            mean = lambda g_, p: np.average(sub.loc[(sub.group == g_) & (sub.post == p), y],
                                            weights=sub.loc[(sub.group == g_) & (sub.post == p), "w"]) \
                if ((sub.group == g_) & (sub.post == p)).any() else np.nan
            res.append(dict(unit=unit, outcome=y, scope=scope,
                            estimate="battery x post" if pre_ok else "battery (post only)",
                            coef=float(m.coef()[term]), se=float(m.se()[term]), n=int(m._N),
                            n_bat_clusters=int(sub.loc[sub.bat == 1, "clus"].nunique()),
                            bat_pre=mean("battery", 0), oth_pre=mean("other", 0),
                            bat_post=mean("battery", 1), oth_post=mean("other", 1),
                            rounds=";".join(f"{a[0][:2]}{a[1]}" for a in rounds.index if a[0] == scope or scope == "all")))
R = pd.DataFrame(res)
pd.set_option("display.width", 200)
print(R.drop(columns="rounds").round(3).to_string(index=False))
R.to_csv(PANELS / "dhs_outcomes.csv", index=False)      # read by 60_balance_table.py

# Balance macros for the paper: pooled battery x post, women's recode
A = R[(R.scope == "all") & (R.estimate == "battery x post")].set_index("outcome")
mac = {"dhsWealthDiff": f"{A.loc['wealth_z', 'coef']:.2f}", "dhsWealthSE": f"{A.loc['wealth_z', 'se']:.2f}",
       "dhsSchoolDiff": f"{A.loc['school_yrs', 'coef']:.2f}", "dhsSchoolSE": f"{A.loc['school_yrs', 'se']:.2f}",
       "dhsNWomen": f"{int(A.loc['wealth_z', 'n']):,}"}
Aall = R[R.scope == "all"].set_index("outcome")
for y, nm in [("anaemic_w", "AnaemW"), ("anaemic_c", "AnaemC"), ("hb_c", "Hb"), ("died_u5", "Died"),
              ("stunted", "Stunt"), ("attends", "Attend")]:
    mac[f"dhs{nm}Diff"] = f"{Aall.loc[y, 'coef']:.2f}"
    mac[f"dhs{nm}SE"] = f"{Aall.loc[y, 'se']:.2f}"
    mac[f"dhs{nm}N"] = f"{int(Aall.loc[y, 'n']):,}"
mac = {k: ("0.00" if v == "-0.00" else v) for k, v in mac.items()}
mac = {k: (v.replace("-", "$-$", 1) if v.startswith("-") else v) for k, v in mac.items()}
lines = ["% Generated by 54_dhs_outcomes.py"] + [r"\newcommand{\%s}{%s}" % kv for kv in mac.items()]
(TABLES / "dhs_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
save_to_overleaf("dhs_values.tex", str(TABLES / "dhs_values.tex"), subfolder="tables")
print("\n".join(lines))
print("saved dhs_outcomes.csv, dhs_values.tex")
