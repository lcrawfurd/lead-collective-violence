r"""
56_site_rings.py  --  Effects closer than 5 km: event-level distance rings around each site.

The 0.1 degree grid cannot resolve distances below about 10 km, but ACLED gives each
event its own coordinates, so we can measure the distance from every event to every site
directly. Unit: site x ring x year, for every battery recycling site and every non-lead
TSIP site in Kenya, Ghana and Senegal, 1997-2024. Rings: 0-1, 1-2, 2-3, 3-5, 5-10 and
10-20 km. Outcome: 1[any event of the type in the ring that year]; for riots also events
per 100 km2 of ring, since rings differ in area (3 km2 for 0-1 km, 940 km2 for 10-20 km)
and the any-event outcome rises with area. An event counts for every site whose ring it
falls in.

    y_srt = a_sr + d_{ring x country x year} + sum_r b_r 1[ring r] x Battery_s x Post_t

b_r is the lead-specific gap in ring r: battery sites v. non-lead sites at the same
distance, after 2007 v. before. The 10-20 km ring is a far comparison. SEs are clustered
by site group (sites within 10 km of each other are chained into one group), because
nearby sites share events.

Riots only. Main sample: ACLED geo_precision 1 (the event's own town or neighbourhood).
Check: dropping the five most-used coordinates in each country (city-centre points where
ACLED puts events it cannot place more precisely).

Outputs: output/figures/fig_site_rings.pdf and
output/tables/site_rings_values.tex (\sr* macros), both copied to Overleaf; and
output/figures/fig_slides_site_rings.pdf (density panel only, large type, for the slides).
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGURES, TABLES, TSIP_PATH, ACLED_GLOBAL, ACLED_KE, ACLED_GH, POST_YR, END_YR, save_to_overleaf

RINGS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20)]
YEARS = range(1997, END_YR + 1)
COUNTRIES = {"Kenya": ACLED_KE, "Ghana": ACLED_GH, "Senegal": None}
AREA = np.array([np.pi * (hi ** 2 - lo ** 2) for lo, hi in RINGS])
OUTCOMES = [("Riots", ["Riots"])]


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


tsip = pd.read_csv(TSIP_PATH)
cols = ["year", "event_type", "latitude", "longitude", "geo_precision", "country"]
S, E = [], []
for cn, path in COUNTRIES.items():
    c = tsip[tsip.country.str.contains(cn, case=False, na=False)]
    is_bat = c.SiteIndustry.str.contains("Battery", case=False, na=False)
    nl = ~c.KeyPollutant.str.contains("Lead", case=False, na=False) & ~is_bat
    s = c[is_bat | nl][["latitude", "longitude"]].copy()
    s["battery"] = is_bat[is_bat | nl].astype(int).values
    s["country"] = cn
    S.append(s)
    a = pd.read_csv(path, usecols=cols) if path else pd.read_csv(ACLED_GLOBAL, usecols=cols)
    a = a[a.country == cn].dropna(subset=["latitude", "longitude", "year"])
    a = a[a.year.between(min(YEARS), max(YEARS))]
    E.append(a)
S = pd.concat(S, ignore_index=True)
S["site"] = np.arange(len(S))
E = pd.concat(E, ignore_index=True)
E["year"] = E.year.astype(int)
E["coord"] = E.latitude.round(4).astype(str) + "_" + E.longitude.round(4).astype(str)
print(f"sites: {S.battery.sum()} battery, {(1 - S.battery).sum()} non-lead; events {len(E):,}")

# site groups: chain sites within 10 km of each other
D = hav(S.latitude.values[:, None], S.longitude.values[:, None],
        S.latitude.values[None, :], S.longitude.values[None, :])
_, S["grp"] = connected_components(D < 10, directed=False)
print(f"site groups: {S.grp.nunique()}")

# drop-check: five most-used coordinates per country
top = (E.groupby(["country", "coord"]).size().rename("n").reset_index()
         .sort_values("n", ascending=False).groupby("country").head(5))
print("most-used coordinates:\n", top.to_string(index=False))
TOP = set(top.coord)


def site_ring_panel(ev):
    """Rows: site x ring x year with 1[any event]."""
    out = []
    for cn in COUNTRIES:
        s, e = S[S.country == cn], ev[ev.country == cn]
        d = hav(e.latitude.values[:, None], e.longitude.values[:, None],
                s.latitude.values[None, :], s.longitude.values[None, :])
        ii, jj = np.nonzero(d < RINGS[-1][1])
        dd = d[ii, jj]
        r = np.searchsorted([hi for _, hi in RINGS], dd, side="right")
        out.append(pd.DataFrame({"site": s.site.values[jj], "ring": r, "year": e.year.values[ii]}))
    hits = pd.concat(out).groupby(["site", "ring", "year"]).size().rename("n").reset_index()
    grid = pd.MultiIndex.from_product([S.site, range(len(RINGS)), YEARS],
                                      names=["site", "ring", "year"]).to_frame(index=False)
    p = grid.merge(hits, how="left").fillna({"n": 0.0}).merge(
        S[["site", "battery", "country", "grp"]], on="site")
    p["any_ev"] = (p.n > 0).astype(float)
    p["dens"] = p.n / AREA[p.ring.values] * 100
    p["post"] = (p.year >= POST_YR).astype(int)
    p["sr"] = p.site.astype(str) + "_" + p.ring.astype(str)
    p["rcy"] = p.ring.astype(str) + "_" + p.country + "_" + p.year.astype(str)
    for k in range(len(RINGS)):
        p[f"g{k}"] = ((p.ring == k) & (p.battery == 1) & (p.post == 1)).astype(int)
    return p


def fit(p, y="any_ev"):
    m = pf.feols(f"{y} ~ " + " + ".join(f"g{k}" for k in range(len(RINGS))) + " | sr + rcy",
                 data=p, vcov={"CRV1": "grp"})
    pre = p[(p.post == 0)].groupby(["battery", "ring"])[y].mean()
    return [(k, float(m.coef()[f"g{k}"]), float(m.se()[f"g{k}"]),
             float(pre.loc[(1, k)]), float(pre.loc[(0, k)])) for k in range(len(RINGS))]


SAMPLES = [("precision 1", lambda e: e[e.geo_precision == 1]),
           ("precision 1, top-5 coordinates dropped", lambda e: e[(e.geo_precision == 1) & ~e.coord.isin(TOP)])]
rows = []
for olab, types in OUTCOMES:
    for slab, f in SAMPLES:
        p = site_ring_panel(f(E[E.event_type.isin(types)]))
        ys = [("any event", "any_ev")] + ([("per 100 km2", "dens")] if olab == "Riots" else [])
        for ylab, y in ys:
            for k, b, se, pb, pn in fit(p, y):
                lo, hi = RINGS[k]
                rows.append(dict(outcome=olab, measure=ylab, sample=slab, ring=f"{lo}-{hi} km",
                                 coef=b, se=se, pre_mean_battery=pb, pre_mean_nonlead=pn))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))

# riots (precision 1) within 1 km of a battery site: how many distinct coordinates
ev = E[(E.event_type == "Riots") & (E.geo_precision == 1)]
b = S[S.battery == 1]
dmin = np.full(len(ev), np.inf)
for cn in COUNTRIES:
    m_e, m_s = (ev.country == cn).values, (b.country == cn).values
    dmin[m_e] = hav(ev.latitude.values[m_e, None], ev.longitude.values[m_e, None],
                    b.latitude.values[None, m_s], b.longitude.values[None, m_s]).min(1)
near = ev[dmin < 1]
print(f"riots within 1 km of a battery site: {len(near)} events at {near.coord.nunique()} coordinates; "
      f"top three coordinates hold {near.coord.value_counts().head(3).sum() / len(near):.0%}")

# macros: riots; A..F = rings 0-1 .. 10-20 km
L = "ABCDEF"
M = {"srNBat": f"{int(S.battery.sum())}", "srNNonlead": f"{int((1 - S.battery).sum())}",
     "srNGroups": f"{S.grp.nunique()}", "srNearOneN": f"{len(near):,}",
     "srNearOneCoords": f"{near.coord.nunique()}"}
_dens = R[(R.measure == "per 100 km2") & (R["sample"] == "precision 1")].reset_index()
for k in range(4):                                   # rings 0-1, 1-2, 2-3, 3-5 km
    M[f"srDens{L[k]}"] = format(_dens.coef[k], ".1f")
M["srDensASE"] = format(_dens.se[0], ".1f")
_drop = R[(R.measure == "per 100 km2") & (R["sample"] == "precision 1, top-5 coordinates dropped")].reset_index()
M["srDensDropA"] = format(_drop.coef[0], ".1f")
M = {k: (v.replace("-", "$-$", 1) if v.startswith("-") else v) for k, v in M.items()}
lines = ["% Generated by 56_site_rings.py"] + [r"\newcommand{\%s}{%s}" % kv for kv in M.items()]
(TABLES / "site_rings_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
save_to_overleaf("site_rings_values.tex", str(TABLES / "site_rings_values.tex"), subfolder="tables")

# figure: riots. (a) any riot; (b) riots per 100 km2. Red: all precision-1 events;
# grey: the five most-used coordinates per country dropped.
SER = [("precision 1", "All events", "#B3261E", -0.09),
       ("precision 1, top-5 coordinates dropped", "City-centre points dropped", "#777777", 0.09)]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
x = np.arange(len(RINGS))
for ax, (meas, title) in zip(axes, [("any event", "(a) Any riot in the ring"),
                                    ("per 100 km2", "(b) Riots per 100 km² of ring")]):
    top = None
    if meas == "per 100 km2":   # the 0-1 km all-events interval is very wide; clip it
        r0 = R[(R.outcome == "Riots") & (R.measure == meas) & R["sample"].isin([q[0] for q in SER])]
        keep = ~((r0["sample"] == "precision 1") & (r0.ring == "0-1 km"))
        top = float((r0.coef + 1.96 * r0.se)[keep].max()) * 1.12
    for smp, lab, col, off in SER:
        r = R[(R.outcome == "Riots") & (R.measure == meas) & (R["sample"] == smp)].reset_index()
        for k in range(len(RINGS)):
            b_, lo_, hi_ = r.coef[k], r.coef[k] - 1.96 * r.se[k], r.coef[k] + 1.96 * r.se[k]
            if top is not None and b_ > top:
                ax.annotate(f"{b_:.1f}", xy=(x[k] + off, top), xytext=(x[k] + off, top * 0.80),
                            ha="center", fontsize=8, color=col,
                            arrowprops=dict(arrowstyle="->", color=col, lw=1))
                continue
            ax.plot([x[k] + off] * 2, [lo_, min(hi_, top) if top else hi_], color=col, lw=1.2)
            ax.plot(x[k] + off, b_, "o", color=col, ms=5)
        if meas == "any event":
            ax.text(x[-1] + 0.35, r.coef.iloc[-1] + (0.012 if off < 0 else -0.012), lab,
                    color=col, fontsize=8, va="center")
    if top is not None:
        ax.set_ylim(top=top)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in RINGS])
    ax.set_xlim(-0.4, len(RINGS) - 0.4 + (1.6 if meas == "any event" else 0))
    ax.set_xlabel("Distance from the site, km")
    ax.set_ylabel("Battery v. non-lead sites, after 2007")
    ax.set_title(title, loc="left", fontsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_site_rings.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_site_rings.pdf", str(out), subfolder="figures")

# slide version: density only, larger type
fig, ax = plt.subplots(figsize=(7.2, 4.2))
r0 = R[(R.outcome == "Riots") & (R.measure == "per 100 km2") & R["sample"].isin([q[0] for q in SER])]
keep = ~((r0["sample"] == "precision 1") & (r0.ring == "0-1 km"))
top = float((r0.coef + 1.96 * r0.se)[keep].max()) * 1.12
for smp, lab, col, off in SER:
    r = R[(R.outcome == "Riots") & (R.measure == "per 100 km2") & (R["sample"] == smp)].reset_index()
    for k in range(len(RINGS)):
        b_, lo_, hi_ = r.coef[k], r.coef[k] - 1.96 * r.se[k], r.coef[k] + 1.96 * r.se[k]
        if b_ > top:
            ax.annotate(f"All events: {b_:.1f}, off the scale", xy=(x[k] + off, top),
                        xytext=(x[k] + off + 0.25, top * 0.93), ha="left", va="center",
                        fontsize=11, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            continue
        ax.plot([x[k] + off] * 2, [lo_, min(hi_, top)], color=col, lw=1.6)
        ax.plot(x[k] + off, b_, "o", color=col, ms=7)
    ax.text(x[3] - 0.2, top * (0.62 if off < 0 else 0.52), lab, color=col, fontsize=12, va="center")
ax.set_ylim(top=top)
ax.axhline(0, color="#999999", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in RINGS], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel("Distance from the site, km", fontsize=12)
ax.set_ylabel("Extra riots per 100 km², battery v. non-lead sites, after 2007", fontsize=12)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_slides_site_rings.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_slides_site_rings.pdf", str(out), subfolder="figures")
print("saved site_rings_values.tex, fig_site_rings, fig_slides_site_rings")
