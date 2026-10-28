"""
45_lead_burden_regions.py  --  Slide figure: share of children with blood lead above 5 ug/dL.

US children aged 1-5: 99.8% at or above 5 ug/dL in 1976-80 and 1.3% in 2011-16 (NHANES;
Egan et al. 2021, EHP 129(3), Table 4). Typed below with that source: published survey estimates.

World regions, 2019: IHME estimates of the number of children aged 0-19 above 5 ug/dL by
country (UNICEF and Pure Earth 2020, "The Toxic Truth", Annex A; PDF in data/), divided by
World Bank population aged 0-19 in 2019 (SP.POP.0014.TO + SP.POP.1519.FE + SP.POP.1519.MA)
and summed to World Bank regions. The World Bank series are read from data/wb_pop0_19_2019.csv;
the World Bank API is queried only if that file is missing.

Outputs: output/figures/fig_lead_burden.pdf (copied to Overleaf figures/),
         output/tables/burden_values.tex (copied to Overleaf tables/).
"""
import json, re, subprocess, urllib.request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGURES, TABLES, save_to_overleaf

US = {"1976-80": 99.8, "2011-16": 1.3}          # Egan et al. (2021), Table 4, ages 1-5
YEAR = 2019

# ── IHME counts by country from the report annex ──────────────────────────────
txt = subprocess.run(["pdftotext", "-layout", str(DATA / "toxic_truth_unicef_pure_earth_2020.pdf"), "-"],
                     capture_output=True, text=True, check=True).stdout
seg = txt[txt.index("     ANNEX\n"):]
rows, last = [], 0
for n, name, mean in re.findall(r"^\s*(\d{1,3})\s{2,}(\S.*?\S)\s{2,}([\d,]+)\**\s+(?:[\d,]+|-)\s+(?:[\d,]+|-)", seg, flags=re.M):
    if int(n) <= last:
        break                                     # numbering restarts in the regional summary
    last = int(n)
    rows.append((name.strip().rstrip("*").strip(), int(mean.replace(",", ""))))
ihme = pd.DataFrame(rows, columns=["name", "n_above5"])
print(f"annex: {len(ihme)} countries, {ihme.n_above5.sum()/1e6:.0f} million children above 5 ug/dL")

# ── World Bank population 0-19 and regions (cached) ───────────────────────────
cache = DATA / f"wb_pop0_19_{YEAR}.csv"
if not cache.exists():
    def wb(ind):
        u = f"https://api.worldbank.org/v2/country/all/indicator/{ind}?date={YEAR}&format=json&per_page=400"
        return {d["countryiso3code"]: d["value"] for d in json.load(urllib.request.urlopen(u))[1] if d["countryiso3code"]}
    meta = json.load(urllib.request.urlopen("https://api.worldbank.org/v2/country?format=json&per_page=400"))[1]
    meta = pd.DataFrame([{"iso3": c["id"], "wb_name": c["name"], "region": c["region"]["value"]}
                         for c in meta if c["region"]["value"] != "Aggregates"])
    a, f, m = wb("SP.POP.0014.TO"), wb("SP.POP.1519.FE"), wb("SP.POP.1519.MA")
    meta["pop0_19"] = [(a.get(i) or 0) + (f.get(i) or 0) + (m.get(i) or 0) if a.get(i) else None for i in meta.iso3]
    meta.to_csv(cache, index=False)
wbp = pd.read_csv(cache)

# report names that differ from World Bank names
ALIAS = {"Bolivia (Plurinational State of)": "Bolivia", "Côte d'Ivoire": "Cote d'Ivoire",
         "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
         "Democratic Republic of the Congo": "Congo, Dem. Rep.", "Congo": "Congo, Rep.",
         "Egypt": "Egypt, Arab Rep.", "Gambia": "Gambia, The", "Bahamas": "Bahamas, The",
         "Iran (Islamic Republic of)": "Iran, Islamic Rep.", "Kyrgyzstan": "Kyrgyz Republic",
         "Lao People's Democratic Republic": "Lao PDR", "Micronesia (Federated States of)": "Micronesia, Fed. Sts.",
         "Republic of Korea": "Korea, Rep.", "Republic of Moldova": "Moldova", "Saint Kitts and Nevis": "St. Kitts and Nevis",
         "Saint Lucia": "St. Lucia", "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
         "Slovakia": "Slovak Republic", "United Republic of Tanzania": "Tanzania", "United States of America": "United States",
         "Venezuela (Bolivarian Republic of)": "Venezuela, RB", "Viet Nam": "Viet Nam", "Yemen": "Yemen, Rep.",
         "Turkey": "Turkiye", "Czechia": "Czechia", "Syrian Arab Republic": "Syrian Arab Republic",
         "United Kingdom": "United Kingdom", "Sao Tome and Principe": "Sao Tome and Principe",
         "Eswatini": "Eswatini", "Cabo Verde": "Cabo Verde", "State of Palestine": "West Bank and Gaza",
         "Palestine": "West Bank and Gaza", "North Macedonia": "North Macedonia", "Russian Federation": "Russian Federation",
         "Brunei Darussalam": "Brunei Darussalam", "Somalia": "Somalia, Fed. Rep."}
ihme["wb_name"] = ihme.name.map(lambda x: ALIAS.get(x, x))
df = ihme.merge(wbp, on="wb_name", how="left")
miss = df[df.pop0_19.isna()]
share_missing = miss.n_above5.sum() / ihme.n_above5.sum()
print(f"unmatched or no population: {len(miss)} countries, {100*share_missing:.2f}% of children above 5:",
      ", ".join(miss.name.head(25)))
if share_missing > 0.01:
    raise SystemExit("45: more than 1% of the annex total did not match a World Bank country; extend ALIAS")
df = df.dropna(subset=["pop0_19"])
df["region"] = df.region.str.strip()
reg = df.groupby("region")[["n_above5", "pop0_19"]].sum()
reg["share"] = 100 * reg.n_above5 / reg.pop0_19
reg = reg.sort_values("share", ascending=False)
world = 100 * df.n_above5.sum() / df.pop0_19.sum()
print(reg.round(1)); print(f"world {world:.1f}%")

# ── Figure: US then and now, world regions in 2019 ────────────────────────────
SHORT = {"Sub-Saharan Africa": "Sub-Saharan Africa", "South Asia": "South Asia",
         "Middle East, North Africa, Afghanistan & Pakistan": "Middle East, N. Africa, Afghanistan & Pakistan",
         "Middle East & North Africa": "Middle East & N. Africa", "East Asia & Pacific": "East Asia & Pacific",
         "Latin America & Caribbean": "Latin America", "Europe & Central Asia": "Europe & Central Asia",
         "North America": "North America"}
fig, ax = plt.subplots(figsize=(6.8, 4.6))
x0, x1 = 0, 1
ax.plot([x0, x1], [US["1976-80"], US["2011-16"]], "-o", color="#555555", lw=2, ms=6, label="_us")
ax.annotate(f"US children, 1976-80: {US['1976-80']:.1f}%", xy=(x0, US["1976-80"]), xytext=(10, 0),
            textcoords="offset points", ha="left", va="center", fontsize=10, color="#555555")
pts = [(SHORT.get(r, r), v) for r, v in reg.share.items()]
for name, v in pts:
    c = "#B3261E" if name in ("Sub-Saharan Africa", "South Asia") else "#006970"
    ax.plot([x1 + 0.9], [v], "o", color=c, ms=6)
# label the 2019 points to the right, spread so they never overlap
ys = sorted([(v, n) for n, v in pts], reverse=True); placed = []; gap = 6.0
for v, n in ys:
    y = v if not placed or placed[-1] - v >= gap else placed[-1] - gap
    placed.append(y)
    c = "#B3261E" if n in ("Sub-Saharan Africa", "South Asia") else "#006970"
    ax.annotate(f"{n} {v:.0f}%", xy=(x1 + 0.9, v), xytext=(x1 + 1.0, y), textcoords="data",
                ha="left", va="center", fontsize=10, color=c,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=c) if abs(y - v) > 0.5 else None)
ax.annotate(f"2011-16: {US['2011-16']:.1f}%", xy=(x1, US["2011-16"]), xytext=(-8, 8), textcoords="offset points",
            ha="right", va="bottom", fontsize=10, color="#555555")
ax.set_xticks([x0, x1, x1 + 0.9], ["US\n1976-80", "US\n2011-16", f"World regions\n{YEAR}"])
ax.set_xlim(-0.75, 3.6)
ax.set_ylim(-4, 105)
ax.set_ylabel("Children with blood lead above 5 µg/dL (%)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_lead_burden.pdf"
fig.savefig(out, bbox_inches="tight")
save_to_overleaf("fig_lead_burden.pdf", str(out), subfolder="figures")
plt.close(fig)

v = ["% Generated by 45_lead_burden_regions.py -- do not edit by hand",
     r"\newcommand{\burdenWorld}{" + f"{world:.0f}" + "}"]
(TABLES / "burden_values.tex").write_text("\n".join(v) + "\n", encoding="utf-8")
save_to_overleaf("burden_values.tex", str(TABLES / "burden_values.tex"), subfolder="tables")
print("saved fig_lead_burden.pdf, burden_values.tex")
