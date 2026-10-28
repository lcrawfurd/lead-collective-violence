"""
11_generate_values.py
Generate a LaTeX file of \\newcommand definitions for the statistics cited in the paper
prose that no other script writes, so no values are hardcoded.

Reads:
  output/panels/panel_{kenya,ghana,senegal}.parquet   (01_build_panels.py)
  output/panels/botec_extensive.csv                   (29_extensive_margin_main.py)
  output/panels/ri_results.csv                        (10_randomisation_inference.py)
  output/panels/vehicle_dose_summary.csv              (23_vehicle_dose_main.py)
  output/panels/cutoff_sweep.csv                      (32_cutoff_sweep.py)
  data/tsip_sites_cleaned.csv, data/ACLED_global.csv

Writes (also copied to Overleaf tables/):
  output/tables/generated_values.tex

Usage: python3 code/11_generate_values.py
"""

import os

import numpy as np
import pandas as pd

from config import TABLES, PANELS, OVERLEAF, TSIP_PATH, ACLED_GLOBAL
PANEL = str(PANELS)
TABS  = str(TABLES)


# ── Strict inputs: a missing or unreadable input stops the build ─────────────
# Earlier versions fell back to numbers typed in from a previous run whenever an input
# was missing. Those went into the paper silently, and some had drifted far from the
# current estimates. Every macro now comes from a current input, or the build stops.
def need(value, what, source):
    """Return value, or stop the build if it is missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise SystemExit(f"\n11_generate_values: cannot compute {what}: {source} is missing "
                         f"or unreadable.\nRun the script that produces it first (see run_all.sh).")
    return value


def need_file(path, producer):
    """Stop the build if an input file is missing, naming the script that writes it."""
    if not os.path.exists(str(path)):
        raise SystemExit(f"\n11_generate_values: missing input {path}\n"
                         f"Run {producer} first (see run_all.sh).")
    return path


vals = {}  # command_name -> value_string


def add(name, value, fmt=None):
    """Register a LaTeX command."""
    if fmt:
        vals[name] = fmt.format(value)
    elif isinstance(value, float):
        vals[name] = f"{value:.2f}"
    elif isinstance(value, int):
        vals[name] = f"{value:,}"
    else:
        vals[name] = str(value)


# ── Load panels ──────────────────────────────────────────────────────────────

print("Loading panels...")
ke = pd.read_parquet(f"{PANEL}/panel_kenya.parquet")
gh = pd.read_parquet(f"{PANEL}/panel_ghana.parquet")
sn = pd.read_parquet(f"{PANEL}/panel_senegal.parquet")
countries = {"Kenya": ke, "Ghana": gh, "Senegal": sn}

pooled = pd.concat([df.assign(country=cn) for cn, df in countries.items()], ignore_index=True)
pooled["cell_id"] = pooled["country"] + "_" + pooled["cell_id"].astype(str)

# ── Table 1: Sample descriptives ─────────────────────────────────────────────

print("Table 1 descriptives...")
tsip = pd.read_csv(str(TSIP_PATH))

for cname, panel in countries.items():
    prefix = cname[:2].lower()  # ke, gh, se
    c_tsip = tsip[tsip["country"] == cname]
    n_bat = c_tsip[c_tsip["SiteIndustry"].str.contains("Battery", case=False, na=False)].shape[0]
    # Non-lead must match the panel-building filter in 01_build_panels.py: exclude
    # both Lead-pollutant sites AND battery-industry sites. One Kenyan site is
    # SiteIndustry "Lead - Battery Recycling" but KeyPollutant "Chromium (Total)";
    # a KeyPollutant-only filter would count it in BOTH the treatment and the
    # comparison group (Kenya 46 rather than the correct 45).
    n_nl = c_tsip[~c_tsip["KeyPollutant"].str.contains("Lead", case=False, na=False)
                  & ~c_tsip["SiteIndustry"].str.contains("Battery", case=False, na=False)].shape[0]
    add(f"sites{prefix}Bat", n_bat)
    add(f"sites{prefix}NL", n_nl)
    add(f"cells{prefix}", panel["cell_id"].nunique())
    add(f"obs{prefix}", len(panel))
    add(f"riots{prefix}", int(panel["n_riots"].sum()))
    add(f"protests{prefix}", int(panel["n_protests"].sum()))

add("pooledCells", pooled["cell_id"].nunique())
add("pooledObs", len(pooled))

# ── BOTEC (extensive margin) ─────────────────────────────────────────────────
# Read the extensive-margin BOTEC computed by 29_extensive_margin_main.py.
print("BOTEC (extensive margin)...")
_botec_csv = need_file(PANELS / "botec_extensive.csv", "29_extensive_margin_main.py")
_b = pd.read_csv(str(_botec_csv)).iloc[0]
add("botecRiots",        f"{_b['riot_attr']:,.0f}")
add("botecRiotsPerYear", f"{_b['riot_per_year']:,.0f}")
add("botecCILo",         f"{_b['riot_lo']:,.0f}")
add("botecCIHi",         f"{_b['riot_hi']:,.0f}")
add("botecShareRiots",   f"{_b['riot_share']:.0f}")
add("botecNearBaseRate", f"{_b['near_base_pre']:.0f}")
add("botecFatalities",   f"{_b['fat_attr']:,.0f}")
add("botecFatDDDD", f"{need(_b.get('fat_dddd'), 'fatal-riot DDDD', 'botec_extensive.csv'):.3f}")
print(f"  riot-active cell-years attributable: {_b['riot_attr']:.0f} "
      f"(CI {_b['riot_lo']:.0f}-{_b['riot_hi']:.0f}), {_b['riot_share']:.0f}% share")
# Battery sites in the three study countries (was a typed 67; now counted).
add("nBatTotal", int((tsip["SiteIndustry"].str.contains("Battery", case=False, na=False)
                      & tsip["country"].isin(["Kenya", "Ghana", "Senegal"])).sum()))

# ── RI p-values ──────────────────────────────────────────────────────────────

print("RI values...")
ri_csv_path = need_file(PANELS / "ri_results.csv", "10_randomisation_inference.py")
ri_df = pd.read_csv(str(ri_csv_path))
ri_by_country = {row["country"]: row for _, row in ri_df.iterrows()}
for cname, prefix in [("Kenya", "Ke"), ("Ghana", "Gh"), ("Senegal", "Se")]:
    p = float(need(ri_by_country.get(cname), f"RI p-value for {cname}", "ri_results.csv")["p_value"])
    add(f"riP{prefix}", "$<$0.001" if p < 0.001 else f"{p:.3f}")
add("riNPerm", f"{int(need(ri_by_country.get('Pooled'), 'RI draws', 'ri_results.csv')['n_valid']):,}")

# ── Vehicle dose and binary escalating differences (from 23) ─────────────────
# vehicle_dose_summary.csv: pooled escalating differences (extensive margin, riots)
# DD = battery gradient; DDD = lead-specific gap; DDDD = conflict-type contrast.
print("Vehicle-dose and escalating-difference macros...")
_veh_csv = need_file(PANELS / "vehicle_dose_summary.csv", "23_vehicle_dose_main.py")
_v = pd.read_csv(str(_veh_csv)).iloc[0]
add("vehDDCoef",    float(_v["dd_coef"]),   "{:.3f}")
add("vehDDDGap",    float(_v["ddd_gap"]),   "{:.3f}")
add("vehDDDSE",     float(_v["ddd_se"]),    "{:.3f}")
add("vehNlCoef",    float(_v["nl_coef"]),   "{:.3f}")
add("vehDDDDDiff",  float(_v["dddd_diff"]), "{:.3f}")
add("vehDDDDSE",    float(_v["dddd_se"]),   "{:.3f}")
# Binary 1[<5km] x Post(2007) version of the same three models
add("binDDCoef",    float(_v["bin_dd_coef"]),   "{:.3f}")
add("binDDSE",      float(_v["bin_dd_se"]),     "{:.3f}")
add("binDDDGap",    float(_v["bin_ddd_gap"]),   "{:.3f}")
add("binDDDSE",     float(_v["bin_ddd_se"]),    "{:.3f}")
add("binDDDDDiff",  float(_v["bin_dddd_diff"]), "{:.3f}")
add("binDDDDSE",    float(_v["bin_dddd_se"]),   "{:.3f}")
print(f"  vehDD={vals['vehDDCoef']}, vehDDD={vals['vehDDDGap']}, "
      f"vehDDDD={vals['vehDDDDDiff']}; binDD={vals['binDDCoef']}")

# ── Treatment-timing sweep (from 32_cutoff_sweep.py) ──────────────────────────
print("Cutoff-sweep macros...")
_cut_csv = need_file(PANELS / "cutoff_sweep.csv", "32_cutoff_sweep.py")
_c = pd.read_csv(str(_cut_csv))
_tmin = float((_c["coef"].abs() / _c["se"]).min())
add("cutoffMinYr",  int(_c["cutoff"].min()), "{:d}")
add("cutoffMaxYr",  int(_c["cutoff"].max()), "{:d}")
add("cutoffMinT",   _tmin, "{:.1f}")
print(f"  {len(_c)} estimates, cutoffs {_c['cutoff'].min()}-{_c['cutoff'].max()}, "
      f"min |t| = {_tmin:.1f}")

# ── Documented battery sites in Africa (conclusion: undercount argument) ─────
_afr = ("Kenya|Ghana|Senegal|Tanzania|Nigeria|Zimbabwe|Madagascar|Uganda|Ethiopia|Cameroon|Mali|"
        "Niger|Benin|Togo|Guinea|Zambia|Mozambique|Rwanda|South Africa|Gabon|Congo|Tunisia|Swaziland")
_b = tsip[tsip.SiteIndustry.astype(str).str.contains("Battery", case=False, na=False)]
_ab = _b[_b.country.astype(str).str.contains(_afr, case=False, na=False)]
add("nBatAfrica", f"{len(_ab)}")
_others = _ab[~_ab.country.astype(str).str.contains("Kenya|Ghana|Senegal", case=False, na=False)]
add("nBatNextAfrica", f"{int(_others.country.value_counts().max())}")   # largest count outside the three
_idn = tsip[tsip.country.astype(str).str.contains("Indonesia", case=False, na=False)]
add("tsipIdnN", f"{len(_idn)}")
add("tsipIdnBat", f"{int(_idn.SiteIndustry.astype(str).str.contains('Battery', case=False, na=False).sum())}")

# ── Battery sites in the three study countries, by TSIP industry label ─────────
# The code treats every TSIP site whose industry mentions "Battery" as a battery site.
_b3 = _b[_b.country.astype(str).isin(["Kenya", "Ghana", "Senegal"])]
_ind = _b3.SiteIndustry.astype(str)
add("nBatSitesThree", f"{len(_b3)}")
add("nBatRecycThree", f"{int((_ind == 'Lead - Battery Recycling').sum())}")
add("nBatMfgThree", f"{int(_ind.str.contains('manufacturing or repair', case=False).sum())}")
add("nBatOtherThree", f"{int(_ind.str.contains('not Lead Acid', case=False).sum())}")
print(f"  battery sites in Africa: {len(_ab)} across {_ab.country.nunique()} countries")

# ── Sub-Saharan Africa conflict burden (intro framing) ───────────────────────
# Deaths, not event counts, are the right basis: SSA is ~21% of world political-violence
# EVENTS but a much larger share of deaths.
print("SSA conflict burden...")
_ssa_regions = ["Eastern Africa", "Middle Africa", "Southern Africa", "Western Africa"]
_pol = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
try:
    _a = pd.read_csv(str(ACLED_GLOBAL), usecols=["year", "region", "event_type", "fatalities"],
                     low_memory=False)
except Exception as _e:
    raise SystemExit(f"\n11_generate_values: ACLED global not readable: {_e}")
_a["fatalities"] = pd.to_numeric(_a.fatalities, errors="coerce").fillna(0)
_YR = 2024                      # last complete year in the extract
_w = _a[(_a.year == _YR) & (_a.event_type.isin(_pol))]
_s = _w[_w.region.isin(_ssa_regions)]
add("ssaYear", f"{_YR}")
add("ssaDeaths", f"{_s.fatalities.sum():,.0f}")
add("ssaDeathShare", f"{100*_s.fatalities.sum()/_w.fatalities.sum():.0f}")
print(f"  {_YR}: SSA {_s.fatalities.sum():,.0f} political-violence deaths "
      f"({100*_s.fatalities.sum()/_w.fatalities.sum():.0f}% of world)")


# ── Derived numbers the text quotes as words or ratios ───────────────────────
def _f(name):
    return float(str(need(vals.get(name), name, "a macro computed above")).replace(",", ""))


add("binDDCoefPP", f"{100*_f('binDDCoef'):.0f}")
add("binDDDGapPP", f"{100*_f('binDDDGap'):.0f}")
add("binDDDDDiffPP", f"{100*_f('binDDDDDiff'):.0f}")
add("binDDDSEPP", f"{100*_f('binDDDSE'):.1f}")
add("binDDDDSEPP", f"{100*_f('binDDDDSE'):.1f}")
# Totals row of the sample-overview table (Table 1)
add("sitesTotNL", f"{sum(int(_f(f'sites{k}NL')) for k in ('ke', 'gh', 'se'))}")
add("riotsTot", f"{sum(int(_f(f'riots{k}')) for k in ('ke', 'gh', 'se')):,}")
add("protestsTot", f"{sum(int(_f(f'protests{k}')) for k in ('ke', 'gh', 'se')):,}")

# ── Write output ─────────────────────────────────────────────────────────────

print(f"\nWriting {len(vals)} commands...")
lines = ["% Auto-generated by 11_generate_values.py — do not edit by hand\n"]
for name, value in sorted(vals.items()):
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
tex = "\n".join(lines) + "\n"

dests = [TABS] + ([str(OVERLEAF / "tables")] if OVERLEAF else [])
for base in dests:
    with open(f"{base}/generated_values.tex", "w") as f:
        f.write(tex)
    print(f"  Saved: {base}/generated_values.tex")

# Placeholders for value files that main.tex \inputs but no longer draws a macro from.

print("Done.")
