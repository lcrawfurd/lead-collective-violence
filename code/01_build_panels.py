"""
01_build_panels.py
DATA ASSEMBLY: Build balanced grid-cell × year panels for each country.

Inputs:
  - TSIP cleaned sites: output/tsip_sites_cleaned.csv
  - ACLED: acled_kenya.csv, acled_ghana.csv, ACLED_global.csv (global)

Outputs:
  - output/panels/panel_{country}.parquet  (one per country)

Each panel contains:
  cell_id, year, cell_lat, cell_lon,
  inv_dist_battery, inv_dist_nonlead,
  post,
  n_riots, n_protests, n_org_violence, n_battles, n_mob_violence, n_vac,
  fat_riots,
  log_riots, log_protests, log_org_violence,
  log_mob_violence, log_vac,
  log_fat_riots

Organised violence = Battles + Violence against civilians (combined).

Sample: ALL 0.1° cells within 50 km of any TSIP site (battery or non-lead),
including zero-event cells. This avoids selection bias from conditioning the
sample on cells that happened to have ≥1 ACLED event (which would skew the
sample toward urbanised cells near battery sites).
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
from utils import haversine_km
from config import PROJECT, DATA, OUTPUT, PANELS, OVERLEAF, TSIP_PATH, ACLED_GLOBAL, ACLED_KE, ACLED_GH, GRID, POST_YR, BASE_YR, END_YR, save_to_overleaf
TSIP_PATH, ACLED_GLOBAL, ACLED_KE, ACLED_GH = (str(p) for p in (TSIP_PATH, ACLED_GLOBAL, ACLED_KE, ACLED_GH))
OUT          = str(PANELS)
os.makedirs(OUT, exist_ok=True)

GRID     = 0.1    # degrees (~11 km)
MAX_DIST = 50.0   # km — include cells within this distance of any TSIP site

# ── Country configuration ──────────────────────────────────────────────────────
COUNTRIES = [
    dict(name="Kenya",   post_year=2007, acled_path=ACLED_KE),
    dict(name="Ghana",   post_year=2007, acled_path=ACLED_GH),
    dict(name="Senegal", post_year=2007, acled_path=None),
    # Bangladesh excluded from analysis: pre-treatment riot rate ~30× higher than SSA
    # countries and no clean identification strategy. Data in data/ACLED_Bangladesh.csv
    # is retained for reference but not used.
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def haversine_matrix(clats, clons, slats, slons):
    """Return (n_cells × n_sites) distance matrix in km."""
    clats_r = np.radians(clats)[:,None]; clons_r = np.radians(clons)[:,None]
    slats_r = np.radians(slats)[None,:]; slons_r = np.radians(slons)[None,:]
    dlat = slats_r - clats_r; dlon = slons_r - clons_r
    a = np.sin(dlat/2)**2 + np.cos(clats_r)*np.cos(slats_r)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_cell_grid(bat_sites, nl_sites, max_dist_km=MAX_DIST):
    """
    Generate all 0.1° cells within max_dist_km of any TSIP site.
    Returns DataFrame with cell_lat, cell_lon, cell_id,
    dist_battery, dist_nonlead, inv_dist_battery, inv_dist_nonlead.
    Includes zero-event cells — no ACLED conditioning.
    """
    all_sites = pd.concat([bat_sites, nl_sites])
    lat_min = all_sites["latitude"].min()  - max_dist_km/111.0 - GRID
    lat_max = all_sites["latitude"].max()  + max_dist_km/111.0 + GRID
    lon_min = all_sites["longitude"].min() - max_dist_km/111.0 - GRID
    lon_max = all_sites["longitude"].max() + max_dist_km/111.0 + GRID

    # All candidate cell centroids
    lats = np.arange(np.floor(lat_min/GRID)*GRID + GRID/2,
                     lat_max + GRID, GRID).round(4)
    lons = np.arange(np.floor(lon_min/GRID)*GRID + GRID/2,
                     lon_max + GRID, GRID).round(4)
    grid = pd.DataFrame([(la, lo) for la in lats for lo in lons],
                        columns=["cell_lat","cell_lon"])

    bat_lats = bat_sites["latitude"].values; bat_lons = bat_sites["longitude"].values
    nl_lats  = nl_sites["latitude"].values;  nl_lons  = nl_sites["longitude"].values

    clats = grid["cell_lat"].values; clons = grid["cell_lon"].values
    d_bat = haversine_matrix(clats, clons, bat_lats, bat_lons).min(axis=1)
    d_nl  = haversine_matrix(clats, clons, nl_lats,  nl_lons ).min(axis=1)

    grid["dist_battery"] = d_bat
    grid["dist_nonlead"] = d_nl
    grid["dist_any"]     = np.minimum(d_bat, d_nl)

    # Keep cells within max_dist of any site
    grid = grid[grid["dist_any"] <= max_dist_km].copy()
    grid["inv_dist_battery"] = 1.0 / (grid["dist_battery"] + 1)
    grid["inv_dist_nonlead"] = 1.0 / (grid["dist_nonlead"] + 1)
    grid["cell_id"] = (grid["cell_lat"].astype(str) + "_" +
                       grid["cell_lon"].astype(str))
    return grid.reset_index(drop=True)


def assign_cell(df):
    """Add cell_lat, cell_lon, cell_id columns (0.1° grid)."""
    df = df.copy()
    df["cell_lat"] = (np.floor(df["latitude"]  / GRID) * GRID + GRID/2).round(4)
    df["cell_lon"] = (np.floor(df["longitude"] / GRID) * GRID + GRID/2).round(4)
    df["cell_id"]  = df["cell_lat"].astype(str) + "_" + df["cell_lon"].astype(str)
    return df


def load_acled(country_name, country_path, global_path, exclude_years=None):
    """Load ACLED events for a single country."""
    needed_cols = ["year", "event_type", "sub_event_type", "latitude", "longitude", "fatalities"]
    if country_path and os.path.exists(country_path):
        df = pd.read_csv(country_path)
        for col in needed_cols:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = pd.read_csv(global_path, usecols=needed_cols + ["country"])
        df = df[df["country"] == country_name].copy()
    df = df.dropna(subset=["latitude", "longitude", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] <= END_YR].copy()          # drop the partial final year
    df["fatalities"] = pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0)
    if exclude_years:
        df = df[~df["year"].isin(exclude_years)].copy()
        print(f"  Excluded years {exclude_years}: {len(df)} events remain")
    return df


def build_panel(acled, bat_sites, nl_sites, post_year):
    """
    Build balanced cell × year panel for one country.
    Skeleton = ALL 0.1° cells within 50 km of any TSIP site (Option A).
    Zero-event cells are included with n=0. ACLED counts merged via left join.
    """
    acled = assign_cell(acled)
    years = sorted(acled["year"].unique())

    # ── Spatial skeleton: all cells within 50 km ───────────────────────────────
    cells = build_cell_grid(bat_sites, nl_sites)

    # ── Balanced skeleton: all cells × all years ───────────────────────────────
    idx = pd.MultiIndex.from_product(
        [cells["cell_id"], years], names=["cell_id", "year"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(
        cells[["cell_id","cell_lat","cell_lon","inv_dist_battery","inv_dist_nonlead"]],
        on="cell_id")
    panel["post"] = (panel["year"] >= post_year).astype(int)

    # ── Event counts by type ───────────────────────────────────────────────────
    type_map = {
        "Riots":    "riots",
        "Protests": "protests",
    }
    for etype, label in type_map.items():
        sub    = acled[acled["event_type"] == etype]
        counts = sub.groupby(["cell_id","year"]).size().reset_index(name=f"n_{label}")
        panel  = panel.merge(counts, on=["cell_id","year"], how="left")
        panel[f"n_{label}"]   = panel[f"n_{label}"].fillna(0)
        panel[f"log_{label}"] = np.log1p(panel[f"n_{label}"])

    # Organised violence: Battles + Violence against civilians
    org_types = ["Battles", "Violence against civilians"]
    org       = acled[acled["event_type"].isin(org_types)]
    org_cnt   = org.groupby(["cell_id","year"]).size().reset_index(name="n_org_violence")
    panel     = panel.merge(org_cnt, on=["cell_id","year"], how="left")
    panel["n_org_violence"]   = panel["n_org_violence"].fillna(0)
    panel["log_org_violence"] = np.log1p(panel["n_org_violence"])
    # Battles alone (ACLED event type), for the appendix sub-type table
    bat_cnt = acled[acled["event_type"] == "Battles"].groupby(["cell_id","year"]).size().reset_index(name="n_battles")
    panel   = panel.merge(bat_cnt, on=["cell_id","year"], how="left")
    panel["n_battles"] = panel["n_battles"].fillna(0)

    # ── Appendix sub-event breakdowns ─────────────────────────────────────────
    for sub_label, sub_type in [
        ("mob_violence",         "Mob violence"),
        ("violent_demo",         "Violent demonstration"),
        ("peaceful_protest",     "Peaceful protest"),
        ("protest_intervention", "Protest with intervention"),
        ("vac_attack",           "Attack"),
        ("vac_abduction",        "Abduction/forced disappearance"),
    ]:
        sub_cnt = (acled[acled["sub_event_type"] == sub_type]
                   .groupby(["cell_id","year"]).size()
                   .reset_index(name=f"n_{sub_label}"))
        panel   = panel.merge(sub_cnt, on=["cell_id","year"], how="left")
        panel[f"n_{sub_label}"]   = panel[f"n_{sub_label}"].fillna(0)
        panel[f"log_{sub_label}"] = np.log1p(panel[f"n_{sub_label}"])

    # ── Fatalities (riots only) ────────────────────────────────────────────────
    riots_fat = (acled[acled["event_type"] == "Riots"]
                 .groupby(["cell_id","year"])["fatalities"]
                 .sum().reset_index(name="fat_riots"))
    panel = panel.merge(riots_fat, on=["cell_id","year"], how="left")
    panel["fat_riots"]     = panel["fat_riots"].fillna(0)
    panel["log_fat_riots"] = np.log1p(panel["fat_riots"])

    return panel


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tsip_all = pd.read_csv(TSIP_PATH)
    built = {}

    for c in COUNTRIES:
        cname     = c["name"]
        post_year = c["post_year"]
        print(f"\n{'='*60}")
        print(f"{cname} (post={post_year})")

        # TSIP sites
        cty_sites = tsip_all[tsip_all["country"].str.contains(cname, case=False, na=False)]
        bat_sites = cty_sites[cty_sites["SiteIndustry"].str.contains("Battery", case=False, na=False)]
        # Non-lead: exclude sites with Lead as key pollutant AND exclude battery sites
        # to prevent any overlap between treatment and placebo groups
        nl_sites  = cty_sites[
            ~cty_sites["KeyPollutant"].str.contains("Lead", case=False, na=False)
            & ~cty_sites["SiteIndustry"].str.contains("Battery", case=False, na=False)
        ]
        print(f"  TSIP: {len(bat_sites)} battery, {len(nl_sites)} non-lead sites")

        # ACLED
        acled = load_acled(cname, c["acled_path"], ACLED_GLOBAL,
                           exclude_years=c.get("exclude_years"))
        print(f"  ACLED: {len(acled)} events ({acled['year'].min()}–{acled['year'].max()})")
        print(f"  Event types: {acled['event_type'].value_counts().to_dict()}")

        # Panel
        panel = build_panel(acled, bat_sites, nl_sites, post_year)
        n_cells = panel["cell_id"].nunique()
        n_years = panel["year"].nunique()
        print(f"  Panel: {n_cells} cells × {n_years} years = {len(panel)} obs")

        # Save
        out_path = os.path.join(OUT, f"panel_{cname.lower()}.parquet")
        panel.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")
        built[cname] = panel

    # ── Sample checks: stop if the panels do not match the sample the paper describes ──
    # A silent change in the TSIP or ACLED inputs, or in the grid code, would otherwise
    # flow into every estimate. Counts verified against the 2026-10-22 panels.
    EXPECTED_CELLS = {"Kenya": 1319, "Ghana": 626, "Senegal": 1050}
    for cname, panel in built.items():
        n = panel["cell_id"].nunique()
        assert n == EXPECTED_CELLS[cname], f"{cname}: {n} cells, expected {EXPECTED_CELLS[cname]}"
        assert len(panel) == n * panel["year"].nunique(), f"{cname}: panel is not balanced"
    n_cells = sum(p["cell_id"].nunique() for p in built.values())
    n_obs = sum(len(p) for p in built.values())
    assert n_cells == 2995, f"pooled sample has {n_cells} cells, expected 2,995"
    n_years = END_YR - 1997 + 1
    for cname, panel in built.items():
        assert sorted(panel["year"].unique()) == list(range(1997, END_YR + 1)), f"{cname}: years not 1997-{END_YR}"
    assert n_obs == 2995 * n_years, f"pooled sample has {n_obs} cell-years, expected {2995 * n_years:,}"
    # Treated cells: within 5 km of a battery site, i.e. 1/(d+1) > 1/6
    n_near = sum(p.loc[p["inv_dist_battery"] > 1.0 / 6.0, "cell_id"].nunique() for p in built.values())
    assert n_near == 30, f"{n_near} cells within 5 km of a battery site, expected 30"
    print(f"\nChecks passed: {n_cells} cells, {n_obs} cell-years, {n_near} cells within 5 km.")
