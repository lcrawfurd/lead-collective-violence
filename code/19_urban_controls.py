#!/usr/bin/env python3
"""
19_urban_controls.py — Time-varying urbanicity controls (population, night lights) per cell
==========================================================================================
Sources
  1. WorldPop (data/worldpop/): population per 0.1° cell
     Years: 2000, 2005, 2007, 2010, 2015, 2020 → linearly interpolated annual
  2. Li et al. (2020) harmonised DMSP+VIIRS nighttime lights, figshare 9828827,
     1997–2024, as Africa-clipped GeoTIFFs in data/nightlights/li_harmonised/
     (ntl_{year}_africa.tif), aggregated to 0.1° cell means; 2025 carried forward.

Inputs
  output/panels/panel_{kenya,ghana,senegal}.parquet (01_build_panels.py)
  data/worldpop/*.tif, data/nightlights/li_harmonised/ntl_{year}_africa.tif

Output
  output/panels/urban_controls.parquet — cell_id × year × log_pop_cell × log_ntl
  (read by 20, 33, 35, 36, 57 and 60)

Offline by default: the local night-light files are used first. The figshare/Zenodo
APIs are queried, and a global GeoTIFF downloaded and clipped, only for a year whose
clipped file is missing.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
CODE_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import config  # noqa: F401  (creates output/panels/)
ROOT       = CODE_DIR.parent
DATA       = ROOT / "data"
WP_DIR     = DATA / "worldpop"
NTL_DIR    = DATA / "nightlights" / "li_harmonised"
PANELS_DIR = ROOT / "output" / "panels"
NTL_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = {
    "KE": ("ken", "panel_kenya.parquet"),
    "GH": ("gha", "panel_ghana.parquet"),
    "SN": ("sen", "panel_senegal.parquet"),
}

WORLDPOP_YEARS = [2000, 2005, 2007, 2010, 2015, 2020]
PANEL_YEARS    = list(range(1997, 2026))
HALF           = 0.05   # half-width of 0.1° cell

# Bounding box covering all three countries with buffer
AFRICA_BBOX = (-18.0, -6.0, 43.0, 17.0)   # (west, south, east, north)

# Years covered by the Li et al. series used here (the figshare article lists 1992-2024).
NTL_YEARS = [y for y in PANEL_YEARS if y <= 2024]

# Li et al. figshare article ID + Zenodo for 2019-2023 extension
FIGSHARE_ARTICLE = "9828827"
# Files on figshare are named Harmonized_DN_NTL_{year}_calDMSP.tif (1992-2013)
# and Harmonized_DN_NTL_{year}_simVIIRS.tif (2014-2018)
# Extension (Chen et al. 2023) covers 2019-2023 from Zenodo record 7750175
ZENODO_RECORD = "7750175"


# ── 1. WorldPop extraction ──────────────────────────────────────────────────
def extract_worldpop(code, iso3):
    print(f"  WorldPop {code}...")
    fmap = {"KE": "panel_kenya.parquet", "GH": "panel_ghana.parquet",
            "SN": "panel_senegal.parquet"}
    panel = pd.read_parquet(PANELS_DIR / fmap[code])
    cells = panel[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates()

    pop_records = []
    for wp_year in WORLDPOP_YEARS:
        tif = WP_DIR / f"{iso3}_ppp_{wp_year}_1km_Aggregated_UNadj.tif"
        if not tif.exists():
            print(f"    skip {tif.name}")
            continue
        with rasterio.open(tif) as src:
            nodata = src.nodata
            for _, row in cells.iterrows():
                win = from_bounds(
                    row.cell_lon - HALF, row.cell_lat - HALF,
                    row.cell_lon + HALF, row.cell_lat + HALF,
                    src.transform,
                )
                data = src.read(1, window=win).astype(float)
                if nodata is not None:
                    data[data == nodata] = np.nan
                data[data < 0] = np.nan
                pop_records.append({"cell_id": row.cell_id,
                                    "wp_year": wp_year,
                                    "pop": np.nansum(data)})

    pop_df = pd.DataFrame(pop_records)
    rows = []
    for cell_id, grp in pop_df.groupby("cell_id"):
        grp = grp.sort_values("wp_year")
        xp  = grp["wp_year"].to_numpy()
        fp  = grp["pop"].to_numpy()
        interp = np.interp(PANEL_YEARS, xp, fp,
                           left=float(fp[0]), right=float(fp[-1]))
        for yr, p in zip(PANEL_YEARS, interp):
            rows.append({"cell_id": cell_id, "year": yr,
                         "pop_cell": max(p, 0)})

    annual = pd.DataFrame(rows)
    annual["log_pop_cell"] = np.log1p(annual["pop_cell"])
    return annual[["cell_id", "year", "log_pop_cell"]]


# ── 2. Nighttime lights: Li et al. download + extraction ───────────────────

def _figshare_file_list(article_id):
    """Return list of {year, url, fname} dicts from figshare API.

    Li et al. (2020) extended dataset on figshare now covers 1992-2024
    (including the VIIRS extension formerly on Zenodo). Use page_size=100
    to retrieve all files (default page size is 10).
    Only 'Harmonized_' files are included to avoid the duplicate
    non-harmonized DN_NTL_2013_simVIIRS.tif entry.
    """
    import requests
    url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    r = requests.get(url, timeout=30, params={"page_size": 100})
    r.raise_for_status()
    files = r.json()
    result = []
    for f in files:
        name = f["name"]
        if not name.endswith(".tif"):
            continue
        if not name.startswith("Harmonized_"):
            continue
        # Extract year from filename e.g. Harmonized_DN_NTL_2007_calDMSP.tif
        parts = name.replace(".tif", "").split("_")
        year_str = [p for p in parts if p.isdigit() and len(p) == 4]
        if not year_str:
            continue
        yr = int(year_str[0])
        result.append({"year": yr, "url": f["download_url"], "fname": name})
    return result


def _zenodo_file_list(record_id):
    """Return list of {year, url, fname} for Zenodo extension (2019-2023)."""
    import requests
    url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    result = []
    for f in data.get("files", []):
        name = f["key"]
        if not name.endswith(".tif"):
            continue
        parts = name.replace(".tif", "").split("_")
        year_str = [p for p in parts if p.isdigit() and len(p) == 4]
        if not year_str:
            continue
        yr = int(year_str[0])
        if 2019 <= yr <= 2025:
            result.append({"year": yr,
                           "url": f["links"]["self"],
                           "fname": name})
    return result


def _download_and_clip(url, fname, year):
    """Download global GeoTIFF, clip to Africa bbox, save locally."""
    clipped = NTL_DIR / f"ntl_{year}_africa.tif"
    if clipped.exists():
        return clipped

    import requests
    from tqdm import tqdm
    raw = NTL_DIR / fname
    if not raw.exists():
        print(f"    Downloading NTL {year}...")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(raw, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=f"NTL {year}", leave=False) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    # Clip to Africa bbox using rasterio window
    west, south, east, north = AFRICA_BBOX
    with rasterio.open(raw) as src:
        win = from_bounds(west, south, east, north, src.transform)
        data = src.read(1, window=win)
        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update(width=data.shape[1], height=data.shape[0],
                       transform=transform)
        with rasterio.open(clipped, "w", **profile) as dst:
            dst.write(data, 1)

    raw.unlink()   # delete global file after clipping
    print(f"    Clipped → {clipped.name}  ({clipped.stat().st_size // 1e6:.0f} MB)")
    return clipped


def extract_ntl_annual(all_cells):
    """Mean NTL per cell per year, from the local clipped files; download only missing years."""
    file_list = [{"year": yr, "clipped": NTL_DIR / f"ntl_{yr}_africa.tif"} for yr in NTL_YEARS]
    missing = [e["year"] for e in file_list if not e["clipped"].exists()]
    if missing:
        print(f"  NTL years missing locally: {missing}; fetching file list online...")
        online = _figshare_file_list(FIGSHARE_ARTICLE) + _zenodo_file_list(ZENODO_RECORD)
        by_year = {}
        for o in online:
            by_year.setdefault(o["year"], o)
        for yr in missing:
            if yr not in by_year:
                raise SystemExit(f"19_urban_controls: no night-light file for {yr} locally or online.")
            _download_and_clip(by_year[yr]["url"], by_year[yr]["fname"], yr)
    print(f"  Using {len(file_list)} NTL years "
          f"({file_list[0]['year']}–{file_list[-1]['year']}) from {NTL_DIR}")

    ntl_records = []
    for entry in file_list:
        yr, clipped = entry["year"], entry["clipped"]
        with rasterio.open(clipped) as src:
            nodata = src.nodata
            for _, row in all_cells.iterrows():
                win = from_bounds(
                    row.cell_lon - HALF, row.cell_lat - HALF,
                    row.cell_lon + HALF, row.cell_lat + HALF,
                    src.transform,
                )
                data = src.read(1, window=win).astype(float)
                if nodata is not None:
                    data[data == nodata] = np.nan
                data[data < 0] = np.nan
                ntl_records.append({"cell_id": row.cell_id, "year": yr,
                                    "ntl": np.nanmean(data)
                                    if data.size > 0 else np.nan})

    if not ntl_records:
        return None

    ntl_df = pd.DataFrame(ntl_records)
    ntl_df["log_ntl"] = np.log1p(ntl_df["ntl"].clip(lower=0))

    # Extend to full panel years by forward/back-filling within each cell
    full = (pd.MultiIndex.from_product(
                [ntl_df.cell_id.unique(), PANEL_YEARS],
                names=["cell_id", "year"])
            .to_frame(index=False))
    ntl_df = full.merge(ntl_df, on=["cell_id", "year"], how="left")
    ntl_df = (ntl_df.sort_values(["cell_id", "year"])
                    .groupby("cell_id", group_keys=False)
                    .apply(lambda g: g.ffill().bfill()))
    return ntl_df[["cell_id", "year", "log_ntl"]]


# ── 3. Build controls parquet ───────────────────────────────────────────────
def build_controls():
    print("=== Building urban controls ===")

    fmap = {"KE": "panel_kenya.parquet", "GH": "panel_ghana.parquet",
            "SN": "panel_senegal.parquet"}

    # WorldPop
    wp_parts = []
    for code, (iso3, _) in COUNTRIES.items():
        wp_parts.append(extract_worldpop(code, iso3))
    wp_all = pd.concat(wp_parts, ignore_index=True)

    # All cell centroids for NTL extraction
    all_cells_list = []
    for code, (_, pfile) in COUNTRIES.items():
        p = pd.read_parquet(PANELS_DIR / pfile)
        all_cells_list.append(p[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates())
    all_cells = pd.concat(all_cells_list).drop_duplicates("cell_id")

    # NTL
    ntl_df = extract_ntl_annual(all_cells)

    # Merge
    controls = wp_all.copy()
    if ntl_df is not None:
        controls = controls.merge(ntl_df, on=["cell_id", "year"], how="left")
    else:
        controls["log_ntl"] = np.nan

    out = PANELS_DIR / "urban_controls.parquet"
    controls.to_parquet(out, index=False)
    years_ntl = controls.dropna(subset=["log_ntl"])["year"].nunique()
    print(f"✓  Saved {len(controls):,} rows, {years_ntl} NTL years → {out}")
    return controls


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_controls()
    print("\nDone.")
