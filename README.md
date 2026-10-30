# Replication package: Lead Exposure and Collective Violence: Evidence from Informal Battery Recycling in Africa

Lee Crawfurd, Center for Global Development (lcrawfurd@cgdev.org)

This package reproduces every table, figure and in-text number in the paper, and the
figures and numbers in the accompanying seminar slides. One script, `run_all.sh`, runs
everything in order. Every estimate and data count in the paper is a LaTeX macro written by
a script into a `*_values.tex` file. The only numbers typed into the text are design
settings stated there (years, distance cut-offs, bootstrap draws, matching ratios).

## Overview

| | |
|---|---|
| Language | Python 3 (all analysis) and R (one sensitivity analysis, called from Python) |
| Master script | `run_all.sh` |
| Run time | about 11 minutes on a MacBook Air (Apple M4, 16 GB, macOS); clean-room run of 28 October 2026 |
| Disk | 2.6 GB of input data; outputs under 20 MB |
| Randomness | Fixed seeds: randomisation inference 20260410, wild cluster bootstrap 20261014; `run_all.sh` sets PYTHONHASHSEED=2 |

## Data availability

Two of the sources, ACLED and DHS, cannot be redistributed. The replicator must download
them after free registration and place them at the paths below. One public report used for
a slide (UNICEF and Pure Earth 2020) is not included for copyright reasons and must be
downloaded. Everything else is included.

The author had legitimate access to every dataset used and permission to use it. Data
included in this package are redistributed under the licences listed below.

| Data | Source | File(s) in `data/` | Included? | Licence / access |
|---|---|---|---|---|
| Conflict events | ACLED, all countries, 1997–2024, one export downloaded through the ACLED API on 22 October 2026 by `code/00_download_acled.py` (counts in `data/ACLED_export_info.txt`) | `ACLED_global.csv`; `acled_kenya.csv` and `acled_ghana.csv` are its Kenya and Ghana rows | No | Free registration at acleddata.com; ACLED terms forbid redistribution of the raw event data |
| Contaminated sites | Pure Earth Toxic Sites Identification Program (TSIP), March 2026 | `tsip_sites_cleaned.csv` | Yes | Shared with permission of Pure Earth |
| Household surveys | Demographic and Health Surveys: Kenya 2003, 2008–09, 2014, 2022; Ghana 2003, 2008, 2014, 2022; Senegal 2005, 2010–11, 2015, 2019 (women's, children's and household-member recodes, Stata format, plus the GPS shapefile) | One folder per survey: `DHS/KE_2003_DHS_*`, `KE_2008-09_DHS_*`, `KE_2014_DHS_*`, `KE_2022_DHS`, `GH_2003`, `GH_2008`, `GH_2014`, `GH_2022`, `SN_2005`, `SN_2010-11`, `SN_2015`, `SN_2019`. Each must contain, at any depth, one `<CC>IR*FL.DTA`, `<CC>KR*FL.DTA`, `<CC>PR*FL.DTA` and `<CC>GE*.shp` (with its .dbf, .shx, .prj), where `<CC>` is KE, GH or SN | No | Free registration at dhsprogram.com, with a separate request for the GPS datasets; DHS terms forbid redistribution |
| Population | WorldPop, 1 km aggregated, UN-adjusted, 2000–2020 | `worldpop/` | Yes | CC BY 4.0 |
| Night lights | Li et al. (2020) harmonised DMSP–VIIRS (figshare 9828827), 1997–2024, clipped to Africa | `nightlights/li_harmonised/ntl_<year>_africa.tif` | Yes | CC BY 4.0 |
| Country boundaries (map) | Natural Earth 1:110m admin-0 countries | `ne_110m_admin_0_countries.zip` | Yes | Public domain |
| Vehicle sales | OICA new vehicle sales by country | `total_sales_2018.xlsx` | Yes | Published by OICA |
| Commodity prices | World Bank Commodity Price Data (Pink Sheet), annual, real 2010 US$ | `wb_lead_food_prices.csv` | Yes | CC BY 4.0 |
| Blood lead near battery sites | Crawfurd et al. (2026), *Beyond Hotspots* replication files: soil samples, AALM lookup table, ring estimates | `beyond_hotspots/` | Yes | Author's own |
| Background blood lead | IHME Global Burden of Disease 2023 | `beyond_hotspots/IHME_GBD_2023_LEAD_RISK_*.CSV` | Yes | IHME free-of-charge non-commercial licence |
| Published effects of battery sites by distance (Figure A10b) | Ipapa (2026), Berkhout et al. (2025), Litzow et al. (2026), Kundu, Mahzab and Plambeck (2026); the table or figure for each value is in the file. The Mexico and Bangladesh values are read from the papers' figures (Mexico from the PDF's vector graphics, Bangladesh digitised from the image) | `published_distance_effects.csv` | Yes | Author's compilation |
| Published riot and conflict effect sizes (Section 6 benchmarks) | McGuirk and Burke (2020), Hsiang, Burke and Miguel (2013); citation and table location for each value in the file | `literature_benchmarks.csv` | Yes | Author's compilation |
| Slides only: causal lead studies | Author's literature review | `lead_review_studies.csv` | Yes | Author's own |
| Slides only: child blood lead by region | UNICEF and Pure Earth (2020), *The Toxic Truth*; World Bank population aged 0–19, 2019 | `toxic_truth_unicef_pure_earth_2020.pdf` (not included: download from https://www.unicef.org/reports/toxic-truth-childrens-exposure-to-lead-pollution-2020 and save under this name), `wb_pop0_19_2019.csv` (included) | Report: No. Population: Yes | UNICEF copyright; World Bank data CC BY 4.0 |

`data/` holds raw inputs only. Everything derived from them, including the cell-year
population and night-light controls (`urban_controls.parquet`, from `19_urban_controls.py`),
is written to `output/panels/`. The pipeline makes no network call when `data/` is
complete: `17`, `19` and `45` read their local files and go online only if a file is
missing.

ACLED revises its data between downloads, so a new download will not match this export
exactly. To reproduce the paper's numbers exactly, ask ACLED for the data as of 22 October
2026, or contact the author.

## Sample period

The sample runs from 1997 to 2024 (`END_YR` in `code/config.py`). ACLED's standard access ends 12 months before the download date, so 2025 is incomplete and is excluded.

`code/00_download_acled.py` downloads one ACLED export for all countries and writes every ACLED file the pipeline reads, so all scripts use the same data version. It needs an ACLED account (credentials read from the environment variables `ACLED_EMAIL` and `ACLED_PASSWORD`) and is not part of `run_all.sh`.

## Instructions

1. Install Python 3.12 or later and the packages in `requirements.txt` (exact versions;
   tested with Python 3.14.0), preferably in a fresh virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Install R 4.3 or later with `HonestDiD`, `Matrix`, `ggplot2` and `patchwork` (tested
   with R 4.5.1, HonestDiD 0.2.8, Matrix 1.7.3, ggplot2 4.0.1). R is required: step `47`
   calls it, and the run stops if it is missing.
   ```bash
   Rscript -e 'install.packages(c("HonestDiD","Matrix","ggplot2","patchwork"), repos="https://cloud.r-project.org")'
   ```
3. Install poppler, which provides `pdftotext` (used by `45`): `brew install poppler` on
   macOS, `apt install poppler-utils` on Debian or Ubuntu.
4. Run `code/00_download_acled.py` (see Sample period), place the DHS files at the paths in
   the table above (inside the folder set by `PUBLIC_DATA` in `code/config.py`, or the
   `PUBLIC_DATA` environment variable), and download the UNICEF report. `run_all.sh` stops at the first missing
   input, so all three are needed for a complete run.
5. Run, from the package root:
   ```bash
   NO_OVERLEAF=1 bash run_all.sh
   ```
   `NO_OVERLEAF=1` keeps every output inside `output/`. Without it, the scripts also copy
   outputs into the author's Overleaf folder if it exists on the machine.

Outputs go to `output/tables/` (LaTeX tables and the `*_values.tex` macro files),
`output/figures/` and `output/panels/` (intermediate data). Every table, figure and macro
the scripts write is used by the paper or the slides, with one exception:
`fig_slides_price_raw.pdf` (`59`), made for the slides but not yet in them. Every script
stops with an error if an input it needs is missing; none falls back to a stored number.
`01_build_panels.py` also stops if the sample differs from the paper's (2,995 cells,
83,860 cell-years, 30 cells within 5 km of a battery site).

## Exhibits and the scripts that produce them

Exhibit numbers are those in the paper. Unnumbered exhibits are in the slides only.

| Exhibit | Content | File | Script |
|---|---|---|---|
| Table 1 | Sample overview by country | in `main.tex`, from macros | `11_generate_values.py` |
| Figure 1 | Motorisation and the world lead price | `motorisation_combined.pdf` | `04_motorisation_figure.py` |
| Figure 2 | Map of sites and riots | `fig_sites_map.pdf` | `17_summary_and_map.py` |
| Table 2 | Summary statistics | `tab_summary_stats.tex` | `17_summary_and_map.py` |
| Figure 3 | Share of cells with any riot, by proximity | `fig_raw_rates.pdf` | `61_raw_rates_figure.py` |
| Table 3 | The effect in three differences | `tab_escalating_diff.tex` | `23_vehicle_dose_main.py` |
| Table 4 | Main results by country and conflict type | `tab_main_binary.tex` | `55_binary_main_table.py` |
| Figure 4 | Lead-specific gap by year | `fig_ddd_gap_event.pdf` | `47_slides_binary.py` |
| Figure 5 | Lead-specific gap by distance from the site | `fig_rings_gap_pooled.pdf` | `51_slides_rings_pooled.py` |
| Table 5 | Lead-specific gap by riot trigger | `tab_trigger.tex` | `36_trigger_coding.py` |
| Figure A1 | Distribution of riots per cell-year | `fig_outcome_distribution.pdf` | `43_outcome_distribution.py` |
| Table A1 | Alternative outcome transformations | `tab_alt_estimators.tex` | `05_robustness.py` |
| Table A2 | Standard errors under different clustering choices | `tab_ppml_conley.tex` | `12_ppml_conley.py` |
| Figure A2 | Randomisation inference | `fig_randomisation_inference.png` | `10_randomisation_inference.py` |
| Table A3 | Leave-one-out | `tab_leave_one_out.tex` | `37_leave_one_out.py` |
| Figure A3 | Event studies by country and conflict type | `fig_slides_dd_event.pdf` | `47_slides_binary.py` |
| Table A4 | Pre-trend tests | `tab_pretrends.tex` | `47_slides_binary.py` |
| Figures A4, A5 | Honest DiD sensitivity, Kenya and pooled gap | `fig_slides_honest_rm.pdf`, `fig_slides_honest_sd.pdf`, `fig_honest_did_gap.pdf` | `47_slides_binary.py` (calls `31_honest_did_extensive.R`) |
| Figure A6 | Onset-year sweep | `fig_cutoff_sweep.pdf` | `32_cutoff_sweep.py` |
| Table A5, Figure A7 | Vehicle-dose treatment | `tab_vehicle_dose.tex`, `fig_vehicle_event.pdf` | `23_vehicle_dose_main.py` |
| Figure A8 | The non-lead comparison sites | `fig_slides_sites_desc.pdf` | `50_slides_descriptives.py` |
| Table A6 | Comparison group and sample robustness | `tab_comparison.tex` | `57_comparison_table.py` |
| Table A7 | Placebo test: fake onset in 2002 | `tab_placebo.tex` | `58_placebo_table.py` |
| Table A8 | Changes in other outcomes after 2007 | `tab_balance_changes.tex` | `60_balance_table.py` (DHS rows from `54_dhs_outcomes.py`) |
| Figure A9, Table A9 | Lead and food prices | `fig_prices.pdf`, `tab_price_placebo.tex` | `38_price_placebo.py` |
| Figure A10 | Soil lead and published effects by distance | `ulab_soil_decay.pdf`; `fig_published_effects.pdf` | `48_slides_soil_decay.py`; `62_published_effects.py` |
| Figure A11 | Distance rings by country | `fig_binary_distance_extensive.pdf` | `30_extensive_margin_figures.py` |
| Figure A12 | Event study by distance band | `eventstudy_by_band.png` | `02_did_analysis.py` |
| Figure A13 | Event-level distance rings | `fig_site_rings.pdf` | `56_site_rings.py` |
| Table A10 | Continuous proximity | `tab_main_extensive.tex` | `29_extensive_margin_main.py` |
| Figure A14 | Binned dose-response | `callaway_binned_scatter.png` | `02_did_analysis.py` |
| Figure A15 | Identifying weights | `fig_callaway_weights.pdf` | `22_callaway_weights.py` |
| Table A11 | Sub-event types | `tab_sub_events.tex` | `55_binary_main_table.py` |
| Tables A12, A13 | Trigger composition and robustness | `tab_trigger_comp.tex`, `tab_trigger_robust.tex` | `36_trigger_coding.py` |
| Table A14 | Remediation break-even | `tab_remediation.tex` | `41_remediation_threshold.py` |
| Table A15 | Effect-size plausibility | `tab_plausibility.tex` | `44_plausibility.py` |
| In-text numbers | All macros | `*_values.tex` | written by the script named in each file's first line |
| Slides only | Slide figures | `fig_slides_*.pdf`, `fig_gap_map.pdf`, `fig_acled_africa.pdf`, `fig_lead_burden.pdf` | `18`, `45`, `46`, `47`, `49`, `50`, `51`, `56`, `59` |

## Order of scripts

`run_all.sh` runs the scripts in this order, because later scripts read earlier outputs:

1. `01_build_panels.py` builds the cell-year panels from TSIP and ACLED.
2. `19_urban_controls.py` adds population and night lights by cell and year.
3. Descriptives and slide exhibits: `04`, `17`, `18`, `45`, `46`, `43`.
4. Main results: `02`, `29` (writes `botec_extensive.csv`), `30`, `23`, `36` (writes
   `trigger_classified.csv`, read by `47`).
5. Inference and robustness: `05`, `10`, `12`, `22`, `28`, `32`, `33`, `35`, `37`, `38`,
   `47` (calls `31_honest_did_extensive.R`), `48`, `49`, `50`, `51`, `54` (writes
   `dhs_outcomes.csv`, read by `60`), `55`, `56`, `57`, `58`, `59`, `60`, `61`, `62`.
6. `11_generate_values.py`, then `41` (reads `generated_values.tex`) and `44` (reads
   `binary_main_values.tex`).

Script numbers have gaps because scripts for analyses dropped from the paper are not part
of this package.

## Software

`requirements.txt` pins every Python package version. Key packages: pandas 2.3.3,
numpy 2.4.1, pyfixest 0.50.1, geopandas 1.1.2, rasterio 1.5.0, matplotlib 3.10.8,
scipy 1.17.0. Standard errors are clustered on 1° spatial blocks using pyfixest's CRV1
estimator. The wild cluster bootstrap is implemented in `code/utils.py`.

## References

### Data

ACLED (2026). Armed Conflict Location & Event Data, all countries, 1997–2024. Downloaded 22 October 2026 from https://acleddata.com.

ICF (2003–2022). Demographic and Health Surveys: Ghana 2003, 2008, 2014, 2022; Kenya 2003, 2008–09, 2014, 2022; Senegal 2005, 2010–11, 2015, 2019 [datasets]. Rockville, MD: ICF. https://dhsprogram.com.

Pure Earth (2026). Toxic Sites Identification Program database, March 2026 extract. https://contaminatedsites.org.

WorldPop (2018). Global high resolution population denominators, 1 km, UN-adjusted, 2000–2020. University of Southampton. https://doi.org/10.5258/SOTON/WP00671.

Natural Earth (2024). 1:110m Cultural Vectors, Admin 0 – Countries. https://www.naturalearthdata.com.

OICA (2019). Sales of new vehicles 2005–2018. International Organization of Motor Vehicle Manufacturers. https://www.oica.net/category/sales-statistics/.

World Bank (2026). Commodity Price Data (The Pink Sheet), annual prices in real 2010 US dollars. https://www.worldbank.org/en/research/commodity-markets.

World Bank (2026). World Development Indicators: population aged 0–14 and 15–19, 2019. https://data.worldbank.org.

Institute for Health Metrics and Evaluation (2025). Global Burden of Disease Study 2023: lead exposure, blood lead levels 1990–2023. Seattle: IHME. https://ghdx.healthdata.org.

UNICEF and Pure Earth (2020). *The Toxic Truth: Children's Exposure to Lead Pollution Undermines a Generation of Future Potential*. New York: UNICEF.

### Methods and papers


Li, X., Zhou, Y., Zhao, M. and Zhao, X. (2020). A harmonized global nighttime light dataset 1992–2018. *Scientific Data* 7, 168.

Raleigh, C., Linke, A., Hegre, H. and Karlsen, J. (2010). Introducing ACLED: An Armed Conflict Location and Event Dataset. *Journal of Peace Research* 47(5), 651–660.

## Licence

Code: MIT (see `LICENSE`). Data in `data/` keep the licences listed under Data availability.
