"""
17_summary_and_map.py
Summary statistics table and site-location map.

Inputs:
  data/tsip_sites_cleaned.csv, data/ACLED_global.csv, output/panels/panel_*.parquet,
  data/ne_110m_admin_0_countries.zip (Natural Earth 110m countries; downloaded only if missing)
Outputs (+ Overleaf):
  output/tables/tab_summary_stats.tex  – 3-column panel summary table
  output/figures/fig_sites_map.pdf     – 3-panel map (Kenya / Ghana / Senegal)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
# TrueType (Type 42) fonts in the PDF keep a Unicode map, so the en dash in the legend
# year ranges survives text extraction (Type 3 fonts extracted it as a space).
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import urllib.request

from config import DATA, OUTPUT, FIGURES, TABLES, END_YR, ACLED_GLOBAL, save_to_overleaf

FIGS = str(FIGURES)
TABS = str(TABLES)

COUNTRIES = ['Kenya', 'Ghana', 'Senegal']

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(x, dp=3):
    if pd.isna(x): return '--'
    return f'{x:,.{dp}f}'

def fmt0(x):
    if pd.isna(x): return '--'
    return f'{int(round(x)):,}'


# ── 1. Load data ──────────────────────────────────────────────────────────────

tsip  = pd.read_csv(str(DATA / 'tsip_sites_cleaned.csv'))
acled = pd.read_csv(str(ACLED_GLOBAL), low_memory=False)
acled = acled[acled['year'] <= END_YR]

panels = {}
for c in COUNTRIES:
    fname = f'panel_{c.lower()}.parquet'
    panels[c] = pd.read_parquet(str(OUTPUT / 'panels' / fname))

# Site subsets
bat_sites = {c: tsip[(tsip.country == c) &
                      tsip.SiteIndustry.str.contains('Battery', case=False, na=False)]
             for c in COUNTRIES}
nl_sites  = {c: tsip[(tsip.country == c) &
                      ~tsip.KeyPollutant.str.contains('Lead', case=False, na=False) &
                      ~tsip.SiteIndustry.str.contains('Battery', case=False, na=False)]
             for c in COUNTRIES}


# ── 2. Summary statistics table ───────────────────────────────────────────────

def make_summary_table():
    rows = []    # (label, [Kenya_val, Ghana_val, Senegal_val], fmt)

    # ── Panel A: Sample structure ──
    rows.append(('PANEL_A', 'Panel A: Sample Structure', None))
    rows.append(('cells',       'Grid cells',          [fmt0(panels[c].cell_id.nunique()) for c in COUNTRIES]))
    rows.append(('obs',         'Cell$\\times$year obs.', [fmt0(len(panels[c])) for c in COUNTRIES]))
    rows.append(('years',       'Years covered',       [f'1997--{END_YR}'] * 3))
    rows.append(('bat_sites',   'Battery recycling sites (TSIP)', [fmt0(len(bat_sites[c])) for c in COUNTRIES]))
    rows.append(('nl_sites',    'Non-lead comparison sites (TSIP)', [fmt0(len(nl_sites[c])) for c in COUNTRIES]))

    # ── Panel B: Treatment intensity (cross-sectional, per cell) ──
    rows.append(('PANEL_B', 'Panel B: Distance to Battery Sites and Clusters', None))

    def ring_share(c, lo, hi):
        sch = panels[c].groupby('cell_id').first().reset_index()
        d   = np.where(sch.inv_dist_battery > 0, 1.0/sch.inv_dist_battery - 1, np.nan)
        n_ring = ((d > lo) & (d <= hi)).sum()
        return f'{int(n_ring):,}'

    rows.append(('cells_5km',   'Cells within 5 km of battery site',
                 [ring_share(c, 0, 5)  for c in COUNTRIES]))
    rows.append(('cells_10km',  'Cells 5--10 km from battery site',
                 [ring_share(c, 5, 10) for c in COUNTRIES]))
    rows.append(('cells_20km',  'Cells 10--20 km from battery site',
                 [ring_share(c, 10, 20) for c in COUNTRIES]))

    # Non-lead comparison cells, same 5 km rule as the main design: 1/(d+1) > 1/6
    def n_near(c, both=False):
        sch = panels[c].groupby('cell_id').first().reset_index()
        k = sch.inv_dist_nonlead > 1.0 / 6.0
        if both:
            k &= sch.inv_dist_battery > 1.0 / 6.0
        return f'{int(k.sum()):,}'
    rows.append(('cells_nl5',   'Cells within 5 km of a non-lead site',
                 [n_near(c) for c in COUNTRIES]))
    rows.append(('cells_both5', 'Cells within 5 km of both',
                 [n_near(c, True) for c in COUNTRIES]))

    # 1-degree blocks: the clusters for every standard error in the paper
    def blocks(c, treated_only=False):
        sch = panels[c].groupby('cell_id').first().reset_index()
        if treated_only:
            sch = sch[sch.inv_dist_battery > 1.0 / 6.0]          # d < 5 km
        b = np.floor(sch.cell_lat).astype(int).astype(str) + '_' + np.floor(sch.cell_lon).astype(int).astype(str)
        return f'{b.nunique():,}'
    rows.append(('blocks',      '1\\textdegree{} blocks (clusters)',
                 [blocks(c) for c in COUNTRIES]))
    rows.append(('blocks_treat', '\\quad of which contain a cell within 5 km of a battery site',
                 [blocks(c, True) for c in COUNTRIES]))

    # ── Panel C: Pre-period means (1997–2006) per cell-year ──
    rows.append(('PANEL_C', 'Panel C: Pre-Period Means, 1997--2006 (per cell$\\times$year)', None))
    for label, col in [('Riots', 'n_riots'), ('Protests', 'n_protests'),
                       ('Organised violence', 'n_org_violence')]:
        vals = []
        for c in COUNTRIES:
            pre = panels[c][panels[c].year < 2007][col]
            vals.append(f'{pre.mean():.3f}')
        rows.append((col + '_pre', label, vals))

    # ── Panel D: Post-period means (2007–END_YR) per cell-year ──
    rows.append(('PANEL_D', f'Panel D: Post-Period Means, 2007--{END_YR} (per cell$\\times$year)', None))
    for label, col in [('Riots', 'n_riots'), ('Protests', 'n_protests'),
                       ('Organised violence', 'n_org_violence')]:
        vals = []
        for c in COUNTRIES:
            post = panels[c][panels[c].year >= 2007][col]
            vals.append(f'{post.mean():.3f}')
        rows.append((col + '_post', label, vals))

    # ── Panel E: cell-years with no events ──
    # The share of cell-years with no events (and of cells with no events in any
    # year) is the central feature of the outcome distribution: it is why the
    # main specification is the extensive margin, why fixed-effects Poisson is
    # unstable (it discards the always-zero cells), and why quantile methods are
    # not identified below the zero share.
    rows.append(('PANEL_E', 'Panel E: Cell-Years with No Events', None))
    for label, col in [('Riots', 'n_riots'), ('Protests', 'n_protests'),
                       ('Organised violence', 'n_org_violence')]:
        vals = [f'{100*(panels[c][col]==0).mean():.1f}\\%' for c in COUNTRIES]
        rows.append((col + '_cyzero', 'Cell-years with zero ' + label.lower(), vals))
    # Share of cells with zero riots in ALL years (the cells FE Poisson drops)
    vals = []
    for c in COUNTRIES:
        ever = panels[c].groupby('cell_id')['n_riots'].sum()
        vals.append(f'{100*(ever==0).mean():.1f}\\%')
    rows.append(('cells_zero_riot', 'Cells with no riot in any year', vals))

    # ── Build LaTeX ──
    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\caption{Summary Statistics}')
    tex.append(r'\label{tab:summary_stats}')
    tex.append(r'\begin{threeparttable}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lccc}')
    tex.append(r'\toprule')
    tex.append(r' & Kenya & Ghana & Senegal \\')
    tex.append(r'\midrule')

    for key, label, vals in rows:
        if key.startswith('PANEL_'):
            tex.append(r'\addlinespace[4pt]')
            tex.append(f'\\textit{{{label}}} \\\\[2pt]')
        else:
            tex.append(f'\\quad {label} & {vals[0]} & {vals[1]} & {vals[2]} \\\\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\setstretch{1}\footnotesize')
    # Table note lives in main.tex (\setTabNote{tab_summary_stats}{...}).
    tex.append(r'\item \TabNote{tab_summary_stats}')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{threeparttable}')
    tex.append(r'\end{table}')

    tex_str = '\n'.join(tex)
    path = f'{TABS}/tab_summary_stats.tex'
    with open(path, 'w') as f:
        f.write(tex_str)
    save_to_overleaf('tab_summary_stats.tex', path, subfolder='tables')
    print('  Saved: tab_summary_stats.tex')


# ── 3. Map ────────────────────────────────────────────────────────────────────

NE_URL = 'https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip'
NE_ZIP = DATA / 'ne_110m_admin_0_countries.zip'


def get_world_boundaries():
    """Return Natural Earth 110m country boundaries, read from the copy in data/.
    Downloads it only if that copy is missing."""
    if not NE_ZIP.exists():
        print(f'  {NE_ZIP.name} not in data/; downloading from Natural Earth...')
        urllib.request.urlretrieve(NE_URL, str(NE_ZIP))
    return gpd.read_file(f'zip://{NE_ZIP}!ne_110m_admin_0_countries.shp')


# bounding boxes for each country panel (lon_min, lon_max, lat_min, lat_max)
BOUNDS = {
    'Kenya':   (33.5, 42.5, -5.5, 5.5),
    'Ghana':   (-4.0,  1.5,  4.0, 11.5),
    'Senegal': (-18.0, -11.0, 11.0, 16.5),
}


def make_map(world):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    for ax, country in zip(axes, COUNTRIES):
        lon_min, lon_max, lat_min, lat_max = BOUNDS[country]

        # Country + neighbours
        sub = world.cx[lon_min:lon_max, lat_min:lat_max]
        target = world[world['NAME'] == country]

        sub.plot(ax=ax, color='#f5f5f0', edgecolor='#888', linewidth=0.5)
        target.plot(ax=ax, color='#e8f4e8', edgecolor='#555', linewidth=1.0)

        # ACLED riot events (pre-treat lighter, post darker)
        acled_c = acled[acled['country'] == country].copy()
        acled_c = acled_c[acled_c['longitude'].between(lon_min, lon_max) &
                          acled_c['latitude'].between(lat_min, lat_max)]
        riots_pre  = acled_c[(acled_c['year'] < 2007) &
                              (acled_c['event_type'] == 'Riots')]
        riots_post = acled_c[(acled_c['year'] >= 2007) &
                              (acled_c['event_type'] == 'Riots')]

        if len(riots_pre) > 0:
            ax.scatter(riots_pre['longitude'], riots_pre['latitude'],
                       s=7, color='#808080', alpha=0.8, linewidths=0,
                       zorder=2, label='Riot pre-2007')
        if len(riots_post) > 0:
            ax.scatter(riots_post['longitude'], riots_post['latitude'],
                       s=8, color='#2244bb', alpha=0.6, linewidths=0,
                       zorder=3, label='Riot 2007+')

        # Non-lead sites (grey squares behind)
        nl = nl_sites[country]
        if len(nl) > 0:
            ax.scatter(nl['longitude'], nl['latitude'],
                       s=35, marker='s', color='#999999',
                       edgecolors='#666', linewidths=0.5, alpha=0.8,
                       zorder=4, label='Non-lead site')

        # Battery sites (red circles, larger)
        bat = bat_sites[country]
        if len(bat) > 0:
            ax.scatter(bat['longitude'], bat['latitude'],
                       s=55, marker='o', color='#cc2222',
                       edgecolors='#800000', linewidths=0.6, alpha=0.9,
                       zorder=5, label='Battery site')


        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(country, fontsize=16, fontweight='bold', pad=6)
        ax.set_aspect('equal')
        ax.set_axis_off()   # a map needs no lat/lon ticks

    # Shared legend below all panels
    legend_elements = [
        mpatches.Patch(facecolor='#cc2222', edgecolor='#800000',
                       label='Battery recycling site'),
        mpatches.Patch(facecolor='#999999', edgecolor='#666',
                       label='Non-lead comparison site'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2244bb',
                   markersize=6, alpha=0.9, label=f'Riot event, 2007–{END_YR}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080',
                   markersize=5, alpha=0.9, label='Riot event, 1997–2006'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    pdf_path = f'{FIGS}/fig_sites_map.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    save_to_overleaf('fig_sites_map.pdf', pdf_path, subfolder='figures')
    plt.close()
    print('  Saved: fig_sites_map')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Summary Stats and Map ===\n')

    print('Building summary stats table...')
    make_summary_table()

    print('Building map...')
    world = get_world_boundaries()
    make_map(world)

    print('\nDone.')
