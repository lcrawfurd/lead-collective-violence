"""
02_did_analysis.py
FIGURES: two dose-response diagnostics in the appendix (the Callaway et al. checks
on the continuous treatment).

Requires panels built by 01_build_panels.py.

Outputs (output/figures/, copied to Overleaf figures/):
  - callaway_binned_scatter.png  post-minus-pre change in Pr(any riot) by quintile of 1/(d+1)
  - eventstudy_by_band.png       event study by distance band (0-5, 5-10, 10-20 km vs >20 km),
                                 pooled Kenya, Ghana, Senegal, cell + country-year FE,
                                 CIs from SEs clustered on 1-degree blocks
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
from config import FIGURES, PANELS, OVERLEAF_FIGS
from utils import add_block, tcrit
PANEL_DIR = str(PANELS)
FIGS      = str(FIGURES)


def _fig_dests(name):
    return [FIGS, OVERLEAF_FIGS] if OVERLEAF_FIGS else [FIGS]


COUNTRIES = ["Kenya", "Ghana", "Senegal"]

# ── Callaway et al. diagnostic plots ──────────────────────────────────────────

def plot_binned_scatter(panels_dict, post_year, fname):
    """
    Callaway et al. check 1: non-parametric dose-response.
    Plot the post-minus-pre change in the probability of any riot in a cell-year
    (cell mean of 1[any riot] over 2007-2024 minus 1997-2006) against 1/(d+1),
    binned into quintiles of 1/(d+1), for each country. Bars: 95% CI of the bin mean
    (1.96 x SD / sqrt(cells in bin)); dashed line: OLS fit across bin ranks.
    """
    fig, axes = plt.subplots(1, len(panels_dict), figsize=(5*len(panels_dict), 4),
                             sharey=False)
    if len(panels_dict) == 1:
        axes = [axes]
    for ax, (cname, panel) in zip(axes, panels_dict.items()):
        anyr = (panel["n_riots"] > 0).astype(float)
        pre  = anyr[panel["year"] < post_year].groupby(panel["cell_id"]).mean()
        post = anyr[panel["year"] >= post_year].groupby(panel["cell_id"]).mean()
        delta = (post - pre).dropna().reset_index()
        delta.columns = ["cell_id", "delta_log_riots"]
        meta = panel[["cell_id","inv_dist_battery"]].drop_duplicates()
        delta = delta.merge(meta, on="cell_id")
        # Bin into 5 quintiles of inv_dist_battery
        delta["quintile"] = pd.qcut(delta["inv_dist_battery"], 5, labels=False)
        binned = delta.groupby("quintile").agg(
            x=("inv_dist_battery", "mean"),
            y=("delta_log_riots", "mean"),
            se=("delta_log_riots", lambda v: v.std() / np.sqrt(len(v)))
        ).reset_index()
        # Plot on evenly-spaced quintile ranks so bunched x-values spread out
        ranks = np.arange(1, len(binned) + 1)
        ax.errorbar(ranks, binned["y"], yerr=1.96*binned["se"],
                    fmt="o", color="tab:blue", capsize=4, markersize=6)
        # OLS trend line across ranks
        m = np.polyfit(ranks, binned["y"], 1)
        ax.plot(ranks, np.polyval(m, ranks), color="tab:red", linewidth=1.2, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        # Label each tick with the mean 1/(d+1) for that quintile
        ax.set_xticks(ranks)
        ax.set_xticklabels([f"{v:.3f}" for v in binned["x"]], fontsize=8)
        ax.set_xlim(0.3, len(binned) + 0.7)
        ax.set_xlabel("Treatment-intensity quintile\n(mean $1/(d+1)$ per bin)")
        ax.set_ylabel("Post $-$ pre change in Pr(any riot)")
        ax.set_title(cname)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for dest in _fig_dests(fname):
        plt.savefig(os.path.join(dest, fname), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {fname}")


def plot_event_study_by_distance_band(countries_dict, post_year, fname):
    """
    Event study stratified by distance band (0-5, 5-10, 10-20 km vs >20 km reference).
    Uses pooled panel across all countries with country×year FE absorbed via within-group
    demeaning. Bands have a clear spatial interpretation: cells at increasing distance
    from the nearest battery recycling site.
    """
    import pyfixest as pf
    base_yr = post_year - 1

    # Build pooled panel
    parts = []
    for cname, panel in countries_dict.items():
        p = panel.copy()
        p["country"] = cname
        parts.append(p)
    pooled = pd.concat(parts, ignore_index=True)
    pooled["cell_id_str"] = pooled["country"] + "_" + pooled["cell_id"].astype(str)
    pooled["cty_year"] = pooled["country"] + "_" + pooled["year"].astype(str)
    add_block(pooled, "country")    # 1-degree blocks: the cluster for the plotted CIs

    # Compute distance in km from inv_dist_battery = 1/(d+1)
    pooled["dist_km"] = 1.0 / pooled["inv_dist_battery"].clip(lower=1e-6) - 1.0
    # Distance band labels and thresholds
    bands = [("0–5 km", 0, 5), ("5–10 km", 5, 10), ("10–20 km", 10, 20)]
    ref_label = ">20 km (ref.)"
    ref_mask = pooled["dist_km"] >= 20

    years = sorted(pooled["year"].unique())
    pre_years  = [y for y in years if y < post_year and y != base_yr]
    post_years = [y for y in years if y >= post_year]

    colours = ["tab:red", "tab:orange", "tab:blue"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    BIN_OUTCOMES = [("n_riots", "Riots"), ("n_protests", "Protests"),
                    ("n_org_violence", "Organised violence")]
    pooled["any_n_riots"] = (pooled["n_riots"] > 0).astype(float)
    pooled["any_n_protests"] = (pooled["n_protests"] > 0).astype(float)
    pooled["any_n_org_violence"] = (pooled["n_org_violence"] > 0).astype(float)
    ends = {}
    for i, (ax, (outcome_col, outcome_label)) in enumerate(zip(axes, BIN_OUTCOMES)):
        outcome_col = "any_" + outcome_col
        for (band_label, lo, hi), colour in zip(bands, colours):
            band_mask = (pooled["dist_km"] >= lo) & (pooled["dist_km"] < hi)
            n_cells = pooled[band_mask]["cell_id_str"].nunique()
            if n_cells < 3:
                continue
            # Use band cells + reference cells only
            sub = pooled[band_mask | ref_mask].copy()
            sub["band_treat"] = ((sub["dist_km"] >= lo) & (sub["dist_km"] < hi)).astype(float)
            # Event-study interactions: band_treat × year_dummy (omit base_yr)
            for y in years:
                if y == base_yr:
                    continue
                sub[f"bty{y}"] = sub["band_treat"] * (sub["year"] == y).astype(float)
            yvars = [f"bty{y}" for y in years if y != base_yr]
            try:
                fml = f"{outcome_col} ~ {' + '.join(yvars)} | cell_id_str + cty_year"
                m = pf.feols(fml, data=sub, vcov={"CRV1": "block"})
                # CIs: block-clustered SE x t(G-1) critical value
                se_c, cr = m.se(), tcrit(m)
                coefs = {int(v.replace("bty","")): m.coef()[v] for v in yvars if v in m.coef().index}
                ses   = {int(v.replace("bty","")): se_c[v]     for v in yvars if v in se_c.index}
                plot_years = sorted(coefs.keys()) + [base_yr]
                plot_coef  = [coefs.get(y, 0.0) for y in sorted(coefs.keys())] + [0.0]
                plot_se    = [ses.get(y,   0.0)  for y in sorted(coefs.keys())] + [0.0]
                plot_years_sorted = sorted(zip(plot_years, plot_coef, plot_se))
                ys_  = [t[0] for t in plot_years_sorted]
                cs_  = [t[1] for t in plot_years_sorted]
                ss_  = [t[2] for t in plot_years_sorted]
                ci_lo = [c - cr*s for c, s in zip(cs_, ss_)]
                ci_hi = [c + cr*s for c, s in zip(cs_, ss_)]
                ax.plot(ys_, cs_, marker="o", markersize=3, color=colour,
                        label=f"{band_label} (n={n_cells})")
                ends[(i, band_label)] = (ys_[-1], cs_[-1], colour, f"{band_label} ({n_cells} cells)")
                ax.fill_between(ys_, ci_lo, ci_hi, alpha=0.12, color=colour)
            except Exception as e:
                print(f"    Band {band_label} {outcome_label} failed: {e}")
                continue
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(post_year - 0.5, color="#888888", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(outcome_label)
        ax.set_xlabel("Year")
        if i == 0:
            ax.set_ylabel("Change in Pr(any event) vs cells >20 km")
    # Direct labels at the line ends of the right-hand panel, spread so they do not
    # overlap each other, with thin leader lines back to the line ends.
    ax = axes[-1]
    pts = sorted([v for (j, _), v in ends.items() if j == len(axes) - 1], key=lambda t: -t[1])
    lo_, hi_ = ax.get_ylim()
    gap = 0.075 * (hi_ - lo_)
    ypos, last = [], np.inf
    for x_, y_, col, lab in pts:
        y = min(y_, last - gap)
        ypos.append(y); last = y
    for (x_, y_, col, lab), y in zip(pts, ypos):
        ax.annotate(lab, xy=(x_, y_), xytext=(x_ + 1.2, y), textcoords="data", fontsize=10,
                    color=col, va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=2))
    fig._direct_labels_done = True
    plt.tight_layout()
    for dest in _fig_dests(fname):
        plt.savefig(os.path.join(dest, fname), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {fname}")



# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panels_loaded = {}
    for cname in COUNTRIES:
        panel_path = os.path.join(PANEL_DIR, f"panel_{cname.lower()}.parquet")
        if not os.path.exists(panel_path):
            raise FileNotFoundError(f"Panel not found: {panel_path}\nRun 01_build_panels.py first.")
        panels_loaded[cname] = pd.read_parquet(panel_path)
        print(f"{cname}: {panels_loaded[cname]['cell_id'].nunique()} cells")

    print("\nCallaway et al. diagnostic plots...")
    # Check 1: binned scatter (dose-response)
    plot_binned_scatter(panels_loaded, 2007, "callaway_binned_scatter.png")
    # Check 2: event study by distance band (pooled Kenya, Ghana, Senegal)
    plot_event_study_by_distance_band(panels_loaded, 2007, "eventstudy_by_band.png")
    print("\nDone.")
