"""
38_price_placebo.py

Food-price placebo for the lead-specific riot gap, plus a diagnostic showing why
commodity prices cannot be used to test the exposure channel over this period.

MOTIVATION. Food-price spikes are a well-established riot trigger. If cells near
battery recycling sites are more food-price sensitive than the non-lead comparison
cells -- plausible, since they are poorer and more urban -- then the post-2007 riot
increase could reflect the food-price crises of 2007-08 and 2011 rather than lead.
This tests that channel by name.

We also examine the real lead price, since higher lead prices raise the return to
collecting and smelting used batteries.

DIAGNOSTIC, which governs how everything here is read: over 1997-2024 the real lead
price correlates 0.92 with the post-2007 indicator and the real food index 0.88. The
2000s commodity supercycle makes both series near-duplicates of the paper's own
treatment timing, so interacting them with proximity simply re-identifies the main
difference-in-differences and tells us nothing new. We therefore residualise each
price on the post-2007 indicator and use only WITHIN-period price variation.

Prices: World Bank Commodity Price Data (the Pink Sheet), annual real 2010 US dollars.
Lead is $/mt; food is the Bank's food price index, 2010 = 100.

Outputs (+ Overleaf): output/tables/tab_price_placebo.tex, output/tables/price_values.tex,
         output/figures/fig_prices.pdf
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import DATA, PANELS, TABLES, POST_YR, save_to_overleaf
from utils import stars, add_block, lincom, coef_sep

NEAR5 = 1.0 / 6.0     # 1/(d+1) > 1/6  <=>  within 5 km (the main design's proximity)

COUNTRIES = ["Kenya", "Ghana", "Senegal"]

px = pd.read_csv(str(DATA / "wb_lead_food_prices.csv"))
px = px[(px.year >= 1997) & (px.year <= 2024)].copy()
px["post"] = (px.year >= POST_YR).astype(int)
for c in ["lead_price", "food_index"]:
    z = (px[c] - px[c].mean()) / px[c].std()
    px[c + "_z"] = z
    # within-period variation only: remove the pre/post mean shift that mirrors treatment
    px[c + "_r"] = z - z.groupby(px.post).transform("mean")

corr_lead = px.lead_price_z.corr(px.post)
corr_food = px.food_index_z.corr(px.post)
peak_yr = int(px.loc[px.lead_price.idxmax(), "year"])
p2006 = float(px.loc[px.year == 2006, "lead_price"].iloc[0])
p2007 = float(px.loc[px.year == 2007, "lead_price"].iloc[0])
print(f"corr(real lead price, post-2007) = {corr_lead:.3f}")
print(f"corr(real food index, post-2007) = {corr_food:.3f}")
print(f"real lead price peaks in {peak_yr}: {p2006:,.0f} -> {p2007:,.0f} $/mt (2010 USD)")

parts = []
for cn in COUNTRIES:
    p = pd.read_parquet(str(PANELS / f"panel_{cn.lower()}.parquet")).copy()
    p["cell_id_str"] = cn[:2] + "_" + p.cell_id.astype(str)
    add_block(p, cn)       # 1-degree blocks: cluster for every reported SE
    p["cty_year"] = cn + "_" + p.year.astype(str)
    parts.append(p)
P = pd.concat(parts, ignore_index=True).merge(px, on="year", how="inner")
P["y"] = (P.n_riots > 0).astype(float)
P["bat5"] = (P.inv_dist_battery > NEAR5).astype(float)
P["nl5"] = (P.inv_dist_nonlead > NEAR5).astype(float)


def gaps(lead_col, food_col):
    d = P.copy()
    # Proximity = within 5 km of a battery / non-lead site, as in the main design
    d["bat_lead"] = d.bat5 * d[lead_col]
    d["nl_lead"] = d.nl5 * d[lead_col]
    d["bat_food"] = d.bat5 * d[food_col]
    d["nl_food"] = d.nl5 * d[food_col]
    m = pf.feols("y ~ bat_lead + nl_lead + bat_food + nl_food | cell_id_str + cty_year",
                 data=d, vcov={"CRV1": "block"})
    def g(a, b):                                     # (battery, non-lead, gap), block-clustered
        return coef_sep(m, a), coef_sep(m, b), lincom(m, {a: 1, b: -1})
    return g("bat_lead", "nl_lead"), g("bat_food", "nl_food"), int(m._N)

raw_lead, raw_food, n_raw = gaps("lead_price_z", "food_index_z")
w_lead, w_food, n_w = gaps("lead_price_r", "food_index_r")
for nm, (bl, nl, gp) in [("raw lead", raw_lead), ("raw food", raw_food),
                         ("within lead", w_lead), ("within food", w_food)]:
    print(f"  {nm:12s} battery {bl[0]:+.3f} ({bl[1]:.3f})  non-lead {nl[0]:+.3f} ({nl[1]:.3f})  "
          f"gap {gp[0]:+.3f}{stars(gp[2])} ({gp[1]:.3f})")


def _c(t):
    return f"{'$-$' if t[0] < 0 else ''}{abs(t[0]):.3f}{stars(t[2])}"


def _panel(title, food, lead):
    out = [r"\textit{" + title + r"} \\[2pt]"]
    for i, lab in enumerate([r"Battery (within 5\,km) $\times$ price", r"Non-lead (within 5\,km) $\times$ price",
                             "Lead-specific gap"]):
        out.append(r"\quad " + lab + f" & {_c(food[i])} & {_c(lead[i])}" + r" \\")
        out.append(f" & ({food[i][1]:.3f}) & ({lead[i][1]:.3f})" + r" \\")
    return out

t = ([r"\begin{table}[htbp]",
      r"\caption{Food and Lead Price Interactions: Lead-Specific Riot Gap}",
      r"\label{tab:price_placebo}", r"\begin{threeparttable}", r"\small",
      r"\begin{tabular}{lcc}", r"\toprule",
      r" & Food price & Lead price \\", r"\midrule"]
     + _panel("Raw prices", raw_food, raw_lead) + [r"[3pt]"]
     + _panel("Within-period variation only", w_food, w_lead)
     + [r"\midrule",
        r"Cell-years & \multicolumn{2}{c}{" + f"{n_w:,}" + r"} \\",
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\setstretch{1}\footnotesize",
        # Table note lives in main.tex (\setTabNote{tab_price_placebo}{...}).
        r"\item \TabNote{tab_price_placebo}",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"])
t = [x for x in t if x != "[3pt]"]
(TABLES / "tab_price_placebo.tex").write_text("\n".join(t), encoding="utf-8")
save_to_overleaf("tab_price_placebo.tex", str(TABLES / "tab_price_placebo.tex"), subfolder="tables")

v = ["% auto-generated by 38_price_placebo.py",
     r"\newcommand{\priceCorrLead}{" + f"{corr_lead:.2f}" + "}",
     r"\newcommand{\priceCorrFood}{" + f"{corr_food:.2f}" + "}",
     r"\newcommand{\leadPeakYr}{" + f"{peak_yr}" + "}",
     r"\newcommand{\leadPriceSix}{" + f"{p2006:,.0f}" + "}",
     r"\newcommand{\leadPriceSeven}{" + f"{p2007:,.0f}" + "}",
     r"\newcommand{\leadPriceRise}{" + f"{100*(p2007/p2006-1):.0f}" + "}"]
(TABLES / "price_values.tex").write_text("\n".join(v) + "\n", encoding="utf-8")
save_to_overleaf("price_values.tex", str(TABLES / "price_values.tex"), subfolder="tables")
print("\nSaved: tab_price_placebo.tex, price_values.tex")

# ── Figure: the two price series, and why they cannot identify the channel ────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIGURES

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

# (a) levels, each in its own units
ax = axes[0]
ax.plot(px.year, px.lead_price, "o-", ms=3.5, lw=1.8, color="tab:blue", label="Lead price (left)")
ax.text(0, 1.03, "Real lead price ($/mt, 2010 USD)", transform=ax.transAxes, ha="left", va="bottom",
        color="tab:blue", fontsize=9)
ax.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax.twinx()
ax2.plot(px.year, px.food_index, "s-", ms=3.5, lw=1.8, color="tab:orange", label="Food index (right)")
ax2.text(1, 1.03, "Food price index (2010 = 100)", transform=ax2.transAxes, ha="right", va="bottom",
         color="tab:orange", fontsize=9)
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax.axvline(POST_YR - 0.5, color="grey", ls="--", lw=0.9)
ax.annotate(f"{peak_yr} peak", xy=(peak_yr, px.lead_price.max()),
            xytext=(peak_yr + 1.5, px.lead_price.max()), fontsize=8, color="tab:blue",
            va="center", arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.7))
ax.set_xlabel("Year")
ax.set_title("(a) Levels", fontsize=10, pad=22)
for s in ("top",):
    ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)

# (b) standardised, with pre/post means drawn in -- this is the diagnostic
ax = axes[1]
ax.axhline(0, color="grey", lw=0.5)
for col, colr, mk, lab in [("lead_price_z", "tab:blue", "o", "Lead price"),
                           ("food_index_z", "tab:orange", "s", "Food price index")]:
    ax.plot(px.year, px[col], mk + "-", ms=3.5, lw=1.8, color=colr, label=lab, alpha=0.9)
    for pv in (0, 1):
        sub = px[px.post == pv]
        ax.hlines(sub[col].mean(), sub.year.min() - 0.4, sub.year.max() + 0.4,
                  color=colr, lw=2.6, alpha=0.35)
ax.axvline(POST_YR - 0.5, color="grey", ls="--", lw=0.9)
ax.set_xlabel("Year"); ax.set_ylabel("Standardised (mean 0, sd 1)")
ax.set_title("(b) Standardised, with pre- and post-2007 means", fontsize=10, pad=22)
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIGURES}/fig_prices.pdf", bbox_inches="tight", dpi=200)
save_to_overleaf("fig_prices.pdf", f"{FIGURES}/fig_prices.pdf", subfolder="figures")
plt.close()
print("Saved: fig_prices.pdf")
