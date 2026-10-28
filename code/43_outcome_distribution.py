"""
43_outcome_distribution.py
Figure motivating the extensive-margin outcome: the distribution of riots per cell-year.

Panel (a): share of cell-years by riot count (log scale). 95-99% are zero, and the few
           non-zero cell-years spread over a long tail.
Panel (b): concentration among cell-years with at least one riot. A small share of
           cell-years accounts for most riot events, so any count or log-count model is
           driven by them. The extensive margin weights every cell-year equally.

Colours: Kenya/Ghana/Senegal keep the blue/orange/green families used elsewhere, but at
steps that pass a colour-vision-deficiency check (the matplotlib tab10 orange and green
are indistinguishable under protanopia). Each country also has its own marker.

Outputs: output/figures/fig_outcome_distribution.pdf (copied to Overleaf figures/)
         output/tables/outcome_dist_values.tex (copied to Overleaf tables/). Its own file, so
         re-running 11_generate_values.py cannot wipe these macros.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import PANELS, TABLES, FIGURES, save_to_overleaf

COUNTRIES = [("Kenya", "ke", "#1f5fa8", "o", "-"),
             ("Ghana", "gh", "#d9730d", "s", "-."),
             ("Senegal", "se", "#1b9e77", "^", "--")]   # Kenya and Senegal nearly coincide in (b):
                                                        # dashes let the blue line show through
BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3–5"), (6, 10, "6–10"),
        (11, 20, "11–20"), (21, 10**9, "21+")]
TOP = 0.10   # concentration statistic: share of riots in the top 10% of riot cell-years

panels = {name: pd.read_parquet(PANELS / f"panel_{name.lower()}.parquet") for name, *_ in COUNTRIES}

# ── statistics (feed both the figure and the macros) ─────────────────────────
stats = {}
for name, key, *_ in COUNTRIES:
    y = panels[name]["n_riots"].astype(int)
    nz = np.sort(y[y > 0].values)[::-1]
    shares = [((y >= lo) & (y <= hi)).mean() for lo, hi, _ in BINS]
    k = max(1, int(np.ceil(TOP * len(nz))))
    stats[name] = dict(
        n=len(y), zero=(y == 0).mean(), le_one=(y <= 1).mean(), max=int(y.max()), n_nz=len(nz),
        one_of_nz=(nz == 1).mean(),                       # riot cell-years with exactly one riot
        top_share=nz[:k].sum() / nz.sum(),                 # riots in top 10% of riot cell-years
        bins=shares,
        lorenz_x=np.arange(1, len(nz) + 1) / len(nz),
        lorenz_y=np.cumsum(nz) / nz.sum(),
    )
    s = stats[name]
    print(f"{name:8s} zero {100*s['zero']:.1f}%  max {s['max']}  riot cell-years {s['n_nz']}  "
          f"exactly-one {100*s['one_of_nz']:.0f}%  top-10% share {100*s['top_share']:.0f}%")

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))  # drawn near text width so labels print at ~8pt

ax = axes[0]
x = np.arange(len(BINS))
for name, key, col, mk, ls in COUNTRIES:
    sh = np.array(stats[name]["bins"])
    keep = sh > 0                                   # no marker for empty bins (log scale)
    ax.plot(x[keep], sh[keep], ls=ls, marker=mk, color=col, lw=1.6, ms=6.5, mec="white", mew=1.2, label=name)
ax.set_yscale("log")
ax.set_ylim(1e-5, 1.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:g}%"))
ax.set_xticks(x, [b[2] for b in BINS])
ax.set_xlabel("Riots recorded in the cell-year")
ax.set_ylabel("Share of cell-years (log scale)")
ax.set_title("(a) Distribution of riots per cell-year", fontsize=10)
zmin = min(stats[n]["zero"] for n, *_ in COUNTRIES); zmax = max(stats[n]["zero"] for n, *_ in COUNTRIES)
ax.annotate(f"{100*zmin:.0f}–{100*zmax:.0f}% of\ncell-years are zero", xy=(0.12, 0.93),
            xytext=(1.0, 0.25), fontsize=8.5, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7))
ax.legend(loc="upper right", fontsize=8.5, frameon=False)

ax = axes[1]
ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.9)
ax.text(0.62, 0.52, "equal counts", rotation=38, fontsize=8, color="#777777")
for name, key, col, mk, ls in COUNTRIES:
    s = stats[name]
    lx = np.r_[0, s["lorenz_x"]]; ly = np.r_[0, s["lorenz_y"]]
    ax.plot(lx, ly, ls, color=col, lw=1.8, label=name)
    ax.plot(TOP, np.interp(TOP, lx, ly), mk, color=col, ms=6.5, mec="white", mew=1.2)
ax.axvline(TOP, color="#bbbbbb", lw=0.7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.set_xlabel("Cell-years with a riot, ranked from most riots")
ax.set_ylabel("Cumulative share of riot events")
ax.set_title("(b) Concentration among cell-years with a riot", fontsize=10)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)

for a in axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.tick_params(labelsize=8.5)
plt.tight_layout()
plt.savefig(FIGURES / "fig_outcome_distribution.pdf", bbox_inches="tight", dpi=200)
save_to_overleaf("fig_outcome_distribution.pdf", str(FIGURES / "fig_outcome_distribution.pdf"), subfolder="figures")
plt.close()

# ── macros ───────────────────────────────────────────────────────────────────
v = ["% Generated by 43_outcome_distribution.py -- do not edit by hand"]
def add(k, val): v.append(f"\\newcommand{{\\{k}}}{{{val}}}")
add("odTopPct", f"{100*TOP:.0f}")
_all = pd.concat(panels.values(), ignore_index=True)
add("odMeanRiot", f"{(_all.n_riots > 0).mean():.3f}")                       # Table 3 outcome mean, cols 1-2
_near = _all[(_all.inv_dist_battery > 1/6) & (_all.year < 2007)]               # treated cells, pre-period
add("odPreNearRiot", f"{(_near.n_riots > 0).mean():.3f}")
add("odMeanStack", f"{pd.concat([(_all.n_riots > 0), (_all.n_org_violence > 0)]).mean():.3f}")  # col 3 (stacked)
add("odZeroMin", f"{100*zmin:.0f}"); add("odZeroMax", f"{100*zmax:.0f}")
add("odLeOneMin", f"{100*min(stats[n]['le_one'] for n, *_ in COUNTRIES):.1f}")
add("odTopMin", f"{100*min(stats[n]['top_share'] for n, *_ in COUNTRIES):.0f}")
add("odTopMax", f"{100*max(stats[n]['top_share'] for n, *_ in COUNTRIES):.0f}")
(TABLES / "outcome_dist_values.tex").write_text("\n".join(v) + "\n", encoding="utf-8")
save_to_overleaf("outcome_dist_values.tex", str(TABLES / "outcome_dist_values.tex"), subfolder="tables")
print("Saved: fig_outcome_distribution.pdf, outcome_dist_values.tex")
