#!/usr/bin/env python3
"""
18_gap_map.py  --  Evidence gap map for the slides: causal lead studies by world region x outcome.

Data: data/lead_review_studies.csv, copied on 2026-10-11 from the study list behind
Crawfurd (2026) "The Causal Effects of Lead Poisoning: A Review" (Overleaf project
"Lead causal review", studies.csv). Counts are computed here from that list, not typed.
A study listed under two outcomes counts in both columns. Multi-country studies
(US/Europe, 44 developing countries, GWAS) get their own row.

Output: output/figures/fig_gap_map.pdf (copied to Overleaf figures/).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from config import FIGURES as FIGS, OVERLEAF_FIGS, DATA
FIGS.mkdir(parents=True, exist_ok=True)

TEAL, DARK, PALE, WHITE = "#006970", "#0B4C5B", "#EFF6F6", "#FFFFFF"
HILITE = "#E8C060"

REGION = {
    "United States": "North America", "United States (NHANES)": "North America",
    "Sweden": "Europe", "Scotland": "Europe",
    "Mexico": "Latin America", "Chile": "Latin America",
    "Indonesia": "East Asia & Pacific",
    "Bangladesh": "South Asia",
    "Kenya": "Sub-Saharan Africa", "Zambia": "Sub-Saharan Africa",
    "United States / Europe": "Multi-country", "44 developing countries": "Multi-country",
    "International (GWAS)": "Multi-country",
}
ROWS = ["North America", "Europe", "Latin America", "East Asia & Pacific", "South Asia",
        "Middle East & N. Africa", "Sub-Saharan Africa", "Multi-country"]
COLS = ["Education /\nIQ & test scores", "Crime &\nbehaviour", "Health"]

def outcome_cols(cat):
    out = []
    if "IQ" in cat or "Test" in cat: out.append(COLS[0])
    if "Crime" in cat: out.append(COLS[1])
    if "Health" in cat: out.append(COLS[2])
    return out

df = pd.read_csv(DATA / "lead_review_studies.csv")
unmapped = sorted(set(df.country) - set(REGION))
if unmapped:
    raise SystemExit(f"18_gap_map: countries with no region: {unmapped}")
counts = pd.DataFrame(0, index=ROWS, columns=COLS)
for _, r in df.iterrows():
    cols = outcome_cols(r.outcome_category)
    if not cols:
        raise SystemExit(f"18_gap_map: unmapped outcome {r.outcome_category!r}")
    for c in cols:
        counts.loc[REGION[r.country], c] += 1
print(counts)

fig, ax = plt.subplots(figsize=(8.0, 5.6))
n_rows, n_cols = counts.shape
max_count = counts.values.max()
for i, row in enumerate(ROWS):
    y = n_rows - 1 - i
    for j, col in enumerate(COLS):
        n = counts.loc[row, col]
        ax.add_patch(patches.Rectangle((j - 0.5, y - 0.5), 1, 1, linewidth=0.6,
                                       edgecolor="#D0D8D8", facecolor=PALE, zorder=1))
        if n > 0:
            alpha = 0.30 + 0.60 * (n / max_count)
            ax.scatter([j], [y], s=180 + 1500 * n / max_count, color=TEAL, alpha=alpha,
                       linewidths=0, zorder=3)   # marker area grows with the count; stays round
            ax.text(j, y, str(n), ha="center", va="center", fontsize=13, fontweight="bold",
                    color=WHITE if alpha > 0.62 else DARK, zorder=4)
        else:
            ax.text(j, y, "0", ha="center", va="center", fontsize=12, color="#C8D4D4", zorder=4)

ssa = n_rows - 1 - ROWS.index("Sub-Saharan Africa")
ax.add_patch(patches.Rectangle((-0.54, ssa - 0.5), n_cols + 0.08, 1, linewidth=1.6,
                               edgecolor=HILITE, facecolor="none", zorder=5, clip_on=False))

ax.set_xlim(-0.5, n_cols - 0.5); ax.set_ylim(-0.5, n_rows - 0.5)
ax.set_xticks(range(n_cols)); ax.set_xticklabels(COLS, fontsize=11, fontweight="bold", color=DARK, linespacing=1.3)
ax.set_yticks([n_rows - 1 - i for i in range(n_rows)]); ax.set_yticklabels(ROWS, fontsize=11, color=DARK)
for t in ax.get_yticklabels():
    if t.get_text() == "Sub-Saharan Africa":
        t.set_fontweight("bold")
ax.tick_params(length=0, pad=8)
ax.xaxis.set_ticks_position("top")
for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout()
path = FIGS / "fig_gap_map.pdf"
plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=WHITE)
if OVERLEAF_FIGS:
    import shutil
    shutil.copy(path, OVERLEAF_FIGS / path.name)
print(f"saved {path.name}")
plt.close()
