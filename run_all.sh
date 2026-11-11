#!/bin/bash
# Master run-all script for "Lead Exposure and Collective Violence: Evidence from Unsafe
# Battery Recycling in Africa"
# Usage: cd lead-crime && bash run_all.sh
#
# Runs every script that produces a table, figure or number used in the current draft
# (paper + York EHPE slides), in dependency order. Scripts for analyses no longer in the
# draft were moved to _archive/offcuts_2026-09-24/ on 2026-09-24, and on 2026-10-22 to
# _archive/offcuts_2026-10-22/ (03, 20, 26, 27, 42, 52). Every output the scripts write is
# used by main.tex or slides_york_ehpe.tex, or read by a later script (output/panels/);
# the one exception is fig_slides_price_raw (59), made for the slides at the author's request.
#
# Prerequisites:
#   pip install -r requirements.txt
#   R with HonestDiD, Matrix, ggplot2 and patchwork (step 31, called by 47; required)
#   poppler (pdftotext, used by 45)
#   ACLED data and the UNICEF report in data/; DHS in PUBLIC_DATA/DHS (see README)

set -e  # Stop on first error

PROJ="$(cd "$(dirname "$0")" && pwd)"
export LEAD_CRIME_ROOT="$PROJ"
# Ignore packages installed in the user folder (~/Library/Python/...). On 8 Oct 2026 pandas 3,
# NumPy 2.5 and scikit-learn 1.9 were installed there, which broke step 19. The pipeline was
# verified with the system install: NumPy 2.4.1, pandas 2.3.3, scikit-learn 1.8.0, pyfixest 0.50.1.
export PYTHONNOUSERSITE=1
# Fix Python's string-hash seed. Without it, two figures (fig_slides_dd_event.pdf,
# eventstudy_by_band.png) differ at the level of floating-point noise from run to run:
# fixed-effect ordering inside the estimator depends on hash order. No table or macro does.
export PYTHONHASHSEED=2      # 2 reproduces the 2026-10-22 versions of both figures
cd "$PROJ/code"

echo "=== Lead-Crime Replication Pipeline ==="
echo "Project root: $PROJ"
echo ""

# ── Data ─────────────────────────────────────────────────────────────────────
echo ">>> 01: Building panels..."
python3 01_build_panels.py
echo ">>> 19: Urban controls (population, night lights) used by 33, 36, 57, 60..."
python3 19_urban_controls.py

# ── Setting and descriptives ─────────────────────────────────────────────────
echo ">>> 04: Motorisation figure..."
python3 04_motorisation_figure.py
echo ">>> 17: Summary statistics and site map..."
python3 17_summary_and_map.py
echo ">>> 18: Evidence gap map (slides)..."
python3 18_gap_map.py
echo ">>> 45: Lead burden by world region (slides)..."
python3 45_lead_burden_regions.py
echo ">>> 46: ACLED descriptives (slides)..."
python3 46_acled_descriptive.py
echo ">>> 43: Outcome distribution figure..."
python3 43_outcome_distribution.py

# ── Main results ─────────────────────────────────────────────────────────────
echo ">>> 02: Dose-response diagnostic figures..."
python3 02_did_analysis.py
echo ">>> 29: Extensive-margin main table, conflict-type test, BOTEC..."
python3 29_extensive_margin_main.py
echo ">>> 30: Extensive-margin event studies, pre-trends, distance rings..."
python3 30_extensive_margin_figures.py
echo ">>> 23: Vehicle-dose specification and escalating differences..."
python3 23_vehicle_dose_main.py
echo ">>> 36: Riot trigger coding..."
python3 36_trigger_coding.py

# ── Inference and robustness ─────────────────────────────────────────────────
echo ">>> 05: Alternative outcome transformations..."
python3 05_robustness.py
echo ">>> 10: Randomisation inference..."
python3 10_randomisation_inference.py
echo ">>> 12: Standard-error comparison (main design)..."
python3 12_ppml_conley.py
echo ">>> 22: Callaway continuous-treatment diagnostics..."
python3 22_callaway_weights.py
echo ">>> 28: Share of cells FE Poisson drops (Appendix A)..."
python3 28_poisson_diagnostics.py
echo ">>> 32: Treatment-timing sweep..."
python3 32_cutoff_sweep.py
echo ">>> 33: Comparison-group robustness macros (non-mining, urban cells)..."
python3 33_comparison_robustness.py
echo ">>> 35: Placement and macro-cell fixed-effect macros..."
python3 35_placement_robustness.py
echo ">>> 37: Leave-one-out..."
python3 37_leave_one_out.py
echo ">>> 38: Food and lead price interactions..."
python3 38_price_placebo.py
echo ">>> 47: Binary versions of slide exhibits (calls 31 R script for Honest DiD)..."
python3 47_slides_binary.py
echo ">>> 48: Soil-lead distance decay (slides)..."
python3 48_slides_soil_decay.py
echo ">>> 49: Lead price (slides)..."
python3 49_slides_lead_price.py
echo ">>> 50: Site and violence descriptives (slides)..."
python3 50_slides_descriptives.py
echo ">>> 51: Distance rings, pooled..."
python3 51_slides_rings_pooled.py
echo ">>> 54: DHS outcomes..."
python3 54_dhs_outcomes.py
echo ">>> 55: Main binary table and sub-events..."
python3 55_binary_main_table.py
echo ">>> 56: Event-level site rings..."
python3 56_site_rings.py
echo ">>> 57: Comparison table..."
python3 57_comparison_table.py
echo ">>> 58: Placebo table..."
python3 58_placebo_table.py
echo ">>> 59: Lead price and raw riot rates (slides; requested, not yet in the deck)..."
python3 59_slides_price_raw.py
echo ">>> 60: Balance table..."
python3 60_balance_table.py
echo ">>> 61: Raw riot rates by distance group (paper figure)..."
python3 61_raw_rates_figure.py
echo ">>> 62: Published effects by distance (Figure A10b)..."
python3 62_published_effects.py

# 31_honest_did_extensive.R is called by 47_slides_binary.py (needs R + HonestDiD).

# ── Macros: run after everything above, since 11 reads their outputs ─────────
echo ">>> 11: Generating LaTeX values..."
python3 11_generate_values.py
echo ">>> 41: Remediation break-even (reads generated_values.tex)..."
python3 41_remediation_threshold.py
echo ">>> 44: Effect-size plausibility (reads generated_values.tex)..."
python3 44_plausibility.py

echo ""
echo "=== Pipeline complete ==="
echo "Outputs in: $PROJ/output/  (tables and figures also copied to Overleaf tables/ and figures/)"
