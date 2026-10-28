"""
10_randomisation_inference.py
Randomisation inference for the main design's lead-specific gap.

Statistic (same as Table 3, tab_main_binary): the lead-specific gap b_B - b_N from
    1[any riot]_ct = a_c + d_t + b_B * B_c x Post_t + b_N * N_c x Post_t + e_ct
with B_c = 1[cell within 5 km of a battery site], N_c = 1[within 5 km of a non-lead
TSIP site] (1/(d+1) > 1/6), Post = year >= 2007; cell + year FE by country, cell +
country-year FE pooled. Each country panel is balanced, so the two-way within
transformation is exact and the estimate is two small normal equations.

Permutation: within each country, the TSIP site-type labels (battery / non-lead /
other lead site) are randomly permuted across all TSIP sites in the country, keeping
the number of each type. B_c and N_c are rebuilt from the permuted battery and
non-lead sites and the gap is re-estimated. 1,000 permutations, seed 20260410; the
pooled statistic uses the same draws in every country. p = share of permuted gaps
>= the observed gap (one-sided, as before); the two-sided share (|perm| >= |obs|) is
saved too.

Outputs:
  output/figures/fig_randomisation_inference.png  (copied to Overleaf; the paper uses the PNG)
  output/panels/ri_results.csv                    (read by 11_generate_values.py)
"""
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import haversine_km
from config import FIGURES, PANELS, TSIP_PATH, POST_YR, save_to_overleaf

N_PERMS = 1000
SEED = 20260410
NEAR5 = 1.0 / 6.0                  # 1/(d+1) > 1/6  <=>  d < 5 km
COUNTRIES = ["Kenya", "Ghana", "Senegal"]

tsip = pd.read_csv(str(TSIP_PATH))


def parts(Yd, pc, b, n):
    """X'X (2x2) and X'y (2) for x = (b_i, n_i) x post_t after two-way demeaning."""
    bc, nc = b - b.mean(), n - n.mean()
    spp = float(pc @ pc)
    XX = spp * np.array([[bc @ bc, bc @ nc], [bc @ nc, nc @ nc]])
    Xy = np.array([bc @ Yd @ pc, nc @ Yd @ pc])
    return XX, Xy


def gap_from(XX, Xy):
    beta = np.linalg.solve(XX, Xy)
    return float(beta[0] - beta[1])


rng = np.random.default_rng(SEED)
DATA = {}
for cn in COUNTRIES:
    p = pd.read_parquet(str(PANELS / f"panel_{cn.lower()}.parquet")).sort_values(["cell_id", "year"])
    cells = p.drop_duplicates("cell_id")[["cell_id", "cell_lat", "cell_lon",
                                          "inv_dist_battery", "inv_dist_nonlead"]].reset_index(drop=True)
    years = np.sort(p.year.unique())
    if len(p) != len(cells) * len(years):
        raise SystemExit(f"{cn}: panel is not balanced; the demeaning shortcut needs a balanced panel")
    Y = (p.n_riots.values > 0).astype(float).reshape(len(cells), len(years))
    Yd = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    pc = (years >= POST_YR).astype(float)
    pc = pc - pc.mean()

    # Site types as in 01_build_panels.py
    sites = tsip[tsip.country.str.contains(cn, case=False, na=False)].reset_index(drop=True)
    is_bat = sites.SiteIndustry.str.contains("Battery", case=False, na=False).values
    is_lead = sites.KeyPollutant.str.contains("Lead", case=False, na=False).values
    label = np.where(is_bat, "B", np.where(is_lead, "L", "N"))
    D = haversine_km(cells.cell_lat.values[:, None], cells.cell_lon.values[:, None],
                     sites.latitude.values[None, :], sites.longitude.values[None, :])

    def indicators(lab, D=D):
        b = (1 / (D[:, lab == "B"].min(1) + 1) > NEAR5).astype(float)
        n = (1 / (D[:, lab == "N"].min(1) + 1) > NEAR5).astype(float)
        return b, n

    b_obs, n_obs = indicators(label)
    b_pan = (cells.inv_dist_battery.values > NEAR5).astype(float)
    n_pan = (cells.inv_dist_nonlead.values > NEAR5).astype(float)
    if not ((b_obs == b_pan).all() and (n_obs == n_pan).all()):
        raise SystemExit(f"{cn}: recomputed 5 km indicators do not match the panel")
    XX, Xy = parts(Yd, pc, b_pan, n_pan)

    # Check against the regression in 55_binary_main_table.py
    q = p.copy()
    q["y"] = (q.n_riots > 0).astype(float)
    q["bt"] = (q.inv_dist_battery > NEAR5).astype(int) * (q.year >= POST_YR)
    q["nt"] = (q.inv_dist_nonlead > NEAR5).astype(int) * (q.year >= POST_YR)
    m = pf.feols("y ~ bt + nt | cell_id + year", data=q)
    g_reg = float(m.coef()["bt"] - m.coef()["nt"])
    if abs(g_reg - gap_from(XX, Xy)) > 1e-8:
        raise SystemExit(f"{cn}: demeaned gap {gap_from(XX, Xy)} != regression gap {g_reg}")

    DATA[cn] = dict(Yd=Yd, pc=pc, label=label, indicators=indicators, XX=XX, Xy=Xy,
                    n_bat=int(is_bat.sum()), n_nl=int((label == "N").sum()), n_sites=len(sites))
    print(f"{cn}: {len(cells)} cells, {len(sites)} TSIP sites ({is_bat.sum()} battery, "
          f"{(label == 'N').sum()} non-lead, {(label == 'L').sum()} other lead); observed gap {g_reg:.3f}")

# Permutations: one relabelling per country per draw
PXX = {cn: np.empty((N_PERMS, 2, 2)) for cn in COUNTRIES}
PXy = {cn: np.empty((N_PERMS, 2)) for cn in COUNTRIES}
for i in range(N_PERMS):
    for cn in COUNTRIES:
        d = DATA[cn]
        b, n = d["indicators"](rng.permutation(d["label"]))
        PXX[cn][i], PXy[cn][i] = parts(d["Yd"], d["pc"], b, n)


def gaps(XXs, Xys):
    out = np.full(len(XXs), np.nan)
    for i, (A, v) in enumerate(zip(XXs, Xys)):
        if np.linalg.matrix_rank(A) == 2:        # a draw with no treated or no comparison cell
            out[i] = gap_from(A, v)
    return out


results = {}
for cn in COUNTRIES + ["Pooled"]:
    if cn == "Pooled":
        obs = gap_from(sum(DATA[c]["XX"] for c in COUNTRIES), sum(DATA[c]["Xy"] for c in COUNTRIES))
        perm = gaps(sum(PXX[c] for c in COUNTRIES), sum(PXy[c] for c in COUNTRIES))
        n_bat = sum(DATA[c]["n_bat"] for c in COUNTRIES)
        n_sites = sum(DATA[c]["n_sites"] for c in COUNTRIES)
    else:
        obs = gap_from(DATA[cn]["XX"], DATA[cn]["Xy"])
        perm = gaps(PXX[cn], PXy[cn])
        n_bat, n_sites = DATA[cn]["n_bat"], DATA[cn]["n_sites"]
    ok = perm[~np.isnan(perm)]
    p1 = float((ok >= obs).mean())
    p2 = float((np.abs(ok) >= abs(obs)).mean())
    results[cn] = dict(obs_coef=obs, perm=ok, p_value=p1, p_two=p2, n_bat=n_bat,
                       n_sites=n_sites, n_valid=len(ok))
    print(f"  {cn:8s} observed gap {obs:.3f}; RI p (one-sided) {p1:.3f}, two-sided {p2:.3f}; "
          f"{len(ok)} valid draws; permuted mean {ok.mean():.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
for i, (ax, cn) in enumerate(zip(axes, COUNTRIES + ["Pooled"])):
    r = results[cn]
    ax.hist(r["perm"], bins=40, color="#B8C4C8", edgecolor="white", density=True)
    ax.axvline(r["obs_coef"], color="#C8102E", lw=1.8)
    ax.text(r["obs_coef"], ax.get_ylim()[1] * 1.0, f"observed {r['obs_coef']:.3f} ",
            color="#C8102E", fontsize=8, ha="right", va="top")
    pv = "p < 0.001" if r["p_value"] < 0.001 else f"p = {r['p_value']:.3f}"
    ax.set_title(f"{cn} ({pv})", fontsize=10)
    ax.set_xlabel("Lead-specific gap, Pr(any riot)", fontsize=9)
    if i == 0:
        ax.set_ylabel("Density")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
out = FIGURES / "fig_randomisation_inference.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
save_to_overleaf("fig_randomisation_inference.png", str(out), subfolder="figures")
print("Saved figure")

pd.DataFrame([{"country": cn, "obs_coef": r["obs_coef"], "p_value": r["p_value"],
               "p_two_sided": r["p_two"], "n_bat": r["n_bat"], "n_sites": r["n_sites"],
               "n_valid": r["n_valid"]} for cn, r in results.items()]
             ).to_csv(str(PANELS / "ri_results.csv"), index=False)
print(f"Saved: {PANELS / 'ri_results.csv'}")
