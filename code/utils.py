"""
Shared utility functions for the lead-crime project.
"""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km. Inputs in degrees. Vectorised over lat2/lon2."""
    R = 6371.0
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_dist_km(clat, clon, site_lats, site_lons):
    """Minimum haversine distance from one point to a set of sites."""
    if len(site_lats) == 0:
        return np.inf
    return haversine_km(clat, clon, site_lats, site_lons).min()


def stars(p):
    """Significance stars."""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def pval_str(p):
    """Format p-value for LaTeX prose."""
    if p < 0.01:
        return r"p$<$0.01"
    elif p < 0.05:
        return r"p$<$0.05"
    elif p < 0.10:
        return r"p$<$0.10"
    else:
        return f"p$=${p:.2f}"


def conley_vcov(model, data, cell_col, lat_col, lon_col, cutoff_km):
    """Conley (1999) spatial HAC variance-covariance matrix.

    Algorithm
    ---------
    1. Aggregate scores (e * X_tilde) to cell level by summing over years.
       Valid because spatial kernel weight depends only on cell distance,
       not on the year pair, so all cross-year terms for the same pair of
       cells receive the same weight.
    2. Build n_cells × n_cells haversine distance matrix.
    3. Apply uniform kernel: w_{cc'} = 1 if d_{cc'} ≤ cutoff_km.
    4. Conley meat = S' @ W @ S  (k × k, where k = n_params).
    5. V = bread @ meat @ bread / n^2   (matches sandwich formula for IID-
       normalised scores, as used by pyfixest).

    Parameters
    ----------
    model     : fitted pyfixest Feols object
    data      : DataFrame aligned with the model observations (same row order)
    cell_col  : column name for cell identifier
    lat_col   : column with cell centroid latitude (degrees)
    lon_col   : column with cell centroid longitude (degrees)
    cutoff_km : spatial bandwidth (km); uniform kernel zeros beyond this

    Returns
    -------
    vcov : ndarray (k, k) — Conley vcov matrix
    se   : ndarray (k,)   — Conley standard errors
    """
    import pandas as pd

    scores = model._scores          # (n_obs, k)
    # pyfixest ≥0.18 stores (X'X) in _hessian and leaves _bread=None;
    # older versions store (X'X)^{-1} in _bread directly.
    if model._bread is not None:
        bread = np.array(model._bread)  # (k, k)
    else:
        bread = np.linalg.inv(np.array(model._hessian))  # (k, k)
    n      = scores.shape[0]

    # --- 1. Aggregate scores to cell level ---------------------------------
    scores_df = pd.DataFrame(scores, columns=[f"s{i}" for i in range(scores.shape[1])])
    scores_df[cell_col] = data[cell_col].values
    S = scores_df.groupby(cell_col)[[c for c in scores_df.columns if c != cell_col]].sum()

    # --- 2. Cell coordinates -----------------------------------------------
    cell_coords = (
        data[[cell_col, lat_col, lon_col]]
        .drop_duplicates(cell_col)
        .set_index(cell_col)
        .loc[S.index]   # align order
    )
    lats = np.radians(cell_coords[lat_col].values)
    lons = np.radians(cell_coords[lon_col].values)
    S_arr = S.values   # (n_cells, k)

    # --- 3. Haversine distance matrix (vectorised, n_cells × n_cells) ------
    R = 6371.0
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # (n_cells, n_cells)

    # --- 4. Uniform kernel --------------------------------------------------
    W = (dist <= cutoff_km).astype(float)  # (n_cells, n_cells)

    # --- 5. Meat = S' @ W @ S ----------------------------------------------
    meat = S_arr.T @ W @ S_arr   # (k, k)

    # --- 6. Sandwich vcov ---------------------------------------------------
    # pyfixest scores are already un-normalised (not divided by n), so
    # the sandwich is simply bread @ meat @ bread — no extra 1/n factor.
    # (Verified: formula recovers CRV1 SE exactly when W = I_{n_cells}.)
    vcov = bread @ meat @ bread
    # Guard: negative diagonal means Conley meat is numerically non-PSD
    # (can happen when spatial cross-products of scores cancel near zero).
    # Return NaN rather than clipping to 0 so callers can detect and fall back.
    diag = np.diag(vcov)
    se   = np.where(diag > 0, np.sqrt(diag), np.nan)
    return vcov, se




# ── Reported inference: CRV1 clustered on 1-degree spatial blocks ────────────
# Every SE, CI, star, p-value and t-stat the paper or slides report for a grid-cell
# regression comes from one matrix: pyfixest CRV1 clustered on 1-degree blocks
# (add_block). Fit with vcov={"CRV1": "block"}; model._vcov is then that matrix.
# t-tests use t(G-1) and joint tests F(q, G-1), G = number of blocks (pyfixest's
# own convention for CRV1). Small-G cases (Ghana: 3 treated blocks) are also
# checked with a wild cluster bootstrap (wild_p).

def add_block(df, country):
    """Add `block` = country + "_" + floor(cell_lat) + "_" + floor(cell_lon).

    `country` is a column name in df, or a literal country name for a
    single-country panel. Blocks never merge across countries.
    """
    c = df[country].astype(str) if (isinstance(country, str) and country in df.columns) \
        else pd_series_const(df, str(country))
    df["block"] = (c + "_" + np.floor(df["cell_lat"]).astype(int).astype(str)
                   + "_" + np.floor(df["cell_lon"]).astype(int).astype(str))
    return df


def pd_series_const(df, value):
    import pandas as pd
    return pd.Series(value, index=df.index)


def n_clusters(model):
    G = model._G
    return int(G[0] if isinstance(G, (list, tuple)) else G)


def tcrit(model, level=0.95):
    """Critical value for a two-sided CI: t(G-1)."""
    from scipy import stats as _st
    return float(_st.t.ppf(0.5 + level / 2, n_clusters(model) - 1))


def lincom(model, weights):
    """(estimate, SE, p) of sum_k w_k * coef_k from the model's (block-clustered)
    vcov; p from t(G-1). weights: {term: weight}, e.g. {"bt": 1, "nt": -1}."""
    from scipy import stats as _st
    names = list(model.coef().index)
    w = np.zeros(len(names))
    for t, wt in weights.items():
        w[names.index(t)] = wt
    est = float(w @ model.coef().values)
    var = float(w @ np.asarray(model._vcov) @ w)
    se = float(np.sqrt(var)) if var > 0 else np.nan
    p = float(2 * _st.t.sf(abs(est / se), n_clusters(model) - 1)) if se > 0 else np.nan
    return est, se, p


def coef_sep(model, term):
    """(coef, SE, p) of one term."""
    return lincom(model, {term: 1.0})


def wald_F(model, terms):
    """Joint test that `terms` are zero: F(q, G-1) from the model's vcov.
    Terms with no variation (coefficient and variance ~0, e.g. a pre-period year
    with no events near any site) are dropped; a rank-deficient matrix uses a
    generalised inverse with q = rank. Returns (F, q, p)."""
    from scipy import stats as _st
    names = list(model.coef().index)
    idx = [names.index(t) for t in terms]
    V = np.asarray(model._vcov)
    dv = np.diag(V)[idx]
    idx = [i for i, v in zip(idx, dv) if v > 1e-12 * max(dv.max(), 1e-300)]
    if not idx:
        return np.nan, 0, np.nan
    b = model.coef().values[idx]
    Vs = V[np.ix_(idx, idx)]
    ev, U = np.linalg.eigh((Vs + Vs.T) / 2)
    keep = ev > 1e-10 * max(abs(ev).max(), 1e-300)
    z = U[:, keep].T @ b
    q = int(keep.sum())
    F = float((z ** 2 / ev[keep]).sum()) / q
    return F, q, float(_st.f.sf(F, q, n_clusters(model) - 1))


# ── Wild cluster bootstrap (WCR, Rademacher) ────────────────────────────────
def _demean(M, fe_codes, tol=1e-12, maxit=1000):
    """Remove fixed effects from the columns of M by alternating projections
    (exact after one sweep for a balanced two-way panel)."""
    M = np.array(M, dtype=float, copy=True)
    counts = [np.bincount(g) for g in fe_codes]
    for _ in range(maxit):
        delta = 0.0
        for g, c in zip(fe_codes, counts):
            for j in range(M.shape[1]):
                mu = np.bincount(g, weights=M[:, j], minlength=len(c)) / np.maximum(c, 1)
                M[:, j] -= mu[g]
                delta = max(delta, float(np.abs(mu).max()))
        if delta < tol:
            break
    return M


def wild_p(df, y, xcols, weights, fe, cluster="block", B=9999, seed=20261014, dist="rademacher"):
    """Wild cluster restricted bootstrap p-value (symmetric, on the t)
    for H0: sum_k w_k * beta_k = 0 in  y ~ xcols | fe,  clustered on `cluster`.

    The null is imposed by reparametrisation: with pivot j (w_j != 0),
    theta = sum w_k beta_k is the coefficient on X_j / w_j once every other
    X_k is replaced by X_k - X_j w_k / w_j; the restricted model drops it.
    Data are FE-demeaned first (FWL). Cluster sums are vectorised over draws.
    dist: "rademacher" (default, +-1) or "webb" (Webb 2023 six-point weights
    +-sqrt(1/2), +-1, +-sqrt(3/2), each with probability 1/6; better with few clusters).
    """
    import pandas as pd
    d = df[[y] + list(xcols) + list(fe) + [cluster]].dropna()
    fe_codes = [pd.factorize(d[f])[0] for f in fe]
    g = pd.factorize(d[cluster])[0]
    G = int(g.max()) + 1
    X = d[list(xcols)].to_numpy(float)
    names = list(xcols)
    j = next(k for k, n in enumerate(names) if weights.get(n, 0) != 0)
    wj = weights[names[j]]
    Z = np.empty_like(X)
    Z[:, 0] = X[:, j] / wj
    others = [k for k in range(len(names)) if k != j]
    for c, k in enumerate(others, start=1):
        Z[:, c] = X[:, k] - X[:, j] * weights.get(names[k], 0) / wj
    M = _demean(np.column_stack([d[y].to_numpy(float), Z]), fe_codes)
    yt, Zt = M[:, 0], M[:, 1:]
    k = Zt.shape[1]
    ssc = G / (G - 1)

    def cluster_sum(A):            # (n, m) -> (G, m)
        return np.column_stack([np.bincount(g, weights=A[:, i], minlength=G) for i in range(A.shape[1])])

    A = np.linalg.inv(Zt.T @ Zt)
    a = A[0]                       # row giving theta
    # Original t
    beta = A @ (Zt.T @ yt)
    e = yt - Zt @ beta
    q = cluster_sum(Zt * e[:, None]) @ a
    t0 = beta[0] / np.sqrt(ssc * (q ** 2).sum())
    # Restricted fit (theta = 0)
    if k > 1:
        Zr = Zt[:, 1:]
        br = np.linalg.lstsq(Zr, yt, rcond=None)[0]
        yhat = Zr @ br
    else:
        yhat = np.zeros_like(yt)
    ur = yt - yhat
    S = cluster_sum(Zt * ur[:, None])                 # (G, k): Z_g' u_r,g
    C = cluster_sum(Zt * yhat[:, None])               # (G, k): Z_g' yhat_g
    H = np.stack([cluster_sum(Zt * Zt[:, [i]]) for i in range(k)], axis=2)  # (G, k, k)
    rng = np.random.default_rng(seed)
    if dist == "rademacher":
        v = rng.choice([-1.0, 1.0], size=(G, B))
    elif dist == "webb":
        w6 = np.sqrt(np.array([0.5, 1.0, 1.5]))
        v = rng.choice(np.concatenate([-w6, w6]), size=(G, B))
    else:
        raise ValueError(f"unknown wild bootstrap weights: {dist}")
    ZY = C.sum(0)[:, None] + S.T @ v                  # (k, B) = Z'y*
    bstar = A @ ZY                                    # (k, B)
    # cluster scores of bootstrap residuals: C_g + v_g S_g - H_g b*
    sc = C[:, :, None] + S[:, :, None] * v[:, None, :] - np.einsum("gkl,lb->gkb", H, bstar)
    qb = np.einsum("k,gkb->gb", a, sc)
    tb = bstar[0] / np.sqrt(ssc * (qb ** 2).sum(0))
    return float(np.mean(np.abs(tb) >= abs(t0)))
