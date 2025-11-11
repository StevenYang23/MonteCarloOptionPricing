import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib as mpl
np.random.seed(8309)

def dupire_local_vol(calls, spline, n_t, n_y):
    """
    Compute Dupire local volatility on a grid using precomputed spline objects.

    Parameters
    ----------
    calls : pandas.DataFrame
        Option dataset containing implied volatilities and associated metadata.
        Only out-of-the-money quotes are used when constructing the surface.
    spline : list
        Collection of spline interpolants for each maturity slice.
    n_t : int
        Number of points in the maturity direction of the output grid.
    n_y : int
        Number of points in the moneyness direction of the output grid.

    Returns
    -------
    numpy.ndarray
        Local volatility surface sampled on the grid produced by
        ``get_local_variance``.
    """
    calls_ = calls.copy()
    calls_ = calls_[calls_["in_out"] == "out"].copy()  # Use only out-of-the-money options
    local_vol_surface = get_local_variance(calls_, spline, n_t, n_y)
    return local_vol_surface

def get_spline(calls):
    """
    Build cubic splines in strike (or log-moneyness) for each maturity.

    Parameters
    ----------
    calls : pandas.DataFrame
        Option dataset containing total variance by moneyness and maturity.

    Returns
    -------
    list
        List of `scipy.interpolate.CubicSpline` objects indexed by maturity.
        When fewer than two unique strikes exist for a maturity, the spline
        entry is set to ``None`` so downstream code can decide how to handle it.
    """
    # Sort by maturity and strike to ensure proper ordering for spline construction.
    calls = calls.sort_values(["ttm","y"])
    spline = []
    for m in calls["ttm"].unique():
        subset = calls.loc[calls["ttm"] == m, ["y", "w"]].dropna()
        # Keep only the first occurrence for each unique 'y' (you can use 'last' or custom logic too)
        unique_pairs = subset.drop_duplicates(subset=["y"], keep="first").sort_values("y")
        moneyness = unique_pairs["y"].values
        if len(moneyness) < 2:
            spline.append(None)
            continue
        sample_volatility = unique_pairs["w"].values
        # moneyness = calls.loc[calls["ttm"] == m]["y"]
        # sample_volatility = calls.loc[calls["ttm"] == m]["w"]
        cs_k = CubicSpline(x=moneyness, y=sample_volatility, extrapolate=True)
        spline.append(cs_k)
    return spline

def get_total_v(data, spline, y, t):
    """
    Interpolate total variance across both strike and maturity dimensions.

    Parameters
    ----------
    data : pandas.DataFrame
        Option dataset containing total variance information.
    spline : list
        Precomputed splines from ``get_spline``.
    y : float
        Log-moneyness variable.
    t : float
        Time to maturity.

    Returns
    -------
    float
        Total variance at the specified log-moneyness and maturity.
    """
    maturities = data["ttm"].unique()
    total_v = []
    valid_t = []
    for m, cs in zip(maturities, spline):
        if cs is None:
            continue
        total_v.append(float(cs(y)))
        valid_t.append(m)
    if not total_v:
        return float("nan")
    if len(total_v) == 1:
        return total_v[0]
    f = interp1d(
        x=np.asarray(valid_t),
        y=np.asarray(total_v),
        kind="linear",
        fill_value="extrapolate",
    )  # Linear interpolation across maturities for total variance
    v = float(f(t))
    return v

def diff(data, spline, y, t):
    """
    Approximate the partial derivatives of the total variance surface using
    symmetric finite differences.

    Parameters
    ----------
    data : pandas.DataFrame
        Option dataset containing total variance information.
    spline : list
        Precomputed splines from ``get_spline``.
    y : float
        Log-moneyness location at which to evaluate the derivatives.
    t : float
        Time to maturity.

    Returns
    -------
    tuple[float, float, float]
        Approximations of ``∂w/∂t``, ``∂w/∂y`` and ``∂²w/∂y²`` evaluated at
        ``(y, t)``.
    """
    yt = get_total_v(data,spline,y,t)
    y_up = get_total_v(data,spline,y*(1+0.001),t)
    y_down = get_total_v(data,spline,y*(1-0.001),t)
    t_up = get_total_v(data,spline,y,t*(1+0.001))

    dw_dt = (t_up - yt)/(t*0.001)
    dw_dy = (y_up - y_down)/(y*0.001*2)
    dw_dy2 = (y_up + y_down - 2*yt)/(y*0.001)**2
    return dw_dt,dw_dy,dw_dy2

def local_v(data, spline, y, t):
    """
    Evaluate the Dupire local variance formula at a single (log-moneyness, maturity)
    point using the supplied total variance surface.

    Parameters
    ----------
    data : pandas.DataFrame
        Option dataset containing total variance information.
    spline : list
        Precomputed splines from ``get_spline``.
    y : float
        Log-moneyness coordinate.
    t : float
        Time to maturity.

    Returns
    -------
    float
        Local volatility level implied by Dupire's formula. Negative values,
        which may occur because of numerical noise or arbitrage, are clipped to
        a small positive floor before taking the square root.
    """
    w = get_total_v(data,spline,y,t)
    dw_dt,dw_dy,dw_dy2 = diff(data,spline,y,t)
    numerator = dw_dt
    denominator = 1 - y / w * dw_dy + 0.25 * (-0.25 - 1 / w + y**2 / w**2) * (dw_dy**2) + 0.5 * dw_dy2
    local_variance = numerator / denominator
    if local_variance < 0:  # Negative values may appear when arbitrage is present; clamp to a small positive value
        local_variance  = 1e-8
    vol = np.sqrt(local_variance)
    return vol
    # return 0.1*np.random.rand()

def get_local_variance(data, spline,n_t,n_y):
    """
    Generate a grid of local variance values over log-moneyness and maturity.

    Parameters
    ----------
    data : pandas.DataFrame
        Option dataset used by ``local_v`` to evaluate total variance.
    spline : list
        Precomputed splines from ``get_spline``.
    n_t : int
        Number of grid points along the maturity axis.
    n_y : int
        Number of grid points along the log-moneyness axis.

    Returns
    -------
    numpy.ndarray
        Local variance values arranged as an array of shape ``(n_y, n_t)``.
    """
    t_array = np.linspace(data["ttm"].min(),data["ttm"].max(),n_t)
    y_array = np.linspace(data["y"].min(),data["y"].max(),n_y)
    t, y = np.meshgrid(t_array,y_array)
    v = np.zeros_like(y)

    # Loop over the grid and evaluate local volatility using the precomputed interpolants
    for t_idx, t1 in enumerate(t_array):
        for y_idx, y1 in enumerate(y_array):
            v[y_idx, t_idx] = local_v(data, spline, y1, t1)
    return v
    
def get_local_vol_path(calls, spline, y, T, N, n_t):
    """
    Sample the local volatility term structure along a single (y, T) path.

    Parameters
    ----------
    calls : pandas.DataFrame
        Option data used by ``local_v`` when evaluating total variance.
    spline : list
        Precomputed volatility interpolants per maturity.
    y : float
        Moneyness (e.g., log(K/F) or K/S0) for the path.
    T : float
        Total maturity (in years).
    N : int
        Number of time steps in the output ``local_vol_path`` (e.g., for SDE simulation).
    n_t : int
        Number of intermediate samples used before interpolating down to ``N`` points.

    Returns
    -------
    numpy.ndarray
        Local volatility values at times ``t_i = i * T / (N - 1)``, ``i = 0, …, N - 1``.
    """
    t_fine = np.linspace(0, T, n_t)
    v = np.empty(n_t)
    
    for i, t in enumerate(t_fine):
        t = max(t, 1e-8)
        val = local_v(data=calls, spline=spline, y=y, t=t)
        v[i] = val if np.isfinite(val) else np.nan

    v_series = pd.Series(v, index=t_fine)
    v_clean = (
        v_series
        .interpolate(method='linear')   # linear fill internal gaps
        .fillna(method='ffill')         # forward fill start
        .fillna(method='bfill')         # backward fill end
    )
    
    # Fallback if all NaN
    if v_clean.isna().all():
        fallback = np.nanmean(calls['imp_vol'])
        fallback = fallback if np.isfinite(fallback) else 0.2  # 20% vol default
        v_clean = pd.Series(np.full(n_t, fallback), index=t_fine)

    v_clean = v_clean.values

    t_out = np.linspace(0, T, N)
    local_vol_path = np.interp(t_out, t_fine, v_clean)

    return local_vol_path

def dupire_simulation(mu_annual, y, S0, T, N, M, calls, spline, n_t):
    """
    Simulate price paths under the Dupire local volatility framework.

    Parameters
    ----------
    mu_annual : float
        Annualized drift of the underlying asset.
    y : float
        Log-moneyness level used to extract the local volatility path.
    S0 : float
        Initial asset price.
    T : float
        Time horizon in years.
    N : int
        Number of time steps.
    M : int
        Number of simulation paths.
    calls : pandas.DataFrame
        Dataset containing market option information.
    spline : list
        Spline interpolants generated by ``get_spline``.
    n_t : int
        Number of intermediate samples used when constructing the local volatility path.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Simulated asset price paths with shape ``(N + 1, M)`` and the local
        volatility path used in the simulation.
    """
    np.random.seed(8309)  # For reproducibility
    V_path = get_local_vol_path(calls, spline, y, T, N, n_t)  # Assumed shape: (N,)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    dW = np.random.normal(0, np.sqrt(dt), (M, N))  # Shape: (M, N)
    
    # Integrated variance: ∫ σ(s)^2 ds ≈ sum σ_j^2 Δt for left Riemann
    sigma = V_path  # (N,)
    inc_var = sigma**2 * dt  # (N,)
    var_int = np.cumsum(np.concatenate(([0.], inc_var)))  # Shape: (N+1,)
    
    # Stochastic integral: ∫ σ(s) dW_s ≈ sum σ_j ΔW_{j+1}
    stoch_dW = sigma[None, :] * dW  # Shape: (M, N)
    stoch_int = np.cumsum(np.insert(stoch_dW, 0, 0, axis=1), axis=1)  # Shape: (M, N+1)
    
    # Drift term
    drift_term = mu_annual * t[None, :] - 0.5 * var_int[None, :]  # Shape: (1, N+1)
    
    # Simulate paths
    S = S0 * np.exp(drift_term + stoch_int)  # Shape: (M, N+1)
    
    return S.T, V_path



    
    
    
