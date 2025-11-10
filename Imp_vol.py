import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
np.random.seed(8309)
def imp_vol_simulation(mu_annual, S0, T, N, M, reg):
    """
    Simulate price paths using implied volatility supplied by a regression model.

    Parameters
    ----------
    mu_annual : float
        Annualized drift of the underlying asset.
    S0 : float
        Initial asset price.
    T : float
        Time horizon in years.
    N : int
        Number of timesteps.
    M : int
        Number of Monte Carlo paths.
    reg : callable
        Regression function returning implied volatility for given strikes and maturities.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Simulated asset price paths with shape ``(N + 1, M)`` and the implied
        volatility term structure along the path.
    """
    np.random.seed(8309)
    dt = T / N
    t_grid = np.linspace(0, T, N + 1)          # shape (N+1,)
    ttm_grid = T - t_grid[:-1]                 # shape (N,) — time-to-maturity at each step

    # Use ATM strike ≈ forward from S0: K(ttm) = S0 * exp(mu * ttm)
    k_atm = S0 * np.exp(mu_annual * ttm_grid)

    # Get implied vol for ATM strike at each ttm
    sigma_vals = reg(k_atm, ttm_grid)  # shape (N,)

    # Ensure sigma_vals is 1D and positive
    sigma_vals = np.maximum(sigma_vals, 1e-6)

    # Draw all normals: (M, N)
    Z = np.random.randn(M, N)

    # Drift and diffusion (note: use mu_annual, not undefined 'mu')
    drift = (mu_annual - 0.5 * sigma_vals**2) * dt          # shape (N,)
    diffusion = sigma_vals * np.sqrt(dt) * Z                # shape (M, N)

    # Cumulative log-price
    logS0 = np.log(S0)
    logS = logS0 + np.hstack([np.zeros((M, 1)), np.cumsum(drift + diffusion, axis=1)])

    S = np.exp(logS)
    return S.T, sigma_vals