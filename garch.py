import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def Garch_simulation(omega, alpha, beta, mu_annual, vol_annual, N, M, T, S0):
    """
    Simulate asset paths using a GARCH(1,1) volatility process.

    Parameters
    ----------
    omega, alpha, beta : float
        GARCH(1,1) parameters governing the variance dynamics.
    mu_annual : float
        Annualized drift of the underlying asset.
    vol_annual : float
        Annualized volatility used to initialize the variance path.
    N : int
        Number of discrete time steps.
    M : int
        Number of Monte Carlo paths.
    T : float
        Simulation horizon in years.
    S0 : float
        Initial asset price.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Simulated price paths and conditional variances with shapes
        ``(N + 1, M)`` and ``(N + 1, M)`` respectively.
    """
    np.random.seed(8309)
    dt = T / N  # Time increment per step
    noise = np.zeros((M, N + 1))
    sigt = np.zeros((M, N + 1))
    CRN = np.random.normal(0, 1, (M, N + 1))
    sigt[:, 0] = (vol_annual ** 2 * dt)  # Initial per-step conditional variance
    noise[:, 0] = np.sqrt(sigt[:, 0]) * CRN[:, 0]  # Fixed: Use sqrt(sigt) consistently

    for t in range(1, N + 1):
        sigt[:, t] = omega + alpha * noise[:, t - 1] ** 2 + beta * sigt[:, t - 1]
        noise[:, t] = np.sqrt(sigt[:, t]) * CRN[:, t]

    rtn = np.zeros((M, N + 1))
    rtn[:, 1:] = (mu_annual * dt) + noise[:, 1:]  # Per-step drift scaled by dt
    paths = S0 * np.exp(np.cumsum(rtn, axis=1))
    return paths.T, sigt.T  # Transposed for (time, paths) and (time, M) shapes

def get_param_garch(ticker):
    """
    Estimate GARCH(1,1) parameters from historical data downloaded via yfinance.

    Parameters
    ----------
    ticker : str
        Symbol for the asset to calibrate.

    Returns
    -------
    numpy.ndarray
        Array containing ``[omega, alpha, beta]`` estimates.
    """
    ticker = ticker
    start_date = "2025-10-01"
    df = yf.download(ticker, start=start_date)['Close']

    # Step 2: Compute log-returns (in decimal; scale to % if preferred)
    returns = np.log(df / df.shift(1)).dropna().values
    n = len(returns)

    # Step 3: GARCH(1,1) negative log-likelihood function
    def garch_loglik(params, returns):
        """
        Compute the negative log-likelihood for a GARCH(1,1) process.
        """
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Initial unconditional variance
        loglik = 0.0
        
        for t in range(1, n):
            # minimize the negative log-likelihood
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            if sigma2[t] <= 0:  # Penalize invalid variances
                return 1e10
            # Normal log-pdf (up to constants; mean=0 under risk-neutral, but here unconditional)
            loglik -= 0.5 * (np.log(2 * np.pi * sigma2[t]) + (returns[t]**2 / sigma2[t]))
        
        return -loglik / n  # Average for stability

    def neg_loglik(params):
        return garch_loglik(params, returns)

    # Step 4: Optimize (use sample std as starting ω)
    init_omega = np.var(returns) * (1 - 0.05 - 0.90)  # Rough: ω = V * (1 - α - β)
    init_params = [init_omega, 0.05, 0.90]  # Typical equity starts
    bounds = [(1e-8, None), (0, 0.999), (0, 0.999)]  # Ensure α + β < 1 via soft constraint

    res = minimize(neg_loglik, init_params, method='L-BFGS-B', bounds=bounds)
    return res.x