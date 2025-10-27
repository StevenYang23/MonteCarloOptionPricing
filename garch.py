import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def Garch_simulation(omega, alpha, beta, N, M, T, S0, r, q):
    np.random.seed(8309)
    dt = T / N
    uncond_var = omega / (1 - alpha - beta)
    paths = np.empty((M, N + 1))
    paths[:, 0] = S0  # Vectorized init
    for m in range(M):
        sigma2 = np.zeros(N + 1)
        log_returns = np.zeros(N)
        sigma2[0] = uncond_var
        for t in range(1, N + 1):
            # GARCH update
            prev_return_sq = 0 if t == 1 else log_returns[t - 2] ** 2
            sigma2[t] = omega + alpha * prev_return_sq + beta * sigma2[t - 1]
            # Risk-neutral log-return
            drift = (r - q) * dt - 0.5 * sigma2[t]
            shock = np.sqrt(sigma2[t]) * np.random.normal()
            log_returns[t - 1] = drift + shock
            # Update path
            paths[m, t] = paths[m, t - 1] * np.exp(log_returns[t - 1])
    return paths.T

def get_param_garch(ticker):
    # Step 1: Fetch historical SPX data (adjust dates as needed)
    ticker = ticker
    start_date = "2020-01-01"  # ~5 years; increase for more data
    end_date = "2025-10-27"    # Up to current date
    df = yf.download(ticker, start=start_date, end=end_date)['Close']

    # Step 2: Compute log-returns (in decimal; scale to % if preferred)
    returns = np.log(df / df.shift(1)).dropna().values
    n = len(returns)
    log_return = np.mean(returns)
    std_return = np.std(returns)

    # Step 3: GARCH(1,1) negative log-likelihood function
    def garch_loglik(params, returns):
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Initial unconditional variance
        loglik = 0.0
        
        for t in range(1, n):
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