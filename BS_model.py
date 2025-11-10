# Import all required libraries
import yfinance as yf
import pandas as pd
import numpy as np
np.random.seed(8309)
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib as mpl
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator
from plotly.subplots import make_subplots
from scipy.optimize import fmin
from scipy.optimize import minimize,least_squares,dual_annealing,differential_evolution

def black_scholes_call(S, K, T, r, sigma):
    """
    Price a European call option using the Black-Scholes closed-form formula.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.

    Returns
    -------
    float
        Theoretical price of the European call option.
    """
    # Handle edge cases
    if T <= 1e-10:
        # At maturity, return intrinsic value
        return max(S - K, 0)
    if sigma <= 1e-10:
        return max(S - K * np.exp(-r * T), 0)
    if K <= 1e-10:
        # No meaningful strike price
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def newton_implied_vol(S, K, T, r, option_price, sigma_init=0.2, tol=1e-6, max_iter=100):
    """
    Estimate implied volatility using the Newton-Raphson method.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    option_price : float
        Observed market price of the option.
    sigma_init : float, optional
        Initial guess for volatility, default is 0.2.
    tol : float, optional
        Convergence tolerance for the algorithm, default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations, default is 100.

    Returns
    -------
    float
        Implied volatility estimate, or ``np.nan`` if convergence fails.
    """
    # Handle edge case when T is 0 or very small
    if T <= 1e-10:
        return np.nan
    sigma = sigma_init
    for i in range(max_iter):
        if sigma <= 1e-10:
            sigma = tol
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        model_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        diff = model_price - option_price
        if abs(diff) < tol:
            return sigma
        if vega == 0:
            # Vega is zero, Newton-Raphson breaks down
            return np.nan
        sigma = sigma - diff / vega
    return np.nan

def implied_vol(S, K, T, r, option_price, sigma_low=1e-6, sigma_high=5.0):
    """
    Compute implied volatility using a hybrid bracketing/Newton approach.

    The function first attempts to locate the root with Brent's method. If that
    fails, it falls back to a Newton-Raphson search.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    option_price : float
        Observed market price of the option.
    sigma_low : float, optional
        Lower bound for the volatility search interval, default is 1e-6.
    sigma_high : float, optional
        Upper bound for the volatility search interval, default is 5.0.

    Returns
    -------
    float
        Implied volatility estimate, or ``np.nan`` if the root-finding fails.
    """
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - option_price
    try:
        return brentq(objective, sigma_low, sigma_high)
    except ValueError:
        return newton_implied_vol(S, K, T, r, option_price, sigma_init=0.2, tol=1e-6, max_iter=100)
