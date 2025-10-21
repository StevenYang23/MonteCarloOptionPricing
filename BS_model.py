# Import all required libraries
import yfinance as yf
import pandas as pd
import numpy as np
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
    if sigma == 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def newton_implied_vol(S, K, T, r, option_price, sigma_init=0.2, tol=1e-6, max_iter=100):
    sigma = sigma_init
    for i in range(max_iter):
        if sigma <= 0:
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
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - option_price
    try:
        return brentq(objective, sigma_low, sigma_high)
    except ValueError:
        return newton_implied_vol(S, K, T, r, option_price, sigma_init=0.2, tol=1e-6, max_iter=100)
