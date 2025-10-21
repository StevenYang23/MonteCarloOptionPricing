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
from BS_model import black_scholes_call

def C(K,T,params):
    S,r,impvol_interp = params
    return black_scholes_call(S, K, T, r, impvol_interp(K,T))
def dupire(K,T,params):
    eps = 1e-4
    dc_dt = (C(K,T+eps,params) - C(K,T,params))/eps
    d2c_dk2 = (C(K+eps,T,params) - 2*C(K,T,params) + C(K-eps,T,params))/eps**2
    if d2c_dk2 <= 0 or dc_dt <= 0:
        S,r,impvol_interp = params
        return impvol_interp(K,T)
    return dc_dt/(0.5*K**2*d2c_dk2)
def get_local_vol_surface(calls,params):
    K_grid = np.linspace(calls["strike"].min(), calls["strike"].max(), 60)
    T_grid = np.linspace(calls["ttm"].min(), calls["ttm"].max(), 60)
    KK, TT = np.meshgrid(K_grid, T_grid)
    # Make a grid of local volatilities over K and T
    local_vol_surface = np.zeros_like(KK)
    for i in range(KK.shape[0]):
        for j in range(KK.shape[1]):
            local_vol_surface[i,j] = dupire(KK[i,j],TT[i,j],params)
    return local_vol_surface