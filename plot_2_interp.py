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

def plot_two_interp(interp, regres, calls):
    # Create shared grid
    strike_range = np.linspace(calls['strike'].min(), calls['strike'].max(), 40)
    ttm_range = np.linspace(calls['ttm'].min(), calls['ttm'].max(), 40)
    strike_grid, ttm_grid = np.meshgrid(strike_range, ttm_range)
    # Evaluate surfaces
    cubic_surface = interp(strike_grid.ravel(), ttm_grid.ravel()).reshape(strike_grid.shape)
    linear_surface = regres(strike_grid.ravel(), ttm_grid.ravel()).reshape(strike_grid.shape)
    # Data point scatter (shared)
    scatter_template = dict(
        x=calls['strike'],
        y=calls['ttm'],
        z=calls['imp_vol'],
        mode='markers',
        marker=dict(size=3, color=calls['imp_vol'], colorscale='Viridis', opacity=0.8),
        showlegend=False
    )
    # Subplots: 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            "Cubic (Constrained Z)",
            "Cubic (Unconstrained Z)",
            "Linear (Unconstrained Z)"
        )
    )
    # Common surface settings
    common_surf = dict(colorscale='Viridis', opacity=0.7, showscale=False, showlegend=False)
    # Left: Cubic + constrained
    fig.add_trace(go.Scatter3d(**scatter_template), row=1, col=1)
    fig.add_trace(go.Surface(x=strike_grid, y=ttm_grid, z=cubic_surface, **common_surf), row=1, col=1)
    # Middle: Cubic + unconstrained
    fig.add_trace(go.Scatter3d(**scatter_template), row=1, col=2)
    fig.add_trace(go.Surface(x=strike_grid, y=ttm_grid, z=cubic_surface, **common_surf), row=1, col=2)
    # Right: Linear + unconstrained
    fig.add_trace(go.Scatter3d(**scatter_template), row=1, col=3)
    fig.add_trace(go.Surface(x=strike_grid, y=ttm_grid, z=linear_surface, **common_surf), row=1, col=3)
    # Constrain Z only for left plot
    zmin = float(calls['imp_vol'].min())
    zmax = float(calls['imp_vol'].max())
    fig.update_scenes(
        dict(zaxis=dict(range=[zmin, zmax])),
        row=1, col=1
    )
    # Middle and right: auto z-range (no update needed)
    # Axis labels
    scene_template = dict(
        xaxis_title='Strike',
        yaxis_title='Time to Maturity (years)',
        zaxis_title='Implied Volatility'
    )
    for col in [1, 2, 3]:
        fig.update_scenes(scene_template, row=1, col=col)

    fig.update_layout(
        height=500,
        width=1500,
        title_text="Implied Volatility Surface: Cubic vs Linear Interpolation (Constrained vs Unconstrained)"
    )
    fig.show()