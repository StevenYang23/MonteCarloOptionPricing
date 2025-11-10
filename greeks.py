import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def payoff_shark_fin_call(S_array, K, B):
    """
    Single-step approximation of a shark fin call option payoff.

    If the terminal price ``S_T`` exceeds the barrier ``B``, the payoff is zero.
    Otherwise the payoff equals ``max(S_T - K, 0)``.
    """
    return np.where(S_array >= B, 0.0, np.maximum(S_array - K, 0.0))



def compute_mc_greeks_on_grid(
    payoff_func,
    S_vals,
    T_vals,
    n_sim,
    r,
    sigma,
    K,
    B,
    h,
    vol_h,
):
    """
    Estimate Delta, Gamma, and Vega on a grid using Monte Carlo finite differences.

    Parameters
    ----------
    payoff_func : callable
        Payoff function mapping terminal spot values to payoffs.
    S_vals : array-like
        Grid of spot prices used for finite-difference evaluation.
    T_vals : array-like
        Grid of option maturities.
    n_sim : int
        Number of Monte Carlo samples per grid point.
    r : float
        Risk-free rate.
    sigma : float
        Base volatility level for simulations.
    K : float
        Strike price.
    B : float
        Barrier level for the shark fin structure.
    h : float
        Finite-difference step in the spot dimension.
    vol_h : float
        Finite-difference step in the volatility dimension.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays containing Delta, Gamma, and Vega values over the grid.
    """
    n_T = len(T_vals)
    n_S = len(S_vals)
    delta_grid = np.zeros((n_T, n_S))
    gamma_grid = np.zeros((n_T, n_S))
    vega_grid  = np.zeros((n_T, n_S))

    for i, T in enumerate(T_vals):
        sqrt_T = np.sqrt(T)
        # Resample noise for each maturity to maintain independence
        for j, S in enumerate(S_vals):
            # ========== Delta & Gamma ==========
            # Draw n_sim standard normal shocks
            Z = np.random.standard_normal(n_sim)

            # Simulate terminal prices in a single-step model
            # 1) S + h
            S_plus = (S + h) * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)
            # 2) S
            S_center = S * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)
            # 3) S - h
            S_minus = (S - h) * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)

            # Discount the average payoff
            V_plus   = np.mean(payoff_func(S_plus,   K, B))   * np.exp(-r * T)
            V_center = np.mean(payoff_func(S_center, K, B))   * np.exp(-r * T)
            V_minus  = np.mean(payoff_func(S_minus,  K, B))   * np.exp(-r * T)
            
            # Central finite differences for Delta & Gamma
            delta = (V_plus - V_minus) / (2 * h)
            gamma = (V_plus - 2 * V_center + V_minus) / (h ** 2)
            
            delta_grid[i, j] = delta
            gamma_grid[i, j] = gamma

            # ========== Vega ==========
            # Perturb volatility by ±vol_h with spot held constant
            Z_vol = np.random.standard_normal(n_sim)
            # sigma + vol_h
            S_sigma_plus = S * np.exp((r - 0.5 * (sigma + vol_h)**2) * T
                                    + (sigma + vol_h) * sqrt_T * Z_vol)
            # sigma - vol_h
            S_sigma_minus = S * np.exp((r - 0.5 * (sigma - vol_h)**2) * T
                                    + (sigma - vol_h) * sqrt_T * Z_vol)

            V_sigma_plus  = np.mean(payoff_func(S_sigma_plus,  K, B)) * np.exp(-r * T)
            V_sigma_minus = np.mean(payoff_func(S_sigma_minus, K, B)) * np.exp(-r * T)

            vega = (V_sigma_plus - V_sigma_minus) / (2 * vol_h)
            vega_grid[i, j] = vega
            
    return delta_grid, gamma_grid, vega_grid

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_greek_surfaces_3d(delta, gamma, vega, S_grid, T_grid):
    """
    Plot Delta, Gamma, and Vega as three side-by-side 3D surfaces in a 1×3 grid.
    
    Parameters:
    - delta, gamma, vega: 2D arrays of Greek values (shape: len(T_grid) × len(S_grid))
    - S_grid: 1D array of spot prices (x-axis)
    - T_grid: 1D array of maturities (y-axis)
    """
    # Create 1×3 subplot layout with 3D scene types
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=["Delta", "Gamma", "Vega"],
        horizontal_spacing=0.02
    )

    greeks = [("Delta", delta), ("Gamma", gamma), ("Vega", vega)]

    for i, (name, z_data) in enumerate(greeks, start=1):
        fig.add_trace(
            go.Surface(
                z=z_data,
                x=S_grid,
                y=T_grid,
                colorscale='Viridis',
                showscale=False  # avoid repeating colorbar 3 times
            ),
            row=1, col=i
        )

        # Configure each 3D scene
        fig.update_scenes(
            xaxis_title="Spot Price S",
            yaxis_title="Time to Maturity T",
            zaxis_title=name,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),  # consistent view angle
            row=1, col=i
        )

    # Optional: add a shared colorbar (e.g., on the rightmost plot)
    fig.data[-1].update(showscale=True)  # enable colorbar only for last subplot

    fig.update_layout(
        title="Shark Fin Call – Greek Surfaces (1×3)",
        title_x=0.5,
        margin=dict(l=10, r=10, t=60, b=10),
        height=500,
        width=1500  # ~500px per subplot
    )

    fig.show()

def plot_greeks(r, sigma, K, B, S0, h=1, vol_h=0.005, n_sim=100000):
    """
    Compute and visualize Delta, Gamma, and Vega for a shark fin call option.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    sigma : float
        Base volatility used in the simulations.
    K : float
        Strike price of the option.
    B : float
        Barrier level of the shark fin structure.
    S0 : float
        Current spot price.
    h : float, optional
        Finite-difference step size in the spot dimension, default 1.
    vol_h : float, optional
        Finite-difference step size in the volatility dimension, default 0.005.
    n_sim : int, optional
        Number of Monte Carlo paths per grid point, default 100000.
    """
    np.random.seed(8309)

    n_S = 1000
    n_T = 100
    S_vals = np.linspace(S0 * 0.4, S0 * 1.6, n_S)   # Spot grid
    T_vals = np.linspace(0.1, 2.0, n_T)            # Maturity grid in years

    delta_grid, gamma_grid, vega_grid = compute_mc_greeks_on_grid(
        payoff_shark_fin_call,
        S_vals, T_vals,
        n_sim, r, sigma, K, B,
        h, vol_h
    )

    # Prepare grids for 3D plotting
    S_grid, T_grid = np.meshgrid(S_vals, T_vals)   
    plot_greek_surfaces_3d(delta=delta_grid, gamma=gamma_grid, vega=vega_grid, S_grid=S_grid, T_grid=T_grid)