import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_two_interp(interp, regres, calls):
    """
    Visualize implied volatility surfaces from spline and regression interpolators.

    Parameters
    ----------
    interp : callable
        Interpolant returning implied volatility for strike and maturity grids.
    regres : callable
        Polynomial regression model for implied volatility.
    calls : pandas.DataFrame
        Option dataset containing strike, time-to-maturity, and implied volatility.
    """
    # Create shared grid
    strike_range = np.linspace(calls['strike'].min(), calls['strike'].max(), 60)
    ttm_range = np.linspace(calls['ttm'].min(), calls['ttm'].max(), 60)
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