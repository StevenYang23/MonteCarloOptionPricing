import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_local_vol_surface(calls, local_vol_surface):
    # Grid for local vol surface (assumed to match KK, TT from caller)
    K_grid = np.linspace(calls["strike"].min(), calls["strike"].max(), 60)
    T_grid = np.linspace(calls["ttm"].min(), calls["ttm"].max(), 60)
    KK, TT = np.meshgrid(K_grid, T_grid)
    # Shared scatter trace for data points (implied vol)
    scatter = go.Scatter3d(
        x=calls['strike'],
        y=calls['ttm'],
        z=calls['imp_vol'],
        mode='markers',
        marker=dict(
            size=3,
            color=calls['imp_vol'],
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Data Points',
        showlegend=False
    )
    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Local Vol (Constrained Z)", "Local Vol (Unconstrained Z)")
    )
    # Surface trace (same for both, but z-axis handling differs)
    surface_trace = go.Surface(
        x=KK,
        y=TT,
        z=local_vol_surface,
        colorscale='Inferno',
        opacity=0.7,
        showscale=False,
        showlegend=False
    )
    # Add traces to both subplots
    fig.add_trace(scatter, row=1, col=1)
    fig.add_trace(surface_trace, row=1, col=1)
    fig.add_trace(scatter, row=1, col=2)
    fig.add_trace(surface_trace, row=1, col=2)
    # Constrain z-axis only in left subplot
    zmin = float(calls['imp_vol'].min())
    zmax = float(calls['imp_vol'].max())
    fig.update_scenes(
        dict(zaxis=dict(range=[zmin, zmax])),
        row=1, col=1
    )
    # Right subplot: leave z-axis auto (no update)
    # Axis labels for both
    scene_template = dict(
        xaxis_title='Strike',
        yaxis_title='Time to Maturity (years)',
        zaxis_title='Volatility'
    )
    fig.update_scenes(scene_template, row=1, col=1)
    fig.update_scenes(scene_template, row=1, col=2)
    fig.update_layout(
        height=600,
        width=1200,
        title='Local Volatility Surface: Constrained vs Unconstrained View',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()