import numpy as np
import plotly.graph_objects as go

def plot_local_vol_surface(calls, local_vol_surface):
    """
    Render a Dupire local volatility surface over moneyness and maturity.
    """
    # Grid for local vol surface (must match the shape of local_vol_surface)
    K_grid = np.linspace(calls["y"].min(), calls["y"].max(), local_vol_surface.shape[1])
    T_grid = np.linspace(calls["ttm"].min(), calls["ttm"].max(), local_vol_surface.shape[0])
    KK, TT = np.meshgrid(K_grid, T_grid)

    fig = go.Figure(data=[
        go.Surface(
            x=KK,
            y=TT,
            z=local_vol_surface,
            colorscale='Inferno',
            opacity=0.7,
            showscale=True,
            colorbar=dict(title="Volatility", len=0.75, x=1.02)
        )
    ])

    fig.update_layout(
        title='Local Volatility Surface',
        scene=dict(
            xaxis_title='Moneyness',
            yaxis_title='Time to Maturity (years)',
            zaxis_title='Volatility',
            zaxis=dict(range=[0, 1.3]),
            aspectratio=dict(x=1, y=0.8, z=0.6)
        ),
        height=600,
        width=800,
        margin=dict(l=0, r=50, b=0, t=40)
    )
    
    fig.show()