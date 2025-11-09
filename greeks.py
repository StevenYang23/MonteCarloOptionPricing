import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def payoff_shark_fin_call(S_array, K, B):
    """
    单步近似：
    若最终价 S_T >= B，则认为障碍被触及 => Payoff = 0
    否则 Payoff = max(S_T - K, 0)
    """
    return np.where(S_array >= B, 0.0, np.maximum(S_array - K, 0.0))



def compute_mc_greeks_on_grid(payoff_func, S_vals, T_vals,
                            n_sim, r, sigma, K, B,
                            h, vol_h):
    """
    :param payoff_func:   给定S终值时的Payoff函数, 例如 payoff_shark_fin_call
    :param S_vals:        现货价格网格
    :param T_vals:        到期时间网格
    :param n_sim:         每个网格点的蒙特卡洛模拟次数
    :param r, sigma, K, B 等期权参数
    :param h:             现货方向有限差分步长
    :param vol_h:         波动率方向有限差分步长
    :return: (delta_grid, gamma_grid, vega_grid)
    """
    n_T = len(T_vals)
    n_S = len(S_vals)
    delta_grid = np.zeros((n_T, n_S))
    gamma_grid = np.zeros((n_T, n_S))
    vega_grid  = np.zeros((n_T, n_S))

    for i, T in enumerate(T_vals):
        sqrt_T = np.sqrt(T)
        # 对每个 T，重新生成随机数或使用相同随机数均可，这里为了稳定性而每次重采样
        for j, S in enumerate(S_vals):
            # ========== Delta & Gamma ========== 
            # 生成 n_sim 个标准正态随机数
            Z = np.random.standard_normal(n_sim)

            # 在单步模型下，模拟终值:
            # 1) S + h
            S_plus = (S + h) * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)
            # 2) S
            S_center = S * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)
            # 3) S - h
            S_minus = (S - h) * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)

            # 计算贴现期权价值(取平均后再折现)
            V_plus   = np.mean(payoff_func(S_plus,   K, B))   * np.exp(-r * T)
            V_center = np.mean(payoff_func(S_center, K, B))   * np.exp(-r * T)
            V_minus  = np.mean(payoff_func(S_minus,  K, B))   * np.exp(-r * T)
            
            # 利用中央差分求 Delta & Gamma
            delta = (V_plus - V_minus) / (2 * h)
            gamma = (V_plus - 2 * V_center + V_minus) / (h ** 2)
            
            delta_grid[i, j] = delta
            gamma_grid[i, j] = gamma

            # ========== Vega ==========
            # 对波动率做 ±vol_h 扰动，Spot不变
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

def plot_greeks(r,sigma,K,B,S0,h=1,vol_h=0.005,n_sim=100000):
    """
    计算并绘制看涨鲨鱼鳍期权在(S, T)网格上的希腊值 (Delta, Gamma, Vega)
    :param r:         无风险利率
    :param sig:       波动率
    :param K:         行权价
    :param B:         鲨鱼鳍障碍价
    :param S0:        当前现货价格
    :param h:         现货方向有限差分步长
    :param vol_h:     波动率方向有限差分步长
    :param n_sim:     每个网格点的蒙特卡洛模拟路径数
    """
    np.random.seed(8309)

    n_S = 1000
    n_T = 100
    S_vals = np.linspace(S0 * 0.4, S0 * 1.6, n_S)   # [40, 160]
    T_vals = np.linspace(0.1, 2.0, n_T)            # [0.1, 2.0] 年

    delta_grid, gamma_grid, vega_grid = compute_mc_greeks_on_grid(
        payoff_shark_fin_call,
        S_vals, T_vals,
        n_sim, r, sigma, K, B,
        h, vol_h
    )

    # 方便后续 3D 作图
    S_grid, T_grid = np.meshgrid(S_vals, T_vals)   
    plot_greek_surfaces_3d(delta=delta_grid, gamma=gamma_grid, vega=vega_grid, S_grid=S_grid, T_grid=T_grid)