import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib as mpl
np.random.seed(8309)

def dupire_local_vol(calls,spline):
    calls_ = calls.copy()
    calls_ = calls[calls["in_out"]=="out"].copy()  #只用虚值期权来计算局部波动率
    local_vol_surface = get_local_variance(calls,spline)
    return local_vol_surface

def get_spline(calls):  #用来事先计算好期限的样条函数值，后续就不用反复计算了
    spline = []
    for m in calls["ttm"].unique():
        moneyness = calls.loc[calls["ttm"] == m]["y"]
        sample_volatility = calls.loc[calls["ttm"] == m]["w"]
        cs_k = CubicSpline(x = moneyness,y = sample_volatility,extrapolate=True)
        spline.append(cs_k)
    return spline

def get_total_v(data,spline,y,t):
    total_v = [float(cs(y)) for cs in spline]
    f = interp1d(x =data["ttm"].unique(),y = total_v,kind = "linear",fill_value="extrapolate") #平远期插值相当于总方差的线性插值
    v = float(f(t))
    return v

# 计算w对t，y的导数
def diff(data,spline,y,t):
    yt = get_total_v(data,spline,y,t)
    y_up = get_total_v(data,spline,y*(1+0.001),t)
    y_down = get_total_v(data,spline,y*(1-0.001),t)
    t_up = get_total_v(data,spline,y,t*(1+0.001))

    dw_dt = (t_up - yt)/(t*0.001)
    dw_dy = (y_up - y_down)/(y*0.001*2)
    dw_dy2 = (y_up + y_down - 2*yt)/(y*0.001)**2
    return dw_dt,dw_dy,dw_dy2

def local_v(data,spline,y,t):
    # w = get_total_v(data,spline,y,t)
    # dw_dt,dw_dy,dw_dy2 = diff(data,spline,y,t)
    # numetator = dw_dt
    # denonimator = 1-y/w*dw_dy + 0.25*(-0.25 - 1/w + y**2/w**2) * (dw_dy**2) + 0.5* dw_dy2
    # local_variance = numetator / denonimator
    # if local_variance < 0:  #若存在套利机会，很可能会出现算出的结果为负数，这里简单处理一下
    #     local_variance  = 1e-8
    # vol = np.sqrt(local_variance)
    # return vol
    return 0.1*np.random.rand()

def get_local_variance(data,spline):
    t_array = np.linspace(data["ttm"].min(),data["ttm"].max(),60)
    y_array = np.linspace(data["y"].min(),data["y"].max(),60)   
    t, y = np.meshgrid(t_array,y_array) 
    v = np.zeros_like(y)

    # 循环确定每个点的隐含波动率，上一步已经存储了k维度上的样条函数，循环的时候还需要t维度的样条插值，这样可以计算任意点的隐含波动率值
    for t_idx,t1 in enumerate(t_array):
        for y_idx,y1 in enumerate(y_array):
            v[y_idx,t_idx] = local_v(data,spline,y1,t1)
    return v

def get_local_vol_path(calls, spline, K, T, N):
    S0 = calls["S0"].iloc[0]
    r = calls["r"].iloc[0]
    t_array = np.linspace(0, T, N)
    local_vol_path = np.zeros(N)
    for i in range(N):
        ttm = T - t_array[i]
        y = np.log(S0/K)
        local_vol_path[i] = local_v(data=calls, spline=spline, y=y, t=ttm)
    return local_vol_path


def dupire_simulation(mu_annual, K, S0, T, N, M, calls, spline):
    np.random.seed(8309)  # For reproducibility
    V_path = get_local_vol_path(calls, spline, K, T, N)  # Assumed shape: (N,)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    dW = np.random.normal(0, np.sqrt(dt), (M, N))  # Shape: (M, N)
    
    # Integrated variance: ∫ σ(s)^2 ds ≈ sum σ_j^2 Δt for left Riemann
    sigma = V_path  # (N,)
    inc_var = sigma**2 * dt  # (N,)
    var_int = np.cumsum(np.concatenate(([0.], inc_var)))  # Shape: (N+1,)
    
    # Stochastic integral: ∫ σ(s) dW_s ≈ sum σ_j ΔW_{j+1}
    stoch_dW = sigma[None, :] * dW  # Shape: (M, N)
    stoch_int = np.cumsum(np.insert(stoch_dW, 0, 0, axis=1), axis=1)  # Shape: (M, N+1)
    
    # Drift term
    drift_term = mu_annual * t[None, :] - 0.5 * var_int[None, :]  # Shape: (1, N+1)
    
    # Simulate paths
    S = S0 * np.exp(drift_term + stoch_int)  # Shape: (M, N+1)
    
    return S.T, V_path



    
    
    
