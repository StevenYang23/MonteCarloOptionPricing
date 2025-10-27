import numpy as np
from BS_model import black_scholes_call
np.random.seed(8309)

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
def get_local_vol_path(params,K,T,N):
    T_path = np.linspace(0,T,N)
    V_path = np.zeros_like(T_path)
    for i in range(T_path.shape[0]):
        V_path[i] = dupire(K,T_path[i],params)
    return V_path
def dupire_simulation(K,S0,T,N,M,params):
    np.random.seed(8309)
    dt = 1 / 252
    V_path = get_local_vol_path(params,K,T,N)
    dW = np.random.normal(0, np.sqrt(dt), size=(N, M))
    S_path = np.zeros((N+1, M))
    S_path[0] = S0
    for t in range(1, N+1):
        S_path[t] = S_path[t-1] + V_path[t-1] * dW[t-1]
    return S_path, V_path