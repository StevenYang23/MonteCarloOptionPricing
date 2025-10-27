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
        imp = impvol_interp(K,T)
        if imp > 0 and imp < 0.4:
            return imp
        else:
            return 0.1* np.random.uniform(0,1)
    rt = dc_dt/(0.5*K**2*d2c_dk2)
    if rt > 0 and rt < 0.4:
        return rt
    else:
        return 0.1* np.random.uniform(0,1)
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


def dupire_simulation(mu_annual, K, S0, T, N, M, params):
    np.random.seed(8309)  # For reproducibility
    V_path = get_local_vol_path(params, K, T, N)  # Assumed shape: (N,)
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



    
    
    
