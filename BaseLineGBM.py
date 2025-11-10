import numpy as np

def GBM_simulation(vol_annual, mu_annual, S0, T, N, M):
    """
    Simulate geometric Brownian motion price paths and corresponding volatility.

    Parameters
    ----------
    vol_annual : float
        Annualized volatility of the underlying asset.
    mu_annual : float
        Annualized drift of the underlying asset.
    S0 : float
        Initial asset price.
    T : float
        Time horizon of the simulation in years.
    N : int
        Number of time steps in the simulation.
    M : int
        Number of Monte Carlo paths to simulate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Matrix of simulated asset paths with shape ``(N + 1, M)`` and
        an array containing the (constant) volatility path.
    """
    np.random.seed(8309)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    B = np.zeros((M, N + 1))
    for i in range(M):
        for j in range(1, N + 1):
            B[i, j] = B[i, j - 1] + np.random.normal(0, np.sqrt(dt))
    S = S0 * np.exp((mu_annual - 0.5 * vol_annual**2) * t[None, :] + vol_annual * B)
    v_path = np.ones(N + 1) * vol_annual
    return S.T, v_path

