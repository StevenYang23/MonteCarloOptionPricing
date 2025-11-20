import matplotlib.pyplot as plt
import numpy as np

def plot_smiles(calls_out, reg, ttm = 0.8):
    """
    Plot observed and fitted implied volatility smiles for a given maturity.

    Parameters
    ----------
    calls_out : pandas.DataFrame
        Option quotes containing strike, time-to-maturity, and implied vol columns.
    reg : callable
        Regression surface that maps strike and maturity to implied volatility.
    ttm : float, optional
        Target time-to-maturity (in years) around which to visualize the smile.
    """
    n = 60
    tol = 0.005
    # Filter option quotes whose time-to-maturity is close to the target ttm
    filtered = calls_out[np.abs(calls_out['ttm'] - ttm) < tol].copy()
    strike = filtered['strike']
    imp_vol = filtered['imp_vol']
    strike_range = np.linspace(strike.min(),strike.max(),n)
    smail = reg(strike_range,ttm*np.ones(n))
    plt.figure(figsize=(8,5))
    plt.scatter(strike, imp_vol, label="Market IV", color='r', alpha=0.7)
    plt.plot(strike_range, smail, label="Regression Smile", color='b')
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title(f"Implied Volatility Smile at TTM={ttm:.3f}")
    plt.legend()
    plt.show()