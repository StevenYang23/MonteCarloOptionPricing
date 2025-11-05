import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats as stats
def plot_s_paths(S_path_dupire,S_path_Heston,S_path_Garch,S_path_GBM,m=50):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(S_path_dupire[:,:m])
    axs[0].set_title("Dupire Local Vol Simulated Paths")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Asset Price")
    axs[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axs[1].plot(S_path_Heston[:,:m])
    axs[1].set_title("Heston Simulated Paths")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Asset Price")
    axs[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axs[2].plot(S_path_Garch[:,:m])
    axs[2].set_title("GARCH Simulated Paths")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Asset Price")
    axs[2].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axs[3].plot(S_path_GBM[:,:m])
    axs[3].set_title("Constant Vol Paths")
    axs[3].set_xlabel("Time Step")
    axs[3].set_ylabel("Asset Price")
    axs[3].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.tight_layout()
    plt.show()
def plot_v_paths(V_path_dupire,V_path_Heston,V_path_Garch,V_path_GBM):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(V_path_dupire, label="Dupire Local Vol")
    axs[0, 0].set_title("Dupire Local Vol")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Volatility or Variance")
    axs[0, 0].legend()
    axs[0, 1].plot(np.mean(V_path_Heston, axis=1), label="Heston Vol (Avg)")
    axs[0, 1].set_title("Heston Vol (Avg)")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Volatility or Variance")
    axs[0, 1].legend()
    axs[1, 0].plot(np.mean(V_path_Garch, axis=1), label="GARCH Vol (Avg)")
    axs[1, 0].set_title("GARCH Vol (Avg)")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Volatility or Variance")
    axs[1, 0].legend()
    axs[1, 1].plot(V_path_GBM, label="GBM Vol (constant)")
    axs[1, 1].set_title("Constant Vol GBM")
    axs[1, 1].set_xlabel("Time Step")
    axs[1, 1].set_ylabel("Volatility or Variance")
    axs[1, 1].legend()
    plt.tight_layout()
    plt.show()

def plot_payouts(local_vol_payouts, heston_payouts, Garch_Vol_payouts, Constant_vol_payouts,bigtitle):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    distributions = [
        (local_vol_payouts, 'Dupire Local Vol Payouts', 'tab:blue', axs[0]),
        (heston_payouts, 'Heston Model Payouts', 'tab:orange', axs[1]),
        (Garch_Vol_payouts, 'GARCH Vol Payouts', 'tab:green', axs[2]),
        (Constant_vol_payouts, 'Constant Vol GBM Payouts', 'tab:red', axs[3])
    ]
    for payouts, title, color, ax in distributions:
        # Reduce to strictly positive payouts for lognorm fit
        payouts_fit = np.array(payouts)
        payouts_fit = payouts_fit[payouts_fit > 0]
        # Plot hist (all payouts, still shows zero bucket)
        ax.hist(payouts, bins=50, alpha=0.7, color=color, density=True, label="Empirical Hist")
        # Fit log-normal only if there are >1 strictly positive observations
        if len(payouts_fit) > 1:
            shape, loc, scale = stats.lognorm.fit(payouts_fit, floc=0)
            xmin = min(payouts_fit)
            xmax = np.percentile(payouts_fit, 99.5)
            x = np.linspace(xmin, xmax, 500)
            pdf_vals = stats.lognorm.pdf(x, shape, loc, scale)
            ax.plot(x, pdf_vals, color='black', lw=2, label="LogNormal fit")
            mu = np.mean(payouts)
            std = np.std(payouts)
            ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}\nlognorm: s={shape:.2f}, scale={scale:.2f}')
        else:
            mu = np.mean(payouts)
            std = np.std(payouts)
            ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}\nlognorm: not enough >0')
        ax.set_xlabel("Payout")
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle(bigtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_payouts_gamma(local_vol_payouts, heston_payouts, Garch_Vol_payouts, Constant_vol_payouts,bigtitle):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    distributions = [
        (local_vol_payouts, 'Dupire Local Vol Payouts', 'tab:blue', axs[0]),
        (heston_payouts, 'Heston Model Payouts', 'tab:orange', axs[1]),
        (Garch_Vol_payouts, 'GARCH Vol Payouts', 'tab:green', axs[2]),
        (Constant_vol_payouts, 'Constant Vol GBM Payouts', 'tab:red', axs[3])
    ]
    for payouts, title, color, ax in distributions:
        payouts = np.array(payouts)
        # Use all data for histogram (including zeros)
        ax.hist(payouts, bins=50, alpha=0.7, color=color, density=True, label="Empirical Hist")
        
        # Filter strictly positive payouts for Gamma fit
        payouts_fit = payouts[payouts > 0]
        
        if len(payouts_fit) > 1:
            # Fit Gamma distribution with loc fixed at 0
            shape, loc, scale = stats.gamma.fit(payouts_fit, floc=0)
            xmin = min(payouts_fit)
            xmax = np.percentile(payouts_fit, 99.5)
            x = np.linspace(xmin, xmax, 500)
            pdf_vals = stats.gamma.pdf(x, shape, loc=loc, scale=scale)
            ax.plot(x, pdf_vals, color='black', lw=2, label="Gamma fit")
            mu = np.mean(payouts)
            std = np.std(payouts)
            ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}\nGamma: α={shape:.2f}, θ={scale:.2f}')
        else:
            mu = np.mean(payouts)
            std = np.std(payouts)
            ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}\nGamma: not enough >0')
        
        ax.set_xlabel("Payout")
        ax.set_ylabel("Density")
        ax.legend()
    
    fig.suptitle(bigtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
