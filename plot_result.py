import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
def plot_s_paths(S_path_dupire,S_path_Heston,S_path_Garch,S_path_GBM,m=50):
    """
    Compare simulated asset paths from different stochastic volatility models.
    """
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
    """
    Plot the evolution of volatility or variance across simulation frameworks.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(V_path_dupire, label="Dupire Local Vol")
    axs[0, 0].set_title("Dupire Local Vol")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Volatility or Variance")
    axs[0, 0].legend()
    axs[0, 1].plot(V_path_Heston, label="Heston Vol")
    axs[0, 1].set_title("Heston Vol (A Random Path)")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Volatility or Variance")
    axs[0, 1].legend()
    axs[1, 0].plot(V_path_Garch, label="GARCH Vol")
    axs[1, 0].set_title("GARCH Vol (A Random Path)")
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

def plot_log_norm(local_vol_payouts, heston_payouts, Garch_Vol_payouts, Constant_vol_payouts, bigtitle, dist=True, K=None):
    """
    Visualize payout distributions and optional log-normal fits.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    distributions = [
        (local_vol_payouts, 'Dupire Local Vol', 'tab:blue', axs[0]),
        (heston_payouts, 'Heston Model', 'tab:orange', axs[1]),
        (Garch_Vol_payouts, 'GARCH Vol', 'tab:green', axs[2]),
        (Constant_vol_payouts, 'Constant Vol GBM', 'tab:red', axs[3])
    ]
    for payouts, title, color, ax in distributions:
        # Plot histogram for all payouts (including zeros)
        ax.hist(payouts, bins=50, alpha=0.7, color=color, density=True, label="Empirical Hist")
        mu = np.mean(payouts)
        std = np.std(payouts)
        if dist:
            # Fit log-normal only on strictly positive payouts
            payouts_fit = np.array(payouts)
            payouts_fit = payouts_fit[payouts_fit > 0]
            if len(payouts_fit) > 1:
                shape, loc, scale = stats.lognorm.fit(payouts_fit, floc=0)
                xmin = min(payouts_fit)
                xmax = np.percentile(payouts_fit, 99.5)
                x = np.linspace(xmin, xmax, 500)
                pdf_vals = stats.lognorm.pdf(x, shape, loc, scale)
                ax.plot(x, pdf_vals, color='black', lw=2, label="LogNormal fit")
                ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}\nlognorm: s={shape:.2f}, scale={scale:.2f}')
            else:
                ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}\nlognorm: not enough >0')
        else:
            # Only show mean and std; no lognorm info
            ax.set_title(f'{title}\nμ={mu:.2f}, σ={std:.2f}')

        # Add vertical line at K if provided
        if K is not None:
            ax.axvline(K, color='gray', linestyle='--', linewidth=1.5, label=f'Strike K={K}')

        ax.set_xlabel("Payout")
        ax.set_ylabel("Density")
        ax.legend()
    
    fig.suptitle(bigtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_CI(local_vol_val,heston_val,Garch_Vol_val,Constant_vol_val,alpha = 0.05):
    """
    Display confidence intervals for option valuations under multiple models.
    """
    mean_local = np.mean(local_vol_val)
    mean_heston = np.mean(heston_val)
    mean_garch = np.mean(Garch_Vol_val)
    mean_constant = np.mean(Constant_vol_val)
    n = local_vol_val.shape[0]
    std_err_local = np.std(local_vol_val, ddof=1) / np.sqrt(n)
    std_err_heston = np.std(heston_val, ddof=1) / np.sqrt(n)
    std_err_garch = np.std(Garch_Vol_val, ddof=1) / np.sqrt(n)
    std_err_constant = np.std(Constant_vol_val, ddof=1) / np.sqrt(n)
    z = norm.ppf(1-alpha/2)
    ci_local = (mean_local - z * std_err_local, mean_local + z * std_err_local)
    ci_heston = (mean_heston - z * std_err_heston, mean_heston + z * std_err_heston)
    ci_garch = (mean_garch - z * std_err_garch, mean_garch + z * std_err_garch)
    ci_constant = (mean_constant - z * std_err_constant, mean_constant + z * std_err_constant)
    means = [mean_local, mean_heston, mean_garch, mean_constant]
    cis = [ci_local, ci_heston, ci_garch, ci_constant]
    stds = [std_err_local, std_err_heston, std_err_garch, std_err_constant]
    labels_with_stats = [
        f'Local Vol (\u03BC={mean_local:.4f}, \u03C3={std_err_local:.4f})',
        f'Heston (\u03BC={mean_heston:.4f}, \u03C3={std_err_heston:.4f})',
        f'GARCH (\u03BC={mean_garch:.4f}, \u03C3={std_err_garch:.4f})',
        f'Const. Vol (\u03BC={mean_constant:.4f}, \u03C3={std_err_constant:.4f})'
    ]
    labels = ['Local Vol', 'Heston', 'GARCH', 'Const. Vol']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mean, ci, label, color) in enumerate(zip(means, cis, labels, colors)):
        ax.hlines(y=i, xmin=ci[0], xmax=ci[1], color=color, linewidth=4, label=label)
        ax.plot(mean, i, 'o', color=color, markersize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean Discounted Option Value')
    ax.set_title('95% Confidence Intervals for Each Model')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    # Add legend with mean and STD next to names, in lower right
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_with_stats, loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_sharkfin(B,K):
    """
    Plot the shark fin payoff profile as a function of terminal price.
    """
    # Define S_T range for payoff plot
    S_T = np.linspace(0, B * 1.1, 500)  # go slightly beyond B
    # Sharkfin payoff (simplified static version: assumes knock-out iff S_T >= B)
    # More accurate: depends on max(S_t), but static diagrams use this convention.
    payoff = np.where(
        S_T < K, 
        0,
        np.where(S_T < B, S_T - K, 0)  # linear between K and B, zero beyond B
    )
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(S_T, payoff, color='darkblue', lw=2.5, label='Sharkfin Payoff')
    plt.axvline(K, color='gray', linestyle='--', lw=1, label=f'Strike K = {K}')
    plt.axvline(B, color='red', linestyle='--', lw=1, label=f'Barrier B = {B:.0f}')

    plt.title('Sharkfin Option Payoff at Maturity\n(Knock-Out with Zero Rebate)', fontsize=14)
    plt.xlabel('Underlying Price at Maturity ($S_T$)', fontsize=12)
    plt.ylabel('Payoff', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(0, B * 1.1)
    plt.ylim(-100, max(payoff) * 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()
def DCF(payout,r,T):
    """
    Discount a payoff from maturity back to present value.
    """
    return payout*np.exp(-r*T)

def valuation(path,K,B,r,AMR=True):
    """
    Value a shark fin option by applying the knock-out barrier to simulated paths.
    """
    ST = path[-1, :]   
    T = path.shape[0]/252
    max_path = ST
    if AMR:                    
        max_path = np.max(path, axis=0)
    call_payoff = np.maximum(ST - K, 0)
    # Knock-out if barrier breached at ANY time
    payoff = np.where(max_path >= B, 0.0, call_payoff)
    DCF_payoff = DCF(payoff,r,T)
    return DCF_payoff
