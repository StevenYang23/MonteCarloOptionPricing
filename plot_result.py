import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
def plot_s_paths(S_path_dupire,S_path_Heston,S_path_Garch,S_path_GBM,m=50):
    """
    Compare simulated asset paths from different stochastic volatility models.

    Parameters
    ----------
    S_path_dupire, S_path_Heston, S_path_Garch, S_path_GBM : numpy.ndarray
        Arrays of shape ``(N + 1, M)`` containing simulated asset prices for
        each framework.
    m : int, optional
        Number of paths to display from each simulation, default 50.
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

    Parameters
    ----------
    V_path_dupire, V_path_Heston, V_path_Garch, V_path_GBM : numpy.ndarray
        Arrays capturing the volatility or variance paths generated under each
        model. Shapes may differ depending on the simulation framework.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(V_path_dupire, label="Local Volatility")
    axs[0, 0].set_title("Dupire Local Vol")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Volatility")
    axs[0, 0].legend()
    axs[0, 1].plot(V_path_Heston, label="Heston Vol")
    axs[0, 1].set_title("Heston Vol (A Random Path)")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Volatility")
    axs[0, 1].legend()
    axs[1, 0].plot(V_path_Garch, label="GARCH Vol")
    axs[1, 0].set_title("GARCH Vol (A Random Path)")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Volatility")
    axs[1, 0].legend()
    axs[1, 1].plot(V_path_GBM, label="GBM Vol (constant)")
    axs[1, 1].set_title("Constant Vol GBM")
    axs[1, 1].set_xlabel("Time Step")
    axs[1, 1].set_ylabel("Volatility")
    axs[1, 1].legend()
    plt.tight_layout()
    plt.show()

def plot_log_norm(local_vol_payouts, heston_payouts, Garch_Vol_payouts, Constant_vol_payouts, bigtitle, dist=True, K=None, B=None):
    """
    Visualize payout distributions and optionally overlay log-normal fits.

    Parameters
    ----------
    local_vol_payouts, heston_payouts, Garch_Vol_payouts, Constant_vol_payouts : array-like
        Discounted payout samples generated from the respective models.
    bigtitle : str
        Title applied to the full figure.
    dist : bool, optional
        If ``True``, attempt to fit and overlay a log-normal distribution on
        positive payout observations, default ``True``.
    K : float, optional
        Strike level to mark with a vertical reference line, if provided.
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
        if B is not None:
            ax.axvline(B, color='red', linestyle='--', linewidth=1.5, label=f'Knock-out barrier B={B}')

        ax.set_xlabel("Payout")
        ax.set_ylabel("Density")
        ax.legend()
    
    fig.suptitle(bigtitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_CI(local_vol_val, heston_val, Garch_Vol_val, Constant_vol_val, alpha=0.05):
    """
    Display confidence intervals for option valuations under multiple models.

    Parameters
    ----------
    local_vol_val, heston_val, Garch_Vol_val, Constant_vol_val : array-like
        Collections of discounted option values produced by each model.
    alpha : float, optional
        Significance level used when computing two-sided confidence intervals,
        default ``0.05`` for 95% CIs.
    """
    # Compute stats
    vals = [local_vol_val, heston_val, Garch_Vol_val, Constant_vol_val]
    labels = ['Local Vol', 'Heston', 'GARCH', 'Const. Vol']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    means = []
    cis = []
    std_errs = []
    for v in vals:
        n = len(v)
        mean = np.mean(v)
        std_err = np.std(v, ddof=1) / np.sqrt(n)
        z = norm.ppf(1 - alpha / 2)
        ci = (mean - z * std_err, mean + z * std_err)
        means.append(mean)
        cis.append(ci)
        std_errs.append(std_err)

    # Labels with stats (for legend)
    labels_with_stats = [
        f'{lab} (μ={m:.4f}, σ={s:.4f})'
        for lab, m, s in zip(labels, means, std_errs)
    ]

    # Determine global x-range for smart placement
    all_x = [ci[0] for ci in cis] + [ci[1] for ci in cis] + means
    x_min, x_max = min(all_x), max(all_x)
    x_range = x_max - x_min
    margin = 0.05 * x_range  # 5% margin
    plot_x_min, plot_x_max = x_min - margin, x_max + margin

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (mean, ci, label, color) in enumerate(zip(means, cis, labels, colors)):
        # Draw CI line
        ax.hlines(y=i, xmin=ci[0], xmax=ci[1], color=color, linewidth=4)
        # Draw mean marker
        ax.plot(mean, i, 'o', color=color, markersize=8, zorder=5)

        # 🎯 Smart label placement:
        # If mean is in left 40% → label on right; else on left
        if mean < x_min + 0.4 * x_range:
            ha = 'left'
            x_label = ci[1] + 0.02 * x_range  # just right of CI
        else:
            ha = 'right'
            x_label = ci[0] - 0.02 * x_range  # just left of CI

        # Place label with background for readability
        ax.text(
            x_label, i,
            labels_with_stats[i],
            va='center',
            ha=ha,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="lightgray", alpha=0.8),
            zorder=10
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_xlabel('Mean Discounted Option Value')
    ax.set_title(f'{int((1-alpha)*100)}% Confidence Intervals for Each Model')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_sharkfin(B,K):
    """
    Plot the shark fin payoff profile as a function of terminal price.

    Parameters
    ----------
    B : float
        Barrier level at which the option knocks out.
    K : float
        Strike price of the call payoff.
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

    Parameters
    ----------
    payout : array-like | float
        Payoff amount(s) to discount.
    r : float
        Continuous risk-free rate used for discounting.
    T : float
        Time to maturity in years.
    """
    return payout*np.exp(-r*T)

def valuation(path,K,B,r,AMR=True):
    """
    Value a shark fin option by applying the knock-out barrier to simulated paths.

    Parameters
    ----------
    path : numpy.ndarray
        Simulated price paths with shape ``(N + 1, M)``.
    K : float
        Strike price of the call option.
    B : float
        Knock-out barrier level.
    r : float
        Continuous discount rate.
    AMR : bool, optional
        If ``True``, treat the payoff as path-dependent (American monitoring of
        the barrier); if ``False``, only the terminal price is inspected.

    Returns
    -------
    numpy.ndarray
        Discounted payoff for each path.
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
