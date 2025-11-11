import matplotlib.pyplot as plt


def plot_hist(calls):
    """
    Plot histograms of time-to-maturity, log-moneyness, and strike from an option dataset.

    Parameters
    ----------
    calls : pandas.DataFrame
        Processed option data containing at least the ``ttm``, ``y``, and
        ``strike`` columns.
    """
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.hist(calls["ttm"],bins=20,color='skyblue',edgecolor='black')
    plt.title("Histogram of TTM")
    plt.xlabel("TTM (years)")
    plt.ylabel("Frequency")
    plt.subplot(1,3,2)
    plt.hist(calls["y"],bins=20,color='salmon',edgecolor='black')
    plt.title("Histogram of Moneyness")
    plt.xlabel("Moneyness (y)")
    plt.ylabel("Frequency")
    plt.subplot(1,3,3)
    plt.hist(calls["strike"],bins=20,color='salmon',edgecolor='black')
    plt.title("Histogram of Strike")
    plt.xlabel("Strike (K)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()