import pandas as pd
import numpy as np
from BS_model import implied_vol
np.random.seed(8309)
import datetime
import tkinter as tk
from tkinter import filedialog

def idx_min(data):
    """
    Return the row with strike closest to the underlying price for a group.
    """
    moneyness = np.abs(data["S0"] - data["strike"])
    return data.loc[moneyness.idxmin()]

def get_Today_data(Ticker, r):
    """
    Retrieve and preprocess daily option chain data (calls only) for a given ticker,
    computing key derived quantities needed for volatility surface modeling and model calibration
    (e.g., Heston, Local Vol, GARCH).

    Parameters
    ----------
    Ticker : yfinance.Ticker
        A yfinance Ticker object (e.g., yf.Ticker("SPY")) providing access to option chains and price history.
    r : float
        Constant risk-free interest rate (annualized, continuously compounded), used to compute forward prices and dividend yields.

    Returns
    -------
    calls : pandas.DataFrame
        Filtered and enriched DataFrame containing **call options only**, with the following columns:

        - `strike`               : Strike price of the option.
        - `lastPrice`            : Last traded price of the option.
        - `lastTradeDate`        : Timestamp of the most recent trade for this contract.
        - `CP`                   : Option type indicator (0 = call, 1 = put); retained for filtering (here always 0).
        - `exp`                  : Expiration date string (YYYY-MM-DD).
        - `S0`                   : Spot price of the underlying asset (closing price from the most recent trading day).
        - `ttm`                  : Time to maturity in **trading years** (actual days / 252).
        - `exp_month`            : Expiration month-year in format 'YYMM' (e.g., '2511' for November 2025).
        - `r`                    : Risk-free rate used (same as input `r`, broadcasted per row).
        - `in_out`               : Moneyness label: 'in' = in-the-money, 'out' = out-of-the-money.
        - `F`                    : Implied forward price for the expiry month, computed via put-call parity using ATM contracts.
        - `q`                    : Implied continuous dividend yield: q = r - (1/T) * log(F/S0).
        - `imp_vol`              : Implied volatility (annualized, Black-Scholes), solved numerically from market price.
        - `w`                    : Total variance: w = σ² * T (used in SVI and calibration contexts).
        - `y`                    : Log-forward moneyness: y = log(K / F), standard normalization for volatility surface.

        Only strikes within ±2.5σ of the 1-year mean price are retained.
        Only expiries with both call and put ATM contracts are used (to ensure reliable forward estimation).
        Rows with missing or invalid data (e.g., failed IV root-finding) are dropped.

    today : datetime.date
        The reference trade date used (adjusted to most recent weekday if today is weekend).

    Notes
    -----
    - ATM forward (`F`) is estimated per expiry using the *nearest-to-ATM* call–put pair:
          F = exp(r*T) * (C - P) + K
      where C, P, K, T correspond to the ATM call and put with same strike and expiry.
    - Dividend yield `q` is derived consistently from the forward: F = S0 * exp((r - q) * T)
    - `imp_vol` uses a user-defined `implied_vol` function (expected to take S, K, T, r, option_price).
    - Time-to-maturity uses 252 trading days/year (standard in equity options).
    """
    # Determine the most recent trading day if today falls on a weekend
    today = datetime.date.today()
    if today.weekday() == 5:  # Saturday
        Today = (today - datetime.timedelta(days=1)).isoformat()
    elif today.weekday() == 6:  # Sunday
        Today = (today - datetime.timedelta(days=2)).isoformat()
    else:
        Today = today.isoformat()

    # Fetch historical data to estimate typical strike bounds
    hist_1y = Ticker.history(period="1y")
    price = hist_1y['Close']
    mean = price.mean()
    var = price.var()
    std = np.sqrt(var)

    hist = Ticker.history(period="1d")
    underly = hist['Close'].iloc[-1]
    exp_list = Ticker.options
    all_options = []
    for exp in exp_list:
        # Skip expiry dates with zero time to maturity
        if exp == Today:
            continue
        if (pd.to_datetime(exp) - pd.to_datetime(Today)).days >= 365:
            break
        chain = Ticker.option_chain(exp)
        # Extract call and put options separately, add appropriate columns, and merge into a single DataFrame
        calls = chain.calls[['strike', 'lastPrice', 'lastTradeDate']].copy()
        calls['CP'] = 0  # 0 for call options
        puts = chain.puts[['strike', 'lastPrice', 'lastTradeDate']].copy()
        puts['CP'] = 1  # 1 for put options
        option = pd.concat([calls, puts], ignore_index=True)
        # Remove strikes outside a +/- 2.5σ band around the 1-year average price
        lower_bound = mean - 2.5 * std
        upper_bound = mean + 2.5 * std
        option.loc[option['strike'] < lower_bound, 'strike'] = np.nan
        option.loc[option['strike'] > upper_bound, 'strike'] = np.nan
        option = option.dropna()
        # Annotate contract metadata
        option['lastTradeDate'] = pd.to_datetime(option['lastTradeDate'])
        option['exp'] = exp
        option['S0'] = underly
        # Calculate time-to-maturity in trading years
        option['ttm'] = ((pd.to_datetime(option['exp']) - pd.to_datetime(Today)).dt.days)/252
        option["exp_month"] = pd.to_datetime(option['exp']).dt.strftime('%y%m')
        option["r"] = r
        option["in_out"] = np.where(
            ((option["CP"] == 0) & (option["strike"] > option["S0"])) |
            ((option["CP"] == 1) & (option["strike"] < option["S0"])),
            "out",
            "in",
        )
        if not option.empty:
            all_options.append(option)
    option = pd.concat(all_options, ignore_index=True)

    # Find at-the-money contracts for each expiry
    atm_contract = option.groupby(["exp_month","CP"]).apply(idx_min, include_groups=False)
    # Keep expiries where both call and put are present
    counts = atm_contract.reset_index().groupby('exp_month')['CP'].nunique()
    valid_exp_months = counts[counts == 2].index
    atm_contract = atm_contract.loc[atm_contract.index.get_level_values(0).isin(valid_exp_months)]

    # Compute forward prices using parity
    idx = pd.IndexSlice
    C = atm_contract.loc[idx[:,0],"lastPrice"].values
    P = atm_contract.loc[idx[:,1],"lastPrice"].values
    K = atm_contract.loc[idx[:,0],"strike"].values
    T = atm_contract.loc[idx[:,0],"ttm"].values
    r = atm_contract.loc[idx[:,0],"r"].values
    F = np.exp(r*T)*(C-P)+K
    # Extract exp_month from the MultiIndex
    exp_months = atm_contract.index.get_level_values(0)[::2].unique()  # Unique call maturities
    F_data =  pd.DataFrame({"exp_month":exp_months,"F":F})
    option = pd.merge(option,F_data)
    option["q"] = option["r"] - np.log(option["F"]/option["S0"])/option["ttm"]
    calls = option[option["CP"]==0].copy()
    # Calculate implied volatility
    calls['imp_vol'] = calls.apply(
        lambda row: implied_vol(S=row['S0'], K=row['strike'], T=row['ttm'], r=row['r'], option_price=row['lastPrice']), axis=1
    )
    calls = calls.dropna()
    calls["w"] = calls["imp_vol"]**2 * calls["ttm"]  # Total variance
    calls["y"] = np.log(calls["strike"]/calls["F"])  # Forward moneyness
    calls = calls.sort_values(["exp_month","y"])
    return calls, today

def save_to_csv(calls, today):
    """
    Persist the processed option data to a dated CSV file in the DataSet folder.
    """
    today = today.strftime('%Y-%m-%d')
    name = "DataSet/"+today+".csv"
    calls.to_csv(name, index=False)

def read_from_csv(name):
    """
    Load previously saved option data from the DataSet folder.
    """
    name = "DataSet/"+name+".csv"
    return pd.read_csv(name)