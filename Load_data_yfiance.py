import pandas as pd
import numpy as np
from BS_model import implied_vol
np.random.seed(8309)
import datetime
import tkinter as tk
from tkinter import filedialog

def idx_min(data):
    moneyness = np.abs(data["S0"]-data["strike"])
    return data.loc[moneyness.idxmin()]

def get_Today_data(Ticker, r):
  # Get today's date
  today = datetime.date.today()
  if today.weekday() == 5:  # Saturday
      Today = (today - datetime.timedelta(days=1)).isoformat()
  elif today.weekday() == 6:  # Sunday
      Today = (today - datetime.timedelta(days=2)).isoformat()
  else:
      Today = today.isoformat()
  # Today's underling price (assume lastest)
  hist_1y = Ticker.history(period="1y")
  price = hist_1y['Close']
  mean = price.mean()
  var = price.var()
  std = np.sqrt(var)
  hist = Ticker.history(period="1d")
  underly = hist['Close'].iloc[-1]
  exp_list = Ticker.options
  all = []
  for exp in exp_list:
    # don't want ttm to be 0
    if exp == Today:
        continue
    if (pd.to_datetime(exp) - pd.to_datetime(Today)).days >= 365:
        break
    chain = Ticker.option_chain(exp)
    # Extract call and put options separately, add appropriate columns, and merge into a single DataFrame
    calls = chain.calls[['strike', 'lastPrice', 'lastTradeDate']].copy()
    calls['CP'] = 0 # 0 for call options
    puts = chain.puts[['strike', 'lastPrice', 'lastTradeDate']].copy()
    puts['CP'] = 1 # 1 for put options
    option = pd.concat([calls, puts], ignore_index=True)
    # if strike is too far from ann_mean, delete the option
    lower_bound = mean - 2.5 * std
    upper_bound = mean + 2.5 * std
    option.loc[option['strike'] < lower_bound, 'strike'] = np.nan
    option.loc[option['strike'] > upper_bound, 'strike'] = np.nan
    option = option.dropna()
    # Only Today
    option['lastTradeDate'] = pd.to_datetime(option['lastTradeDate'])
    option['exp'] = exp
    option['S0'] = underly
    # Calculate TTM
    option['ttm'] = ((pd.to_datetime(option['exp']) - pd.to_datetime(Today)).dt.days)/252
    option["exp_month"] = pd.to_datetime(option['exp']).dt.strftime('%y%m')
    option["r"] = r
    option["in_out"] = np.where(((option["CP"]==0) & (option["strike"]>option["S0"]))|((option["CP"]==1) & (option["strike"]<option["S0"])),"out","in")
    if not option.empty:
      all.append(option)
  option = pd.concat(all, ignore_index=True)
  # find at-the-money contract
  atm_contract = option.groupby(["exp_month","CP"]).apply(idx_min, include_groups=False)
  # For each exp_month, keep only those where both C and P are present
  counts = atm_contract.reset_index().groupby('exp_month')['CP'].nunique()
  valid_exp_months = counts[counts == 2].index
  atm_contract = atm_contract.loc[atm_contract.index.get_level_values(0).isin(valid_exp_months)]
  # Forward price
  idx = pd.IndexSlice
  C = atm_contract.loc[idx[:,0],"lastPrice"].values
  P = atm_contract.loc[idx[:,1],"lastPrice"].values
  K = atm_contract.loc[idx[:,0],"strike"].values
  T = atm_contract.loc[idx[:,0],"ttm"].values
  r = atm_contract.loc[idx[:,0],"r"].values
  F = np.exp(r*T)*(C-P)+K
  # Extract exp_month from the MultiIndex
  exp_months = atm_contract.index.get_level_values(0)[::2].unique()  # Get unique exp_months for calls
  F_data =  pd.DataFrame({"exp_month":exp_months,"F":F})
  option = pd.merge(option,F_data)
  option["q"] = option["r"] - np.log(option["F"]/option["S0"])/option["ttm"]
  calls = option[option["CP"]==0].copy()
  # Calulate Imp Vol
  calls['imp_vol'] = calls.apply(
        lambda row: implied_vol(S=row['S0'], K=row['strike'], T=row['ttm'], r=row['r'], option_price=row['lastPrice']), axis=1
    )
  calls = calls.dropna()
  calls["w"] = calls["imp_vol"]**2 * calls["ttm"]  #计算总方差
  calls["y"] = np.log(calls["strike"]/calls["F"])  #计算远期在值程度
  calls = calls.sort_values(["exp_month","y"])
  return calls, today

def save_to_csv(calls,today):
  today = today.strftime('%Y-%m-%d')
  name = "DataSet/"+today+".csv"
  calls.to_csv(name, index=False)

def read_from_csv(name):
  name = "DataSet/"+name+".csv"
  return pd.read_csv(name)