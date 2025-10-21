import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CloughTocher2DInterpolator

def get_poly_features(strike, ttm):
    strike = np.asarray(strike)
    ttm = np.asarray(ttm)
    features = np.column_stack([
        np.sqrt(ttm),
        ttm,
        ttm**2,
        ttm**3,
        np.sqrt(strike),
        strike,
        strike**2,
        strike**3,
        strike*ttm,
        strike*ttm**2,
        strike**2*ttm,
        strike**2*ttm**2,
        np.ones_like(ttm)  # intercept
    ])
    return features

def get_interp(calls):
  X = get_poly_features(calls["strike"].values, calls["ttm"].values)
  y = calls["imp_vol"].values
  model = LinearRegression().fit(X, y)
  def regres(K, T):
      # Convert inputs to arrays for consistent processing
      K = np.asarray(K)
      T = np.asarray(T)
      # Check if inputs are scalars
      is_scalar = K.ndim == 0 and T.ndim == 0
      features = get_poly_features(K, T)
      result = model.predict(features)
      # Return scalar if inputs were scalars, otherwise return array
      if is_scalar:
          return result.item()
      else:
          return result
  points = np.column_stack([calls["strike"].values, calls["ttm"].values])
  vals = calls["imp_vol"].values
  interp = CloughTocher2DInterpolator(points, vals)
  return interp,regres