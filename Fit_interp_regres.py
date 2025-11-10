import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CloughTocher2DInterpolator
np.random.seed(8309)

def get_poly_features(strike, ttm):
    """
    Construct polynomial features of strike and time-to-maturity.

    Parameters
    ----------
    strike : array-like
        Strike prices associated with the option data.
    ttm : array-like
        Time-to-maturity values expressed in years.

    Returns
    -------
    numpy.ndarray
        Design matrix containing engineered polynomial features.
    """
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
    """
    Build interpolators for implied volatility using regression and triangulation.

    Parameters
    ----------
    calls : pandas.DataFrame
        Option dataset containing strike, maturity, and implied volatility.

    Returns
    -------
    tuple
        ``(interp, regres)`` containing the Clough-Tocher interpolator and a
        polynomial regression callable that both approximate implied volatility.
    """
    X = get_poly_features(calls["strike"].values, calls["ttm"].values)
    y = calls["imp_vol"].values
    model = LinearRegression().fit(X, y)

    def regres(K, T):
        """
        Predict implied volatility using the fitted polynomial regression model.
        """
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
        return result

    points = np.column_stack([calls["strike"].values, calls["ttm"].values])
    vals = calls["imp_vol"].values
    interp = CloughTocher2DInterpolator(points, vals)
    return interp, regres