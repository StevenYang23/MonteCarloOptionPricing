
import numpy as np
from scipy.optimize import minimize
np.random.seed(8309)
i = complex(0,1)

def heston_simulation(S0, r, q, K, T, v0, kappa, theta, rho, sigma, M, N):
    """
    Simulate asset and variance paths under the Heston stochastic volatility model.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield. Currently unused in the drift term.
    K : float
        Strike price; included for API symmetry but unused in simulation.
    T : float
        Time horizon in years.
    v0 : float
        Initial variance level.
    kappa : float
        Mean-reversion speed of the variance process.
    theta : float
        Long-run variance.
    rho : float
        Correlation between the Brownian motions of price and variance.
    sigma : float
        Volatility of volatility.
    M : int
        Number of Monte Carlo paths.
    N : int
        Number of discrete timesteps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Asset price paths and variance paths, each with shape ``(N + 1, M)``.
    """
    np.random.seed(8309)
    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])

    # arrays for storing prices and variances
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)

    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)

    return S, v

# def heston_simulation(S0,r,q,K,T,v0,kappa,theta,rho,sigma,M,N):
#     dt = T/N
#     S_path = np.zeros((N+1,M))
#     V_path = np.zeros((N+1,M))
#     S_path[0] = S0
#     V_path[0] = v0
#     for step in range(1,N+1):
#         rn1 = np.random.standard_normal(M)
#         rn2 = np.random.standard_normal(M)
#         rn2 = rho*rn1 + np.sqrt(1-rho**2)*rn2
#         S_path[step] = S_path[step - 1] * np.exp((r-q-0.5*V_path[step-1])*dt + np.sqrt(V_path[step-1])*np.sqrt(dt)*rn1) # Solution to the geometric Brownian motion
#         V_path[step] = (np.sqrt(V_path[step - 1]) + sigma/2*np.sqrt(dt)*rn2)**2 + kappa*(theta-V_path[step - 1])*dt - sigma**2/4*dt
#         V_path[step] = np.maximum(V_path[step],0)
#     return S_path,V_path

class Heston_SLSQP:
    """
    Calibrate the Heston model parameters using SLSQP optimization.
    """

    def __init__(self,S0,r,q,K,T,option_value,opt_paras):
        """
        Parameters
        ----------
        S0 : float
            Spot price.
        r : float
            Risk-free interest rate.
        q : np.ndarray | float
            Dividend yield(s) aligned with the option data.
        K : np.ndarray
            Strike prices.
        T : np.ndarray
            Time-to-maturity values.
        option_value : np.ndarray
            Observed option values used for calibration.
        opt_paras : dict
            Initial guesses for the Heston parameters.
        """
        self.S0 = S0  # Spot price
        self.r = r  # Risk-free rate
        self.q = q  # Dividend yield
        self.K = K  # Strike prices
        self.T = T  # Time to maturity
        self.option_value = option_value  # Observed option values or implied volatility
        # opt_paras: dictionary with initial guesses for v0, kappa, theta, rho, sigma (e.g. {"v0":0.05,"kappa":5,"theta":0.05,"rho":-0.5,"sigma":0.15})
        self.opt_paras = opt_paras.copy()

    def pricing(self,S0,r,q,K,T,v0,kappa,theta,rho,sigma):
        """
        Price a European call option under the Heston model using Fourier inversion.
        """

        def char_function(phi):
            """
            Characteristic function for the log-price under the Heston model.
            """
            rspi = rho*sigma*phi*i  # Common term in the characteristic function
            d = np.sqrt( (rspi-kappa)**2 + sigma**2 * (phi*i+ phi**2) )
            g = (kappa - rspi + d)/(kappa - rspi - d)

            C = kappa * theta/sigma**2 * ((kappa-rspi+d)*T -2*np.log((1-g*np.exp(d*T))/(1-g)))
            D = (kappa-rspi+d)/sigma**2 * ( (1-np.exp(d*T))/(1-g*np.exp(d*T)) )
            func = np.exp(C + D*v0)
            return func

        def rect_integral(func,lower = 0,upper = 100,N = 1000):
            """
            Numerical integration using the rectangle method with vector support.

            Parameters
            ----------
            func : callable
                Function to integrate.
            lower : float
                Lower integration bound.
            upper : float
                Upper integration bound.
            N : int
                Number of integration subintervals.
            """
            k = np.log(S0/K)+(r-q)*self.T
            result = 0
            dx = (upper-lower)/N
            for j in range(0,N):
                phi = lower + dx * (2*j + 1)/2
                func_value = np.exp(i*phi*k)*func(phi-0.5*i)/(phi**2+1/4)
                result = result + func_value * dx
            return result
        real_integral = np.real(rect_integral(char_function))
        value = np.exp(-q*T) * S0 - np.sqrt(S0*K)*np.exp(-(r+q)*T/2)/np.pi*real_integral  # Call option value
        return value
    def obj_func(self,paras):
        """
        Objective function for calibration: squared error between model and market prices.
        """
        v0,kappa,theta,rho,sigma = paras
        err = np.sum( (self.pricing(self.S0,self.r,self.q,self.K,self.T,v0,kappa,theta,rho,sigma) - self.option_value)**2 )
        return err
    def fit(self):
        """
        Run SLSQP optimization to fit the Heston parameters to market data.
        """
        init_point = tuple(self.opt_paras.values())
        bnds = ([1e-6,1],[1e-6,10],[1e-6,1],[-1,1],[1e-6,1])
        # cons = ({'type': 'ineq', 'fun': lambda x: 2*x[1]*x[2] - x[-1]**2})  # Optional: enforce the Feller condition
        result =  minimize(fun = self.obj_func,x0 =init_point,method = "SLSQP",bounds = bnds).x
        self.opt_paras["v0"],self.opt_paras["kappa"],self.opt_paras["theta"],self.opt_paras["rho"],self.opt_paras["sigma"] = result
        return self.opt_paras
    def predict(self,S0,r,q,K,T):
        """
        Evaluate the calibrated Heston model on new option inputs.
        """
        heston_value = self.pricing(S0,r,q,K,T,self.opt_paras["v0"],self.opt_paras["kappa"],self.opt_paras["theta"],self.opt_paras["rho"],self.opt_paras["sigma"])
        return heston_value

def fit_Heston_model(calls):
    """
    Calibrate the Heston model parameters using market option data.
    """
    parity_data = calls.copy()
    parity_data["parity_call_price"] = np.where(
        parity_data["CP"] == 0,
        parity_data["lastPrice"],
        parity_data["lastPrice"] + np.exp(-parity_data["r"]*parity_data["ttm"]) * (parity_data["F"]-parity_data["strike"])
    )
    parity_data = parity_data.sort_values(["exp_month","CP","strike"])
    fit_option = parity_data
    #fit_option = data_raw[data_raw["CP"]==0]
    S0 = fit_option["S0"].values[0]
    r = fit_option["r"].values[0]
    q = fit_option["q"].values
    K = fit_option["strike"].values
    T = fit_option["ttm"].values
    option_value = fit_option["parity_call_price"].values
    opt_paras = {"v0":0.02,"kappa":2,"theta":0.02,"rho":-0.5,"sigma":0.02}
    lewis_slsqp = Heston_SLSQP(S0,r,q,K,T,option_value,opt_paras)
    opt_paras = lewis_slsqp.fit()
    return opt_paras["v0"],opt_paras["kappa"],opt_paras["theta"],opt_paras["rho"],opt_paras["sigma"]