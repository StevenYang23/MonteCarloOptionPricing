
import numpy as np
from scipy.optimize import minimize
np.random.seed(8309)
i = complex(0,1)

def heston_simulation(S0,r,q,K,T,v0,kappa,theta,rho,sigma,M,N):
    np.random.seed(8309)
# def heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M):

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



class Heston_SLSQP:
    def __init__(self,S0,r,q,K,T,option_value,opt_paras):
        self.S0 = S0  #标的价格
        self.r = r  #无风险利率
        self.q = q  #股利
        self.K = K  #行权价格
        self.T = T  #到期时间
        self.option_value = option_value  #隐含波动率
        #opt_paras：优化的v0,kappa,theta,rho,sigma参数，需要字典形式如，{"v0":0.05,"kappa":5,"theta":0.05,"rho":-0.5,"sigma":0.15}
        self.opt_paras = opt_paras.copy()
    def pricing(self,S0,r,q,K,T,v0,kappa,theta,rho,sigma):  #S0,r,q,K,T作为参数，方便后续的predict函数
        def char_function(phi):  #先定义好特征函数
            rspi = rho*sigma*phi*i  # 公式里经常出现这一项，建议先定义好
            d = np.sqrt( (rspi-kappa)**2 + sigma**2 * (phi*i+ phi**2) )
            g = (kappa - rspi + d)/(kappa - rspi - d)

            C = kappa * theta/sigma**2 * ((kappa-rspi+d)*T -2*np.log((1-g*np.exp(d*T))/(1-g)))
            D = (kappa-rspi+d)/sigma**2 * ( (1-np.exp(d*T))/(1-g*np.exp(d*T)) )
            func = np.exp(C + D*v0)
            return func
        def rect_integral(func,lower = 0,upper = 100,N = 1000):  #自定义支持向量运算的积分方式
            """
            lower:积分下限
            upper：积分上限
            N：积分区间数量
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
        value = np.exp(-q*T) * S0 - np.sqrt(S0*K)*np.exp(-(r+q)*T/2)/np.pi*real_integral  #看涨期权价值
        return value
    def obj_func(self,paras):
        v0,kappa,theta,rho,sigma = paras
        err = np.sum( (self.pricing(self.S0,self.r,self.q,self.K,self.T,v0,kappa,theta,rho,sigma) - self.option_value)**2 )
        return err
    def fit(self):
        init_point = tuple(self.opt_paras.values())
        bnds = ([1e-6,1],[1e-6,10],[1e-6,1],[-1,1],[1e-6,1])
        # cons = ({'type': 'ineq', 'fun': lambda x: 2*x[1]*x[2] - x[-1]**2})  #feller condition,可加可不加
        result =  minimize(fun = self.obj_func,x0 =init_point,method = "SLSQP",bounds = bnds).x
        self.opt_paras["v0"],self.opt_paras["kappa"],self.opt_paras["theta"],self.opt_paras["rho"],self.opt_paras["sigma"] = result
        return self.opt_paras
    def predict(self,S0,r,q,K,T):
        heston_value = self.pricing(S0,r,q,K,T,self.opt_paras["v0"],self.opt_paras["kappa"],self.opt_paras["theta"],self.opt_paras["rho"],self.opt_paras["sigma"])
        return heston_value

def fit_Heston_model(calls):
  parity_data = calls.copy()
  parity_data["parity_call_price"] = np.where(parity_data["CP"] == 0,parity_data["lastPrice"],parity_data["lastPrice"] + np.exp(-parity_data["r"]*parity_data["ttm"]) * (parity_data["F"]-parity_data["strike"]))
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