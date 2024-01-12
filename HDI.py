import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable, List, Tuple
import sobol_new as sn
from datetime import datetime
import json

class OptionPricingSimulator: 

    def __init__(self, K: float, S0: float, r: float, sigma: float, T: float, m: int, logfile: str = None ) -> None:
        self.K = K
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.m = m
        self.dt = T/m
        self.t = np.arange(1, m+1)*self.dt

        # logfile init
        if logfile:
            self.logfile = logfile
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.logfile = f"log_{timestamp}.txt"

        with open(self.logfile, 'w') as f:
            f.write(f"K={K}, S0={S0}, r={r}, sigma={sigma}, T={T}, m={m}\n")

    def log(self, log_msg: str) -> str:
        with open(self.logfile, 'a') as f:
            f.write(log_msg + '\n')
        return log_msg

    # update dimensions (an dt by consequence)
    def update_m(self, m: int) -> None:
        self.m = m
        self.dt = self.T/m
        self.t = np.arange(1, self.m+1)*self.dt

    # stock price analytic solution
    def S(self, w: np.ndarray) -> np.ndarray:
        time_dependent_coeff = self.S0*np.exp((self.r - self.sigma**2/2)*self.t) # shape (m,1)
        brownian_motion_coeff = np.exp(self.sigma*np.einsum('ij,j->ij', w, self.t)) # shape (N,m)
        S = np.einsum('ij,j->ij', brownian_motion_coeff, time_dependent_coeff) # shape (N,m)
        return S

    # jump part of payoff function (average of S(t) over time)
    def phi(self, w: np.ndarray) -> np.ndarray:
        assert self.t.shape[0] == w.shape[1], f"should be as many time steps as values in Brownian motion path, got {self.t.shape[0]} time steps and {w.shape[1]} values in Brownian motion path"
        # payoff is average of S(t) over time over strike price 
        return np.sum(self.S(w), axis=1)/self.m - self.K
    
    # Asian call option (Psi_1)
    def Asian(self, w: np.ndarray) -> float:
        mask = (self.phi(w)>0)
        return self.phi(w)*mask
    
    # binary digital Asian option (Psi_2)
    def Asian_binary(self, w: np.ndarray) -> float:
        return (self.phi(w)>0).astype(int)

    # build covariance matrix
    def build_C(self) -> np.ndarray:
        C = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                # min(t_i, t_j) i,j = 1,...,m
                C[i,j] = min(i+1,j+1)*self.dt
        return C
    
    # generate uniform vectors
    def generate_uniform_vectors(self, N:int) -> np.ndarray:
        return st.uniform.rvs(size=(N, self.m))

    # multivariate standard normal CDF inverse
    def CDF_inverse(self, y: np.ndarray) -> np.ndarray:
        return st.norm.ppf(y)
    
    # Monte Carlo simulation with Cholesky decomposition
    def Cholesky_MC_sim(self, psi: Callable, y: np.ndarray) -> float:
        A = np.linalg.cholesky(self.build_C())
        interest_coeff = np.exp(-self.r*self.T)
        Psi_vars = psi(self.CDF_inverse(y)@A)
        MC_mean = np.mean(Psi_vars)
        var = np.var(Psi_vars)
        return interest_coeff * MC_mean, var
    
    """ Crude Monte Carlo simulation
        psi: payoff function
        N: number of points
    """
    def crude_MC(self, psi: Callable, N: int) -> float:
        Vi, var = self.Cholesky_MC_sim(psi, self.generate_uniform_vectors(N))
        mse = var/N
        return Vi, mse

    """ Randomized Quasi Monte Carlo simulation (Sobol sequence)
        psi: payoff function
        N: number of points
        K: number of RQMC simulations
    """
    def randomized_QMC(self, psi: Callable, N: int, K: int) -> float:
        P = sn.generate_points(int(N),int(self.m),0)
        U = self.generate_uniform_vectors(K)
        Vi = np.zeros(K)
        for i in range(K):
            shifted_P = (P + U[i]) % 1
            Vi[i], var = self.Cholesky_MC_sim(psi, shifted_P)
        mse = np.var(Vi)/K
        return np.mean(Vi), mse

def main():
    K = 100 # strike price
    S0 = 100 # intial stock price
    r = 0.1 # interest rate
    sigma = 0.1 # volatility
    T = 1 # time to maturity
    qmc_K = 10 # number of QMC simulations

    # dimensions
    m_list = [32, 64, 128, 256, 512]
    
    N_list = 2**np.arange(7, 13) # number of simulations

    # simulation ================================================================
    OPS = OptionPricingSimulator(K, S0, r, sigma, T, m_list[0])

    # Cholesky MC ---------------------------------------------------------------
    crude_MC_dict = {}
    descr = f"Crude Monte Carlo simulation with Cholesky decomposition\nK={K}, S0={S0}, r={r}, sigma={sigma}, T={T}"
    crude_MC_dict['descr'] = descr
    for m in m_list:
        OPS.update_m(m)
        for N in N_list:
            asian_V, asian_mse = OPS.crude_MC(OPS.Asian, N)
            binary_V, binary_mse = OPS.crude_MC(OPS.Asian_binary, N)
            crude_MC_dict.setdefault(m, []).append(((asian_V, binary_V), (asian_mse, binary_mse)))
        
    QMC_dict = {}
    descr = f"Randomized Quasi Monte Carlo simulation with Sobol sequence\nK={K}, S0={S0}, r={r}, sigma={sigma}, T={T}"
    QMC_dict['descr'] = descr
    for m in m_list:
        OPS.update_m(m)
        for N in N_list:
            asian_V, asian_err = OPS.randomized_QMC(OPS.Asian, N, qmc_K)
            binary_V, binary_err = OPS.randomized_QMC(OPS.Asian_binary, N, qmc_K)
            QMC_dict.setdefault(m, []).append(((asian_V, binary_V), (asian_err, binary_err)))

    # write data to file ========================================================
    # use json format 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"crude_MC_{timestamp}.json", 'w') as f:
        json.dump(crude_MC_dict, f)
    
    with open(f"QMC_{timestamp}.json", 'w') as f:
        json.dump(QMC_dict, f)
    
    # plot results ==============================================================
    plot_results(
        m_list=m_list,
        N_list=N_list,
        data=crude_MC_dict,
        title='Crude Monte Carlo'
    )

    plot_results(
        m_list=m_list,
        N_list=N_list,
        data=QMC_dict,
        title='Quasi Monte Carlo'
    )

def load_data(fname: str) -> dict:
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def plot_results(**kwargs) -> None:
    m_list = kwargs['m_list']
    N_list = kwargs['N_list']
    data = kwargs['data']
    title = kwargs['title']

    fig, axs = plt.subplots(2, 2, figsize=(15,5))
    fig.suptitle(title)
    axs[0,0].set_title('Asian call option')
    axs[0,1].set_title('Binary digital Asian option')
    axs[1,0].set_xlabel('N')
    axs[1,1].set_xlabel('N')
    axs[0,0].set_ylabel('V')
    axs[1,0].set_ylabel('MSE')
    axs[0,1].set_ylabel('V')
    axs[1,1].set_ylabel('MSE')
    for m in m_list:
        axs[0,0].plot(N_list, [data[m][j][0][0] for j in range(len(N_list))], label=f'm={m}')
        axs[0,1].plot(N_list, [data[m][j][0][1] for j in range(len(N_list))], label=f'm={m}')
        axs[1,0].plot(N_list, [data[m][j][1][0] for j in range(len(N_list))], label=f'm={m}')
        axs[1,1].plot(N_list, [data[m][j][1][1] for j in range(len(N_list))], label=f'm={m}')
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    plt.show()



if __name__ == '__main__':
    main()



    
