import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable, List, Tuple

class OptionPricingSimulator: 

    def __init__(self, K: float, S0: float, r: float, sigma: float, T: float, m: int) -> None:
        self.K = K
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.m = m
        self.dt = T/m
        self.t = np.arange(1, m+1)*self.dt

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
        return np.max(self.phi(w), 0)
    
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
        return interest_coeff * np.mean(psi(self.CDF_inverse(y)@A))
    
    # crude Monte Carlo simulation (uniform sampling)
    def crude_MC(self, psi: Callable, N: int) -> float:
        return self.Cholesky_MC_sim(psi, self.generate_uniform_vectors(N))

    def randomized_QMC(self, psi: Callable, N: int) -> float:
        pass

def main():
    K = 100 # strike price
    S0 = 100 # intial stock price
    r = 0.1 # interest rate
    sigma = 0.1 # volatility
    T = 1 # time to maturity

    # dimensions
    m_list = [32, 64, 128, 256, 512]
    
    N_list = 2**np.arange(7, 13) # number of simulations

    # simulation ================================================================
    OPS = OptionPricingSimulator(K, S0, r, sigma, T, m_list[0])

    # Cholesky MC ---------------------------------------------------------------
    crude_MC_dict = {}
    for m in m_list:
        OPS.update_m(m)
        crude_MC_dict[m] = [(
                OPS.crude_MC(OPS.Asian, N),
                OPS.crude_MC(OPS.Asian_binary, N))
            for N in N_list]
        
    QMC_dict = {}
    for m in m_list:
        OPS.update_m(m)
        QMC_dict[m] = [(
                OPS.randomized_QMC(OPS.Asian, N),
                OPS.randomized_QMC(OPS.Asian_binary, N))
            for N in N_list]
        
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
        title='Randomized Quasi Monte Carlo'
    )


def plot_results(**kwargs) -> None:
    m_list = kwargs['m_list']
    N_list = kwargs['N_list']
    data = kwargs['data']
    title = kwargs['title']

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(title, fontsize=16)
    ax[0].set_title('Asian Call Option')
    ax[1].set_title('Binary Digital Asian Option')
    ax[0].set_xlabel('N')
    ax[1].set_xlabel('N')
    ax[0].set_ylabel('Option Price')
    ax[1].set_ylabel('Option Price')
    ax[0].set_xscale('log', base=2)
    ax[1].set_xscale('log', base=2)
    # ax[0].set_yscale('log', base=2)
    # ax[1].set_yscale('log', base=2)
    ax[0].set_xticks(N_list)
    ax[1].set_xticks(N_list)
    ax[0].set_xticklabels(N_list)
    ax[1].set_xticklabels(N_list)
    # ax[0].set_yticks(N_list)
    # ax[1].set_yticks(N_list)
    # ax[0].set_yticklabels(N_list)
    # ax[1].set_yticklabels(N_list)
    ax[0].grid(True)
    ax[1].grid(True)

    for m in m_list:
        ax[0].plot(N_list, [x[0] for x in data[m]], label=f'm={m}')
        ax[1].plot(N_list, [x[1] for x in data[m]], label=f'm={m}')
    ax[0].legend()
    ax[1].legend()
    plt.show()
    

if __name__ == '__main__':
    main()



    
