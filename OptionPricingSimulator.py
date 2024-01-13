import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable, List, Tuple
import sobol_new as sn
from datetime import datetime

class OptionPricingSimulator: 

    def __init__(self, K: float, S0: float, r: float, sigma: float, T: float, m: int = 8) -> None:
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
    
    # build eta matrix (for Levy-Ciesielski transformation)
    def build_eta(self, N: int) -> np.ndarray:
        n_vars = int(2**N)
        eta = np.zeros((self.m, n_vars))
        for t in range(self.m):
            for k in range(n_vars):
                ti = (t+1)*self.dt
                if k == 0:
                    eta[t,k] = ti
                else:
                    # k = 2^(n-1) + i with i = 1,...,2^(n-1)
                    n = np.floor(np.log2(k)) + 1
                    i = (k+1) - 2**(n-1)
                    eta[t,k] = self.eta(n,i,ti)
        return eta

    def eta(self, n: int, i: int, ti: float) -> float:
        left = (2*i-2)/(2**n)
        middle = (2*i-1)/(2**n)
        right = (2*i)/(2**n)
        if ti < left or ti > right:
            return 0
        elif ti < middle:
            return 2**((n-1)/2)*(ti-(2*i-2)/(2**n))
        else:
            return 2**((n-1)/2)*((2*i)/(2**n)-ti) # TODO: what to do with ti=middle
        
    # generate uniform vectors
    def generate_uniform_vectors(self, N:int) -> np.ndarray:
        return st.uniform.rvs(size=(N, self.m))

    # multivariate standard normal CDF inverse
    def CDF_inverse(self, y: np.ndarray) -> np.ndarray:
        return st.norm.ppf(y)
    
    """ Monte Carlo simluation
        psi: payoff function (Asian or Asian_binary)
        y: uniform vectors 
        matrix: transformation matrix (Cholesky or Levy-Ciesielski)
    """
    def MC(self, psi: Callable, y: np.ndarray, matrix: np.ndarray) -> float:
        interest_coeff = np.exp(-self.r*self.T)
        Psi_vars = psi(self.CDF_inverse(y)@matrix)
        MC_mean = np.mean(Psi_vars)
        var = np.var(Psi_vars)
        mse = var/y.shape[0]
        return interest_coeff * MC_mean, mse
    
    """ Crude Monte Carlo simulation
        psi: payoff function
        N: number of points
    """
    def crude_MC(self, psi: Callable, N: int, transformation: str = "Cholesky") -> float:
        if transformation == "Cholesky":
            A = np.linalg.cholesky(self.build_C())
            return self.MC(psi, self.generate_uniform_vectors(N), A)
        elif transformation == "Levy-Ciesielski":
            eta = self.build_eta(np.log2(self.m))
            return self.MC(psi, self.generate_uniform_vectors(N), eta)
        else:
            raise NotImplementedError(f"transformation {transformation} not implemented")

    """ Randomized Quasi Monte Carlo simulation (Sobol sequence)
        psi: payoff function
        N: number of points
        K: number of RQMC simulations
    """
    def randomized_QMC(self, psi: Callable, N: int, K: int, transformation: str = "Cholesky") -> float:
        P = sn.generate_points(N,self.m)
        U = self.generate_uniform_vectors(K)
        Vi_list = np.zeros(K)
        for i in range(K):
            shifted_P = (P + U[i]) % 1
            if transformation == "Cholesky":
                Vi_list[i], var = self.Cholesky_MC(psi, shifted_P)
            elif transformation == "Levy-Ciesielski":
                Vi_list[i], var = self.Levy_Ciesielski_MC(psi, shifted_P)
            else:
                raise NotImplementedError(f"transformation {transformation} not implemented")
        mse = np.var(Vi_list)/K # TODO: should we consider the interest coefficient here?
        Vi = np.mean(Vi_list)
        return Vi, mse
    
if __name__ == "__main__":
    # set parameters
    K = 100
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1
    m = 1

    # create simulator
    simulator = OptionPricingSimulator(K, S0, r, sigma, T, m)

    