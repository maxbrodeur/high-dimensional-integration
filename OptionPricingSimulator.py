import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable
import sobol_new as sn
from datetime import datetime
from scipy.optimize import bisect
from scipy.integrate import quad

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

    """
    Stock price analytic solution
    params:
        w: Brownian motion path (m,N) (column vector for each N)
    return: S(t) (m,N)
    """
    def S(self, w: np.ndarray) -> np.ndarray:
        time_dependent_coeff = self.S0*np.exp((self.r - self.sigma**2/2)*self.t) # shape (m,1)
        brownian_motion_coeff = np.exp(self.sigma*np.einsum('ij,i->ij', w, self.t)) # shape (m,N)
        S = np.einsum('ij,i->ij', brownian_motion_coeff, time_dependent_coeff) # shape (m,N)
        return S

    """
    Jump part of payoff function
    params:
        w: Brownian motion path (m,N) (column vector for each N)
    return: phi(w) (m,N)
    """
    def phi(self, w: np.ndarray) -> np.ndarray:
        assert self.t.shape[0] == w.shape[0], f"should be as many time steps as values in Brownian motion path, got {self.t.shape[0]} time steps and {w.shape[0]} values in Brownian motion path"
        # payoff is average of S(t) over time over strike price 
        return np.sum(self.S(w), axis=0)/self.m - self.K
    
    # Asian call option (Psi_1)
    def Asian(self, w: np.ndarray) -> float:
        mask = self.phi(w)>0
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
        for m, t in enumerate(self.t):
            for k in range(n_vars):
                if k == 0:
                    eta[m,k] = t
                else:
                    # k = 2^(n-1) + i with i = 1,...,2^(n-1)
                    n = np.floor(np.log2(k)) + 1
                    i = (k+1) - 2**(n-1)
                    eta[m,k] = self.eta(n,i,t)
        return eta

    def eta(self, n: int, i: int, t: float) -> float:
        left = (2*i-2)/(2**n)
        middle = (2*i-1)/(2**n)
        right = (2*i)/(2**n)
        if t < left or t > right:
            return 0
        elif t < middle:
            return 2**((n-1)/2)*(t-(2*i-2)/(2**n))
        else:
            return 2**((n-1)/2)*((2*i)/(2**n)-t) # TODO: what to do with ti=middle
        
    """
    Generate uniform column vectors
    params:
        N: number of vectors
    return: y (m,N) (column vector for each N)
    """
    def generate_uniform_vectors(self, N:int) -> np.ndarray:
        return np.random.uniform(size=(self.m,N))
    
    """
    CDF inverse of standard normal distribution
    params:
        y: uniform vectors
    return: CDF_inverse(y)
    """
    def CDF_inverse(self, y: np.ndarray) -> np.ndarray:
        return st.norm.ppf(y)
    
    """ Monte Carlo simluation
        fn: function to apply to each column vector
        y: uniform column vectors (m,N)
        matrix: transformation matrix (Cholesky or Levy-Ciesielski)
    """
    def MC(self, fn: Callable, y: np.ndarray, matrix: np.ndarray) -> float:
        N = y.shape[1]
        interest_coeff = np.exp(-self.r*self.T)
        fn_vars = fn(matrix@self.CDF_inverse(y)) # transpose since its eta @ column vector
        assert len(fn_vars) == N, f"Psi(matrix@CDF_inverse(y)) should be a vector of length {N}, got shape {fn_vars.shape}"
        MC_mean = np.mean(fn_vars)
        var = np.var(fn_vars)
        mse = var/N
        return interest_coeff * MC_mean, mse
    
    """ Crude Monte Carlo simulation
        fn: function we are approximating
        N: number of points
    """
    def crude_MC(self, fn: Callable, N: int, transformation: str = "Cholesky", preintegrated: bool = False) -> float:
        
        if transformation == "Cholesky":
            matrix = np.linalg.cholesky(self.build_C())
        elif transformation == "Levy-Ciesielski":
            LC_N = int(np.log2(self.m))
            matrix = self.build_eta(LC_N)
        else:
            raise NotImplementedError(f"transformation {transformation} not implemented")

        if preintegrated:
            y = st.uniform.rvs(size=(self.m-1,N)) # shape (m-1,N)
            j = 0
            return self.preintegrated_MC(fn, y, matrix, j)
        else:
            y = self.generate_uniform_vectors(N)
            return self.MC(fn, y, matrix)

    """ Randomized Quasi Monte Carlo simulation (Sobol sequence)
        fn: function we are approximating
        N: number of points
        K: number of RQMC simulations
    """
    def randomized_QMC(self, fn: Callable, N: int, K: int, transformation: str = "Cholesky", preintegrated: bool = False) -> float:

        if transformation == "Cholesky":
            matrix = np.linalg.cholesky(self.build_C())
        elif transformation == "Levy-Ciesielski":
            LC_N = int(np.log2(self.m))
            matrix = self.build_eta(LC_N)

        if preintegrated:
            P = sn.generate_points(self.m-1, N) # shape (m-1,N)
            U = st.uniform.rvs(size=(K,self.m-1)) # shape (K,m-1)
        else:
            P = sn.generate_points(self.m, N) # shape (m,N)
            U = self.generate_uniform_vectors(K).T # shape (K,m)

        Vi_list = np.zeros(K)

        for i in range(K):
            shift = U[i].reshape(-1,1) # reshape to column vector
            shifted_P = (P + shift) % 1
            assert shifted_P.shape == P.shape, f"shifted_P shape {shifted_P.shape} should be equal to P shape {P.shape}"
            if preintegrated:
                j = 0
                Vi_list[i], _ = self.preintegrated_MC(fn, shifted_P, matrix, j)
            else:
                Vi_list[i], _ = self.MC(fn, shifted_P, matrix)
            
        mse = np.var(Vi_list)/K # TODO: should we consider the interest coefficient here?
        Vi = np.mean(Vi_list)

        return Vi, mse
    
    # Preintegration ================================================================

    """ Preintegrated Monte Carlo simulation
        fn: function we are approximating
        ymj: uniform column vectors (m-1,N) – ymj as in y_{-j}
        matrix: transformation matrix (Cholesky or Levy-Ciesielski)
        j: index of preintegrated variable
    """
    def preintegrated_MC(self, psi: Callable, ymj: np.ndarray, matrix: np.ndarray, j: int) -> float:
        assert ymj.shape[0] == self.m-1, f"ymj should have shape (m-1,N), got {ymj.shape}"

        N = ymj.shape[1]
        psi_vars = np.zeros(N) 
        
        for i, ymj_column in enumerate(ymj.T): # transpose to iterate over columns
            
            # find root of boundary psi
            def boundary_psi(yj: float) -> float:
                y = np.insert(ymj_column, j, yj, axis=0) # insert y_j at index j
                y = y.reshape(-1,1) # reshape to column vector
                return self.phi(matrix@self.CDF_inverse(y)) 

            yj_root, r = bisect(boundary_psi, 0, 1, xtol=1e-25, full_output=True)
            if not r.converged:
                print(f"Warning: bisect did not converge for yj_root, j={j}, i={i}")
                print(f"\tymj_column: {ymj_column}")
                print(f"\tyj_root: {yj_root}")
            # quadrature of psi
            # wrap psi to take only yj as input
            def psi_j(yj: float) -> float:
                y = np.insert(ymj_column, j, yj, axis=0)
                y = y.reshape(-1,1) # reshape to column vector
                return psi(matrix@self.CDF_inverse(y))

            psi_var, _ = quad(psi_j, yj_root, 1) 
            psi_vars[i] = psi_var

        MC_mean = np.mean(psi_vars)
        var = np.var(psi_vars)
        mse = var/N

        return MC_mean, mse

    def _print_eta(self):
        curr_m = self.m
        with open('eta.txt', 'w') as f:
            for n in range(4):
                m=2**n
                self.update_m(m)
                LC_N = int(np.log2(self.m))
                eta = self.build_eta(LC_N)
                f.write(f"m: {m}\t")
                f.write(f"N: {LC_N}\n")
                f.write(f"eta:\n")
                # Print the matrix
                for row in eta:
                    f.write('\t')
                    for value in row:
                        f.write(f'{value:.2f} ')
                    f.write('\n')
                f.write('\n')
        self.update_m(curr_m)
    
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

    