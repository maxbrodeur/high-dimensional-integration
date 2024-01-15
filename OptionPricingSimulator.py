import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable
import math
import sobol_new as sn
from datetime import datetime
from scipy.optimize import newton
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
        brownian_motion_coeff = np.exp(self.sigma*w) # shape (m,N)
        S = np.einsum('ij,i->ij', brownian_motion_coeff, time_dependent_coeff) # shape (m,N)
        return S
    
    """
    Stock price analytic solution with respect to single variable w_tj
    params:
        wj: Brownian motion path (1,N) | float
    return: S(tj) (m,N)
    """
    def S_j(self, wj: np.ndarray | float, j: int) -> np.ndarray:
        exponent = (self.r - self.sigma**2/2)*self.t[j] + self.sigma*wj*self.t[j]
        return self.S0*np.exp(exponent)

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

    """
    Eta coefficient matrix
    params:
        n,i: indices
        t: time
    return: eta(n,i,t) value
    """
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
    
    # SIMULATION ===================================================================

    """ Launch simulation with given configuration
        args:
            method (str): Crude MC or Randomized QMC
            transformation (str): Cholesky or Levy-Ciesielski
            preintegrated (bool): whether to use preintegration
            N (int): number of simulations
            Psi (str): Asian or Asian binary
            j (int): index of preintegrated variable
    """
    def simulate(self, **args):
        method = args['method']
        transformation = args['transformation']
        preintegrated = args['preintegrated']
        N = args['N']
        Psi = args['Psi']
        if preintegrated:
            j = args['j']
        if method == 'Randomized QMC':
            qmc_K = args['qmc_K']

        # Transformation matrix
        if transformation == "Cholesky":
            matrix = np.linalg.cholesky(self.build_C())
        elif transformation == "Levy-Ciesielski":
            LC_N = int(np.log2(self.m))
            matrix = self.build_eta(LC_N)
        else:
            raise NotImplementedError(f"transformation {transformation} not implemented")
        
        # Psi (remove mask in for preintegration)
        if Psi == 'Asian':
            if preintegrated:
                fn = self.phi
            else:
                fn = self.Asian
        elif Psi == 'Asian binary':
            if preintegrated:
                fn = lambda x: np.ones(x.shape[1]) # collapse columns to 1
            else:
                fn = self.Asian_binary

        if method == 'Crude MC':
            if preintegrated:
                y = st.uniform.rvs(size=(self.m-1,N)) # shape (m-1,N)
                j = 0
                return self.preintegrated_MC(fn, y, matrix, j)
            else:
                y = self.generate_uniform_vectors(N)
                return self.MC(fn, y, matrix)
            
        elif method == 'Randomized QMC':
            if preintegrated:
                P = sn.generate_points(N, self.m-1).T # shape (m-1,N)
                U = st.uniform.rvs(size=(qmc_K,self.m-1)) # shape (K,m-1)
            else:
                P = sn.generate_points(N, self.m).T # shape (m,N)
                U = self.generate_uniform_vectors(qmc_K).T # shape (K,m)

            Vi_list = np.zeros(qmc_K)

            for i in range(qmc_K):
                shift = U[i].reshape(-1,1) # reshape to column vector
                shifted_P = (P + shift) % 1
                assert shifted_P.shape == P.shape, f"shifted_P shape {shifted_P.shape} should be equal to P shape {P.shape}"
                if preintegrated:
                    j = 0
                    Vi_list[i], _ = self.preintegrated_MC(fn, shifted_P, matrix, j)
                else:
                    Vi_list[i], _ = self.MC(fn, shifted_P, matrix)

            Vi = np.mean(Vi_list)
            mse = np.var(Vi_list)/qmc_K
            return Vi, mse


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
            
            # wrapper for phi to take only xj as input
            def phi_wrapper(xj: float) -> float:
                xmj = self.CDF_inverse(ymj_column)
                x = np.insert(xmj, j, xj, axis=0) # insert x_j at index j
                x = x.reshape(-1,1) # reshape to column vector
                w = matrix@x
                return self.phi(w) 
            
            # derivative of phi with respect to xj
            def phi_j(xj: float) -> float:
                xmj = self.CDF_inverse(ymj_column)
                x = np.insert(xmj, j, xj, axis=0)
                x = x.reshape(-1,1)
                w = matrix@x
                S = self.S(w)
                # j-th column of matrix, element-wise product with t
                Mj = matrix[:,j]
                return self.sigma/self.m*np.dot(S.T, Mj) 

            xj_root = newton(phi_wrapper, 0, fprime=phi_j)

            # quadrature of psi
            def psi_wrapper(yj: float) -> float:
                y = np.insert(ymj_column, j, yj, axis=0)
                y = y.reshape(-1,1)
                w = matrix@self.CDF_inverse(y)
                return psi(w)
            
            def psi_transform(t:float) -> float:
                xmj_column = self.CDF.inverse(ymj_column)
                x = np.insert(xmj_column, j, xj_root + t/(1-t),axis = 0)
                w = matrix @ x
                return psi(w)* np.exp(-1/2*(xj_root + t/(t-1))**2 )/(math.sqrt(2*math.pi)*(1-t)**2)


            yj_root = st.norm.cdf(xj_root) # recast

            psi_var, _ = quad(psi_transform, 0, 1) 

            
            # THIS DOESN'T WORK
            # psi_var, _ = quad(psi_wrapper, yj_root, 1) 
            
            # THIS IS BINARY QUADRATUE RESULT, WHICH WORKS
            # psi_var = 1-yj_root

            psi_vars[i] = psi_var

        interest_coeff = np.exp(-self.r*self.T)
        MC_mean = interest_coeff*np.mean(psi_vars)
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
    
    def plot_unit_cube_phi(self):
        # generate uniform vectors
        N = 500
        y = st.uniform.rvs(size=(self.m-1,N)) # shape (m-1,N)
        
        matrix = np.linalg.cholesky(self.build_C())
        
        # plot
        fig = plt.figure()
        ax = fig.add_subplot()
        title = r'$\phi(y_j, \mathbf{y}_{-j})$'
        ax.set_title(title)
        ax.set_xlabel(r'$y_j$')
        ax.set_ylabel(r'$\phi$')
        # ax.set_xlim(0,1e-15)
        ax.set_xscale('log')
        ax.set_ylim(-2,2)

        for i in range(N):
            ymj_column = y[:,i]
            j = 0

            def phi_wrapper(yj: float) -> float:
                y = np.insert(ymj_column, j, yj, axis=0) # insert y_j at index j
                y = y.reshape(-1,1) # reshape to column vector
                w = matrix@self.CDF_inverse(y)
                return self.phi(w) 
            
            yj = np.linspace(0,1e-20,100)
            phi = np.zeros(yj.shape)
            for k, yj_k in enumerate(yj):
                phi[k] = phi_wrapper(yj_k)
            # opacity is 0.1
            ax.plot(yj, phi, alpha=0.6)

        # draw line at 0
        ax.axhline(y=0, color='k')

        # remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig('./assets/precision_roots.pdf')

        plt.show()

    def plot_phi(self):

        # generate uniform vectors
        N = 100
        ymj = st.uniform.rvs(size=(self.m-1,N)) # shape (m-1,N)

        matrix = np.linalg.cholesky(self.build_C())

        j = 0

        # plot
        fig, ax = plt.subplots(2, 1, figsize=(12,5))
        title = r'$\phi(x_j, \mathbf{y}_{-j})$'
        fig.suptitle(title)
        ax[0].set_xlabel(r'$x_j$')
        ax[1].set_xlabel(r'$x_j$')
        ax[0].set_ylabel(r'$\phi$')
        ax[1].set_ylabel(r'$\frac{\partial \phi}{\partial x_j}$')
        # ax[0].set_xlim(-100,100)
        # ax[1].set_xlim(-100,100)

        for ymj_column in ymj.T: 

            def phi_wrapper(xj: float) -> float:
                xmj = self.CDF_inverse(ymj_column)
                x = np.insert(xmj, j, xj, axis=0) # insert x_j at index j
                x = x.reshape(-1,1) # reshape to column vector
                w = matrix@x
                return self.phi(w) 
            
            # derivative of phi with respect to xj
            def phi_j(xj: float) -> float:
                xmj = self.CDF_inverse(ymj_column)
                x = np.insert(xmj, j, xj, axis=0)
                x = x.reshape(-1,1)
                w = matrix@x
                S = self.S(w)
                # j-th column of matrix, element-wise product with t
                Mj = matrix[:,j]
                return self.sigma/self.m*np.dot(S.T, Mj*self.t) 
            
            xj = np.linspace(-100,100,1000)
            phi = np.zeros(xj.shape)
            for k, xj_k in enumerate(xj):
                phi[k] = phi_wrapper(xj_k)

            phij = np.zeros(xj.shape)
            for k, xj_k in enumerate(xj):
                phij[k] = phi_j(xj_k)
            
            ax[0].plot(xj, phi, label=r'$\phi$')
            ax[1].plot(xj, phij, label=r'$\frac{\partial \phi}{\partial x_j}$')            

        # draw line at 0
        ax[0].axhline(y=0, color='k')

        plt.show()

    def plot_Sobol(self):
        P = sn.generate_points(1000, 2)
        print(P)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(P[:,0], P[:,1])
        plt.show()


        
if __name__ == "__main__":
    # set parameters
    # K = 100
    # S0 = 100
    # r = 0.05
    # sigma = 0.2
    # T = 1
    # m = 1

    # create simulator
    # simulator = OptionPricingSimulator(K, S0, r, sigma, T, m)
    pass