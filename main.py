from datetime import datetime
import json 
import numpy as np
from OptionPricingSimulator import OptionPricingSimulator as OPS
import matplotlib.pyplot as plt
import pandas as pd
import time

def simulation():
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
    sim = OPS(K, S0, r, sigma, T)

    # initialize data frames
    df = pd.DataFrame(columns=['m', 'N', 'V', 'MSE', 'time', 'method', 'Psi', 'transformation']) 

    for m in m_list:
        sim.update_m(m)
        for N in N_list:
            # ASIAN CALL OPTION
            # Crude MC
            start = time.time()
            V, MSE = sim.crude_MC(sim.Asian, N, transformation='Cholesky')
            end = time.time()
            new_row = {
                'm': m,
                'N': N,
                'V': V,
                'MSE': MSE,
                'time': end-start,
                'method': 'Crude MC',
                'Psi': 'Asian',
                'transformation': 'Cholesky'
            },
            df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)

            # QMC
            start = time.time()
            V, MSE = sim.randomized_QMC(sim.Asian, N, qmc_K, transformation='Cholesky')
            end = time.time()
            new_row = {
                'm': m,
                'N': N,
                'V': V,
                'MSE': MSE,
                'time': end-start,
                'method': 'Randomized QMC',
                'Psi': 'Asian',
                'transformation': 'Cholesky'
            },
            df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)

            # BINARY DIGITAL ASIAN OPTION
            # Crude MC
            start = time.time()
            V, MSE = sim.crude_MC(sim.Asian_binary, N, transformation='Cholesky')
            end = time.time()
            new_row = {
                'm': m,
                'N': N,
                'V': V,
                'MSE': MSE,
                'time': end-start,
                'method': 'Crude MC',
                'Psi': 'Binary',
                'transformation': 'Cholesky'
            },
            df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)

            # QMC
            start = time.time()
            V, MSE = sim.randomized_QMC(sim.Asian_binary, N, qmc_K, transformation='Cholesky')
            end = time.time()
            new_row = {
                'm': m,
                'N': N,
                'V': V,
                'MSE': MSE,
                'time': end-start,
                'method': 'Randomized QMC',
                'Psi': 'Binary',
                'transformation': 'Cholesky'
            },
            df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)

    # write data to file ========================================================
    # use json format 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"./data/data_{timestamp}.json"
    df.to_json(fname, orient='records')

def data_manipulation():

    # Error Comparison ==========================================================
    name = "data/data_20240112-171038.json"
    data = pd.DataFrame(load_data(name))
    crude_MC_data = data[data['method']=='Crude MC']
    randomized_QMC_data = data[data['method']=='Randomized QMC']
    title = f"Crude MC vs. Randomized QMC error"
    plot_error_comparison(
        data1=crude_MC_data, 
        data2=randomized_QMC_data, 
        title=title,
        label1='Crude MC',
        label2='Randomized QMC'    
    )

def load_data(fname: str) -> dict:
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def plot_error_comparison(**kwargs) -> None:
    data1 = kwargs['data1']
    data2 = kwargs['data2']
    title = kwargs['title']
    label1 = kwargs['label1']
    label2 = kwargs['label2']

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)
    axs[0].set_title('Asian call option')
    axs[1].set_title('Binary digital Asian option')
    axs[0].set_xlabel('N')
    axs[1].set_xlabel('N')
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('MSE')
    axs[0].set_xscale('log', base=2)
    axs[1].set_xscale('log', base=2)
    # axs[0].set_yscale('log', base=10)
    # axs[1].set_yscale('log', base=10)
    axs[0].plot(data1[data1['Psi']=='Asian']['N'], data1[data1['Psi']=='Asian']['MSE'], label=label1)
    axs[0].plot(data2[data2['Psi']=='Asian']['N'], data2[data2['Psi']=='Asian']['MSE'], label=label2)
    axs[1].plot(data1[data1['Psi']=='Binary']['N'], data1[data1['Psi']=='Binary']['MSE'], label=label1)
    axs[1].plot(data2[data2['Psi']=='Binary']['N'], data2[data2['Psi']=='Binary']['MSE'], label=label2)
    axs[0].legend()
    axs[1].legend()
    plt.show()


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
    # simulation()
    data_manipulation()



    
