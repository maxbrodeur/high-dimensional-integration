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
    df = pd.DataFrame(columns=['m', 'N', 'V', 'MSE', 'time', 'method', 'Psi', 'transformation', 'preintegrated', 'qmc_K']) 

    # elapsed time
    t = 0

    # table to store data
    table = []

    for m in m_list:
        sim.update_m(m)
        for N in N_list:
            print(f'm={m}, N={N}\nt={t}', end='\r\r')

            config ={
                'method': 'Crude MC',
                'm': m,
                'N': N,
                'Psi': 'Asian binary',
                'transformation': 'Cholesky',
                'preintegrated': False,
                'j': 0,
            }
            time_start = time.time()
            V, mse = sim.simulate(**config)
            time_end = time.time()
            time_elapsed = time_end - time_start
            data = config.copy()
            data.update({'V': V, 'MSE': mse, 'time': time_elapsed})
            table.append(data)
            
            t += time_elapsed

            config['method'] = 'Randomized QMC'
            config['qmc_K'] = qmc_K
            time_start = time.time()
            V, mse = sim.simulate(**config)
            time_end = time.time()
            time_elapsed = time_end - time_start
            data = config.copy()
            data.update({'V': V, 'MSE': mse, 'time': time_elapsed})
            df = pd.concat([df, pd.DataFrame(data, [0])])
            table.append(data)

            t += time_elapsed

    print(f"Total time taken: {np.floor(t/60)} minutes {t%60} seconds")

    # write data to file ========================================================
    df = pd.DataFrame(table)
    # use json format 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"./data/data_{timestamp}.json"
    df.to_json(fname, orient='records')

def data_manipulation():

    name = "data/data_20240114-143308.json"
    data = pd.DataFrame(load_data(name))

    # PREINTEGRATED BINARY ONLY ==================================================================
    # Crude MC Results
    binary_data = data[data['Psi']=='Asian binary']
    binary_data[binary_data['preintegrated']==False]
    crude_MC_data = binary_data[binary_data['method']=='Crude MC']
    title = f"Crude MC (Asian binary digital option)"
    plot_binary_results(data=crude_MC_data, title=title)

    # Randomized QMC Results
    randomized_QMC_data = binary_data[binary_data['method']=='Randomized QMC']
    randomized_QMC_data = randomized_QMC_data[randomized_QMC_data['preintegrated']==False]
    title = f"Randomized QMC (Asian binary digital option)"
    plot_binary_results(data=randomized_QMC_data, title=title)

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

def plot_binary_results(**kwargs) -> None:
    data = kwargs['data']
    title = kwargs['title']
    m_list = data['m'].unique()
    N_list = data['N'].unique()

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)
    # axs[0].set_title('V')
    # axs[1].set_title('MSE')
    axs[0].set_xlabel('N')
    axs[1].set_xlabel('N')
    axs[0].set_ylabel('V')
    axs[1].set_ylabel('MSE')
    axs[0].set_xscale('log', base=2)
    axs[1].set_xscale('log', base=2)
    for m in m_list:
        axs[0].plot(N_list, data[(data['Psi']=='Asian binary') & (data['m']==m)]['V'], label=f'm={m}')
        axs[1].plot(N_list, data[(data['Psi']=='Asian binary') & (data['m']==m)]['MSE'], label=f'm={m}')
    axs[0].legend()
    axs[1].legend()
    plt.show()

def plot_results(**kwargs) -> None:
    data = kwargs['data']
    title = kwargs['title']
    m_list = data['m'].unique()
    N_list = data['N'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(12,5))
    fig.suptitle(title)
    axs[0,0].set_title('Asian call option')
    axs[0,1].set_title('Binary digital Asian option')
    axs[1,0].set_xlabel('N')
    axs[1,1].set_xlabel('N')
    axs[0,0].set_ylabel('V')
    axs[1,0].set_ylabel('MSE')
    axs[0,1].set_ylabel('V')
    axs[1,1].set_ylabel('MSE')
    axs[0,0].set_xscale('log', base=2)
    axs[1,0].set_xscale('log', base=2)
    axs[0,1].set_xscale('log', base=2)
    axs[1,1].set_xscale('log', base=2)
    for m in m_list:
        axs[0,0].plot(N_list, data[(data['Psi']=='Asian') & (data['m']==m)]['V'], label=f'm={m}')
        axs[1,0].plot(N_list, data[(data['Psi']=='Asian') & (data['m']==m)]['MSE'], label=f'm={m}')
        axs[0,1].plot(N_list, data[(data['Psi']=='Asian binary') & (data['m']==m)]['V'], label=f'm={m}')
        axs[1,1].plot(N_list, data[(data['Psi']=='Asian binary') & (data['m']==m)]['MSE'], label=f'm={m}')
    # axs[1,0].plot(N_list, 1/N_list, label='1/N')
    # axs[1,1].plot(N_list, 1/N_list, label='1/N')
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    plt.show()



if __name__ == '__main__':
    # simulation()
    data_manipulation()



    
