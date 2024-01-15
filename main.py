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
    m_list = [32, 64, 128, 256]
    
    N_list = 2**np.arange(7, 12) # number of simulations
    # N_list = 2**np.arange(7, 11) # number of simulations

    # simulation ================================================================
    sim = OPS(K, S0, r, sigma, T)

    # initialize data frames
    df = pd.DataFrame(columns=['m', 'N', 'V', 'MSE', 'time', 'method', 'Psi', 'transformation', 'preintegrated', 'qmc_K']) 

    # elapsed time
    t = 0

    # table to store data
    table = []

    default_config = {
        'method': 'Crude MC',
        'm': 32,
        'N': 2**7,
        'Psi': 'Asian binary',
        'transformation': 'Cholesky',
        'preintegrated': False,
        'j': 0,
    }

    def config_gen(**kwargs):
        config = default_config.copy()
        config.update(kwargs)
        return config

    for m in m_list:
        sim.update_m(m)
        # sim.plot_Sobol()
        # break
        for N in N_list:
            print(f'm={m}, N=2^{np.log2(N)}\nt={t:0.2f}', end='\r')

            configs = [
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky'),
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky', preintegrated=True),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky', preintegrated=True),

                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', preintegrated=True, qmc_K=qmc_K),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', preintegrated=True, qmc_K=qmc_K),
            ]

            for config in configs:
                time_start = time.time()
                V, mse = sim.simulate(**config)
                time_end = time.time()
                time_elapsed = time_end - time_start
                data = config.copy()
                data.update({'V': V, 'MSE': mse, 'time': time_elapsed})
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

    # Preintegrated vs. Normal ==================================================
    name = "data/data_20240115-171636.json"
    data = pd.DataFrame(load_data(name))

    # Binary

    # QMC
    title = f"QMC"
    data1 = data[data['method']=='Randomized QMC']
    data1 = data1[data1['transformation']=='Cholesky']
    data1 = data1[data1['Psi']=='Asian binary']
    data1 = data1[data1['preintegrated']==False]
    data2 = data[data['method']=='Randomized QMC']
    data2 = data2[data2['transformation']=='Cholesky']
    data2 = data2[data2['Psi']=='Asian binary']
    data2 = data2[data2['preintegrated']==True]

    # plot_binary_error_comparison(data=data1, data2=data2, title=title, title1='Normal', title2='Preintegrated')

    # Crude MC
    title = f"Crude MC"
    data1 = data[data['method']=='Crude MC']
    data1 = data1[data1['transformation']=='Cholesky']
    data1 = data1[data1['Psi']=='Asian binary']
    data1 = data1[data1['preintegrated']==False]
    data2 = data[data['method']=='Crude MC']
    data2 = data2[data2['transformation']=='Cholesky']
    data2 = data2[data2['Psi']=='Asian binary']
    data2 = data2[data2['preintegrated']==True]

    # plot_binary_error_comparison(data=data1, data2=data2, title=title, title1='Normal', title2='Preintegrated')

    # Asian

    # QMC
    title = f"QMC"
    data1 = data[data['method']=='Randomized QMC']
    data1 = data1[data1['transformation']=='Cholesky']
    data1 = data1[data1['Psi']=='Asian']
    data1 = data1[data1['preintegrated']==False]
    data2 = data[data['method']=='Randomized QMC']
    data2 = data2[data2['transformation']=='Cholesky']
    data2 = data2[data2['Psi']=='Asian']
    data2 = data2[data2['preintegrated']==True]

    plot_asian_error_comparison(data=data1, data2=data2, title=title, title1='Normal', title2='Preintegrated')

    # Crude MC
    title = f"Crude MC"
    data1 = data[data['method']=='Crude MC']
    data1 = data1[data1['transformation']=='Cholesky']
    data1 = data1[data1['Psi']=='Asian']
    data1 = data1[data1['preintegrated']==False]
    data2 = data[data['method']=='Crude MC']
    data2 = data2[data2['transformation']=='Cholesky']
    data2 = data2[data2['Psi']=='Asian']
    data2 = data2[data2['preintegrated']==True]

    plot_asian_error_comparison(data=data1, data2=data2, title=title, title1='Normal', title2='Preintegrated')

    # CHOLESKY ==================================================================
    
    # Crude MC Results
    title = f"Crude MC"
    crude_MC_data = data[data['method']=='Crude MC']
    crude_MC_data = crude_MC_data[crude_MC_data['transformation']=='Cholesky']
    crude_MC_data = crude_MC_data[crude_MC_data['preintegrated']==False]
    # plot_results(data=crude_MC_data, title=title)

    # Randomized QMC Results
    title = f"Randomized QMC"
    randomized_QMC_data = data[data['method']=='Randomized QMC']
    randomized_QMC_data = randomized_QMC_data[randomized_QMC_data['transformation']=='Cholesky']
    randomized_QMC_data = randomized_QMC_data[randomized_QMC_data['preintegrated']==False]
    # plot_results(data=randomized_QMC_data, title=title)

    # Preintegrated Binary =======================================================
    title = f"Preintegrated Crude MC (Binary)"
    preintegrated_binary_data = data[data['method']=='Crude MC']
    preintegrated_binary_data = preintegrated_binary_data[preintegrated_binary_data['transformation']=='Cholesky']
    preintegrated_binary_data = preintegrated_binary_data[preintegrated_binary_data['Psi']=='Asian binary']
    preintegrated_binary_data = preintegrated_binary_data[preintegrated_binary_data['preintegrated']==True]
    # plot_binary_results(data=preintegrated_binary_data, title=title)

    title = f"Preintegrated Randomized QMC (Binary)"
    preintegrated_binary_data = data[data['method']=='Randomized QMC']
    preintegrated_binary_data = preintegrated_binary_data[preintegrated_binary_data['transformation']=='Cholesky']
    preintegrated_binary_data = preintegrated_binary_data[preintegrated_binary_data['Psi']=='Asian binary']
    preintegrated_binary_data = preintegrated_binary_data[preintegrated_binary_data['preintegrated']==True]
    # plot_binary_results(data=preintegrated_binary_data, title=title)

    # Preintegrated Asian ========================================================
    title = f"Preintegrated Crude MC (Asian)"
    preintegrated_asian_data = data[data['method']=='Crude MC']
    preintegrated_asian_data = preintegrated_asian_data[preintegrated_asian_data['transformation']=='Cholesky']
    preintegrated_asian_data = preintegrated_asian_data[preintegrated_asian_data['Psi']=='Asian']
    preintegrated_asian_data = preintegrated_asian_data[preintegrated_asian_data['preintegrated']==True]
    plot_asian_results(data=preintegrated_asian_data, title=title)

    title = f"Preintegrated Randomized QMC (Asian)"
    preintegrated_asian_data = data[data['method']=='Randomized QMC']
    preintegrated_asian_data = preintegrated_asian_data[preintegrated_asian_data['transformation']=='Cholesky']
    preintegrated_asian_data = preintegrated_asian_data[preintegrated_asian_data['Psi']=='Asian']
    preintegrated_asian_data = preintegrated_asian_data[preintegrated_asian_data['preintegrated']==True]
    plot_asian_results(data=preintegrated_asian_data, title=title)

    


def load_data(fname: str) -> dict:
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

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

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

def plot_asian_results(**kwargs) -> None:
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
        axs[0].plot(N_list, data[(data['Psi']=='Asian') & (data['m']==m)]['V'], label=f'm={m}')
        axs[1].plot(N_list, data[(data['Psi']=='Asian') & (data['m']==m)]['MSE'], label=f'm={m}')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

def plot_preintegrated_results(**kwargs) -> None:
    data = kwargs['data']
    title = kwargs['title']
    m_list = data['m'].unique()
    N_list = data['N'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(12,5))
    fig.suptitle(title)
    axs[0,0].set_title('Normal')
    axs[0,1].set_title('Preintegrated')
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
        axs[0,0].plot(N_list, data[(data['preintegrated']==False) & (data['m']==m)]['V'], label=f'm={m}')
        axs[1,0].plot(N_list, data[(data['preintegrated']==False) & (data['m']==m)]['MSE'], label=f'm={m}')
        axs[0,1].plot(N_list, data[(data['preintegrated']==True) & (data['m']==m)]['V'], label=f'm={m}')
        axs[1,1].plot(N_list, data[(data['preintegrated']==True) & (data['m']==m)]['MSE'], label=f'm={m}')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

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

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

def plot_binary_error_comparison(**kwargs) -> None:
    data1 = kwargs['data']
    data2 = kwargs['data2']
    title = kwargs['title']
    title1 = kwargs['title1']
    title2 = kwargs['title2']
    m_list = data1['m'].unique()
    N_list = data1['N'].unique()

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)

    axs[0].set_title(title1)
    axs[1].set_title(title2)
    axs[0].set_xlabel('N')
    axs[1].set_xlabel('N')
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('MSE')
    axs[0].set_xscale('log', base=2)
    axs[1].set_xscale('log', base=2)
    axs[0].set_yscale('log', base=2)
    axs[1].set_yscale('log', base=2)
    slopes_1 = []
    slopes_2 = []
    for m in m_list:
        axs[0].plot(N_list, data1[(data1['Psi']=='Asian binary') & (data1['m']==m)]['MSE'], label=f'm={m}')
        axs[1].plot(N_list, data2[(data2['Psi']=='Asian binary') & (data2['m']==m)]['MSE'], label=f'm={m}')

        slope_1, _ = np.polyfit(np.log2(N_list), np.log2(data1[(data1['Psi']=='Asian binary') & (data1['m']==m)]['MSE']), 1)
        slope_2, _ = np.polyfit(np.log2(N_list), np.log2(data2[(data2['Psi']=='Asian binary') & (data2['m']==m)]['MSE']), 1)

        slopes_1.append(slope_1)
        slopes_2.append(slope_2)

    # add slope to graphs
    axs[0].text(0.1, 0.1, f"slope = {np.mean(slopes_1):0.2f}", transform=axs[0].transAxes)
    axs[1].text(0.1, 0.1, f"slope = {np.mean(slopes_2):0.2f}", transform=axs[1].transAxes)

    # plot 1/N as black dotted line
    # plot 1/N^2 as black dashed line
    axs[0].plot(N_list, 1/N_list, 'k:', label=r'$1/N$')
    axs[0].plot(N_list, 1/(N_list**2), 'k--', label=r'$1/N^2$')
    axs[1].plot(N_list, 1/N_list, 'k:', label='1/N')
    axs[1].plot(N_list, 1/(N_list**2), 'k--', label='1/N^2')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

def plot_asian_error_comparison(**kwargs) -> None:
    data1 = kwargs['data']
    data2 = kwargs['data2']
    title = kwargs['title']
    title1 = kwargs['title1']
    title2 = kwargs['title2']
    m_list = data1['m'].unique()
    N_list = data1['N'].unique()

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)

    axs[0].set_title(title1)
    axs[1].set_title(title2)
    axs[0].set_xlabel('N')
    axs[1].set_xlabel('N')
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('MSE')
    axs[0].set_xscale('log', base=2)
    axs[1].set_xscale('log', base=2)
    axs[0].set_yscale('log', base=2)
    axs[1].set_yscale('log', base=2)
    slopes_1 = []
    slopes_2 = []
    for m in m_list:
        axs[0].plot(N_list, data1[(data1['Psi']=='Asian') & (data1['m']==m)]['MSE'], label=f'm={m}')
        axs[1].plot(N_list, data2[(data2['Psi']=='Asian') & (data2['m']==m)]['MSE'], label=f'm={m}')

        slope_1, _ = np.polyfit(np.log2(N_list), np.log2(data1[(data1['Psi']=='Asian') & (data1['m']==m)]['MSE']), 1)
        slope_2, _ = np.polyfit(np.log2(N_list), np.log2(data2[(data2['Psi']=='Asian') & (data2['m']==m)]['MSE']), 1)

        slopes_1.append(slope_1)
        slopes_2.append(slope_2)

    # add slope to graphs
    axs[0].text(0.1, 0.1, f"slope = {np.mean(slopes_1):0.2f}", transform=axs[0].transAxes)
    axs[1].text(0.1, 0.1, f"slope = {np.mean(slopes_2):0.2f}", transform=axs[1].transAxes)

    # plot 1/N as black dotted line
    # plot 1/N^2 as black dashed line
    axs[0].plot(N_list, 1/N_list, 'k:', label=r'$1/N$')
    axs[0].plot(N_list, 1/(N_list**2), 'k--', label=r'$1/N^2$')
    axs[1].plot(N_list, 1/N_list, 'k:', label='1/N')
    axs[1].plot(N_list, 1/(N_list**2), 'k--', label='1/N^2')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

if __name__ == '__main__':
    # simulation()
    data_manipulation()



    
