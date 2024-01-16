from datetime import datetime
import json 
import numpy as np
from OptionPricingSimulator import OptionPricingSimulator as OPS
import matplotlib.pyplot as plt
import pandas as pd
import time

def simulation():

    K = 120 # strike price
    S0 = 100 # intial stock price
    r = 0.1 # interest rate
    sigma = 0.1 # volatility
    T = 1 # time to maturity

    qmc_K = 10 # number of QMC simulations

    # dimensions
    m_list = [32, 64, 128, 256, 512]
    # m_list = [32, 64, 128, 256]
    
    N_list = 2**np.arange(7, 14) # number of simulations
    # N_list = 2**np.arange(7, 12) # number of simulations

    # initialize data frames
    df = pd.DataFrame(columns=['m', 'N', 'V', 'MSE', 'time', 'method', 'Psi', 'transformation', 'preintegrated', 'qmc_K', 'variance_reduction']) 

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
        'variance_reduction': None,
        'qmc_K': qmc_K
    }

    def config_gen(**kwargs):
        config = default_config.copy()
        config.update(kwargs)
        return config

    sim = OPS(K, S0, r, sigma, T)

    for m in m_list:
        sim.update_m(m)
        for N in N_list:
            print(f'm={m}, N=2^{np.log2(N)}\nt={t:0.2f}', end='\r')

            configs = [
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky', variance_reduction='anti'),
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Levy-Ciesielski'),
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky', variance_reduction='anti'),
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Levy-Ciesielski'),
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky', preintegrated=True),
                config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky', preintegrated=True),

                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='anti'),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Levy-Ciesielski', qmc_K=qmc_K),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K,  variance_reduction='anti'),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Levy-Ciesielski', qmc_K=qmc_K),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', preintegrated=True, qmc_K=qmc_K),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', preintegrated=True, qmc_K=qmc_K),
            ]

            # configs = [ config_gen(m=m, N=N, method='Crude MC', Psi='binary', transformation='Cholesky',) ]

            for config in configs:
                sim = OPS(K, S0, r, sigma, T, m)
                time_start = time.time()
                V, mse = sim.simulate(**config)
                time_end = time.time()
                time_elapsed = time_end - time_start
                data = config.copy()
                data.update({'V': V, 'MSE': mse, 'time': time_elapsed})
                table.append(data)
                t += time_elapsed
            
            # results = pool.map(simulate, configs)

            # for result, time_elapsed in results:
            #     table.append(result)
            #     t += time_elapsed
                
        if m>=256:
            print(f"Total time taken: {np.floor(t/60)} minutes {t%60} seconds")
            # write data to file ========================================================
            df = pd.DataFrame(table)
            # use json format 
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"./data/backup/data_{timestamp}_{N}to{np.log2(N)}.json"
            df.to_json(fname, orient='records')

            

    print(f"Total time taken: {np.floor(t/60)} minutes {t%60} seconds")

    # write data to file ========================================================
    df = pd.DataFrame(table)
    # use json format 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"./data/data_{timestamp}.json"
    df.to_json(fname, orient='records')

def data_manipulation():

    # Compare everything ========================================================

    name = "data/data_20240116-021615.json"

    # name = "data/data_20240115-195510_preintegrated_asian.json"
    data = pd.DataFrame(load_data(name))
    PCMC = data[
        (data['method']=='Crude MC') 
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
    ]
    PRQMC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
    ]

    # name = "data/data_20240115-223110_preintegrated_binary.json"
    # data = pd.DataFrame(load_data(name))
    # PCMC = pd.concat([PCMC, data[
    #     (data['method']=='Crude MC') 
    #     & (data['transformation']=='Cholesky')
    #     & (data['preintegrated']==True)
    # ]])

    # PRQMC = pd.concat([PRQMC, data[
    #     (data['method']=='Randomized QMC')
    #     & (data['transformation']=='Cholesky')
    #     & (data['preintegrated']==True)
    # ]])

    # name = "data/data_20240115-220117_ACTUALLY_EVERYTHING_except_preintegration.json"
    # data = pd.DataFrame(load_data(name))

    CMC = data[
        (data['method']=='Crude MC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==False)
        # & (data['variance_reduction']==False)
        & (data['variance_reduction']!='anti')
    ]

    RQMC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==False)
        # & (data['variance_reduction']==False)
        & (data['variance_reduction']!='anti')
    ]

    title = r'CMC Mean Squared Error convergence $K=120$'
    all_data = [
                PCMC, 
                # PRQMC, 
                CMC, 
                # RQMC
    ]
    titles = [
        'Preintegrated CMC',
        # 'Preintegrated RQMC',
        'CMC',
        # 'RQMC'
    ]

    # plot_comparison_mean(data=all_data, 
    #                      titles=titles, 
    #                      title=title, 
    #                      col='MSE', 
    #                      log=True, 
    #                      convergence_lines=True,
    #                      plot_goal_line=False,)
    
    m_list = [32] 
    title = r'CMC Mean Squared Error convergence $K=120$, $m=512$'
    all_data = [PCMC, CMC]
    titles = [
        'Preintegrated CMC',
        'CMC',
    ]

    # plot_comparison(data=all_data, 
    #                 titles=titles, 
    #                 title=title, 
    #                 col='MSE', 
    #                 log=True, 
    #                 convergence_lines=True,
    #                 m_list=m_list)

    title = f"Cholesky vs. Lévy-Ciesielski error"
    cholesky_data = data[
        (data['transformation']=='Cholesky')
        & (data['preintegrated']==False)
        & (data['variance_reduction']==False)
    ]
    levy_data = data[
        (data['transformation']=='Levy-Ciesielski')
        & (data['preintegrated']==False)
        & (data['variance_reduction']==False)
    ]
    all_data = [cholesky_data, levy_data]
    titles = [
        'Cholesky',
        'Lévy-Ciesielski'
    ]
    # plot_comparison_mean(data=all_data, titles=titles, title=title, col='MSE', log=False, convergence_lines=False)

    # Performance Comparison ====================================================
    title = f"Preintegration performance"
    preintegrated_data = data[ 
        (data['preintegrated']==True)
        & (data['variance_reduction']==False)
    ]
    non_preintegrated_data = data[
        (data['preintegrated']==False)
        & (data['variance_reduction']==False)
    ]
    all_data = [preintegrated_data, non_preintegrated_data]
    titles = [
        'Preintegrated',
        'Non-preintegrated'
    ]
    # plot_comparison_mean(data=all_data, 
    #                      titles=titles, 
    #                      title=title, 
    #                      col='time', 
    #                      log=False, 
    #                      convergence_lines=False,
    #                      plot_goal_line=False)

    plot_results(data=CMC, title='All results')
    


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

def plot_error_comparison(**kwargs) -> None:
    data1 = kwargs['data1']
    data2 = kwargs['data2']
    title = kwargs['title']
    title1 = kwargs['title1']
    title2 = kwargs['title2']
    m_list = data1['m'].unique()
    N_list = data1['N'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(12,5))
    fig.suptitle(title)

    axs[0,0].set_title(title1)
    axs[0,1].set_title(title2)
    axs[1,0].set_xlabel('N')
    axs[1,1].set_xlabel('N')
    axs[0,0].set_ylabel('MSE (Asian)')
    axs[1,0].set_ylabel('MSE (Asian binary)')
    axs[0,0].set_xscale('log', base=2)
    axs[1,0].set_xscale('log', base=2)
    axs[0,1].set_xscale('log', base=2)
    axs[1,1].set_xscale('log', base=2)
    axs[0,0].set_yscale('log', base=2)
    axs[1,0].set_yscale('log', base=2)
    axs[0,1].set_yscale('log', base=2)
    axs[1,1].set_yscale('log', base=2)

    slopes_1 = []
    slopes_2 = []
    slopes_3 = []
    slopes_4 = []

    for m in m_list:
        axs[0,0].plot(N_list, data1[(data1['Psi']=='Asian') & (data1['m']==m)]['MSE'], label=f'm={m}')
        axs[1,0].plot(N_list, data1[(data1['Psi']=='Asian binary') & (data1['m']==m)]['MSE'], label=f'm={m}')
        axs[0,1].plot(N_list, data2[(data2['Psi']=='Asian') & (data2['m']==m)]['MSE'], label=f'm={m}')
        axs[1,1].plot(N_list, data2[(data2['Psi']=='Asian binary') & (data2['m']==m)]['MSE'], label=f'm={m}')

        slope_1, _ = np.polyfit(np.log2(N_list), np.log2(data1[(data1['Psi']=='Asian') & (data1['m']==m)]['MSE']), 1)
        slope_2, _ = np.polyfit(np.log2(N_list), np.log2(data1[(data1['Psi']=='Asian binary') & (data1['m']==m)]['MSE']), 1)
        slope_3, _ = np.polyfit(np.log2(N_list), np.log2(data2[(data2['Psi']=='Asian') & (data2['m']==m)]['MSE']), 1)
        slope_4, _ = np.polyfit(np.log2(N_list), np.log2(data2[(data2['Psi']=='Asian binary') & (data2['m']==m)]['MSE']), 1)

        slopes_1.append(slope_1)
        slopes_2.append(slope_2)
        slopes_3.append(slope_3)
        slopes_4.append(slope_4)


    # plot 1/N as black dotted line
    # plot 1/N^2 as black dashed line
    # plot 1/sqrt(N) as black line
    axs[0,0].plot(N_list, 1/N_list, 'k:', label=r'$1/N$')
    axs[0,0].plot(N_list, 1/(N_list**2), 'k--', label=r'$1/N^2$')
    axs[0,0].plot(N_list, 1/(N_list**(1/2)), 'k-', label=r'$1/\sqrt{N}$')
    
    axs[1,0].plot(N_list, 1/N_list, 'k:', label=r'$1/N$')
    axs[1,0].plot(N_list, 1/(N_list**2), 'k--', label=r'$1/N^2$')
    axs[1,0].plot(N_list, 1/(N_list**(1/2)), 'k-', label=r'$1/\sqrt{N}$')

    axs[1,1].plot(N_list, 1/N_list, 'k:', label='1/N')
    axs[1,1].plot(N_list, 1/(N_list**2), 'k--', label='1/N^2')
    axs[1,1].plot(N_list, 1/(N_list**(1/2)), 'k-', label=r'$1/\sqrt{N}$')

    axs[0,1].plot(N_list, 1/N_list, 'k:', label='1/N')
    axs[0,1].plot(N_list, 1/(N_list**2), 'k--', label='1/N^2')
    axs[0,1].plot(N_list, 1/(N_list**(1/2)), 'k-', label=r'$1/\sqrt{N}$')





    # add slope to graphs
    axs[0,0].text(0.7, 0.8, f"slope = {np.mean(slopes_1):0.2f}", transform=axs[0,0].transAxes)
    axs[1,0].text(0.7, 0.8, f"slope = {np.mean(slopes_2):0.2f}", transform=axs[1,0].transAxes)
    axs[0,1].text(0.7, 0.8, f"slope = {np.mean(slopes_3):0.2f}", transform=axs[0,1].transAxes)
    axs[1,1].text(0.7, 0.8, f"slope = {np.mean(slopes_4):0.2f}", transform=axs[1,1].transAxes)


    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

def plot_comparison_mean(**kwargs) -> None:
    data = kwargs['data']
    titles = kwargs['titles']
    title = kwargs['title']
    col = kwargs['col']
    log = kwargs['log']
    convergence_lines = kwargs['convergence_lines']
    plot_goal_line = kwargs['plot_goal_line']

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)
    axs[0].set_title('Asian')
    axs[1].set_title('Binary')

    axs[0].set_xlabel('N')
    axs[1].set_xlabel('N')
    
    axs[0].set_ylabel(col)
    
    axs[0].set_xscale('log', base=2)
    axs[1].set_xscale('log', base=2)

    if log:
        axs[0].set_yscale('log', base=2)
        axs[1].set_yscale('log', base=2)

    for d, t in zip(data, titles):
        
        # collapse m dimensions
        if 'Asian' in d['Psi'].unique():
            asian_d = d[d['Psi']=='Asian']
            y = asian_d.groupby('N')[col].mean()
            x = asian_d['N'].unique()
            axs[0].plot(x, y, label=t)
        if 'Asian binary' in d['Psi'].unique():
            binary_d = d[d['Psi']=='Asian binary']
            y = binary_d.groupby('N')[col].mean()
            x = binary_d['N'].unique()
            axs[1].plot(x, y, label=t)
    
    if convergence_lines:
        # plot 1/N as black dotted line
        # plot 1/N^2 as black dashed line
        # plot 1/N^3 as black line

        y_min, y_max = axs[0].get_ylim()
        axs[0].plot(x, 0.4*1/x , 'k:', label=r'$1/N$')
        axs[0].plot(x, 5e1*1/(x**(2)), 'k-', label=r'$1/N^2$')
        # axs[0].plot(x, 1e0*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[0].set_ylim(y_min, y_max)
        
        y_min, y_max = axs[1].get_ylim()
        axs[1].plot(x, 1e-2*1/x, 'k:', label=r'$1/N$')
        axs[1].plot(x, 1e0*1/(x**(2)), 'k-', label=r'$1/N')
        # axs[1].plot(x, 1e0*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[1].set_ylim(y_min, y_max)

    if plot_goal_line:
        # plot horizontal line at y=10^-2
        axs[0].axhline(y=1e-2, color='k', linestyle='--', label=r'$10^{-2}$')
        axs[1].axhline(y=1e-2, color='k', linestyle='--', label=r'$10^{-2}$')

    # take care of legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(f"figures/{title}.pdf")
    plt.show()

def plot_comparison(**kwargs) -> None:
    data = kwargs['data']
    titles = kwargs['titles']
    title = kwargs['title']
    col = kwargs['col']
    log = kwargs['log']
    m_list = kwargs['m_list']
    convergence_lines = kwargs['convergence_lines']

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)
    axs[0].set_title('Asian')
    axs[1].set_title('Binary')

    axs[0].set_xlabel('N')
    axs[1].set_xlabel('N')
    
    axs[0].set_ylabel(col)
    
    axs[0].set_xscale('log', base=2)
    axs[1].set_xscale('log', base=2)

    if log:
        axs[0].set_yscale('log', base=2)
        axs[1].set_yscale('log', base=2)

    for d, t in zip(data, titles):
        m_list_2 = d['m'].unique()
        for m in set(m_list).intersection(set(m_list_2)):
            if 'Asian' in d['Psi'].unique():
                asian_d = d[(d['Psi']=='Asian') & (d['m']==m)]
                y = asian_d[col]
                x = asian_d['N']
                axs[0].plot(x, y, label=f'{t}')
            if 'Asian binary' in d['Psi'].unique():
                binary_d = d[(d['Psi']=='Asian binary') & (d['m']==m)]
                y = binary_d[col]
                x = binary_d['N']
                axs[1].plot(x, y, label=f'{t}')

    if convergence_lines:
        # fix axis limits, such that plotting lines doesn't change the scaling

        # plot 1/N as black dotted line
        # plot 1/N^2 as black dashed line
        # plot 1/N^3 as black line
        y_min, y_max = axs[0].get_ylim()
        axs[0].plot(x, 2e-2*1/x , 'k:', label=r'$1/N$')
        axs[0].plot(x, 2*1/(x**(2)), 'k-', label=r'$1/N^2$')
        axs[0].plot(x, 2e2*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[0].set_ylim(y_min, y_max)
        
        y_min, y_max = axs[1].get_ylim()
        axs[1].plot(x, 1e-3*1/x, 'k:', label=r'$1/N$')
        axs[1].plot(x, 1e-1*1/(x**(2)), 'k-', label=r'$1/N')
        axs[1].plot(x, 1e1*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[1].set_ylim(y_min, y_max)
    
    # legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(f"figures/{title}.pdf")

    plt.show()

if __name__ == '__main__':
    # simulation()
    data_manipulation()



    
