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
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky', variance_reduction='anti'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky', variance_reduction='control'),

                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky', variance_reduction='anti'),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky', variance_reduction='control'),
                
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian binary', transformation='Cholesky', preintegrated=True),
                # config_gen(m=m, N=N, method='Crude MC', Psi='Asian', transformation='Cholesky', preintegrated = True),

                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='anti'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='control'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='scramble'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='truncated mean'),
                
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='anti'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='control'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='scramble'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, variance_reduction='truncated mean'),

                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', preintegrated=True, qmc_K=qmc_K),
                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, preintegrated = True, variance_reduction='anti'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, preintegrated = True, variance_reduction='scramble'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian binary', transformation='Cholesky', qmc_K=qmc_K, preintegrated = True, variance_reduction='truncated mean'),

                config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, preintegrated = True, variance_reduction='anti'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, preintegrated = True, variance_reduction='scramble'),
                # config_gen(m=m, N=N, method='Randomized QMC', Psi='Asian', transformation='Cholesky', qmc_K=qmc_K, preintegrated = True, variance_reduction='truncated mean'),

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

    # Compare Variance Reduction techniques =====================================
    name = "data/data_20240116-124857_variance_reduction_no_pre_asian.json"
    data = pd.DataFrame(load_data(name))

    # Compare truncated mean ====================================================
    name = "data/data_20240116-175550_truncated_preintegrated_RQMC.json"
    data = pd.DataFrame(load_data(name))

    # Compare all variance reduction methods on CMC =============================
    name = "data/data_20240116-124857_variance_reduction_no_pre_asian.json"
    data = pd.DataFrame(load_data(name))

    ANTI = data[
        (data['method']=='Crude MC')
        & (data['transformation']=='Cholesky')
        & (data['variance_reduction']=='anti')
        & (data['preintegrated']==False)
    ]

    CMC = data[
        (data['method']=='Crude MC')
        & (data['transformation']=='Cholesky')
        & (data['variance_reduction'].isnull())
        & (data['preintegrated']==False)
    ]

    QMC_TRUNC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['variance_reduction']=='truncated mean')
        & (data['preintegrated']==False)
    ]

    QMC_ANTI = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['variance_reduction']=='anti')
        & (data['preintegrated']==False)
    ]

    name = "data/data_20240116-190237_control.json"
    data = pd.DataFrame(load_data(name))

    CONTROL = data[
        (data['method']=='Crude MC')
        & (data['transformation']=='Cholesky')
        & (data['variance_reduction']=='control')
        & (data['preintegrated']==False)
    ]

    QMC_CONTROL = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['variance_reduction']=='control')
        & (data['preintegrated']==False)
    ]

    all_data = [ANTI, CONTROL, CMC]
    
    titles = [
        'Antithetic',
        'Control',
        'None'
    ]

    title = "Variance Reduction Techniques on Crude MC"

    # plot_comparison_mean(data=all_data, titles=titles, title=title, col='MSE', log=True, convergence_lines=False, plot_goal_line=False)
    
    # Compare Preintegration ====================================================
    name = "data/data_20240116-163838_asian_and_binary_RQMC_preintegration.json"
    data = pd.DataFrame(load_data(name))

    title = "Mean Squared Error and Preintegration"

    PRQMC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
    ]

    name = "data/data_20240116-165752_preintegrated_CMC.json"
    data = pd.DataFrame(load_data(name))

    PCMC = data[
        (data['method']=='Crude MC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
    ]

    
    # data1 = pd.concat([PRQMC, RQMC])
    # data2 = pd.concat([PCMC, CMC])

    # performance_by_method(data1=data1,
    #                       data2=data2,
    #                       title="Computational Cost of Preintegration")
    
    name = "data/data_20240116-021615_K120_everything.json"
    data = pd.DataFrame(load_data(name))

    RQMC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==False)
    ]

    PRQMC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
    ]
    
    title =  "test"
    all_data = [PRQMC, RQMC, PCMC, CMC]
    titles = [
        'Preintegrated RQMC',
        'RQMC',
        'Preintegrated CMC',
        'CMC'
    ]
    # plot_comparison_mean(data=all_data, titles=titles, title=title, col='MSE', log=True, convergence_lines=False, plot_goal_line=False)
    
    name="data/data_20240116-175550_truncated_preintegrated_RQMC.json"
    data = pd.DataFrame(load_data(name))

    PTRUNC = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
        & (data['variance_reduction']=='truncated mean')
    ]

    title = "Variation Reduction Techniques on RQMC"

    all_data = [RQMC, QMC_ANTI, QMC_CONTROL, PTRUNC]
    titles = [
        'None',
        'Antithetic',
        'Control',
        'Scrambled'
    ]

    # plot_comparison_mean(data=all_data, titles=titles, title=title, col='MSE', log=True, convergence_lines=False, plot_goal_line=False)

    title = "Variation Reduction Techniques on Preintegrated RQMC"

    name = "data/data_20240116-201941.json"
    data = pd.DataFrame(load_data(name))

    PANTI = data[
        (data['method']=='Randomized QMC')
        & (data['transformation']=='Cholesky')
        & (data['preintegrated']==True)
        & (data['variance_reduction']=='anti')
    ]

    all_data = [PRQMC, PANTI, PTRUNC]
    titles = [
        'None',
        'Antithetic',
        'Scrambled'
    ]

    # plot_comparison_mean(data=all_data, titles=titles, title=title, col='MSE', log=True, convergence_lines=False, plot_goal_line=False)
    data1 = pd.concat([CMC, PCMC])
    data2 = pd.concat([RQMC, PRQMC])
    performance_by_method(data1=data1,
                          data2=data2,
                          title="Computational Cost of CMC vs. RQMC")

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
    axs[0,0].set_title('Asian Call Option')
    axs[0,1].set_title('Binary Digital Asian Option')
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
    
    # set axs[1,0] to exponent notation
    axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    for m in m_list:
        axs[0,0].plot(N_list, data[(data['Psi']=='Asian') & (data['m']==m)]['V'], label=f'm={m}')
        axs[1,0].plot(N_list, data[(data['Psi']=='Asian') & (data['m']==m)]['MSE'], label=f'm={m}')
        axs[0,1].plot(N_list, data[(data['Psi']=='Asian binary') & (data['m']==m)]['V'], label=f'm={m}')
        axs[1,1].plot(N_list, data[(data['Psi']=='Asian binary') & (data['m']==m)]['MSE'], label=f'm={m}')

    # axs[1,0].plot(N_list, 1/N_list, label='1/N')
    # axs[1,1].plot(N_list, 1/N_list, label='1/N')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Dimension')

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
    if 'title1' not in kwargs.keys():
        title1 = 'Asian'
    else:
        title1 = kwargs['title1']
    if 'title2' not in kwargs.keys():
        title2 = 'Binary'
    else:
        title2 = kwargs['title2']
    col = kwargs['col']
    log = kwargs['log']
    convergence_lines = kwargs['convergence_lines']
    plot_goal_line = kwargs['plot_goal_line']
    if 'c1' not in kwargs.keys():
        c1 = 'Asian'
    else:
        c1 = kwargs['c1']
    if 'c2' not in kwargs.keys():
        c2 = 'Asian binary'
    else:
        c2 = kwargs['c2'] 

    if 'col2' not in kwargs.keys():
        col2 = 'Psi'
    else:
        col2 = kwargs['col2']

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(title)
    axs[0].set_title(title1)
    axs[1].set_title(title2)

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
            asian_d = d[d[col2]==c1]
            y = asian_d.groupby('N')[col].mean()
            x = asian_d['N'].unique()
            axs[0].plot(x, y, label=t)
        else:
            axs[0].plot([], [], label=t)
        if 'Asian binary' in d[col2].unique():
            binary_d = d[d[col2]==c2]
            y = binary_d.groupby('N')[col].mean()
            x = binary_d['N'].unique()
            axs[1].plot(x, y, label=t)
        else:
            axs[1].plot([], [], label=t)
    
    if convergence_lines:
        # plot 1/N as black dotted line
        # plot 1/N^2 as black dashed line
        # plot 1/N^3 as black line

        y_min, y_max = axs[0].get_ylim()
        axs[0].plot(x, 0.15*1/x , 'k:', label=r'$1/N$')
        axs[0].plot(x, 1.5*1e1*1/(x**(2)), 'k--', label=r'$1/N^2$')
        # axs[0].plot(x, 1e0*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[0].set_ylim(y_min, y_max)
        
        y_min, y_max = axs[1].get_ylim()
        axs[1].plot(x, 1e-2*1/x, 'k:', label=r'$1/N$')
        axs[1].plot(x, 1e0*1/(x**(2)), 'k--', label=r'$1/N^2$')
        # axs[1].plot(x, 1e0*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[1].set_ylim(y_min, y_max)

    if plot_goal_line:
        # plot horizontal line at y=10^-2
        axs[0].axhline(y=1e-2, color='k', linestyle='--', label=r'$10^{-2}$')
        axs[1].axhline(y=1e-2, color='k', linestyle='--', label=r'$10^{-2}$')

    # take care of legend
    handles, labels = axs[1].get_legend_handles_labels()
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
    if 'c1' not in kwargs.keys():
        c1 = 'Asian'
    else:
        c1 = kwargs['c1']
    if 'c2' not in kwargs.keys():
        c2 = 'Asian binary'
    else:
        c2 = kwargs['c2'] 

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
            if c1 in d['Psi'].unique():
                asian_d = d[(d['Psi']=='Asian') & (d['m']==m)]
                y = asian_d[col]
                x = asian_d['N']
                axs[0].plot(x, y, label=f'{t}')
            else:
                axs[0].plot([], [], label=t)
            if 'Asian binary' in d['Psi'].unique():
                binary_d = d[(d['Psi']=='Asian binary') & (d['m']==m)]
                y = binary_d[col]
                x = binary_d['N']
                axs[1].plot(x, y, label=f'{t}')
            else:
                axs[0].plot([], [], label=t)

    if convergence_lines:
        # fix axis limits, such that plotting lines doesn't change the scaling

        # plot 1/N as black dotted line
        # plot 1/N^2 as black dashed line
        # plot 1/N^3 as black line
        y_min, y_max = axs[0].get_ylim()
        axs[0].plot(x, 2e-2*1/x , 'k:', label=r'$1/N$')
        axs[0].plot(x, 2*1/(x**(2)), 'k-', label=r'$1/N^2$')
        # axs[0].plot(x, 2e2*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[0].set_ylim(y_min, y_max)
        
        y_min, y_max = axs[1].get_ylim()
        axs[1].plot(x, 1e-3*1/x, 'k:', label=r'$1/N$')
        axs[1].plot(x, 1e-1*1/(x**(2)), 'k-', label=r'$1/N')
        # axs[1].plot(x, 1e1*1/(x**(3)), 'k--', label=r'$1/N^{3}$')
        axs[1].set_ylim(y_min, y_max)
    
    # legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(f"figures/{title}.pdf")

    plt.show()

def performance_by_method(**kwargs) -> None:
    data1 = kwargs['data1']
    data2 = kwargs['data2']

    title = kwargs['title']

    fig, axs = plt.subplots(1, 1, figsize=(12,5))
    fig.suptitle(title)

    axs.set_xlabel('N')

    axs.set_ylabel('mean time (s)')

    axs.set_xscale('log', base=2)
    
    pre_1 = data1[data1['preintegrated']==True]
    pre_2 = data2[data2['preintegrated']==True]

    # hide y axis numbers while keeping y axis label
    # axs[0].yaxis.set_major_formatter(plt.NullFormatter())
    # axs[1].yaxis.set_major_formatter(plt.NullFormatter())

    # remove y axis ticks
    # axs[0].tick_params(axis='y', which='both', length=0)
    # axs[1].tick_params(axis='y', which='both', length=0)

    axs.set_yscale('log', base=10)

    y = pre_1.groupby('N')['time'].mean()
    x = pre_1['N'].unique()
    axs.plot(x, y, label=f'{"Crude MC"}')

    y = pre_2.groupby('N')['time'].mean()
    x = pre_2['N'].unique()
    axs.plot(x, y, label=f'{"Randomized QMC"}')
    
    # legend
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(f"figures/{title}.pdf")

    plt.show()

if __name__ == '__main__':
    # simulation()
    # data_manipulation()
    plot_results(
        **{
            'data': pd.DataFrame(load_data("data/data_20240116-163838_asian_and_binary_RQMC_preintegration.json")),
            'title': 'Asian Call Option Value & MSE vs. Sample Size',
        }
    )
    plt.savefig("figures/Asian_option_pricing.svg")


    
