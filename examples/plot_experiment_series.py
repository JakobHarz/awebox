from ftplib import all_errors

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import alpha
import matplotlib.ticker as mtick


# %% Latexify the plots
def latexify():
    import matplotlib
    params_MPL_Tex = {
        'text.usetex': True,
        'font.family': 'serif',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    matplotlib.rcParams.update(params_MPL_Tex)
latexify()

# %% Classes to store the experiment data
class ExperimentInfo:
    def __init__(self, filepath: str):

        print(f'Extracting Experiment Information from: {filepath}')

        data = np.load(filepath, allow_pickle=True)

        self.data_SAM = data['SAM'].item()
        self.data_REC = data['REC'].item()

        self.data_MPC = None
        WITH_MPC = 'MPC' in data.keys()
        if WITH_MPC:
            self.data_MPC = data['MPC'].item()

        self.d = self.data_SAM['d']
        self.N = self.data_SAM['N']
        self.param_regulization = self.data_SAM['regularizationValue']

        # extract solver information
        self.N_var = self.data_SAM['solver_stats']['N_var']
        self.N_eq = self.data_SAM['solver_stats']['N_eq']
        self.t_wall = self.data_SAM['solver_stats']['t_wall']['optimization']
        self.iterations = self.data_SAM['solver_stats']['N_iter']['optimization']
        self.t_iter = self.t_wall/self.iterations

        # extract power optimality
        self.J_SAM = self.data_SAM['x']['e'][0][-1]/self.data_SAM['time'][-1]
        self.J_REC = self.data_REC['x']['e'][0][-1]/self.data_REC['time'][-1]
        if self.data_MPC is not None:
            self.J_MPC = self.data_MPC['x']['e'][0][-1]/self.data_MPC['time'][-1]
            if self.data_MPC.get('mpc_cpu_time') is not None:
                self.t_wall_MPC = self.data_MPC['mpc_cpu_time'][1:]
                self.t_iter_MPC = self.data_MPC['mpc_iter'][1:]
            else: # else set to np.inf
                self.t_wall_MPC = np.inf * np.ones(self.N)
                self.t_iter_MPC = np.inf * np.ones(self.N)

class DefaultExperimentInfo:
    def __init__(self, filepath: str):

        print(f'Extracting Experiment Information from: {filepath}')

        data = np.load(filepath, allow_pickle=True)

        self.data_DEFAULT = data['DEFAULT'].item()
        self.data_MPC = None
        WITH_MPC = 'MPC' in data.keys()
        if WITH_MPC:
            self.data_MPC = data['MPC'].item()

        self.N = self.data_DEFAULT['N']

        # extract solver information
        self.N_var = self.data_DEFAULT['solver_stats']['N_var']
        self.N_eq = self.data_DEFAULT['solver_stats']['N_eq']
        self.t_wall = self.data_DEFAULT['solver_stats']['t_wall']['optimization']
        self.iterations = self.data_DEFAULT['solver_stats']['N_iter']['optimization']
        self.t_iter = self.t_wall/self.iterations

        # extract power optimality
        self.J_DEFAULT = self.data_DEFAULT['x']['e'][0][-1]/self.data_DEFAULT['time'][-1]
        if self.data_MPC is not None:
            self.J_MPC = self.data_MPC['x']['e'][0][-1]/self.data_MPC['time'][-1]
            if self.data_MPC.get('mpc_cpu_time') is not None:
                self.t_wall_MPC = self.data_MPC['mpc_cpu_time'][1:] # remove first iteration since no warmstarting
                self.t_iter_MPC = self.data_MPC['mpc_iter'][1:] # remove first iteration since no warmstarting
            else: # else set to np.inf
                self.t_wall_MPC = np.inf * np.ones(self.N)
                self.t_iter_MPC = np.inf * np.ones(self.N)

# %% Load series of experiements:
import os
base_directory = '_export/0206'

all_experiments = []
for file in os.listdir(f'{base_directory}/toPlot'):
    if file.endswith('.npz'):
        all_experiments.append(ExperimentInfo(f'{base_directory}/toPlot/{file}'))

default_experiments = []
for file in os.listdir(f'{base_directory}/toPlot_default'):
    if file.endswith('.npz'):
        default_experiments.append(DefaultExperimentInfo(f'{base_directory}/toPlot_default/{file}'))


# group the experiments by d
experiments_by_d = {}
for d in [3,4,5,6]:
    d_exp = [exp for exp in all_experiments if exp.d == d]

    # sort the list of experiments by N
    d_exp.sort(key=lambda x: x.N)

    experiments_by_d[d] = d_exp

# delete the keys which are empty
for key in list(experiments_by_d.keys()):
    if len(experiments_by_d[key]) == 0:
        del experiments_by_d[key]

# sort the default list by N
default_experiments.sort(key=lambda x: x.N)

# %% Plot Series
fig, axes = plt.subplot_mosaic("AA;AA;AA;BB;BB;CC;CC", figsize=(4.5,4))
plt.sca(axes['A'])


# plot default results
J_DEF_list = np.array([exp.J_DEFAULT for exp in default_experiments])
N_DEF_list = np.array([exp.N for exp in default_experiments])
J_DEF_MPC_list = np.array([exp.J_MPC for exp in default_experiments])
plt.plot(N_DEF_list, J_DEF_MPC_list/1000, f'r^-', markersize=3, label=f'Full Problem')

plt.sca(axes['B'])
error = (J_DEF_list - J_DEF_MPC_list)/J_DEF_MPC_list
plt.plot(N_DEF_list, np.abs(error)*100, f'r^-', markersize=3, label=f'Full Problem')

# plot sam results
ds_to_plot = list(experiments_by_d.keys())
for index,d in enumerate(ds_to_plot):
    experiments = experiments_by_d[d]
    J_SAM_list = np.array([exp.J_SAM for exp in experiments])
    J_REC_list = np.array([exp.J_REC for exp in experiments])
    J_MPC_list = np.array([exp.J_MPC for exp in experiments])
    N_list = np.array([exp.N for exp in experiments])

    plt.sca(axes['A'])
    plt.plot([],[],f'C{index}.-',label=f'd={d}', alpha=1)
    plt.plot(N_list, J_MPC_list/1000,f'C{index}.-')

    plt.sca(axes['B'])
    error = (J_SAM_list - J_MPC_list)/J_MPC_list
    plt.plot(N_list, np.abs(error)*100, f'C{index}.-', alpha=1)
    plt.plot([],[],f'C{index}.-',label=f'd={d}', alpha=1)


plt.sca(axes['A'])
# plt.ylim([0, np.max(J_SAM_list/1000)*1.1])
plt.xticks(np.arange(-20,50,5))
plt.xlim([0, np.max(N_list)*1.05])
# plt.xlabel('N')
plt.ylabel('P [kW]')
plt.grid(alpha=0.25)
plt.legend(ncol=2,loc='upper right')

plt.sca(axes['B'])
plt.ylim([0, 11])
plt.xticks(np.arange(-20,50,5))
plt.xlim([0, np.max(N_list)*1.05])
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.ylabel('Rel. Error in \%')
plt.grid(alpha=0.25)
plt.legend(ncol=len(ds_to_plot)+1)



plt.sca(axes['C'])

# plot default results
t_iter_DEF_list = np.array([exp.t_iter for exp in default_experiments])
plt.plot(N_DEF_list, t_iter_DEF_list, f'r^-', markersize=3,label=f'Full Problem')


for index,d in enumerate(ds_to_plot):
    experiments = experiments_by_d[d]
    N_list = np.array([exp.N for exp in experiments])
    N_eq_list = np.array([exp.N_eq for exp in experiments])
    t_iter_list = np.array([exp.t_iter for exp in experiments])
    plt.plot(N_list, t_iter_list, f'C{index}.-', label=f'd={d}')
    #
    # t_wall_list = np.array([exp.t_wall for exp in experiments])
    # plt.plot(N_list, t_wall_list, f'C{index}.-', label=f'd={d}')


plt.xticks(np.arange(-20,50,5))
# plt.ylim([0, np.max(t_iter_list)*1.7])
plt.xlim([0, np.max(N_list)*1.05])
plt.yscale('log')
# plt.ylim([1E-1, 3E0])

# plt.xlabel('N')
plt.ylabel('$t_\mathrm{iter}$ [s]')
# plt.ylabel('$t_\mathrm{wall}$ [s]')
plt.grid(alpha=0.25)
plt.legend(ncol=2,loc='upper right')

# third subplot: Number of variables

# plt.sca(axes['D'])
#
# # plot default results
# N_DEF_list = np.array([exp.N for exp in default_experiments])
# t_wall_DEF_list = np.array([exp.t_wall for exp in default_experiments])
# plt.plot(N_DEF_list, t_wall_DEF_list, f'r^-', markersize=3,label=f'Full Problem')
#

# for index,d in enumerate(ds_to_plot):
#     experiments = experiments_by_d[d]
#     N_list = np.array([exp.N for exp in experiments])
#
#     t_wall_list = np.array([exp.t_wall for exp in experiments])
#     plt.plot(N_list, t_wall_list, f'C{index}.-', label=f'd={d}')
#
#
# plt.xticks(np.arange(-20,50,5))
# # plt.ylim([0, np.max(t_iter_list)*1.7])
# # plt.ylim([10, 1000])
# plt.xlim([0, np.max(N_list)*1.05])
# plt.yscale('log')
#
# # plt.xlabel('N')
# plt.ylabel('$t_\mathrm{wall}$ [s]')
# # plt.ylabel('$t_\mathrm{wall}$ [s]')
# plt.grid(alpha=0.25)
# plt.legend(ncol=2,loc='upper right')
#


plt.xlabel(r'Number of Subcycles $N$')
# plt.legend(ncol=len(ds_to_plot)+1,loc = 'lower right')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('figures/experiment_series.pdf')
plt.show()

# %% analyze MPC iterations

# default
for exp in default_experiments:
    if exp.data_MPC is not None:
        print(f'Full Problem: {exp.N} - median: {np.median(exp.t_wall_MPC):0.2f}, min: {np.min(exp.t_wall_MPC):0.2f}, max: {np.max(exp.t_wall_MPC):0.2f}')

# SAM
for exp in all_experiments:
    if exp.data_MPC is not None:
        print(f'SAM d={exp.d} - N={exp.N} - median: {np.median(exp.t_wall_MPC):0.2f}, min: {np.min(exp.t_wall_MPC):0.2f}, max: {np.max(exp.t_wall_MPC):0.2f}')


# %% plot cycle time
# T_cycle_DEFAULT = [exp.data_DEFAULT['time'][-1] for exp in default_experiments if exp.data_MPC is not None]
# plt.plot(N_DEF_list,T_cycle_DEFAULT, 'r.-', label='Full Problem')