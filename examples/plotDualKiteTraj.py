# %% plot the results
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import os


def latexify():
    params_MPL_Tex = {
        'text.usetex': True,
        'font.family': 'serif',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    matplotlib.rcParams.update(params_MPL_Tex)


latexify()

scaling_e = 1.81933e+06

# %% Load the data


# basePath = '_export/varyN'
# basePath = '_export/singleExperiment'
basePath = '_export/MPC'
N = 15

# check that the file with the results exists
filename_SAM = f'{basePath}/dualKiteLongTrajectory_N_{N}_SAM.csv'
filename_REC = f'{basePath}/dualKiteLongTrajectory_N_{N}_REC.csv'
filename_MPC = f'{basePath}/dualKiteLongTrajectory_N_{N}_MPC.csv'

assert os.path.exists(filename_SAM), f"File {filename_SAM} does not exist"
assert os.path.exists(filename_REC), f"File {filename_REC} does not exist"
MPC_AVAILABLE = os.path.exists(filename_MPC)

data_SAM = pandas.read_csv(filename_SAM)
data_REC = pandas.read_csv(filename_REC)

# scale 'e' state
data_SAM['x_e_0'] = data_SAM['x_e_0'] * scaling_e
data_REC['x_e_0'] = data_REC['x_e_0'] * scaling_e

if MPC_AVAILABLE:
    data_MPC = pandas.read_csv(filename_MPC)

# %% compute some stuff
power_SAM = (data_SAM['x_e_0'].iloc[-1] - data_SAM['x_e_0'].iloc[0])  / data_SAM['t'].iloc[-1]
power_REC = (data_REC['x_e_0'].iloc[-1] - data_REC['x_e_0'].iloc[0])  / data_REC['t'].iloc[-1]
if MPC_AVAILABLE: power_MPC = (data_MPC['x_e_0'].iloc[-1] - data_MPC['x_e_0'].iloc[0]) / data_MPC['t'].iloc[-1]

print(f"Power SAM: {power_SAM / 1000:.2f} kW")
print(f"Power REC: {power_REC / 1000:.2f} kW")
if MPC_AVAILABLE: print(f"Power MPC: {power_MPC / 1000:.2f} kW")

# %% Plot the states

# states_list = ['e']
# subplot_n_rows = len(states_list)
# subplot_n_cols = 1
#
# plt.figure(figsize=(10, 5))
# for index, state in enumerate(states_list):
#     plt.subplot(subplot_n_rows, subplot_n_cols, index + 1)
#     plt.plot(data_SAM['t'], data_SAM[f'x_{state}_0'], '-', label=f'SAM {state}')
#     # plt.plot(data_SAM['t_average'], data_SAM[f'X_{state}_0'], '-', label=f'SAM_Average {state}')
#     plt.plot(data_REC['t'], data_REC[f'x_{state}_0'], label=f'REC {state}')
#     if MPC_AVAILABLE: plt.plot(data_MPC['t'], data_MPC[f'x_{state}_0'], label=f'MPC {state}')
#
#     plt.xlabel('Time [s]')
#     plt.ylabel(f'{state}')
#     plt.legend()
# plt.show()


# %% Create a 3D plot of the trajectory

def generate_plot(ax, state_name='q10', PLOT_SAM_REELIN=None, PLOT_SAM_MICRO=None, PLOT_SAM_MACRO=None, PLOT_REC=None,
                  PLOT_MPC=False, elev:float = 18.0, azim:float = 120.0):
    if PLOT_REC is not None: assert type(PLOT_REC) == dict, "PLOT_REC should be a dictionary of plotting options"

    if PLOT_MPC:
        if MPC_AVAILABLE:
            ax.plot3D(*[data_MPC[f'x_{state_name}_{i}'] for i in range(3)], 'C0-', alpha=1, label='MPC')
            # ax.plot3D(*[data_MPC[f'x_q21_{i}'] for i in range(3)], 'C0-', alpha=0.5)

    if PLOT_REC is not None:
        # plot the reconstructed trajectory
        # ax.plot3D(*[q21_rec[i,:] for i in range(3)], 'C1--', alpha=0.5)
        ax.plot3D(*[data_REC[f'x_{state_name}_{i}'] for i in range(3)], **PLOT_REC)

    if PLOT_SAM_REELIN or PLOT_SAM_MACRO or PLOT_SAM_MICRO:

        N_regions = np.max(data_SAM['regionIndex']) + 1
        # colors_regions = ['C0'] + [f'C{i}' for i in range(1, N_regions-1)] + ['C0']
        colors_regions = ['C0'] + [f'C2' for i in range(1, N_regions - 1)] + ['C0']
        alpha_regions = [1] + [0.5 for i in range(1, N_regions - 1)] + [1]

        if PLOT_SAM_MACRO is not None:
            # plot the average
            ax.plot3D(*[data_SAM[f'X_{state_name}_{i}'] for i in range(3)], 'C0-', alpha=1, label='Average')

        for i in range(N_regions):
            region = data_SAM[data_SAM['regionIndex'] == i]

            # plot the start and end of the region
            if i not in [0, N_regions - 1]:
                if PLOT_SAM_MICRO is not None:
                    ax.plot3D(*[region[f'x_{state_name}_{i}'] for i in range(3)],
                              '-',
                              **PLOT_SAM_MICRO)

                    ax.plot3D(*[region[f'x_{state_name}_{i}'].iloc[[0, -1]] for i in range(3)],
                              '.', color=colors_regions[i], alpha=alpha_regions[i])
            else:
                if PLOT_SAM_REELIN is not None:
                    ax.plot3D(*[region[f'x_{state_name}_{i}'] for i in range(3)], **PLOT_SAM_REELIN)

        # ax.plot3D([], [], [], '-', color='C0', label='SAM')

    # set bounds for nice view
    q21_rec = np.array([data_REC[f'x_q21_{i}'] for i in range(3)])
    meanpos = np.mean(q21_rec[:], axis=1)

    bblenght = np.max(np.abs(q21_rec - meanpos.reshape(3, 1)))
    ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
    ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
    ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)

    ax.quiver(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, 1, 0, 0, length=40,
              color='g')
    ax.text(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, "Wind", 'x', color='g',
            size=15)

    ax.set_xlabel(r'$x$ in m')
    ax.set_ylabel(r'$y$ in m')
    ax.set_zlabel(r'$z$ in m')

    # ax.legend()
    # plt.axis('off')
    # ax.view_init(elev=23., azim=-45)
    ax.view_init(elev=elev, azim=azim)
    # ax.set_title(f'State {state_name}')

    # remove duplicates from the legend list
    handles, previous_labels = ax.get_legend_handles_labels()
    label_dict = {label: handle for label, handle in zip(previous_labels, handles)}
    new_handles = list(label_dict.values())
    new_labels = list(label_dict.keys())
    ax.legend(handles=new_handles, labels=new_labels,loc='upper right')

    return ax


# plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')

# %% COMPARISON PLOT
# _, axs = plt.subplots(2,2,figsize=(20,20),subplot_kw=dict(projection='3d'))
# generate_plot(axs[0,0],state_name='q21',PLOT_SAM=True, PLOT_REC=True, PLOT_MPC=False)
# generate_plot(axs[0,1],state_name='q31',PLOT_SAM=True, PLOT_REC=True, PLOT_MPC=False)
# generate_plot(axs[1,0],state_name='q21',PLOT_SAM=False, PLOT_REC=True, PLOT_MPC=True)
# generate_plot(axs[1,1],state_name='q31',PLOT_SAM=False, PLOT_REC=True, PLOT_MPC=True)
#
#
# plt.tight_layout()
# plt.savefig('SAM_REC_MPC_DualKites_N6.pdf')
# plt.show()
#
# _, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection='3d'))
# generate_plot(axs[0], state_name='q21',
#               PLOT_REC={'color': 'C0', 'alpha': 1, 'label': 'System Dynamics'})
# generate_plot(axs[1], state_name='q21',
#               PLOT_SAM_REELIN={'color': 'C0', 'alpha': 0.5,'label': 'System Dynamics'},
#               PLOT_SAM_MACRO={'color': 'C0', 'alpha': 1, 'label': 'Average Dynamics'})
# plt.tight_layout()
# plt.show()

for k in range(3):
    # _, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection='3d'))
    plt.figure(figsize=(6, 5))
    ax = plt.axes(projection='3d')
    if k == 0:
        PLOT_REC = {'color': 'C0', 'alpha': 1}
        PLOT_SAM_MICRO = None
        PLOT_SAM_MACRO = None
    elif k == 1:
        PLOT_REC = None
        PLOT_SAM_MICRO = {'color': 'C2', 'alpha': 1, 'label': 'Micro-Integrations'}
        PLOT_SAM_MACRO = {'color': 'C0', 'alpha': 1, 'label': 'Reelout: Average Dynamics'}
    elif k == 2:
        PLOT_REC = {'color': 'C0', 'alpha': 0.3}
        PLOT_SAM_MICRO = {'color': 'C2', 'alpha': 1, 'label': 'Micro-Integrations'}
        PLOT_SAM_MACRO = {'color': 'C0', 'alpha': 1, 'label': 'Reelout: Average Dynamics'}


    generate_plot(ax, state_name='q31', elev=21, azim=125,
                  PLOT_SAM_REELIN={'color': 'C0', 'alpha': 0.5,'label': 'Reelin: System Dynamics'},PLOT_REC=PLOT_REC,PLOT_SAM_MICRO=PLOT_SAM_MICRO,PLOT_SAM_MACRO=PLOT_SAM_MACRO)

    # generate_plot(axs[1], state_name='q31', elev=21, azim=125,
    #               PLOT_SAM_REELIN={'color': 'C0', 'alpha': 0.5,'label': 'Reelin: System Dynamics'},
    #               PLOT_SAM_MICRO={'color': 'C2', 'alpha': 1, 'label': 'Micro-Integrations'},
    #               PLOT_SAM_MACRO={'color': 'C0', 'alpha': 1, 'label': 'Reelout: Average Dynamics'},PLOT_REC=PLOT_REC)
    plt.tight_layout()
    plt.savefig(f'_export/plots/SAM_Showcase_N{N}_{k}.pdf')
    plt.show()