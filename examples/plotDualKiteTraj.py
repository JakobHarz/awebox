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


# %% Load the data

N = 20

#check that the file with the results exists
filename_SAM = f'_export/dualKiteLongTrajectory_N_{N}_SAM.csv'
filename_REC = f'_export/dualKiteLongTrajectory_N_{N}_REC.csv'
filename_MPC = f'_export/dualKiteLongTrajectory_N_{N}_MPC.csv'

assert os.path.exists(filename_SAM), f"File {filename_SAM} does not exist"
assert os.path.exists(filename_REC), f"File {filename_REC} does not exist"
MPC_AVAILABLE = os.path.exists(filename_MPC)

data_SAM = pandas.read_csv(filename_SAM)
data_REC = pandas.read_csv(filename_REC)
if MPC_AVAILABLE:
    data_MPC = pandas.read_csv(filename_MPC)

# %% compute some stuff
scaling_e = 873280
power_SAM = (data_SAM['x_e_0'].iloc[-1] - data_SAM['x_e_0'].iloc[0])*scaling_e/data_SAM['t'].iloc[-1]
power_REC = (data_REC['x_e_0'].iloc[-1] - data_REC['x_e_0'].iloc[0])*scaling_e/data_REC['t'].iloc[-1]
if MPC_AVAILABLE: power_MPC = (data_MPC['x_e_0'].iloc[-1] - data_MPC['x_e_0'].iloc[0])/data_MPC['t'].iloc[-1]

print(f"Power SAM: {power_SAM/1000:.2f} kW")
print(f"Power REC: {power_REC/1000:.2f} kW")
if MPC_AVAILABLE: print(f"Power MPC: {power_MPC/1000:.2f} kW")

# %% Create a 3D plot of the trajectory

def generate_plot(ax,state_name='q10',PLOT_SAM=False, PLOT_REC=False, PLOT_MPC=False):

    if PLOT_MPC:
        if MPC_AVAILABLE:
            ax.plot3D(*[data_MPC[f'x_{state_name}_{i}'] for i in range(3)], 'C0-', alpha=1, label='MPC')
            # ax.plot3D(*[data_MPC[f'x_q21_{i}'] for i in range(3)], 'C0-', alpha=0.5)

    if PLOT_REC:
        # plot the reconstructed trajectory
        # ax.plot3D(*[q21_rec[i,:] for i in range(3)], 'C1--', alpha=0.5)
        ax.plot3D(*[data_REC[f'x_{state_name}_{i}'] for i in range(3)], 'C1--', alpha=0.5, label='Reconstructed')

    # plot the regions
    if PLOT_SAM:
        N_regions = np.max(data_SAM['regionIndex']) + 1
        colors_regions = ['C0'] + [f'C{i}' for i in range(1, N_regions-1)] + ['C0']
        for i in range(N_regions):
            region = data_SAM[data_SAM['regionIndex'] == i]
            # ax.plot3D(region['x_q10_0'], region['x_q10_1'], region['x_q10_2'], f'C{i}-', alpha=1)
            ax.plot3D(*[region[f'x_{state_name}_{i}'] for i in range(3)], f'-', color = colors_regions[i], alpha=1)
            # ax.plot3D(region['x_q31_0'], region['x_q31_1'], region['x_q31_2'], f'-', color = colors_regions[i],alpha=1)

            # plot the start and end of the region
            if i not in [0, N_regions-1]:
                ax.plot3D(*[region[f'x_{state_name}_{i}'].iloc[[0, -1]] for i in range(3)],
                          'o', color=colors_regions[i], alpha=1)
                # ax.plot3D(*[region[f'x_q21_{i}'].iloc[[0, -1]] for i in range(3)],
                #           'o', color=colors_regions[i], alpha=1)

        ax.plot3D([],[],[],'-',color='C0',label='SAM')


    # set bounds for nice view
    q21_rec = np.array([data_REC[f'x_q21_{i}'] for i in range(3)])
    meanpos = np.mean(q21_rec[:], axis=1)

    bblenght = np.max(np.abs(q21_rec - meanpos.reshape(3, 1)))
    ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
    ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
    ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)

    ax.quiver(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, 1, 0, 0, length=40, color='g')
    ax.text(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, "Wind", 'x', color='g', size=15)

    ax.set_xlabel(r'$x$ in m')
    ax.set_ylabel(r'$y$ in m')
    ax.set_zlabel(r'$z$ in m')

    # ax.legend()
    # plt.axis('off')
    ax.view_init(elev=23., azim=-45)
    ax.set_title(f'State {state_name}')
    ax.legend()

    return ax
# plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')

# %% COMPARISON PLOT
_, axs = plt.subplots(2,2,figsize=(20,20),subplot_kw=dict(projection='3d'))
generate_plot(axs[0,0],state_name='q21',PLOT_SAM=True, PLOT_REC=True, PLOT_MPC=False)
generate_plot(axs[0,1],state_name='q31',PLOT_SAM=True, PLOT_REC=True, PLOT_MPC=False)
generate_plot(axs[1,0],state_name='q21',PLOT_SAM=False, PLOT_REC=True, PLOT_MPC=True)
generate_plot(axs[1,1],state_name='q31',PLOT_SAM=False, PLOT_REC=True, PLOT_MPC=True)


plt.tight_layout()
plt.savefig('SAM_REC_MPC_DualKites_N6.pdf')
plt.show()
