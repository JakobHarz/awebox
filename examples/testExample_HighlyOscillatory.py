#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Rachel Leuthold
"""

from typing import List, Dict
import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

from awebox.ocp.collocation import Collocation
from awebox.ocp.discretization_averageModel import eval_time_grids_SAM, construct_time_grids_SAM_reconstruction, \
    reconstruct_full_from_SAM, originalTimeToSAMTime
from awebox.viz.visualization import build_interpolate_functions_full_solution, dict_from_repeated_struct


# set the logger level to 'DEBUG' to see IPOPT output
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)


DUAL_KITES = False

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}

if DUAL_KITES:
    from examples.paper_benchmarks import reference_options as ref

    options = ref.set_reference_options(user='A')
    options = ref.set_dual_kite_options(options)
else:
    options['user_options.system_model.architecture'] = {1: 0}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'

# indicate desired environment
# here: wind velocity profile according to power-law
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
options['model.system_bounds.x.l_t'] = [10.0, 1500.0]  # [m]

# (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
# note: this may result in slightly slower solution timings
options['nlp.compile_subfunctions'] = True
options['model.integration.method'] = 'constraints'  # use enery as a state, works better with SAM

options['nlp.collocation.u_param'] = 'zoh'
options['nlp.SAM.use'] = True
options['nlp.SAM.MaInt_type'] = 'legendre'
options['nlp.SAM.N'] = 10 # the number of full cycles approximated
options['nlp.SAM.d'] = 4 # the number of cycles actually computed
options['nlp.SAM.ADAtype'] = 'CD'  # the approximation scheme
options['user_options.trajectory.lift_mode.windings'] =  options['nlp.SAM.d'] + 1 # todo: set this somewhere else


# SAM Regularization
single_regularization_param = 1E-2
options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 1E-2*single_regularization_param
options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1*single_regularization_param
options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0*single_regularization_param
options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E-1*single_regularization_param

# smooth the reel in phase (this increases convergence speed x10)
options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)
# options['solver.cost.beta.0'] = 8e0
# options['solver.cost.u_regularisation.0'] = 1e0
# options['solver.max_iter'] = 0
# options['solver.max_iter_hippo'] = 0

# Number of discretization points
n_k = 10 * (options['nlp.SAM.d'] + 1)
options['nlp.n_k'] = n_k

if DUAL_KITES:
    options['model.system_bounds.theta.t_f'] = [10, 10 * options['nlp.SAM.N']]  # [s]
else:
    options['model.system_bounds.theta.t_f'] = [20, 40*options['nlp.SAM.N']] # [s]

options['solver.linear_solver'] = 'ma27'

options['visualization.cosmetics.interpolation.n_points'] = 40* options['nlp.SAM.N'] # high plotting resolution

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'DualKitesLongHorizon')
trial.build()
trial.optimize()
# draw some of the pre-coded plots for analysis

# %% Plot state trajectories

from awebox.tools.struct_operations import calculate_SAM_regions
from awebox.tools.struct_operations import calculate_kdx_SAM


plot_dict_SAM = trial.visualization.plot_dict_SAM
plot_dict_REC = trial.visualization.plot_dict
time_plot_SAM = plot_dict_SAM['time_grids']['ip']
ip_regions_SAM = plot_dict_SAM['SAM_regions_ip']

time_grid_SAM = plot_dict_SAM['time_grids']
time_grid_SAM_x = time_grid_SAM['x']
regions_indeces = calculate_SAM_regions(trial.nlp.options)
delta_ns = [region_indeces.__len__() for region_indeces in regions_indeces]
Ts_opt = [delta_ns[i] / trial.nlp.options['n_k'] * trial.solution_dict['V_opt']['theta', 't_f', i] for i in range(trial.nlp.options['SAM']['d']+1)]
t_ip = plot_dict_REC['time_grids']['ip']


avg_power_REC = plot_dict_SAM['power_and_performance']['avg_power'] / 1e3
avg_power_SAM = plot_dict_REC['power_and_performance']['avg_power'] / 1e3

print('======================================')
print('Average power SAM: {} kW'.format(avg_power_SAM))
print('Average power REC: {} kW'.format(avg_power_REC))
print('======================================')

# print the costs:
cost_dict = trial.visualization.plot_dict['cost']
print('\n======================================')
print('Costs:')
for key, value in cost_dict.items():
    val = float(value)
    if np.abs(val) > 1e-10:
        print(f'\t {key}:  {val:0.4f}')
print('======================================')


# %% Plot outputs
time_plot_REC = plot_dict_REC['time_grids']['ip']
invariants_REC = plot_dict_REC['outputs']['invariants']

time_plot_SAM_nlp = plot_dict_SAM['time_grids']['x_coll']
invariants_SAM = plot_dict_SAM['outputs']['invariants']


plt.figure()
plt.title("Invariants:")


# plt.plot(time_plot, power, '-')
invariants_to_plot = ['c10']
for region_index in np.arange(0, trial.options['nlp']['SAM']['d']+1):
    indices = np.where(ip_regions_SAM == region_index)
    # plt.plot(time_plot_SAM[indices], airspeed_SAM[indices], 'C0-')
    for key in invariants_to_plot:
        plt.plot(time_plot_SAM[indices], np.abs(invariants_SAM[key][0][indices]), 'C0-')
plt.plot([],[],'C0-',label='SAM')

for key in invariants_to_plot:
    plt.plot(time_plot_REC, np.abs(invariants_REC[key][0]), 'k-',alpha=0.3)
plt.plot([],[],'k-',alpha=0.3,label='REC')

# add phase switches
for region in regions_indeces:
    # plt.axvline(x=time_grid_SAM_x[region[0]],color='k',linestyle='dotted',alpha=0.2)
    # plt.axvline(x=time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]),color='k',linestyle='dotted',alpha=0.2)

    # instead: use fill_between to mark regions:
    plt.gca().axvspan(time_grid_SAM_x[region[0]], time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]), color='k', alpha=0.1)

plt.legend()
plt.yscale('log')
plt.xlabel('Time [s]')
plt.show()


# %% Plot the states
plt.figure(figsize=(10, 10))
if DUAL_KITES:
    plot_states = ['q21', 'dq21', 'l_t']
else:
    plot_states = ['q10', 'dq10', 'l_t', 'dl_t','e']
for index, state_name in enumerate(plot_states):
    plt.subplot(3, 2, index + 1)
    state_traj = np.vstack([plot_dict_SAM['x'][state_name][i] for i in range(plot_dict_SAM['x'][state_name].__len__())]).T

    d = trial.options['nlp']['SAM']['d']
    for region_index in range(d+1):
        plt.plot(time_grid_SAM['ip'][np.where(ip_regions_SAM == region_index)],
                    state_traj[np.where(ip_regions_SAM == region_index)],  '-')
        plt.gca().set_prop_cycle(None)  # reset color cycle

    plt.plot([], [], label=state_name)

    plt.gca().set_prop_cycle(None)  # reset color cycle

    state_recon = np.vstack([plot_dict_REC['x'][state_name][i] for i in range(plot_dict_REC['x'][state_name].__len__())]).T
    plt.plot(t_ip, state_recon, label=state_name + '_recon', linestyle='--')

    # add phase switches
    for region in regions_indeces:
        plt.axvline(x=time_grid_SAM_x[region[0]],color='k',linestyle='--')
        plt.axvline(x=time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]),color='k',linestyle='--')
    # plt.axvline(x=time_grid_SAM_x[regions_indeces[-1][-1]],color='k',linestyle='--')

    #
    # for region_indeces in regions_indeces[1:-1]:
    #     plt.axvline(x=time_grid_SAM_x[region_indeces[0]],color='b',linestyle='--')

    plt.xlabel('time [s]')

    plt.legend()
plt.tight_layout()
plt.show()

# %% Initialization and Reference
from casadi.tools import structure3
Vopt = trial.optimization.V_opt
Vref: structure3 = trial.optimization.V_ref
Vinit = trial.optimization.V_init

if True:
    plt.figure('Initialization and Reference')
    time_grid_init = trial.nlp.time_grids['x'](Vinit['theta', 't_f']).full().flatten()

    if DUAL_KITES:
        plot_states = ['q21', 'dq21', 'l_t','e']
    else:
        plot_states = ['q10', 'dq10', 'l_t','e']

    for index, state_name in enumerate(plot_states):
        plt.subplot(2, 2, index + 1)

        state_traj_ref = np.hstack(Vref['x', :, state_name]).T
        plt.plot(time_grid_init, state_traj_ref, label=state_name + '_ref', color=f'C{index}', linestyle='--')

        # add phase switches
        plt.axvline(x=time_grid_init[regions_indeces[1][0]], color='k', linestyle='--')
        plt.axvline(x=time_grid_init[regions_indeces[-1][0]], color='k', linestyle='--')

        for region_indeces in regions_indeces[1:-1]:
            plt.axvline(x=time_grid_init[region_indeces[0]], color='b', linestyle='--')

        plt.xlabel('time [s]')

        plt.legend()
    plt.tight_layout()
    plt.show()

# %% plot the results
import matplotlib
import mpl_toolkits.mplot3d as a3

plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')

_raw_vertices = np.array([[-1.2, 0, -0.4, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0]])
_raw_vertices = _raw_vertices - np.mean(_raw_vertices, axis=1).reshape((3, 1))


def drawKite(pos, rot, wingspan, color='C0', alpha=1):
    rot = np.reshape(rot, (3, 3)).T

    vtx = _raw_vertices * wingspan / 2  # -np.array([[0.5], [0], [0]]) * sizeKite
    vtx = rot @ vtx + pos
    tri = a3.art3d.Poly3DCollection([vtx.T])
    tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
    tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
    # tri.set_alpha(alpha)
    # tri.set_edgealpha(alpha)
    ax.add_collection3d(tri)


nk_reelout = int(options['nlp.n_k'] * trial.options['nlp']['phase_fix_reelout'])
nk_cut = round(options['nlp.n_k'] * trial.options['nlp']['phase_fix_reelout'])

q10_REC = plot_dict_REC['x']['q10']
ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C0-', alpha=0.3)

q10_opt = plot_dict_SAM['x']['q10']
ip_regions_SAM = plot_dict_SAM['SAM_regions_ip']


for region_index, color in zip(np.arange(0, trial.options['nlp']['SAM']['d']), [f'C{i}' for i in range(20)]):
    ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == region_index)],
              q10_opt[1][np.where(ip_regions_SAM == region_index)],
              q10_opt[2][np.where(ip_regions_SAM == region_index)]
              , '-', color=color,
                  alpha=1, markersize=3)

# ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == trial.options['nlp']['SAM']['d'])],
#           q10_opt[1][np.where(ip_regions_SAM == trial.options['nlp']['SAM']['d'])],
#           q10_opt[2][np.where(ip_regions_SAM == trial.options['nlp']['SAM']['d'])], 'C0.',
#           alpha=0.3, markersize=6)
# # ax.plot3D(q10_opt_average[0], q10_opt_average[1], q10_opt_average[2], 'b-', alpha=1)
# ax.plot3D(q10_opt_average_strobo[0], q10_opt_average_strobo[1], q10_opt_average_strobo[2], 'bs', alpha=1)
#
# # reconstructed trajectory
# ax.plot3D(q10_reconstruct[0], q10_reconstruct[1], q10_reconstruct[2], 'C1-', alpha=0.2)

# average trajectory
# ax.plot3D(X_macro_coll[0,:], X_macro_coll[1,:], X_macro_coll[2,:], 'b.-', alpha=1, markersize=6)


# plot important points
# ax.plot3D(q1_opt[0,0], q1_opt[1,0], q1_opt[2,0], 'C1o', alpha=1)
# ax.plot3D(q1_opt[0,strobo_indeces[-1]], q1_opt[1,strobo_indeces[-1]], q1_opt[2,strobo_indeces[-1]], 'C1o', alpha=1)


# tether10 = np.hstack([q21_opt_plot[:, [-1]], np.zeros((3, 1))])
# tether21 = np.hstack([q2_opt[:, [-1]], q1_opt[:, [-1]]])
# tether31 = np.hstack([q3_opt[:, [-1]], q1_opt[:, [-1]]])
# ax.plot3D(tether21[0], tether21[1], tether21[2], '-',color='black')
# ax.plot3D(tether31[0], tether31[1], tether31[2], '-',color='black')
# ax.plot3D(tether10[0], tether10[1], tether10[2], '-',color='black')


# set bounds for nice view
q10_REC_all = np.vstack([q10_REC[0],q10_REC[1],q10_REC[2]])
meanpos = np.mean(q10_REC_all, axis=1)

bblenght = np.max(np.abs(q10_REC_all - meanpos.reshape(3, 1)))
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

# plt.legend()
plt.tight_layout()
plt.savefig('3DReelout.pdf')
plt.show()
