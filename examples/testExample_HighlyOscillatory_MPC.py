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

import numpy

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

# set the logger level to 'DEBUG' to see IPOPT output
from awebox.logger.logger import Logger as awelogger
from examples.paper_benchmarks.reference_options import set_reference_options

awelogger.logger.setLevel(10)

N = 3
d = 3

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1: 0}
options = set_reference_options(options)
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'

# (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
# note: this may result in slightly slower solution timings
options['nlp.compile_subfunctions'] = False
options['nlp.cost.beta'] = False  # penalize side-slip (can improve convergence)
options['model.integration.method'] = 'constraints'  # use enery as a state, works better with SAM

options['nlp.collocation.u_param'] = 'zoh'
options['nlp.SAM.use'] = True
options['nlp.SAM.MaInt_type'] = 'legendre'
options['nlp.SAM.N'] = N  # the number of full cycles approximated
options['nlp.SAM.d'] = d  # the number of cycles actually computed
options['nlp.SAM.ADAtype'] = 'CD'  # the approximation scheme
options['user_options.trajectory.lift_mode.windings'] = options['nlp.SAM.d'] + 1  # todo: set this somewhere else
options['user_options.trajectory.fixed_params'] = {}  # free tether diameter

# SAM Regularization
single_regularization_param = 1E-1
options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 1E1 * single_regularization_param
options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1E0 * single_regularization_param
# options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 1E3*single_regularization_param
options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0 * single_regularization_param
options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E-2 * single_regularization_param

# Number of discretization points
n_k = 20 * (options['nlp.SAM.d']) * 2
options['nlp.n_k'] = n_k

# model bounds
options['model.system_bounds.x.dl_t'] = [-50.0, 20.0]  # [m/s]=
options['model.system_bounds.x.l_t'] = [10.0, 2500.0]  # [m]
options['model.system_bounds.x.ddl_t'] = [-2.4, 2.4]  # [m/s^2]
options['model.system_bounds.theta.t_f'] = [20, 50 + options['nlp.SAM.N'] * 30]  # [s]
# solver and viz options
options['solver.linear_solver'] = 'ma27'
options['visualization.cosmetics.interpolation.n_points'] = 300 * options['nlp.SAM.N']  # high plotting resolution

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'SAM_MPC')
trial.build()
trial.optimize()
# trial.save(fn=f'trial_save_SAM_{"dual" if DUAL_KITES else "single"}Kite')
solution_dict = trial.solution_dict

# draw some of the pre-coded plots for analysis

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power'] / 1e3

print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

# %% Fake the AWEbox into recalibrating its visualz with the reconstructed trajectory
V_reconstruct = trial.visualization.plot_dict['V_plot']
trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = True
trial.options['nlp']['SAM']['use'] = False
n_k_total = len(V_reconstruct['x']) - 1
trial.visualization.plot_dict['n_k'] = n_k_total
# print(calculate_kdx_SAM_reconstruction(trial.options['nlp'], V_reconstruct,30))

# OVERWRITE VOPT OF THE TRIAL
trial.optimization.V_opt = V_reconstruct
trial.optimization.V_final_si = trial.visualization.plot_dict['V_plot_si']

# %% MPC SIMULATION
import copy

# awelogger.logger.setLevel('INFO')

# from awebox.logger.logger import Logger as awelogger
# awelogger.logger.setLevel('INFO')
time_grid_MPC = trial.visualization.plot_dict['time_grids']['x']
T_opt = float(time_grid_MPC[-1])

# set-up closed-loop simulation
T_mpc = 3  # seconds
N_mpc = 20  # MPC horizon
ts = T_mpc / N_mpc  # sampling time

# SAM reconstruct options
options['nlp.SAM.flag_SAM_reconstruction'] = True
options['nlp.SAM.use'] = False

# MPC options
options['mpc.scheme'] = 'radau'
options['mpc.d'] = 3
options['mpc.jit'] = False
options['mpc.cost_type'] = 'tracking'
options['mpc.expand'] = True
options['mpc.linear_solver'] = 'ma27'
options['mpc.max_iter'] = 600
options['mpc.max_cpu_time'] = 2000
options['mpc.N'] = N_mpc
options['mpc.plot_flag'] = False
options['mpc.ref_interpolator'] = 'poly'
options['mpc.homotopy_warmstart'] = True
options['mpc.terminal_point_constr'] = False

# simulation options
options['sim.number_of_finite_elements'] = 50  # integrator steps within one sampling time
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

weights_x = trial.model.variables_dict['x'](1E-6)
weights_x['q10'] = 1
weights_x['dq10'] = 1
weights_x['r10'] = 1
weights_x['e'] = 0

additionalMPCoptions = {}
additionalMPCoptions['Q'] = weights_x
additionalMPCoptions['R'] = trial.model.variables_dict['u'](1)
additionalMPCoptions['P'] = weights_x
additionalMPCoptions['Z'] = trial.model.variables_dict['z'](1E-6)

# make simulator
from awebox import sim

closed_loop_sim = sim.Simulation(trial, 'closed_loop', ts, options, additional_mpc_options=additionalMPCoptions)

#  Run the closed-loop simulation

# T_sim = T_opt//40 # seconds
T_sim = T_opt  # seconds
N_sim = int(T_sim / ts)  # closed-loop simulation steps
# tion steps

startTime = 0
closed_loop_sim.run(N_sim, startTime=startTime)

# %% Debug Plot
import casadi as ca
# # plot_t_grid = np.array(closed_loop_sim.visualization.plot_dict['time_grids']['ip']).flatten()
#
# #evaluate the interpolator of the closed loop simulation
interpolator = closed_loop_sim.mpc.interpolator_si
time_grid_MPC_ref = np.mod(np.linspace(0, T_opt-0.001, 100), T_opt)
x_ref = trial.model.variables_dict['x'].repeated(interpolator(time_grid_MPC_ref,'x'))
# q21_ref = np.vstack([interpolator(time_grid_MPC.full().flatten(),'q21',0,'x').full().flatten(),
#                      interpolator(time_grid_MPC.full().flatten(),'q21',1,'x').full().flatten(),
#                      interpolator(time_grid_MPC.full().flatten(),'q21',2,'x').full().flatten()])
# plt.figure(figsize=(10, 10))
# states_to_plot = ['q10','dq10','l_t','dl_t']
# for index,state in enumerate(states_to_plot):
#     plt.subplot(2, 2, index+1)
#     plt.plot(trial.visualization.plot_dict['time_grids']['x'].full().flatten(), ca.horzcat(*V_reconstruct['x',:,state]).full().T,'.-',alpha=0.2)
#
#     # reset color cycle
#     plt.gca().set_prop_cycle(None)
#     plt.plot(time_grid_MPC_ref, ca.horzcat(*x_ref[:,state]).full().T,'--')
# plt.show()
# #
# # print(asdf)

#  plot the interpolated reference trajectory
plot_t_grid = np.array(closed_loop_sim.visualization.plot_dict['time_grids']['ip']).flatten()

# trajectories
q10_MPC = np.vstack([np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][0]).flatten(),
                     np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][1]).flatten(),
                     np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][2]).flatten()])

plot_dict_CLSIM = closed_loop_sim.visualization.plot_dict
ip_grid = plot_dict_CLSIM['time_grids']['ip']

# reference trajectory of the PMCP at the start
startTime_updated = ip_grid[0]
pmpc_first_time_grid = closed_loop_sim.mpc._Pmpc__compute_time_grids(startTime_updated)
pmpc_first_ref = closed_loop_sim.mpc.get_reference(*pmpc_first_time_grid)

# % Plot the STATES
plt.figure(figsize=(10,10))

# plot the reference
# plt.plot(closed_loop_sim.visualization.plot_dict['time_grids']['ref']['x'].full(), closed_loop_sim.visualization.plot_dict['ref']['x']['q10'][0], label='reference_MPC')

states_to_plot = ['q10','dq10','r10','l_t','dl_t','e']
for index_state, name_state in enumerate(states_to_plot):
    plt.subplot(int(np.ceil((len(states_to_plot)+1)//2)), 2, index_state + 1)

    plt.plot(ip_grid,
             np.vstack(plot_dict_CLSIM['x'][name_state]).T)
    plt.plot([],[],'k-',label='sim')
    # reset color cycle
    plt.gca().set_prop_cycle(None)

    traj_state =  np.vstack(plot_dict_CLSIM['ref_si']['x'][name_state])
    plt.plot(ip_grid,traj_state.T,'--',)
    plt.plot([],[],'k--',label=' mpc reference_recorded')

    # plot reconstructed trajectory
    # plt.gca().set_prop_cycle(None)
    # plt.plot(trial.visualization.plot_dict['time_grids']['x'].full().flatten(), ca.horzcat(*trial.visualization.plot_dict['V_plot']['x',:,name_state]).full().T,'.-',alpha=0.2)
    plt.plot(trial.visualization.plot_dict['time_grids']['x'].full().flatten(), ca.horzcat(*trial.visualization.plot_dict['V_plot_si']['x',:,name_state]).full().T,'.-',alpha=0.2)
    plt.plot([],[],'k.-',label='rec+ip')

    # reset color cycle
    plt.gca().set_prop_cycle(None)
    plt.plot(time_grid_MPC_ref, ca.horzcat(*x_ref[:,name_state]).full().T,'--',alpha=0.3,linestyle='dotted')
    plt.plot([],[],'k',linestyle='dotted',label='mpc interpol. eval')

    # plot first mpc reference trajectory
    # plt.gca().set_prop_cycle(None)
    # plt.plot(pmpc_first_time_grid[1][0::4], ca.horzcat(*pmpc_first_ref['x',:,name_state]).full().T,'-.')

    plt.ylabel(name_state)
    plt.legend()
plt.tight_layout()
plt.show()


# %% 3D plot of the tracked trajectory
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


# nk_reelout = int(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])
# nk_cut = round(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])
#

# else:
q10_REC = trial.visualization.plot_dict['x']['q10']
ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C1-', alpha=0.2)
ax.plot3D(q10_MPC[0], q10_MPC[1], q10_MPC[2], 'C0-', alpha=1)


# set bounds for nice view
meanpos = np.mean(q10_MPC[:], axis=1) + np.array([50, 0, 0])

bblenght = np.max(np.abs(q10_MPC - meanpos.reshape(3, 1)))
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
# plt.savefig('3DReelout.pdf')
plt.show()




