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
import pickle

import awebox as awe
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
from awebox.ocp.discretization_averageModel import eval_time_grids_SAM, construct_time_grids_SAM_reconstruction, \
    reconstruct_full_from_SAM

DUAL_KITES = True

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}

if DUAL_KITES:
    from examples.paper_benchmarks import reference_options as ref

    options = ref.set_reference_options(user='A')
    options = ref.set_dual_kite_options(options)
else:
    options['user_options.system_model.architecture'] = {1: 0}
    options = set_ampyx_ap2_settings(options)

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

# options['nlp.phase_fix_reelout'] = 0.7
options['nlp.useAverageModel'] = True
options['nlp.cost.output_quadrature'] = False  # use enery as a state, works better with SAM
options['nlp.MaInt_type'] = 'radau'
options['nlp.N_SAM'] = 3 # the number of full cycles approximated
options['nlp.d_SAM'] = 4 # the number of cycles actually computed
options['nlp.SAM_ADAtype'] = 'BD'  # the approximation scheme
options['nlp.SAM_Regularization'] = 200  # regularization parameter
# options['nlp.SAM_Regularization'] = 1E-1 * options['nlp.N_SAM']  # regularization parameter


options['user_options.trajectory.lift_mode.windings'] = options['nlp.d_SAM'] + 1
n_k = 15 * options['user_options.trajectory.lift_mode.windings']
options['nlp.n_k'] = n_k

# needed for correct initial tracking phase
options['nlp.phase_fix_reelout'] = (options['user_options.trajectory.lift_mode.windings'] - 1) / options[
    'user_options.trajectory.lift_mode.windings']
options['solver.initialization.groundspeed'] = 30.0 # better initialization

if DUAL_KITES:
    options['model.system_bounds.theta.t_f'] = [5, 10 * options['nlp.N_SAM']]  # [s]
else:
    options['model.system_bounds.theta.t_f'] = [30, 20 * options['nlp.N_SAM']]  # [s]

options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
options['solver.linear_solver'] = 'ma57'
# options['solver.max_iter'] = 0
# options['solver.max_iter_hippo'] = 0

options['visualization.cosmetics.interpolation.N'] = 300  # high plotting resolution
options['visualization.cosmetics.plot_bounds'] = True  # high plotting resolution

# options['solver.callback'] = True

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'DualKitesLongHorizon')
trial.build()

# instead of optimitizing, we load the solution dict from a previous run
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

# # %%
# trial.plot(['states'])
# plt.gcf().tight_layout()



# print(asdf)
# %% Post-Processing


import casadi as ca
from awebox.ocp.discretization_averageModel import OthorgonalCollocation
from awebox.tools.struct_operations import calculate_SAM_regions, evaluate_cost_dict

d_SAM = options['nlp.d_SAM']
N_SAM = options['nlp.N_SAM']

Vopt = solution_dict['V_opt']
Vref = solution_dict['V_ref']
# Vinit = trial.optimization.V_init
# Vopt = trial.optimization.V_init

d = trial.nlp.time_grids['x'](Vopt['theta', 't_f']).full().flatten()

regions_indeces = calculate_SAM_regions(trial.nlp.options)
strobo_indeces = [region_indeces[0] for region_indeces in regions_indeces[1:]]
model = trial.model
macroIntegrator = OthorgonalCollocation(np.array(ca.collocation_points(d_SAM, options['nlp.MaInt_type'])))

t_f_opt = Vopt['theta', 't_f']

# %% Reconstuct into large V structure of the FULL trajectory
time_grid_SAM = eval_time_grids_SAM(trial.nlp.options,Vopt['theta', 't_f'])
time_grid_SAM_x = time_grid_SAM['x']

V_reconstruct, time_grid_recon_eval, output_vals_reconstructed = reconstruct_full_from_SAM(trial.nlp.options, Vopt,
                                                                                           trial.solution_dict['output_vals'])


# %% Fake the AWEbox into recalibrating its visualz with the reconstructed trajectory
trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = True
trial.options['nlp']['SAM']['use'] = False
n_k_total = len(V_reconstruct['x']) - 1
trial.visualization.plot_dict['n_k'] = n_k_total
# print(calculate_kdx_SAM_reconstruction(trial.options['nlp'], V_reconstruct,30))

# OVERWRITE VOPT OF THE TRIAL
trial.optimization.V_opt = V_reconstruct

trial.visualization.recalibrate(V_reconstruct, trial.visualization.plot_dict, output_vals_reconstructed, trial.optimization.integral_outputs_final, trial.options, time_grid_recon_eval,solution_dict['cost'], 'fake', V_reconstruct, trial.optimization.global_outputs_opt)

# find the duration of the regions
n_k = trial.nlp.options['n_k']
regions_indeces = calculate_SAM_regions(trial.nlp.options)
regions_deltans = np.array([region.__len__() for region in regions_indeces])
N_regions = trial.nlp.options['SAM']['d'] + 1
assert len(regions_indeces) == N_regions
T_regions = (Vopt['theta', 't_f'] / n_k * regions_deltans).full().flatten()

# %% MPC SIMULATION
import copy
# from awebox.logger.logger import Logger as awelogger
# awelogger.logger.setLevel('INFO')

T_opt = float(time_grid_recon_eval['x'][-1])


# set-up closed-loop simulation
T_mpc = 1.5 # seconds
N_mpc = 50 # MPC horizon

# T_sim = 3 # seconds
T_sim = T_opt - T_regions[-1] # seconds
ts = T_mpc/N_mpc # sampling time
N_sim = int(T_sim/ts)  # closed-loop simulation steps
#SAM reconstruct options
options['nlp.flag_SAM_reconstruction'] = True
options['nlp.useAverageModel'] = False

# MPC options
options['mpc.scheme'] = 'radau'
options['mpc.d'] = 2
options['mpc.jit'] = False
options['mpc.cost_type'] = 'tracking'
options['mpc.expand'] = True
options['mpc.linear_solver'] = 'ma27'
options['mpc.max_iter'] = 1500
options['mpc.max_cpu_time'] = 2000
options['mpc.N'] = N_mpc
options['mpc.plot_flag'] = False
options['mpc.ref_interpolator'] = 'poly'
options['mpc.homotopy_warmstart'] = True
options['mpc.terminal_point_constr'] = False

# simulation options
options['sim.number_of_finite_elements'] = 20 # integrator steps within one sampling time
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

# reduce average wind speed
# options['sim.sys_params']['wind']['u_ref'] = 1.0*options['sim.sys_params']['wind']['u_ref']

# make simulator
closed_loop_sim = awe.sim.Simulation(trial,'closed_loop', ts, options)

# # %% Debug Plot
# # plot_t_grid = np.array(closed_loop_sim.visualization.plot_dict['time_grids']['ip']).flatten()
#
# #evaluate the interpolator of the closed loop simulation
interpolator = closed_loop_sim.mpc.interpolator
q21_ref = np.vstack([interpolator(time_grid_recon_eval['x'].full().flatten(),'q21',0,'x').full().flatten(),
                     interpolator(time_grid_recon_eval['x'].full().flatten(),'q21',1,'x').full().flatten(),
                     interpolator(time_grid_recon_eval['x'].full().flatten(),'q21',2,'x').full().flatten()])

plt.figure(figsize=(10, 10))
# plt.plot(time_grid_recon_eval['x'].full().flatten(), ca.horzcat(*V_reconstruct['x',:,'q10']).full().T,'.-')

# reset color cycle
plt.gca().set_prop_cycle(None)
plt.plot(time_grid_recon_eval['x'].full().flatten(), q21_ref.T,'--')
plt.show()
#
# # print(asdf)

# Run the closed-loop simulation

closed_loop_sim.run(N_sim)
plt.show()

# %% plot the interpolated reference trajectory
plot_t_grid = np.array(closed_loop_sim.visualization.plot_dict['time_grids']['ip']).flatten()

if DUAL_KITES:
    q21_ref = np.vstack([interpolator(plot_t_grid, 'q21', 0, 'x').full().flatten(),
                         interpolator(plot_t_grid, 'q21', 1, 'x').full().flatten(),
                         interpolator(plot_t_grid, 'q21', 2, 'x').full().flatten()])
    q31_ref = np.vstack([interpolator(plot_t_grid, 'q31', 0, 'x').full().flatten(),
                         interpolator(plot_t_grid, 'q31', 1, 'x').full().flatten(),
                         interpolator(plot_t_grid, 'q31', 2, 'x').full().flatten()])

# trajectories
q10_MPC = np.vstack([np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][0]).flatten(),
                     np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][1]).flatten(),
                     np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][2]).flatten()])

if DUAL_KITES:
    q21_MPC = np.vstack([np.array(closed_loop_sim.visualization.plot_dict['x']['q21'][0]).flatten(),
                         np.array(closed_loop_sim.visualization.plot_dict['x']['q21'][1]).flatten(),
                         np.array(closed_loop_sim.visualization.plot_dict['x']['q21'][2]).flatten()])
    q31_MPC = np.vstack([np.array(closed_loop_sim.visualization.plot_dict['x']['q31'][0]).flatten(),
                         np.array(closed_loop_sim.visualization.plot_dict['x']['q31'][1]).flatten(),
                         np.array(closed_loop_sim.visualization.plot_dict['x']['q31'][2]).flatten()])

# % Plot the STATES
plt.figure(figsize=(10,10))

# plot the reference
# plt.plot(closed_loop_sim.visualization.plot_dict['time_grids']['ref']['x'].full(), closed_loop_sim.visualization.plot_dict['ref']['x']['q10'][0], label='reference_MPC')
for index_state, name_state in enumerate(['q21','dq21']):
    for index_dim in range(3):
        traj = interpolator(plot_t_grid, name_state, index_dim, 'x').full().flatten()
        plt.subplot(3, 2, index_state*3 + index_dim + 1)

        plt.plot(closed_loop_sim.visualization.plot_dict['time_grids']['ip'],
                 np.array(closed_loop_sim.visualization.plot_dict['x'][name_state][index_dim]).flatten(), label='sim')

        plt.plot(plot_t_grid,traj,'--', label='reference_custom')
        plt.legend()
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


nk_reelout = int(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])
nk_cut = round(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])

if DUAL_KITES:

    # referece
    ax.plot3D(q21_ref[0], q21_ref[1], q21_ref[2], 'C0--', alpha=1)
    ax.plot3D(q31_MPC[0], q31_MPC[1], q31_MPC[2], 'C1--', alpha=1)

    # traj
    ax.plot3D(q21_MPC[0], q21_MPC[1], q21_MPC[2], 'C0-', alpha=1)
    ax.plot3D(q31_MPC[0], q31_MPC[1], q31_MPC[2], 'C1-', alpha=1)


# else:
#     ax.plot3D(q10_reconstruct[0], q10_reconstruct[1], q10_reconstruct[2], 'C1-', alpha=0.2)


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

# %% put together the state trajectory for export
import pandas

def sim_to_df(sim):
    df = pandas.DataFrame()

    # timegrids
    df['t'] = sim.visualization.plot_dict['time_grids']['ip']


    for entry_type in ['x', 'u']:
        for entry_name in sim.visualization.plot_dict[entry_type].keys():
            for index_dim in range(sim.trial.model.variables_dict[entry_type][entry_name].shape[0]):
                name = entry_type + '_' + entry_name + '_' + str(index_dim)
                values = np.array(sim.visualization.plot_dict[entry_type][entry_name][index_dim]).flatten()
                df[name] = values
    return df

def interpolate_trajectory(trial,V,N:int, Tend: float) -> pandas.DataFrame:
    assert trial.options['nlp']['SAM']['flag_SAM_reconstruction']
    df = pandas.DataFrame()
    interpolator = trial.nlp.Collocation.build_interpolator(trial.nlp.options, V)
    t_grid = np.linspace(0, Tend, N)
    df['t'] = t_grid

    for entry_type in ['x','u']:
        for entry_name in trial.model.variables_dict[entry_type].keys():
            for index_dim in range(trial.model.variables_dict[entry_type][entry_name].shape[0]):
                name = entry_type + '_' + entry_name + '_' + str(index_dim)
                values = interpolator(t_grid, entry_name,index_dim, entry_type).full().squeeze()
                # print(f'name:{name}, shape:{values.shape}',flush=True)
                df[name] = values
    return df

def interpolate_SAM_trajectory(trial,V,N:int) -> pandas.DataFrame:
    assert trial.options['nlp']['SAM']['use'] == True
    df = pandas.DataFrame()
    interpolator = trial.nlp.Collocation.build_interpolator(trial.nlp.options, V)

    # fake a time struct
    from casadi.tools import struct_symMX,entry
    import casadi as ca
    time_state_struct = struct_symMX([entry('t',shape=(1,1))])
    tf_struct = struct_symMX([entry('t_f',shape=V['theta','t_f'].shape)])
    V_time_struct = struct_symMX([entry('x',repeat=[trial.nlp.n_k],struct=time_state_struct),
                                  entry('coll_var',repeat=[trial.nlp.n_k,trial.nlp.d],struct=struct_symMX([entry('x',struct=time_state_struct)])),
                                  entry('theta',struct = tf_struct)])
    # fill the struct with the time grid
    V_time = V_time_struct(0)
    V_time['x',:,'t',0] = (time_grid_SAM['x'](V['theta', 't_f'])).full().flatten().tolist()
    for k in range(trial.nlp.n_k):
        V_time['coll_var',k,:,'x','t',0] = (time_grid_SAM['coll'](V['theta', 't_f'])[k,:].T).full().flatten().tolist()
    V_time['theta','t_f'] = V['theta','t_f']

    interpolator_time  = trial.nlp.Collocation.build_interpolator(trial.nlp.options, V_time)

    # find the duration of the regions
    n_k = trial.nlp.options['n_k']
    regions_indeces = calculate_SAM_regions(trial.nlp.options)
    regions_deltans = np.array([region.__len__() for region in regions_indeces])
    N_regions = trial.nlp.options['d_SAM'] + 1
    assert len(regions_indeces) == N_regions
    T_regions = (V['theta','t_f'] / n_k * regions_deltans).full().flatten()  # the duration of each discretization region
    T_end_SAM = np.sum(T_regions)
    t_grid_SAM = np.linspace(0, T_end_SAM, N) # in the AWEBOX time grid, not correct
    df['regionIndex'] = [np.argmax(t < np.cumsum(T_regions)+0.0001) for t in t_grid_SAM]


    # fill time
    values_time = interpolator_time(t_grid_SAM, 't', 0, 'x').full().flatten()
    df['t'] = values_time


    for entry_type in ['x','u']:
        for entry_name in trial.model.variables_dict[entry_type].keys():
            for index_dim in range(trial.model.variables_dict[entry_type][entry_name].shape[0]):
                # we evaluate on the AWEBox time grid, not the SAM time grid!
                values = interpolator(t_grid_SAM, entry_name,index_dim, entry_type).full().flatten()

                name = entry_type + '_' + entry_name + '_' + str(index_dim)
                df[name] = values


    # interpolate the average polynomials
    from awebox.ocp.discretization_averageModel import OthorgonalCollocation
    d_SAM = trial.nlp.options['SAM']['d']
    coll_points = np.array(ca.collocation_points(d_SAM,trial.nlp.options['SAM']['MaInt_type']))
    interpolator_average_integrator = OthorgonalCollocation(coll_points)
    interpolator_average = interpolator_average_integrator.getPolyEvalFunction(shape=trial.model.variables_dict['x'].cat.shape, includeZero=True)
    tau_average = np.linspace(0, 1, N)

    # compute the average polynomials and fill the dataframe
    X_average = interpolator_average.map(tau_average.size)(tau_average, V['x_macro',0], *[V['x_macro_coll',i] for i in range(d_SAM)])
    X_average = trial.model.variables_dict['x'].repeated(X_average)
    for entry_name in trial.model.variables_dict['x'].keys():
        for index_dim in range(trial.model.variables_dict['x'][entry_name].shape[0]):
            # we evaluate on the AWEBox time grid, not the SAM time grid!
            values = ca.vertcat(*X_average[:,entry_name, index_dim]).full().flatten()

            name = 'X' + '_' + entry_name + '_' + str(index_dim)
            df[name] = values

    # construct the time grid for the average polynomials
    Tend = float(time_grid_SAM['x'](V['theta', 't_f'])[-1])
    t_grid_average = np.linspace(T_regions[0], Tend - T_regions[-1], N) # in AWEBOX time grid!
    df['t_average'] = t_grid_average

    return df
# export the simulation trajectory
df_sim = sim_to_df(closed_loop_sim)
df_sim.to_csv(f'_export/MPC/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_MPC.csv', index=False)


trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = False
trial.options['nlp']['SAM']['use'] = True
df_SAM = interpolate_SAM_trajectory(trial,Vopt, 3000)
df_SAM.to_csv(f'_export/MPC/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_SAM.csv', index=False)


trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = True
trial.options['nlp']['SAM']['use'] = False
df_reconstruct = interpolate_trajectory(trial,V_reconstruct, 3000, float(time_grid_recon_eval['x'][-1]))
df_reconstruct.to_csv(f'_export/MPC/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_REC.csv', index=False)
