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
import awebox
import awebox as awe
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

from awebox.ocp.collocation import Collocation
from awebox.ocp.discretization_averageModel import construct_time_grids_SAM, construct_time_grids_SAM_reconstruction, \
    reconstruct_full_from_SAM
from awebox.tools.struct_operations import calculate_SAM_regions


# %% Latexify the plots


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
options['nlp.SAM_MaInt_type'] = 'radau'
options['nlp.N_SAM'] = 20 # the number of full cycles approximated
options['nlp.d_SAM'] = 3 # the number of cycles actually computed
options['nlp.SAM_ADAtype'] = 'BD' # the approximation scheme
options['user_options.trajectory.lift_mode.windings'] = options['nlp.d_SAM'] + 1
n_k = 20 * options['user_options.trajectory.lift_mode.windings']
options['nlp.n_k'] = n_k
# options['nlp.phase_fix_reelout'] = (options['user_options.trajectory.lift_mode.windings'] - 1) / options[
#     'user_options.trajectory.lift_mode.windings']
options['nlp.phase_fix_reelout'] = 0.7

if DUAL_KITES:
    options['model.system_bounds.theta.t_f'] = [5, 10 * options['nlp.N_SAM']]  # [s]
else:
    options['model.system_bounds.theta.t_f'] = [40, 40 * options['nlp.N_SAM']]  # [s]

options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
options['solver.linear_solver'] = 'ma57'
# options['solver.max_iter'] = 0
# options['solver.max_iter_hippo'] = 0

options['visualization.cosmetics.interpolation.N'] = 1000  # high plotting resolution
options['visualization.cosmetics.plot_bounds'] = True  # high plotting resolution

options['solver.callback'] = True


# set-up sweep options
# sweep_opts = [('nlp.N_SAM', [20,30])]
#
# sweep = awe.Sweep(name = 'dual_kites_power_curve', options = options, seed = sweep_opts)
# sweep.build()
# sweep.run(apply_sweeping_warmstart = True)
# sweep.plot(['comp_stats', 'comp_convergence'])
# plt.show()


# build and optimize the NLP (trial)
trial = awe.Trial(options, 'DualKitesLongHorizon')
trial.build()
if DUAL_KITES:
    trial.optimize(debug_locations=['initial'])
else:
    trial.optimize()

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
# plot_dict = trial.visualization.plot_dict
# outputs = plot_dict['outputs']
# time = plot_dict['time_grids']['ip']
# avg_power = plot_dict['power_and_performance']['avg_power'] / 1e3
#
# print('======================================')
# print('Average power: {} kW'.format(avg_power))
# print('======================================')


# %% Post-Processing
Vopt = trial.solution_dict['V_opt']
V_reconstruct, time_grid_recon_eval, output_vals_reconstructed = reconstruct_full_from_SAM(trial,Vopt)


# %% Plot state trajectories
time_grid_SAM = construct_time_grids_SAM(trial.nlp.options)
time_grid_SAM_x = time_grid_SAM['x'](Vopt['theta', 't_f']).full().flatten()

# %% put together the state trajectory for export
import pandas
def interpolate_trajectory(trial: awebox.Trial,V,N:int, Tend: float) -> pandas.DataFrame:
    assert trial.options['nlp']['flag_SAM_reconstruction']
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

def interpolate_SAM_trajectory(trial: awebox.Trial,V,N:int) -> pandas.DataFrame:
    assert trial.options['nlp']['useAverageModel'] == True
    df = pandas.DataFrame()
    interpolator = trial.nlp.Collocation.build_interpolator(trial.nlp.options, V)

    # find the duration of the regions
    n_k = trial.nlp.options['n_k']
    regions_indeces = calculate_SAM_regions(trial.nlp.options)
    regions_deltans = np.array([region.__len__() for region in regions_indeces])
    N_regions = trial.nlp.options['d_SAM'] + 2
    assert len(regions_indeces) == N_regions
    T_regions = (V['theta','t_f'] / n_k * regions_deltans).full().flatten()  # the duration of each discretization region
    T_end_SAM = np.sum(T_regions)

    # construct a time grid that correctly represents the SAM regions in physical time
    t_grid_SAM = np.linspace(0, T_end_SAM, N) # in the AWEBOX time grid, not correct
    # df['t_SAM'] = t_grid_SAM
    df['regionIndex'] = [np.argmax(t < np.cumsum(T_regions)+0.0001) for t in t_grid_SAM]
    offsets = np.array([np.sum(T_regions[:df['regionIndex'][i]]) for i in range(N)])
    df['t'] = t_grid_SAM + offsets

    for entry_type in ['x','u']:
        for entry_name in trial.model.variables_dict[entry_type].keys():
            for index_dim in range(trial.model.variables_dict[entry_type][entry_name].shape[0]):
                # we evaluate on the AWEBox time grid, not the SAM time grid!
                values = interpolator(t_grid_SAM, entry_name,index_dim, entry_type).full().flatten()

                name = entry_type + '_' + entry_name + '_' + str(index_dim)
                df[name] = values

    return df


trial.options['nlp']['flag_SAM_reconstruction'] = False
trial.options['nlp']['useAverageModel'] = True
df_SAM = interpolate_SAM_trajectory(trial,Vopt, 2000)
df_SAM.to_csv(f'_export/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_SAM.csv', index=False)


trial.options['nlp']['flag_SAM_reconstruction'] = True
trial.options['nlp']['useAverageModel'] = False
df_reconstruct = interpolate_trajectory(trial,V_reconstruct, 2000, float(time_grid_recon_eval['x'][-1]))
df_reconstruct.to_csv(f'_export/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_REC.csv', index=False)


