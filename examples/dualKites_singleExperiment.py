
import awebox
import awebox as awe
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

from awebox.ocp.collocation import Collocation
from awebox.ocp.discretization_averageModel import eval_time_grids_SAM, construct_time_grids_SAM_reconstruction, \
    reconstruct_full_from_SAM
from awebox.tools.struct_operations import calculate_SAM_regions
from awebox.logger.logger import Logger as awelogger
from examples.paper_benchmarks import reference_options as ref

options = {}
options = ref.set_reference_options(user='A')
options = ref.set_dual_kite_options(options)
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

# SAM OPTIONS
options['nlp.SAM.use'] = True
options['nlp.cost.output_quadrature'] = False  # use enery as a state, works better with SAM
options['nlp.cost.output_quadrature'] = False  # use enery as a state, works better with SAM
options['nlp.SAM.MaInt_type'] = 'legendre'
options['nlp.SAM.N'] = 20 # the number of full cycles approximated
options['nlp.SAM.d'] = 3 # the number of cycles actually computed
options['nlp.SAM.ADAtype'] = 'CD'  # the approximation scheme

# SAM Regularization
single_regularization_param = 1E-4
options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 0*single_regularization_param
options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1*single_regularization_param
options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0*single_regularization_param
options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E-1*single_regularization_param


# smooth the reel in phase (this increases convergence speed x10)
options['solver.cost.beta.0'] = 8e0
options['solver.cost.u_regularisation.0'] = 1e0


options['user_options.trajectory.lift_mode.windings'] = options['nlp.SAM.d'] + 1
n_k = 15 * options['user_options.trajectory.lift_mode.windings']
options['nlp.n_k'] = n_k

# needed for correct initial tracking phase
# options['nlp.phase_fix_reelout'] = (options['user_options.trajectory.lift_mode.windings'] - 1) / options[
#     'user_options.trajectory.lift_mode.windings']
options['solver.initialization.groundspeed'] = 30.0
# options['nlp.phase_fix_reelout'] = 0.7

options['model.system_bounds.theta.t_f'] = [5, 10 * options['nlp.SAM.N']]  # [s]

options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
options['solver.linear_solver'] = 'ma57'
# options['solver.max_iter'] = 0
# options['solver.max_iter_hippo'] = 0

options['visualization.cosmetics.interpolation.N'] = 200  # high plotting resolution
options['visualization.cosmetics.plot_bounds'] = True  # high plotting resolution

options['solver.callback'] = True


# build and optimize the NLP (trial)
trial = awe.Trial(options, 'DualKitesLongHorizon')
trial.build()
# if DUAL_KITES:
#     trial.optimize(debug_locations=['initial'])
# else:
trial.optimize()

# %% Post-Processing
Vopt = trial.solution_dict['V_opt']
V_reconstruct, time_grid_recon_eval, output_vals_reconstructed = reconstruct_full_from_SAM(trial.nlp.options, Vopt,
                                                                                           trial.solution_dict['output_vals'])


# %% Plot state trajectories
time_grid_SAM = eval_time_grids_SAM(trial.nlp.options,Vopt['theta', 't_f'])
time_grid_SAM_x = time_grid_SAM['x']

# %% put together the state trajectory for export
import pandas
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
    assert trial.options['nlp']['SAM']['use']
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
    N_regions = trial.nlp.options['SAM']['d'] + 1
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
    t_grid_average = np.linspace(T_regions[0], Tend - T_regions[-1], N)
    df['t_average'] = t_grid_average

    return df


from awebox.viz.visualization import VisualizationSAM
def exportSAMtoDf(visualization: VisualizationSAM) -> pandas.DataFrame:
    assert trial.options['nlp']['SAM']['use']
    df = pandas.DataFrame()

    # find the duration of the regions
    n_k = trial.nlp.options['n_k']
    regions_indeces = calculate_SAM_regions(trial.nlp.options)
    regions_deltans = np.array([region.__len__() for region in regions_indeces])
    N_regions = trial.nlp.options['SAM']['d'] + 1
    assert len(regions_indeces) == N_regions
    T_regions = (V[
                     'theta', 't_f'] / n_k * regions_deltans).full().flatten()  # the duration of each discretization region
    T_end_SAM = np.sum(T_regions)
    t_grid_SAM = np.linspace(0, T_end_SAM, N)  # in the AWEBOX time grid, not correct
    df['regionIndex'] = visualization.plot_dict_SAM['SAM_regions_ip']

    # fill time
    values_time = interpolator_time(t_grid_SAM, 't', 0, 'x').full().flatten()
    df['t'] = values_time

    for entry_type in ['x', 'u']:
        for entry_name in trial.model.variables_dict[entry_type].keys():
            for index_dim in range(trial.model.variables_dict[entry_type][entry_name].shape[0]):
                # we evaluate on the AWEBox time grid, not the SAM time grid!
                values = interpolator(t_grid_SAM, entry_name, index_dim, entry_type).full().flatten()

                name = entry_type + '_' + entry_name + '_' + str(index_dim)
                df[name] = values

    # interpolate the average polynomials
    from awebox.ocp.discretization_averageModel import OthorgonalCollocation
    d_SAM = trial.nlp.options['SAM']['d']
    coll_points = np.array(ca.collocation_points(d_SAM, trial.nlp.options['SAM']['MaInt_type']))
    interpolator_average_integrator = OthorgonalCollocation(coll_points)
    interpolator_average = interpolator_average_integrator.getPolyEvalFunction(
        shape=trial.model.variables_dict['x'].cat.shape, includeZero=True)
    tau_average = np.linspace(0, 1, N)

    # compute the average polynomials and fill the dataframe
    X_average = interpolator_average.map(tau_average.size)(tau_average, V['x_macro', 0],
                                                           *[V['x_macro_coll', i] for i in range(d_SAM)])
    X_average = trial.model.variables_dict['x'].repeated(X_average)
    for entry_name in trial.model.variables_dict['x'].keys():
        for index_dim in range(trial.model.variables_dict['x'][entry_name].shape[0]):
            # we evaluate on the AWEBox time grid, not the SAM time grid!
            values = ca.vertcat(*X_average[:, entry_name, index_dim]).full().flatten()

            name = 'X' + '_' + entry_name + '_' + str(index_dim)
            df[name] = values

    # construct the time grid for the average polynomials
    Tend = float(time_grid_SAM['x'](V['theta', 't_f'])[-1])
    t_grid_average = np.linspace(T_regions[0], Tend - T_regions[-1], N)
    df['t_average'] = t_grid_average

# EXPORT
# basePath = '_export/varyN'
basePath = '_export/singleExperiment'

trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = False
trial.options['nlp']['SAM']['use'] = True
df_SAM = interpolate_SAM_trajectory(trial,Vopt, 2000)
df_SAM.to_csv(f'{basePath}/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_SAM.csv', index=False)


trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = True
trial.options['nlp']['SAM']['use'] = False
df_reconstruct = interpolate_trajectory(trial,V_reconstruct, 2000, float(time_grid_recon_eval['x'][-1]))
df_reconstruct.to_csv(f'{basePath}/dualKiteLongTrajectory_N_{options["nlp.N_SAM"]}_REC.csv', index=False)

awelogger.logger.info('Exported trajectory data!')
