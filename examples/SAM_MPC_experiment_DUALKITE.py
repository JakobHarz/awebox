#!/usr/bin/python3
"""
Dual-kite pumping trajectory for the Ampyx AP2 aircraft, optimized with SAM,
reconstructed, and validated in closed-loop MPC simulation.

Adapted from SAM_MPC_experiment.py (single-kite) to the dual-kite architecture
used in ampyx_ap2_SAM.py's DUAL_KITES branch.

:author: Jakob Harzer
"""

from typing import List, Dict

import numpy

import awebox as awe
from examples.paper_benchmarks.reference_options import set_reference_options, set_dual_kite_options
import numpy as np

# set the logger level to 'DEBUG' to see IPOPT output
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)


def run_SAM_MPC_experiment_dualkite(d=3, N=5):

    # indicate desired system architecture
    # here: dual kite with 6DOF Ampyx AP2 model, single layer node
    options = set_reference_options(user='A')
    options = set_dual_kite_options(options)

    # indicate desired operation mode
    # here: lift-mode system with pumping-cycle operation
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'

    # indicate numerical nlp details
    options['solver.linear_solver'] = 'ma57'
    options['nlp.cost.beta'] = False  # penalize side-slip (can improve convergence)
    options['nlp.compile_subfunctions'] = False  # significantly decreases construction time for the heavier dual-kite model
    options['model.integration.method'] = 'constraints'  # use energy as a state, works better with SAM
    # options['solver.max_iter_hippo'] = 1000  # dual-kite homotopy sub-problems need more room than the single-kite default (100)

    # omega_bound = 30.0 * np.pi / 180.0
    # options['model.system_bounds.x.omega'] = [np.array(3 * [-omega_bound]), np.array(3 * [omega_bound])]

    # initialization: 2024 "found nice working parameters" run lowered this from the set_dual_kite_options default (50.0)
    # options['solver.initialization.groundspeed'] = 30.0

    options['nlp.collocation.u_param'] = 'zoh'
    options['nlp.SAM.use'] = True
    options['nlp.SAM.MaInt_type'] = 'radau'
    options['nlp.SAM.N'] = N  # the number of full cycles approximated
    options['nlp.SAM.d'] = d  # the number of cycles actually computed
    options['nlp.SAM.ADAtype'] = 'BD'  # the approximation scheme
    options['user_options.trajectory.lift_mode.windings'] = options['nlp.SAM.d'] + 1  # todo: set this somewhere else

    # SAM Regularization: ratio matches the 2024 "found nice working parameters" run
    # (component_costs['SAM_regularization'] = SAM_Regularization*(1E-4*first_deriv + 1*third_deriv + 10*similar_durations)),
    # where similar-cycle-duration regularization dominates by orders of magnitude -- the opposite balance of what we had before.
    single_regularization_param = 1.0
    options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 1E-1 * single_regularization_param
    options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1E0 * single_regularization_param
    options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0 * single_regularization_param
    options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E1 * single_regularization_param

    # Number of discretization points (dual-kite scaling, cf. ampyx_ap2_SAM.py)
    n_k = 10 * (options['nlp.SAM.d']) * 2
    options['nlp.n_k'] = n_k

    # model bounds (dual-kite scaling, cf. ampyx_ap2_SAM.py)
    options['model.system_bounds.x.dl_t'] = [-15.0, 20.0]  # [m/s]
    options['model.system_bounds.x.l_t'] = [10.0, 500.0 + 100 * N]  # [m] (2024 run left this at the reference_options default lower bound)
    options['model.system_bounds.x.ddl_t'] = [-2.4, 2.4]  # [m/s^2]
    options['model.system_bounds.theta.t_f'] = [5, 10* N]  # [s]

    # viz options
    options['visualization.cosmetics.interpolation.n_points'] = 300 * options['nlp.SAM.N']  # high plotting resolution

    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'SAM_MPC_DUALKITE')
    trial.build()
    trial.optimize()
    solution_dict = trial.solution_dict

    # extract information from the solution for independent plotting or post-processing
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

    # OVERWRITE VOPT OF THE TRIAL
    trial.optimization.V_opt = V_reconstruct
    trial.optimization.V_final_si = trial.visualization.plot_dict['V_plot_si']

    # %% MPC SIMULATION
    import copy

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

    # dual-kite MPC tracking weights: track both aircraft nodes, leave the (virtual) branch node q10 at its tiny default
    weights_x = trial.model.variables_dict['x'](1E-6)
    weights_x['q21'] = 1
    weights_x['dq21'] = 1
    weights_x['r21'] = 1
    weights_x['q31'] = 1
    weights_x['dq31'] = 1
    weights_x['r31'] = 1
    weights_x['e'] = 0

    additionalMPCoptions = {}
    additionalMPCoptions['Q'] = weights_x
    additionalMPCoptions['R'] = trial.model.variables_dict['u'](1)
    additionalMPCoptions['P'] = weights_x
    additionalMPCoptions['Z'] = trial.model.variables_dict['z'](1E-6)

    # make simulator
    from awebox import sim

    closed_loop_sim = sim.Simulation(trial, 'closed_loop', ts, options, additional_mpc_options=additionalMPCoptions)

    # Run the closed-loop simulation
    # only simulate a fraction of the full reel-out+reel-in duration: the plotting script only ever
    # shows a short section of the MPC trajectory near the start, and dual-kite MPC steps are expensive.
    T_sim = T_opt / 10  # seconds
    N_sim = int(T_sim / ts)  # closed-loop simulation steps

    startTime = 0
    closed_loop_sim.run(N_sim, startTime=startTime)

    # %% Export Trajectories for fancier Plotting

    # SAM solver stats
    solver_stats = {
        'N_var': trial.nlp.V.size,
        'N_eq': trial.nlp.g.size,
        't_wall': trial.optimization.t_wall,
        'N_iter': trial.optimization.iterations,
    }

    # all the stuff to be plotted from SAM
    plot_dict_SAM = trial.visualization.plot_dict_SAM
    export_dict_SAM = {}
    export_dict_SAM['regions'] = plot_dict_SAM['SAM_regions_ip']
    export_dict_SAM['time'] = plot_dict_SAM['time_grids']['ip']
    export_dict_SAM['time_X'] = plot_dict_SAM['time_grids']['ip_X']
    export_dict_SAM['x'] = plot_dict_SAM['x']
    export_dict_SAM['X'] = plot_dict_SAM['X']
    export_dict_SAM['d'] = trial.options['nlp']['SAM']['d']
    export_dict_SAM['N'] = trial.options['nlp']['SAM']['N']
    export_dict_SAM['regularizationValue'] = single_regularization_param
    export_dict_SAM['n_k'] = trial.options['nlp']['n_k']
    export_dict_SAM['solver_stats'] = solver_stats

    export_dict_REC = {}
    plot_dict_REC = trial.visualization.plot_dict
    export_dict_REC['time'] = plot_dict_REC['time_grids']['ip']
    export_dict_REC['x'] = plot_dict_REC['x']

    export_dict_MPC = {}
    plot_dict_CLSIM = closed_loop_sim.visualization.plot_dict
    export_dict_MPC['time'] = plot_dict_CLSIM['time_grids']['ip']
    export_dict_MPC['x'] = plot_dict_CLSIM['x']

    # store mpc stats
    export_dict_MPC['mpc_cpu_time'] = closed_loop_sim.mpc.log['cpu']
    export_dict_MPC['mpc_iter'] = closed_loop_sim.mpc.log['iter']

    export_dict = {'SAM': export_dict_SAM, 'REC': export_dict_REC, 'MPC': export_dict_MPC}

    # save the data
    from datetime import datetime
    datestr = datetime.now().strftime('%Y%m%d_%H%M')
    N_val = trial.options['nlp']['SAM']['N']
    d_val = trial.options['nlp']['SAM']['d']
    filename = f'{datestr}_AWE_SAM_DUALKITE_N{N_val}_d{d_val}'
    np.savez(f'_export/{filename}.npz', **export_dict)

    awelogger.logger.info(f'Exported data to _export/{filename}.npz')

    return export_dict
