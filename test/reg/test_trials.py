#!/usr/bin/python3
"""Template for trial tests

@author: Thilo Bronnenmeyer, kiteswarms 2018

- edit: Rachel Leuthold, Jochem De Schutter ALU-FR 2020-21
"""

import collections
import copy
import logging
import pdb
import casadi.tools as cas

import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.bubbledancer_data as bubbledancer_data
import awebox.opts.kite_data.boeing747_data as boeing747_data
from ampyx_ap2_settings import set_ampyx_ap2_settings
import awebox.opts.options as options
import awebox.trial as awe_trial
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import numpy as np

from awebox.logger.logger import Logger as awelogger
import matplotlib.pyplot as plt
awelogger.logger.setLevel(10)



logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)


def test_single_kite(final_homotopy_step='final'):
    trial_name = 'single_kite_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_single_kite_basic_health(final_homotopy_step='final'):
    trial_name = 'single_kite_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_single_kite_6_dof(final_homotopy_step='final'):
    trial_name = 'single_kite_6_dof_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_single_kite_6_dof_basic_health(final_homotopy_step='final'):
    trial_name = 'single_kite_6_dof_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_zoh():
    trial_name = 'zoh_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_drag_mode():
    trial_name = 'drag_mode_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_save_trial():
    trial_name = 'save_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_dual_kite(final_homotopy_step='final'):
    trial_name = 'dual_kite_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None

def test_dual_kite_basic_health(final_homotopy_step='final'):
    trial_name = 'dual_kite_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_dual_kite_6_dof(final_homotopy_step='final'):
    trial_name = 'dual_kite_6_dof_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_dual_kite_6_dof_basic_health(final_homotopy_step='final'):
    trial_name = 'dual_kite_6_dof_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_small_dual_kite(final_homotopy_step='final'):
    trial_name = 'small_dual_kite_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None

#
# def test_large_dual_kite(final_homotopy_step='final'):
#     trial_name = 'large_dual_kite_trial'
#     run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
#     return None


def test_actuator_qaxi(final_homotopy_step='final'):
    trial_name = 'actuator_qaxi_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None

def test_actuator_qaxi_basic_health(final_homotopy_step='final'):
    trial_name = 'actuator_qaxi_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_actuator_uaxi():
    trial_name = 'actuator_uaxi_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_actuator_qasym():
    trial_name = 'actuator_qasym_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_actuator_uasym():
    trial_name = 'actuator_uasym_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_actuator_comparison():
    trial_name = 'actuator_comparison_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_dual_kite_tracking():
    trial_name = 'dual_kite_tracking_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_dual_kite_tracking_winch():
    trial_name = 'dual_kite_tracking_winch_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_vortex_force_zero():
    trial_name = 'vortex_force_zero_trial'
    run_a_solve_and_check_test(trial_name)
    return None


def test_vortex_force_zero_basic_health(final_homotopy_step='final'):
    trial_name = 'vortex_force_zero_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


def test_vortex():
    options_dict = generate_options_dict()
    trial_name = 'vortex_trial'
    solve_trial(options_dict[trial_name], trial_name)
    return None



def test_vortex_basic_health(final_homotopy_step='final'):
    options_dict = generate_options_dict()
    trial_name = 'vortex_basic_health_trial'
    solve_trial(options_dict[trial_name], trial_name, final_homotopy_step=final_homotopy_step)
    return None


def make_basic_health_variant(base_options):
    basic_health_options = copy.deepcopy(base_options)

    basic_health_options['user_options.trajectory.lift_mode.windings'] = 1
    basic_health_options['nlp.n_k'] = 15  # try to decrease this.
    basic_health_options['nlp.collocation.d'] = 3
    basic_health_options['nlp.collocation.u_param'] = 'zoh'
    basic_health_options['solver.hippo_strategy'] = False

    basic_health_options['solver.health_check.when'] = 'always'
    basic_health_options['solver.homotopy_method.advance_despite_max_iter'] = False
    basic_health_options['solver.homotopy_method.advance_despite_ill_health'] = False
    basic_health_options['solver.initialization.check_reference'] = True
    basic_health_options['solver.initialization.check_feasibility.raise_exception'] = True
    basic_health_options['solver.max_iter'] = 300
    basic_health_options['solver.ipopt.autoscale'] = False
    basic_health_options['solver.health_check.raise_exception'] = True
    basic_health_options['solver.health_check.spy_matrices'] = False
    basic_health_options['nlp.collocation.name_constraints'] = True
    basic_health_options['solver.health_check.help_with_debugging'] = True
    basic_health_options['quality.when'] = 'never'
    basic_health_options['visualization.cosmetics.variables.si_or_scaled'] = 'scaled'
    basic_health_options['solver.health_check.save_health_indicators'] = True

    return basic_health_options


def generate_options_dict():
    """
    Set options for the trials that should be tested and store them in dictionary
    :return: dictionary with trial options
    """

    # set options
    single_kite_options = {}
    single_kite_options['user_options.system_model.architecture'] = {1: 0}
    single_kite_options = set_ampyx_ap2_settings(single_kite_options)
    single_kite_options['solver.linear_solver'] = 'ma57'
    single_kite_options['visualization.cosmetics.plot_bounds'] = True
    single_kite_options['visualization.cosmetics.trajectory.kite_bodies'] = True
    single_kite_options['visualization.cosmetics.trajectory.kite_aero_dcm'] = True
    single_kite_options['solver.weights.q'] = 1e0
    single_kite_options['solver.weights.dq'] = 1.e0
    single_kite_options['solver.weights.l_t'] = 1e-5
    single_kite_options['solver.weights.dl_t'] = 1e-3
    single_kite_options['solver.weights.ddl_t'] = 1e-1
    single_kite_options['solver.weights.dddl_t'] = 1e-1
    single_kite_options['solver.weights.coeff'] = 1e-6
    single_kite_options['solver.weights.r'] = 1e0
    single_kite_options['solver.cost.fictitious.0'] = 1.e0
    single_kite_options['solver.cost.fictitious.1'] = 1.e3
    single_kite_options['solver.cost.u_regularisation.0'] = 1e-3
    single_kite_options['solver.cost.theta_regularisation.0'] = 1.e1
    single_kite_options['solver.initialization.groundspeed'] = 20.
    single_kite_options['solver.cost.tracking.0'] = 1e0
    single_kite_options['solver.cost.psi.1'] = 1.e2
    single_kite_options['solver.cost_factor.power'] = 1e5  #1e4
    # single_kite_options['solver.cost.theta_regularisation.0'] = 1.e-1
    # single_kite_options['solver.cost.beta.0'] = 1.e0

    single_kite_basic_health_options = make_basic_health_variant(single_kite_options)

    single_kite_6_dof_options = copy.deepcopy(single_kite_options)
    single_kite_6_dof_options['user_options.system_model.kite_dof'] = 6
    single_kite_6_dof_options['solver.weights.r'] = 1e0

    single_kite_6_dof_basic_health_options = make_basic_health_variant(single_kite_6_dof_options)

    zoh_options = copy.deepcopy(single_kite_options)
    zoh_options['nlp.collocation.u_param'] = 'zoh'

    drag_mode_options = copy.deepcopy(single_kite_options)
    drag_mode_options['user_options.trajectory.system_type'] = 'drag_mode'
    drag_mode_options['quality.test_param.power_balance_thresh'] = 2.
    drag_mode_options['model.system_bounds.theta.t_f'] = [20., 70.]  # [s]

    save_trial_options = copy.deepcopy(single_kite_options)
    save_trial_options['solver.save_trial'] = True

    dual_kite_options = copy.deepcopy(single_kite_options)
    dual_kite_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    dual_kite_options['solver.initialization.theta.l_s'] = 75.
    dual_kite_options['solver.initialization.check_reference'] = True
    dual_kite_options['nlp.collocation.u_param'] = 'zoh'
    dual_kite_options['model.system_bounds.theta.t_f'] = [5., 20.]

    dual_kite_basic_health_options = make_basic_health_variant(dual_kite_options)

    dual_kite_6_dof_options = copy.deepcopy(dual_kite_options)
    dual_kite_6_dof_options['user_options.system_model.kite_dof'] = 6

    dual_kite_6_dof_basic_health_options = make_basic_health_variant(dual_kite_6_dof_options)

    small_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    small_dual_kite_options['user_options.kite_standard'] = bubbledancer_data.data_dict()
    small_dual_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    small_dual_kite_options['solver.cost_factor.power'] = 1e7
    # small_dual_kite_options['solver.cost.theta_regularisation.0'] = 1.e-1
    # small_dual_kite_options['solver.cost.beta.0'] = 1.e0

    large_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    large_dual_kite_options['user_options.kite_standard'] = boeing747_data.data_dict()
    large_dual_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    large_dual_kite_options['solver.initialization.theta.l_s'] = 60. * 10.
    large_dual_kite_options['solver.initialization.l_t'] = 2.e3
    large_dual_kite_options['model.system_bounds.theta.t_f'] = [1.e-3, 500.]
    large_dual_kite_options['model.model_bounds.tether_force.include'] = False
    large_dual_kite_options['model.model_bounds.tether_stress.include'] = True
    large_dual_kite_options['solver.cost.tracking.0'] = 1e-1

    actuator_qaxi_options = copy.deepcopy(dual_kite_6_dof_options)
    actuator_qaxi_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1} #, 4: 1}
    actuator_qaxi_options['user_options.kite_standard'] = ampyx_data.data_dict()
    actuator_qaxi_options['user_options.induction_model'] = 'actuator'
    actuator_qaxi_options['model.aero.actuator.steadyness'] = 'quasi-steady'
    actuator_qaxi_options['model.aero.actuator.symmetry'] = 'axisymmetric'
    actuator_qaxi_options['visualization.cosmetics.trajectory.actuator'] = True
    actuator_qaxi_options['visualization.cosmetics.trajectory.kite_bodies'] = True
    actuator_qaxi_options['model.system_bounds.theta.a'] = [-0., 0.5]
    # actuator_qaxi_options['model.aero.actuator.normal_vector_model'] = 'least_squares'

    actuator_qaxi_options['solver.cost.gamma.1'] = 1.e2  # 1e3 fictitious problem by 1e-1
    actuator_qaxi_options['solver.cost.psi.1'] = 1.e2  # 1e4 power problem scaled by 1e-2
    # actuator_qaxi_options['solver.cost.theta_regularisation.0'] = 1.e0
    actuator_qaxi_options['solver.cost.iota.1'] = 1.e2  # 1e3 induction problem scaled by 1e-1
    # actuator_qaxi_options['solver.max_iter'] = 90
    # actuator_qaxi_options['solver.max_iter_hippo'] = 90
    actuator_qaxi_options['solver.cost_factor.power'] = 1e5  # 1e4 -> high reg in final step

    # actuator_qaxi_options['solver.cost.theta_regularisation.0'] = 1.e-1
    actuator_qaxi_options['user_options.trajectory.lift_mode.windings'] = 3
    # actuator_qaxi_options['solver.cost.beta.0'] = 1.e0
    # actuator_qaxi_options['solver.weights.r'] = 1e-1
    # actuator_qaxi_options['solver.weights.q'] = 1e2
    # actuator_qaxi_options['solver.weights.dq'] = 1e2
    # actuator_qaxi_options['solver.cost.u_regularisation.0'] = 1e-5

    actuator_qaxi_basic_health_options = make_basic_health_variant(actuator_qaxi_options)

    actuator_uaxi_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uaxi_options['model.aero.actuator.steadyness'] = 'unsteady'
    actuator_uaxi_options['model.model_bounds.tether_stress.scaling'] = 10.

    actuator_qasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_qasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_qasym_options['solver.cost.psi.1'] = 1.e1

    actuator_uasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_uasym_options['solver.cost.psi.1'] = 1.e1

    actuator_comparison_options = copy.deepcopy(actuator_qaxi_options)
    actuator_comparison_options['model.aero.actuator.steadyness_comparison'] = ['q', 'u']
    actuator_comparison_options['user_options.system_model.kite_dof'] = 6

    vortex_options = copy.deepcopy(single_kite_6_dof_options)
    vortex_options['user_options.trajectory.lift_mode.windings'] = 1
    vortex_options['user_options.induction_model'] = 'vortex'
    vortex_options['model.aero.vortex.far_wake_element_type'] = 'not_in_use'
    vortex_options['model.aero.vortex.wake_nodes'] = 1 ###### rela!!!!!!
    vortex_options['model.aero.vortex.representation'] = 'alg'
    vortex_options['quality.test_param.vortex_truncation_error_thresh'] = 1e20
    vortex_options['nlp.collocation.u_param'] = 'zoh'
    vortex_options['model.aero.vortex.biot_savart_residual_assembly'] = 'lifted'

    # vortex_options['solver.weights.vortex'] = 1e-3
    # vortex_options['solver.cost_factor.power'] = 1e6  # 1e4
    vortex_options['solver.weights.vortex'] = 1e-3
    vortex_options['solver.cost.iota.1'] = 1.e3
    vortex_options['model.aero.induction.final_resi_scaling'] = 1.e-1

    vortex_options['model.aero.vortex.rate_of_change_scaling_factor'] = 1.e-1

    vortex_options['model.scaling.other.position'] = 'altitude'
    vortex_options['model.aero.vortex.position_scaling_method'] = 'convection'  #'convection'
    vortex_options['model.aero.vortex.bound_induction_offset'] = 'r_core' # 'r_core'
    vortex_options['model.aero.vortex.bound_induction_scaling_factor'] = 1.e-1  # 1.e1  #1. #1.e3 #4
    vortex_options['model.aero.vortex.bound_induction_denominator_factor'] = 1.e5  # 1.e1  #1. #1.e3 #4
    vortex_options['model.aero.vortex.core_to_chord_ratio'] = 2. * 5.5 * 1./10. # 2. * b_ref * 1/AR
    # https: // openfast.readthedocs.io / en / main / source / user / aerodyn - olaf / OLAFTheory.html  # regularization


    # bisf = 1e0 and bidf = 1e0
    # initial
    # INFO:	kkt: condition............: 1.195E+53
    # INFO:	kkt: min_red_hessian_eig..: -1.02E+20
    #
    # bisf = 1e1 and bidf = 1e4
    # initial
    # INFO: kkt: condition............: 1.867E+35
    # INFO: kkt: min_red_hessian_eig..: -5.869E+10
    # INFO:	shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36 -> {'[x,0,q10,0]': '5.70e+09', '[x,0,q10,1]': '-9.64e+07', '[x,0,q10,2]': '-4.43e+09', '[z,0,wx_1_ext_0,0]': '-2.90e+09', '[z,0,wx_1_ext_0,1]': '1.45e+08', '[z,0,wx_1_ext_0,2]': '2.28e+09', '[z,0,wx_1_int_0,0]': '-2.81e+09', '[z,0,wx_1_int_0,1]': '-4.87e+07', '[z,0,wx_1_int_0,2]': '2.15e+09', '[z,0,wu_bound_finite_filament_0_1_den,0]': '3.13e+06'}

    # bisf = 1e1 and bidf = 1e8
    # initial
    # INFO:	kkt: condition............: 3.544E+16
    # INFO:	kkt: min_red_hessian_eig..: -0.4413
    # INFO:	... largest (absolute value sense) KKT matrix entry (5.654E+10) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_u_selected_30
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_num', 0)

    # bisf = 1e1 and bidf = 1e7
    # initial
    # INFO: kkt: condition............: 8.668E+14
    # INFO: kkt: min_red_hessian_eig..: -0.4413
    # INFO:	... largest (absolute value sense) KKT matrix entry (5.654E+09) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_u_selected_30
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_num', 0)

    # bisf = 1e1 and bidf = 1e6
    # initial
    # INFO:	kkt: condition............: 8.699E+13
    # INFO:	kkt: min_red_hessian_eig..: -0.1471
    # INFO:	... largest (absolute value sense) KKT matrix entry (5.654E+08) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_u_selected_30
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_num', 0)

    # bisf = 1e1 and bidf = 1e5
    # initial
    # INFO:	kkt: condition............: 8.699E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.007949
    # INFO:	... largest (absolute value sense) KKT matrix entry (5.654E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_u_selected_30
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_num', 0)

    # bisf = 1e0 and bidf = 1e5
    # initial
    # INFO: kkt: condition............: 3.4E+12
    # INFO: kkt: min_red_hessian_eig..: -0.0002168
    # INFO:	... largest (absolute value sense) KKT matrix entry (3.125E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e0 and bidf = 2e5
    # initial
    # INFO:	kkt: condition............: 6.801E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.007438
    # INFO:	... largest (absolute value sense) KKT matrix entry (6.25E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e0 and bidf = 8e4
    # initial
    # INFO:	kkt: condition............: 2.72E+12
    # INFO:	kkt: min_red_hessian_eig..: -9.334E-05
    # INFO:	... largest (absolute value sense) KKT matrix entry (2.5E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e0 and bidf = 5e4
    # initial
    # INFO:	kkt: condition............: 1.7E+12
    # INFO:	kkt: min_red_hessian_eig..: -7.179E-05
    # INFO:	... largest (absolute value sense) KKT matrix entry (1.563E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e-1 and bidf = 5e4. position = radius
    # initial
    # INFO:	kkt: condition............: 1.7E+12
    # INFO:	kkt: min_red_hessian_eig..: -4.926E-05
    # INFO:	... largest (absolute value sense) KKT matrix entry (1.563E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e-1 and bidf = 5e4. position = altitude
    # initial
    # INFO:	kkt: condition............: 1.705E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.0002227
    # INFO:	... largest (absolute value sense) KKT matrix entry (1.563E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e-1 and bidf = 5e4. position = altitude, wx = b_ref
    # initial
    # INFO:	kkt: condition............: 1.705E+12
    # INFO:	kkt: min_red_hessian_eig..: -5.67E-05
    # INFO:	... largest (absolute value sense) KKT matrix entry (1.563E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e0 and bidf = 1e5, position = altitude, wx = b_ref
    # initial
    # INFO: kkt: condition............: 3.41E+12
    # INFO: kkt: min_red_hessian_eig..: -0.0001604
    # INFO:	... largest (absolute value sense) KKT matrix entry (3.125E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e-1 and bidf = 1e5, position = altitude, wx = radius
    # initial
    # INFO:	kkt: condition............: 2.237E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.0002217
    # INFO:	... largest (absolute value sense) KKT matrix entry (2.05E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e-1 and bidf = 1e5, position = altitude, wx = convection, offset = micro
    # initial
    # INFO:	kkt: condition............: 2.237E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.0003004
    # INFO:	... largest (absolute value sense) KKT matrix entry (2.05E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)

    # bisf = 1e-1 and bidf = 1e5, position = altitude, wx = convection, offset = r_core, cvw = 1e-3
    # initial
    # INFO:	kkt: condition............: 3.411E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.001273
    # INFO:	... largest (absolute value sense) KKT matrix entry (3.125E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)







    # bisf = 1e-1 and bidf = 1e5, position = altitude, wx = convection, offset = c_ref
    # initial
    # INFO:	kkt: condition............: 3.199E+12
    # INFO:	kkt: min_red_hessian_eig..: -0.001222
    # INFO:	... largest (absolute value sense) KKT matrix entry (2.932E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)






    # bisf = 1e0 and bidf = 1e5, position = altitude, wx = c_ref
    # initial
    # INFO:	kkt: condition............: 1.186E+14
    # INFO:	kkt: min_red_hessian_eig..: -5.834E-05
    # INFO:	... largest (absolute value sense) KKT matrix entry (2.05E+07) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)


    # bisf = 1e0 and bidf = 1e4
    # initial (mie)
    # INFO:	kkt: condition............: 9.764E+25
    # INFO:	kkt: min_red_hessian_eig..: -17.16
    # INFO:	... largest (absolute value sense) KKT matrix entry (8.469E+06) is associated with:
    # INFO:	    column:      multiplier collocation_4_0_biot_savart_bound_finite_filament_0_1_den_36
    # INFO:	       row:        variable ('coll_var', 4, 0, 'x', 'q10', 2)





    # bisf = 1e0 and bidf = 1e6
    # initial
    # INFO: kkt: condition............: 3.4E+13
    # INFO: kkt: min_red_hessian_eig..: -0.1471
    # INFO:	... largest (absolute value sense) KKT matrix entry (3.125E+08) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_den_selected_36
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_den', 0)


    # bisf = 1e2 and bidf = 1e5
    # initial
    # INFO:	kkt: condition............: 8.699E+13
    # INFO:	kkt: min_red_hessian_eig..: -0.1471
    # INFO:	... largest (absolute value sense) KKT matrix entry (5.654E+08) is associated with:
    # INFO:	    column:      multiplier shooting_0_biot_savart_bound_finite_filament_0_1_u_selected_30
    # INFO:	       row:        variable ('z', 0, 'wu_bound_finite_filament_0_1_num', 0)


    vortex_basic_health_options = make_basic_health_variant(vortex_options)

    vortex_force_zero_options = copy.deepcopy(vortex_options)
    vortex_force_zero_options['model.aero.induction.force_zero'] = True

    vortex_force_zero_basic_health_options = make_basic_health_variant(vortex_force_zero_options)

    dual_kite_tracking_options = copy.deepcopy(dual_kite_6_dof_options)
    dual_kite_tracking_options['user_options.trajectory.type'] = 'tracking'
    dual_kite_tracking_options['user_options.trajectory.lift_mode.windings'] = 1
    # dual_kite_tracking_options['user_options.trajectory.tracking.fix_tether_length'] = True
    dual_kite_tracking_options['nlp.n_k'] = 20

    dual_kite_tracking_winch_options = copy.deepcopy(dual_kite_tracking_options)
    dual_kite_tracking_winch_options['user_options.trajectory.tracking.fix_tether_length'] = False

    # nominal landing
    nominal_landing_options = copy.deepcopy(dual_kite_options)
    nominal_landing_options['user_options.trajectory.type'] = 'nominal_landing'
    nominal_landing_options['user_options.trajectory.transition.initial_trajectory'] = 'dual_kite_trial.dict'
    nominal_landing_options['solver.initialization.initialization_type'] = 'modular'

    # compromised landing
    compromised_landing_options = copy.deepcopy(nominal_landing_options)
    compromised_landing_options['user_options.trajectory.type'] = 'compromised_landing'
    compromised_landing_options['model.model_bounds.dcoeff_compromised_factor'] = 0.0
    compromised_landing_options['user_options.trajectory.compromised_landing.emergency_scenario'] = ('broken_roll', 2)
    compromised_landing_options['user_options.trajectory.compromised_landing.xi_0_initial'] = 0.8

    # define options list
    options_dict = collections.OrderedDict()
    options_dict['single_kite_trial'] = single_kite_options
    options_dict['single_kite_basic_health_trial'] = single_kite_basic_health_options
    options_dict['single_kite_6_dof_trial'] = single_kite_6_dof_options
    options_dict['single_kite_6_dof_basic_health_trial'] = single_kite_6_dof_basic_health_options
    options_dict['zoh_trial'] = zoh_options
    options_dict['drag_mode_trial'] = drag_mode_options
    options_dict['save_trial'] = save_trial_options
    options_dict['dual_kite_trial'] = dual_kite_options
    options_dict['dual_kite_basic_health_trial'] = dual_kite_basic_health_options
    options_dict['small_dual_kite_trial'] = small_dual_kite_options
    options_dict['large_dual_kite_trial'] = large_dual_kite_options
    options_dict['dual_kite_6_dof_trial'] = dual_kite_6_dof_options
    options_dict['dual_kite_6_dof_basic_health_trial'] = dual_kite_6_dof_basic_health_options
    options_dict['actuator_qaxi_trial'] = actuator_qaxi_options
    options_dict['actuator_qaxi_basic_health_trial'] = actuator_qaxi_basic_health_options
    options_dict['actuator_uaxi_trial'] = actuator_uaxi_options
    options_dict['actuator_qasym_trial'] = actuator_qasym_options
    options_dict['actuator_uasym_trial'] = actuator_uasym_options
    options_dict['actuator_comparison_trial'] = actuator_comparison_options
    options_dict['vortex_force_zero_trial'] = vortex_force_zero_options
    options_dict['vortex_force_zero_basic_health_trial'] = vortex_force_zero_basic_health_options
    options_dict['vortex_trial'] = vortex_options
    options_dict['vortex_basic_health_trial'] = vortex_basic_health_options
    options_dict['dual_kite_tracking_trial'] = dual_kite_tracking_options
    options_dict['dual_kite_tracking_winch_trial'] = dual_kite_tracking_winch_options
    # options_dict['nominal_landing_trial'] = nominal_landing_options
    # options_dict['compromised_landing_trial'] = compromised_landing_options

    return options_dict


def run_a_solve_and_check_test(trial_name, final_homotopy_step='final'):
    """
    Solve one individual trial and run tests on it
    :param trial_name: name of the trial
    :return: None
    """

    options_dict = generate_options_dict()
    trial_options = options_dict[trial_name]

    # compute trajectory solution
    trial = solve_trial(trial_options, trial_name, final_homotopy_step=final_homotopy_step)

    # evaluate results
    if hasattr(trial, 'quality') and hasattr(trial.quality, 'results'):
        evaluate_results(trial.quality.results, trial_name)

    if not trial.optimization.solve_succeeded:
        message = 'optimization of trial ' + trial_name + ' failed'
        raise Exception(message)

    return None


def evaluate_results(results, trial_name):
    for test_name in list(results.keys()):
        assert results[test_name], 'Test failed for ' + trial_name + ', Test regarding ' + test_name + ' failed.'

    return None


def solve_trial(trial_options, trial_name, final_homotopy_step='final'):
    """
    Set up and solve trial
    :return: solved trial
    """

    trial = awe_trial.Trial(trial_options, trial_name)
    trial.build()
    trial.optimize(final_homotopy_step=final_homotopy_step)

    trial.print_cost_information()
    trial.plot(['wake_lifted_variables', 'actuator_isometric', 'wake_isometric', 'level_3', 'animation_snapshot', 'induction_factor'])
    plt.show()

    pdb.set_trace()


    return trial

# #
# test_single_kite_basic_health()
# test_single_kite()
# test_single_kite_6_dof_basic_health()
# test_single_kite_6_dof()
# test_zoh()
# test_drag_mode()
# test_save_trial()
# test_dual_kite_basic_health()
# test_dual_kite()
# test_dual_kite_6_dof_basic_health()
# test_dual_kite_6_dof()
# test_small_dual_kite()
# test_large_dual_kite()
# test_actuator_qaxi()
# test_actuator_qaxi_basic_health() #final_homotopy_step='induction')
# test_actuator_qasym()
# test_actuator_uaxi()
# test_actuator_uasym()
# test_actuator_comparison()
# test_vortex_force_zero_basic_health() #final_homotopy_step='initial_guess')
# test_vortex_force_zero()
test_vortex_basic_health() #final_homotopy_step='initial_guess')
# test_vortex()
# test_dual_kite_tracking()
# test_dual_kite_tracking_winch()
