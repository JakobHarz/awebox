#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
'''
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-21
'''

import casadi.tools as cas

import awebox.mdl.aero.induction_dir.vortex_dir.convection as convection
import awebox.mdl.aero.induction_dir.vortex_dir.flow as flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as vortex_element
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as vortex_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as vortex_finite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as vortex_semi_infinite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_cylinder as vortex_semi_infinite_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_cylinder as vortex_semi_infinite_tangential_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_cylinder as vortex_semi_infinite_longitudinal_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as vortex_wake

import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.algebraic_representation as algebraic_representation
import awebox.mdl.aero.induction_dir.vortex_dir.state_repr_dir.state_representation as state_representation

import awebox.mdl.aero.induction_dir.vortex_dir.far_wake as vortex_far_wake
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.ocp.ocp_constraint as ocp_constraint
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import numpy as np


def build(options, architecture, wind, variables_si, parameters):

    tools.check_positive_vortex_wake_nodes(options)

    vortex_representation = tools.get_option_from_possible_dicts(options, 'representation')
    if vortex_representation == 'alg':
        return algebraic_representation.build(options, architecture, wind, variables_si, parameters)
    elif vortex_representation == 'state':
        return state_representation.build(options, architecture, wind, variables_si, parameters)
    else:
        log_and_raise_unknown_representation_error(vortex_representation)

    return None

def get_model_constraints(options, wake_dict, wind, variables_si, architecture):

    if tools.vortices_are_modelled(options):

        vortex_representation = tools.get_option_from_possible_dicts(options, 'representation')
        if vortex_representation == 'alg':
            return algebraic_representation.get_model_constraints(wake_dict, wind, variables_si, architecture)
        elif vortex_representation == 'state':
            return state_representation.get_model_constraints(wake_dict, wind, variables_si, architecture)
        else:
            log_and_raise_unknown_representation_error(vortex_representation)

    return None

def get_ocp_constraints(nlp_options, V, Outputs, model, time_grids):

    ocp_cstr_list = ocp_constraint.OcpConstraintList()

    if tools.vortices_are_modelled(nlp_options):
        vortex_representation = tools.get_option_from_possible_dicts(nlp_options, 'representation')
        if vortex_representation == 'alg':
            return algebraic_representation.get_ocp_constraints(nlp_options, V, Outputs, model, time_grids)
        elif vortex_representation == 'state':
            return state_representation.get_ocp_constraints(nlp_options, V, Outputs, model, time_grids)
        else:
            log_and_raise_unknown_representation_error(vortex_representation)

    return ocp_cstr_list


def log_and_raise_unknown_representation_error(vortex_representation):
    message = 'vortex representation (' + vortex_representation + ') is not recognized'
    awelogger.logger.error(message)
    raise Exception(message)
    return None

def get_vortex_cstr(options, wind, variables_si, parameters, objects, architecture):

    vortex_representation = options['aero']['vortex']['representation']
    cstr_list = cstr_op.ConstraintList()

    if vortex_representation == 'state':
        state_conv_cstr = convection.get_state_repr_convection_cstr(options, wind, variables_si, architecture)
        cstr_list.append(state_conv_cstr)

    superposition_cstr = flow.get_superposition_cstr(options, wind, variables_si, objects, architecture)
    cstr_list.append(superposition_cstr)

    vortex_far_wake_model = options['aero']['vortex']['far_wake_model']
    if ('cylinder' in vortex_far_wake_model):
        radius_cstr = vortex_far_wake.get_cylinder_radius_cstr(options, wind, variables_si, parameters, architecture)
        cstr_list.append(radius_cstr)

    return cstr_list


def test():

    vect_op.test_altitude()

    vortex_element.test()
    vortex_element_list.test()

    vortex_finite_filament.test()
    vortex_semi_infinite_filament.test()
    vortex_semi_infinite_cylinder.test()
    vortex_semi_infinite_tangential_cylinder.test()
    vortex_semi_infinite_longitudinal_cylinder.test()

    vortex_wake.test()

    # freestream_filament_far_wake_test_list = vortex_filament_list.test(far_wake_model = 'freestream_filament')
    # flow.test(freestream_filament_far_wake_test_list)
    # pathwise_filament_far_wake_test_list = vortex_filament_list.test(far_wake_model = 'pathwise_filament')
    # flow.test(pathwise_filament_far_wake_test_list)

    return None

def collect_vortex_outputs(model_options, atmos, wind, variables_si, outputs, vortex_objects, parameters, architecture):

    # break early and loud if there are problems
    test()

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    kite_nodes = architecture.kite_nodes
    for kite_obs in kite_nodes:

        parent_obs = architecture.parent_map[kite_obs]

        u_ind = flow.get_induced_velocity_at_kite(variables_si, vortex_objects, kite_obs)

        n_hat = unit_normal.get_n_hat(model_options, parent_obs, variables_si, parameters, architecture)
        local_a = flow.get_induction_factor_at_kite(model_options, wind, variables_si, vortex_objects, architecture, kite_obs, n_hat=n_hat)

        far_wake_u_ind = flow.get_induced_velocity_at_kite(variables_si, vortex_objects, kite_obs, selection='far_wake')
        far_wake_u_ind_norm = vect_op.norm(far_wake_u_ind)
        far_wake_u_ind_norm_over_ref = far_wake_u_ind_norm / wind.get_speed_ref()

        est_truncation_error = (far_wake_u_ind_norm) / vect_op.norm(u_ind)

        outputs['vortex']['u_ind' + str(kite_obs)] = u_ind
        outputs['vortex']['u_ind_norm' + str(kite_obs)] = vect_op.norm(u_ind)
        outputs['vortex']['local_a' + str(kite_obs)] = local_a

        outputs['vortex']['far_wake_u_ind' + str(kite_obs)] = far_wake_u_ind
        outputs['vortex']['far_wake_u_ind_norm_over_ref' + str(kite_obs)] = far_wake_u_ind_norm_over_ref

        outputs['vortex']['est_truncation_error' + str(kite_obs)] = est_truncation_error

    return outputs

def compute_global_performance(power_and_performance, plot_dict):

    kite_nodes = plot_dict['architecture'].kite_nodes

    max_est_trunc_list = []
    max_est_discr_list = []
    far_wake_u_ind_norm_over_ref_list = []

    all_local_a = None

    for kite in kite_nodes:

        trunc_name = 'est_truncation_error' + str(kite)
        local_max_est_trunc = np.max(np.array(plot_dict['outputs']['vortex'][trunc_name][0]))
        max_est_trunc_list += [local_max_est_trunc]

        kite_local_a = np.ndarray.flatten(np.array(plot_dict['outputs']['vortex']['local_a' + str(kite)][0]))
        if all_local_a is None:
            all_local_a = kite_local_a
        else:
            all_local_a = np.vstack([all_local_a, kite_local_a])

        max_kite_local_a = np.max(kite_local_a)
        min_kite_local_a = np.min(kite_local_a)
        local_max_est_discr = (max_kite_local_a - min_kite_local_a) / max_kite_local_a
        max_est_discr_list += [local_max_est_discr]

        local_far_wake_u_ind_norm_over_ref = np.max(np.array(plot_dict['outputs']['vortex']['far_wake_u_ind_norm_over_ref' + str(kite)]))
        far_wake_u_ind_norm_over_ref_list += [local_far_wake_u_ind_norm_over_ref]

    average_local_a = np.average(all_local_a)
    power_and_performance['vortex_average_local_a'] = average_local_a

    stdev_local_a = np.std(all_local_a)
    power_and_performance['vortex_stdev_local_a'] = stdev_local_a

    max_far_wake_u_ind_norm_over_ref = np.max(np.array(far_wake_u_ind_norm_over_ref_list))
    power_and_performance['vortex_max_far_wake_u_ind_norm_over_ref'] = max_far_wake_u_ind_norm_over_ref

    max_est_trunc = np.max(np.array(max_est_trunc_list))
    power_and_performance['vortex_max_est_truncation_error'] = max_est_trunc

    max_est_discr = np.max(np.array(max_est_discr_list))
    power_and_performance['vortex_max_est_discretization_error'] = max_est_discr

    return power_and_performance

# test()