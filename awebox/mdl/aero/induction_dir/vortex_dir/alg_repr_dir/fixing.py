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
constraints to create "intermediate condition" fixing constraints on the positions of the wake nodes,
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
'''
import copy
import pdb

import numpy as np
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger


################# define the actual constraint

def get_constraint(nlp_options, V, Outputs, Integral_outputs, model, time_grids):

    cstr_list = cstr_op.ConstraintList()

    cstr_list.append(get_specific_constraint('wx', nlp_options, V, Outputs, Integral_outputs, model, time_grids))
    cstr_list.append(get_specific_constraint('wg', nlp_options, V, Outputs, Integral_outputs, model, time_grids))

    far_wake_element_type = general_tools.get_option_from_possible_dicts(nlp_options, 'far_wake_element_type', 'vortex')
    if far_wake_element_type == 'semi_infinite_cylinder':
        cstr_list.append(get_specific_constraint('wx_center', nlp_options, V, Outputs, Integral_outputs, model, time_grids))
        cstr_list.append(get_specific_constraint('wh', nlp_options, V, Outputs, Integral_outputs, model, time_grids))

    return cstr_list

def get_specific_constraint(abbreviated_var_name, nlp_options, V, Outputs, Integral_outputs, model, time_grids):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    wake_nodes = general_tools.get_option_from_possible_dicts(nlp_options, 'wake_nodes', 'vortex')
    rings = general_tools.get_option_from_possible_dicts(nlp_options, 'rings', 'vortex')
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    if abbreviated_var_name == 'wx':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = wingtips
        wake_node_or_ring_list = range(wake_nodes)
    elif abbreviated_var_name == 'wg':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = [None]
        wake_node_or_ring_list = range(rings)
    elif abbreviated_var_name == 'wh':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = [None]
        wake_node_or_ring_list = [None]
    elif abbreviated_var_name == 'wx_center':
        kite_shed_or_parent_shed_list = set([model.architecture.parent_map[kite] for kite in model.architecture.kite_nodes])
        tip_list = [None]
        wake_node_or_ring_list = [None]
    else:
        message = 'get_specific_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
        awelogger.logger.error(message)
        raise Exception(message)

    cstr_list = cstr_op.ConstraintList()

    print_op.warn_about_temporary_functionality_removal(location='alg_repr.fixing.shooting_cstr_at_ndx=n_k?')

    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                for ndx in range(n_k):
                    local_cstr = get_specific_local_constraint(abbreviated_var_name, nlp_options, V, Outputs,
                                                               Integral_outputs, model, time_grids,
                                                               kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx)
                    cstr_list.append(local_cstr)

                    for ddx in range(d):
                        local_cstr = get_specific_local_constraint(abbreviated_var_name, nlp_options, V, Outputs,
                                                                   Integral_outputs, model,
                                                                   time_grids, kite_shed_or_parent_shed, tip,
                                                                   wake_node_or_ring, ndx, ddx)
                        cstr_list.append(local_cstr)

    return cstr_list


def get_specific_local_constraint(abbreviated_var_name, nlp_options, V, Outputs, Integral_outputs, model, time_grids, kite_shed_or_parent_shed, tip,
                                  wake_node_or_ring, ndx, ddx=None):

    var_name = vortex_tools.get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)
    cstr_name = 'fixing_' + var_name + '_' + str(ndx)

    if ddx is None:
        var_local_scaled = V['xl', ndx, var_name]
    else:
        cstr_name += ',' + str(ddx)
        var_local_scaled = V['coll_var', ndx, ddx, 'xl', var_name]

    if ddx is None:
        var_val_scaled = V['coll_var', ndx - 1, -1, 'xl', var_name]
        var_val_si = struct_op.var_scaled_to_si('xl', var_name, var_val_scaled, model.scaling)
    else:
        # look-up the actual value from the Outputs. Keep the computing here minimal.
        if abbreviated_var_name == 'wx':
            var_val_si = get_local_convected_position_value(nlp_options, V, Outputs, model, time_grids, kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wg':
            var_val_si = get_local_average_circulation_value(nlp_options, V, Integral_outputs, model, time_grids, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wh':
            var_val_si = get_local_cylinder_pitch_value(nlp_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wx_center':
            var_val_si = get_local_cylinder_center_value(nlp_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        else:
            message = 'get_specific_local_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
            awelogger.logger.error(message)
            raise Exception(message)

    var_local_si = struct_op.var_scaled_to_si('xl', var_name, var_local_scaled, model.scaling)

    resi_si = var_local_si - var_val_si
    resi_scaled = struct_op.var_si_to_scaled('xl', var_name, resi_si, model.scaling)

    local_cstr = cstr_op.Constraint(expr=resi_scaled,
                                    name=cstr_name,
                                    cstr_type='eq')

    return local_cstr


def get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx=None):

    if ddx is None:
        ndx = ndx - 1
        ddx = -1

    n_k = nlp_options['n_k']
    ddx_shed = ddx

    # # if wake_node = 0, then ndx_shed = ndx
    # # if wake_node = 1, then ndx_shed = (ndx - 1)
    # # .... if ndx_shed is 1, then ndx_shed -> 1
    # # ....  if ndx_shed is 0, then ndx_shed -> n_k
    # # ....  if ndx_shed is -1, then ndx_shed -> n_k - 1
    # # .... so, ndx_shed -> np.mod(ndx - wake_node, n_k)
    subtracted_ndx = ndx - wake_node
    ndx_shed = np.mod(subtracted_ndx, n_k)
    periods_passed = np.floor(subtracted_ndx / n_k)

    return ndx_shed, ddx_shed, periods_passed


########## wake node position

def get_local_convected_position_value(nlp_options, V, Outputs, model, time_grids, kite_shed, tip, wake_node, ndx, ddx):
    t_f_scaled = V['theta', 't_f']
    t_f_si = struct_op.var_scaled_to_si('theta', 't_f', t_f_scaled, model.scaling)
    tgrid_coll = time_grids['coll'](t_f_si)
    wx_convected = get_the_convected_position_from_the_current_indices_and_wake_node(nlp_options, Outputs, model, tgrid_coll, kite_shed, tip,
                                                                      wake_node, ndx, ddx)

    return wx_convected


def get_the_convected_position_from_the_current_indices_and_wake_node(nlp_options, Outputs, model, tgrid_coll, kite, tip, wake_node, ndx, ddx=None):

    ndx_shed, ddx_shed, periods_passed = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)

    wingtip_pos = get_the_wingtip_position_at_shedding_indices(Outputs, kite, tip, ndx_shed, ddx_shed)

    delta_t = get_the_convection_time_from_the_current_indices_and_wake_node(nlp_options, tgrid_coll, wake_node, ndx, ddx)
    u_local = model.wind.get_velocity(wingtip_pos[2])
    wx_convected = wingtip_pos + delta_t * u_local

    return wx_convected


def get_the_convection_time_from_the_current_indices_and_wake_node(nlp_options, tcoll, wake_node, ndx, ddx=None):

    if ddx == None:
        ndx = ndx - 1
        ddx = -1

    ndx_shed, ddx_shed, periods_passed = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options,
                                                                                                         wake_node, ndx,
                                                                                                         ddx)

    t_period = tcoll[-1, -1]
    shedding_time = t_period * periods_passed + tcoll[ndx_shed, ddx_shed]
    current_time = tcoll[ndx, ddx]
    delta_t = current_time - shedding_time

    return delta_t

def get_the_wingtip_position_at_shedding_indices(Outputs, kite, tip, ndx_shed, ddx_shed):
    wingtip_pos = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'wingtip_' + tip + str(kite)]
    return wingtip_pos


############## ring strength

def get_local_average_circulation_value(nlp_options, V, Integral_outputs, model, time_grids, kite_shed, ring, ndx, ddx):
    int_name = 'integrated_circulation' + str(kite_shed)
    local_scaling = model.scaling['xd'][int_name]

    n_k = nlp_options['n_k']

    t_f_scaled = V['theta', 't_f']
    t_f_si = struct_op.var_scaled_to_si('theta', 't_f', t_f_scaled, model.scaling)
    tgrid_coll = time_grids['coll'](t_f_si)

    optimization_period = tgrid_coll[-1, -1]
    total_integrated_circulation_scaled = Integral_outputs['coll_int_out', -1, -1, int_name]

    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, ring, ndx,
                                                                                            ddx)

    integrated_circulation_shed_scaled = Integral_outputs['coll_int_out', ndx_shed, ddx_shed, int_name]
    integrated_circulation_shed_si = integrated_circulation_shed_scaled * local_scaling
    time_shed = tgrid_coll[ndx_shed, ddx_shed]

    ddx_before_shed = ddx_shed
    if ndx_shed == 0:
        ndx_before_shed = ndx_shed - 1 + n_k
        time_before_shed = tgrid_coll[ndx_before_shed, ddx_before_shed] - optimization_period
        integrated_circulation_before_shed_scaled = Integral_outputs['coll_int_out',
            ndx_before_shed, ddx_before_shed, int_name] - total_integrated_circulation_scaled

    else:
        ndx_before_shed = ndx_shed - 1
        time_before_shed = tgrid_coll[ndx_before_shed, ddx_before_shed]
        integrated_circulation_before_shed_scaled = Integral_outputs[
            'coll_int_out', ndx_before_shed, ddx_before_shed, int_name]

    integrated_circulation_before_shed_si = integrated_circulation_before_shed_scaled * local_scaling

    delta_integrated_circulation_si = integrated_circulation_shed_si - integrated_circulation_before_shed_si
    delta_t = time_shed - time_before_shed

    average_circulation = delta_integrated_circulation_si / delta_t

    return average_circulation


################ cylinder center

def get_local_cylinder_center_value(nlp_options, Outputs, parent_shed, wake_node, ndx, ddx=None):
    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)
    wx_center = get_the_cylinder_center_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed)
    return wx_center

def get_the_cylinder_center_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed):
    wingtip_pos = Outputs['coll_outputs', ndx_shed, ddx_shed, 'performance', 'actuator_center' + str(parent_shed)]
    return wingtip_pos


################ cylinder pitch

def get_local_cylinder_pitch_value(nlp_options, Outputs, parent_shed, wake_node, ndx, ddx=None):
    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)
    wx_center = get_the_cylinder_pitch_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed)
    return wx_center

def get_the_cylinder_pitch_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed):
    pitch = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'far_wake_cylinder_pitch' + str(parent_shed)]
    return pitch

###############


########## test

def test_the_convection_time(epsilon=1.e-4):

    ndx = 5 # some number larger than the number of collocation nodes

    nlp_options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures()

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    scheme = nlp_options['collocation']['scheme']
    tau_root = cas.vertcat(cas.collocation_points(d, scheme))

    width = 1.73 # some number that's not likely to arise 'naturally'

    tcoll = []
    for ndx in range(n_k):
        for ddx in range(d):
            tcoll = cas.vertcat(tcoll, width * (ndx + tau_root[ddx]) )

    tcoll = tcoll.reshape((d, n_k)).T
    optimization_period = tcoll[-1, -1]

    for ddx in [None, 2]:

        if ddx is None:
            case_description_string = ' at a shooting node'
        else:
            case_description_string = ' at a collocation node'

        wake_nodes = {}
        found = {}
        expected = {}
        conditions = {}
        total_condition = 0

        # if wake_node = 0 -> delta_t == 0
        # if wake_node = n_k -> delta_t = t_final
        # if wake_node = 2 n_k -> delta_t = 2 * t_final

        wake_nodes[0] = 0 * n_k
        expected[0] = 0

        wake_nodes[1] = 1 * n_k
        expected[1] = 1 * optimization_period

        wake_nodes[2] = 2 * n_k
        expected[2] = 2 * optimization_period

        wake_nodes['partial'] = 1
        expected['partial'] = width

        for name, wake_node in wake_nodes.items():
            found[name] = get_the_convection_time_from_the_current_indices_and_wake_node(nlp_options, tcoll, wake_node, ndx, ddx)
            diff = found[name] - expected[name]

            local_condition = (diff**2. < epsilon**2.)
            conditions[name] = local_condition
            total_condition += local_condition

        criteria = (total_condition == len(wake_nodes.keys()))

        if not criteria:
            message = 'something went wrong when computing how long a given wake node has been convecting, ' + case_description_string
            awelogger.logger.error(message)
            raise Exception(message)

    return None

def test():
    test_the_convection_time()

# test()
