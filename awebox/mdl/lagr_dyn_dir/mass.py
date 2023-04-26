#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
node mass computation
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''
import pdb

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger
import awebox.mdl.lagr_dyn_dir.tools as tools
import awebox.mdl.aero.tether_dir.tether_aero as tether_aero

# ======================
# MASS OUTPUTS CREATION
# ======================

def generate_mass_outputs(options, variables_si, outputs, parameters, architecture, scaling):

    if 'masses' not in list(outputs.keys()):
        outputs['masses'] = {}

    for node in range(1, architecture.number_of_nodes):
        seg_props = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, upper_node=node)
        outputs['masses']['m_tether{}'.format(node)] = seg_props['seg_mass']

    outputs['masses']['ground_station'] = get_ground_station_mass(options, architecture, scaling, variables_si, parameters)

    return outputs

def get_ground_station_mass(options, architecture, scaling, variables_si, parameters):

    ground_station_mass = parameters['theta0', 'ground_station', 'm_gen']
    if options['tether']['use_wound_tether']:
        main_props = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, upper_node=1)
        wound_length = variables_si['theta']['l_t_full'] - main_props['seg_length']
        wound_mass = main_props['cross_section_area'] * \
            parameters['theta0', 'tether', 'rho'] * wound_length
        ground_station_mass += wound_mass

    return ground_station_mass

# =====================
#  MASS SCALING METHODS
# =====================

def generate_kite_to_node_mass_ratio(options, variables_si, parameters, architecture, scaling):

    number_of_nodes = architecture.number_of_nodes
    kite_mass = parameters['theta0', 'geometry', 'm_k']

    # this will be used to scale the forces,
    # so we need to repeat the scaling forces 3x, once per dimension for force.
    # and, only on those nodes for which we will construct dynamics equations via. Lagrangian mechanics
    mass_ratio_stacked = []
    for node in range(1, number_of_nodes):

        node_mass = cas.DM(0.)

        segment_under_node_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling,
                                                                                  variables_si, parameters, node)
        if (node == 1) and options['tether']['use_wound_tether'] and ('l_t_full' in variables_si['theta'].keys()):

            tether_density = segment_under_node_properties['density']
            cross_section = segment_under_node_properties['scaling_area']
            length = scaling['theta', 'l_t_full', 0]
            mass_tether = tether_density * cross_section * length
            mass_generator = parameters['theta0', 'ground_station', 'm_gen']
            mass_under_node = mass_tether + mass_generator
        else:
            mass_under_node = segment_under_node_properties['scaling_mass']
        node_mass += mass_under_node / 2.

        if node in architecture.children_map.keys():
            for child in architecture.children_map[node]:
                segment_above_node_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, child)
                mass_above_node = segment_above_node_properties['scaling_mass']
                node_mass += mass_above_node / 2.

        if node in architecture.kite_nodes:
            node_mass += kite_mass

        mass_ratio = kite_mass / node_mass
        three_dimensional_mass_ratio = cas.DM.ones((3, 1)) * mass_ratio

        mass_ratio_stacked = cas.vertcat(mass_ratio_stacked, three_dimensional_mass_ratio)

    return mass_ratio_stacked


# def remove_wound_tether_entry(node_masses):
#     if 'm00' in node_masses.keys():
#         node_masses.pop('m00')
#
#     return node_masses

# def initialize_mass_dictionary(options, architecture):
#
#     number_of_nodes = architecture.number_of_nodes
#
#     empty = cas.DM.zeros((1, 1))
#
#     # initialize dictionary
#     node_masses = {}
#     for node in range(1, number_of_nodes):
#         node_masses['m' + architecture.node_label(node)] = empty
#
#     node_masses['m00'] = empty  # mass added to the ground-station
#     node_masses['groundstation'] = empty
#
#     return node_masses


# def generate_mass_dictionary_for_all_nodes(options, variables, parameters, architecture, scaling):
#
#     number_of_nodes = architecture.number_of_nodes
#     kite_nodes = architecture.kite_nodes
#
#     use_wound_tether = options['tether']['use_wound_tether']
#
#     node_masses = initialize_mass_dictionary(options, architecture)
#
#     upper_nodes_of_above_ground_segments = range(1, number_of_nodes)
#     for upper_node in upper_nodes_of_above_ground_segments:
#         node_masses = add_above_ground_tether_segment_mass(upper_node, node_masses, options,
#                                                            architecture, scaling, variables, parameters)
#
#     if use_wound_tether:
#         node_masses = add_wound_tether_mass(node_masses, options, architecture, scaling, variables, parameters)
#
#     for kite in kite_nodes:
#         node_masses = add_kite_mass(kite, node_masses, architecture, parameters)
#
#     node_masses = add_groundstation_mass(node_masses, parameters)
#
#     if not use_wound_tether:
#         node_masses = remove_wound_tether_entry(node_masses)
#
#     return node_masses


# def add_groundstation_mass(node_masses, parameters):
#     m_groundstation = parameters['theta0', 'ground_station', 'm_gen']
#     node_masses['groundstation'] += m_groundstation
#
#     return node_masses


# def add_wound_tether_mass(node_masses, options, architecture, scaling, variables, parameters):
#
#     main_props = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables, parameters, upper_node=1)
#     wound_length = options['scaling']['theta']['l_t_full'] - main_props['scaling_length']
#     wound_cross_section = main_props['scaling_area']
#
#     wound_mass = wound_cross_section * parameters['theta0', 'tether', 'rho'] * wound_length
#     node_masses['m00'] += wound_mass
#
#     return node_masses


# def add_above_ground_tether_segment_mass(upper_node, node_masses, options, architecture, scaling, variables, parameters):
#
#     seg_props = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables, parameters, upper_node=upper_node)
#     seg_mass = seg_props['scaling_mass']
#     top_mass_alloc_frac = options['tether']['top_mass_alloc_frac']
#
#     # attribute (the fraction of the segment mass that belong to the top node) to the top node
#     node_masses['m' + architecture.node_label(upper_node)] += top_mass_alloc_frac * seg_mass
#
#     # attribute (the fraction of the segment mass that doesn't belong to the top node) to the bottom node
#     node_masses['m' + architecture.parent_label(upper_node)] += (1. - top_mass_alloc_frac) * seg_mass
#
#     return node_masses


# def add_kite_mass(node, node_masses, architecture, parameters):
#
#     kite_mass = parameters['theta0', 'geometry', 'm_k']
#     if node in architecture.kite_nodes:
#         node_masses['m' + architecture.node_label(node)] += kite_mass
#
#     return node_masses
