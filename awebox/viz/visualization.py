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
##################################
# Class Visualization contains plotting functions to visualize data
# of trials and sweeps
###################################
from typing import Dict
from awebox.tools import struct_operations as struct_op
import casadi
import numpy as np
import casadi as ca
import casadi.tools as cas
from . import tools
from . import trajectory
from . import variables
from . import animation
from . import output
from . import wake

import os


import matplotlib

import matplotlib.pyplot as plt

from awebox.logger.logger import Logger as awelogger
from ..ocp.collocation import Collocation
from ..ocp.discretization_averageModel import eval_time_grids_SAM, reconstruct_full_from_SAM, \
    originalTimeToSAMTime, OthorgonalCollocation, construct_time_grids_SAM_reconstruction, \
    constructPiecewiseCasadiExpression
from ..opti import diagnostics
from ..tools.struct_operations import calculate_SAM_regions, calculate_SAM_regionIndexArray


#todo: compare to initial guess for all plots as option
#todo: options for saving plots


class Visualization(object):

    def __init__(self):

        self.__plot_dict = None

    def build(self, model, nlp, name, options):
        """
        Generate plot dictionary with all relevant plot information.
        :param model: system model
        :param nlp: NLP formulation
        :param visualization_options: visualization related options
        :return: None
        """

        self.__plot_dict = tools.calibrate_visualization(model, nlp, name, options)
        self.create_plot_logic_dict()
        self.__options = options

        return None

    def recalibrate(self, V_plot, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref, global_outputs):

        self.__plot_dict = tools.recalibrate_visualization(V_plot, self.plot_dict, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref, global_outputs)

        return None

    def plot(self, V_plot, parametric_options, output_vals, integral_outputs_final, flags, time_grids, cost, name, sweep_toggle, V_ref, global_outputs, fig_name='plot', fig_num = None, recalibrate = True):
        """
        Generate plots with given parametric and visualization options
        :param V_plot: plot data (scaled)
        :param parametric_options: values for parametric options
        :param visualization_options: visualization related options
        :return: None
        """

        # recalibrate plot_dict
        if recalibrate:
            self.recalibrate(V_plot, self.__plot_dict, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref, global_outputs)

        if type(flags) is not list:
            flags = [flags]

        # define special flags
        if 'all' in flags:
            flags = list(self.__plot_logic_dict.keys())
            flags.remove('animation')
            flags.remove('animation_snapshot')
            flags = [flag for flag in flags if 'outputs:' not in flag]

        level_1 = ['states', 'controls', 'isometric']
        level_2 = level_1 + ['invariants', 'algebraic_variables', 'lifted_variables', 'constraints']
        level_3 = level_2 + ['aero_dimensionless', 'aero_coefficients', 'projected_xy', 'projected_xz', 'projected_yz']

        if 'level_1' in flags:
            flags.remove('level_1')
            flags += level_1

        if 'level_2' in flags:
            flags.remove('level_2')
            flags += level_2

        if 'level_3' in flags:
            flags.remove('level_3')
            flags += level_3

        # iterate over flags
        for flag in flags:
            if flag[:5] == 'comp_':
                awelogger.logger.warning('Comparison plots are only supported for sweeps. Flag "' + flag + '" ignored.')
            else:
                self.__produce_plot(flag, fig_name, parametric_options['visualization']['cosmetics'], fig_num)

        if parametric_options['visualization']['cosmetics']['show_when_ready'] == True and sweep_toggle == False:
            plt.show(block=True)

        return None

    def create_plot_logic_dict(self):
        """
        Create a dict for selecting the correct plotting function for a given flag.
        Notation for adding entries:
        (FUNCTION, TUPLE_WITH_ADDITIONAL_ARGS/None)
        :return: dictionary for plot function selection
        """

        outputs = self.plot_dict['outputs_dict']
        variables_dict = self.plot_dict['variables_dict']
        integral_variables = self.plot_dict['integral_variables']

        plot_logic_dict = {}
        plot_logic_dict['isometric'] = (trajectory.plot_trajectory, {'side':'isometric'})
        plot_logic_dict['projected_xy'] = (trajectory.plot_trajectory, {'side':'xy'})
        plot_logic_dict['projected_yz'] = (trajectory.plot_trajectory, {'side':'yz'})
        plot_logic_dict['projected_xz'] = (trajectory.plot_trajectory, {'side':'xz'})
        plot_logic_dict['quad'] = (trajectory.plot_trajectory, {'side':'quad'})
        plot_logic_dict['animation'] = (animation.animate_monitor_plot, None)
        plot_logic_dict['animation_snapshot'] = (animation.animate_snapshot, None)
        plot_logic_dict['vortex_verification'] = (wake.plot_vortex_verification, None)
        plot_logic_dict['induction_factor'] = (output.plot_induction_factor, None)
        plot_logic_dict['relative_radius'] = (output.plot_relative_radius, None)
        plot_logic_dict['loyd_comparison'] = (output.plot_loyd_comparison, None)
        plot_logic_dict['aero_coefficients'] = (output.plot_aero_coefficients, None)
        plot_logic_dict['aero_dimensionless'] = (output.plot_aero_validity, None)
        plot_logic_dict['wake_isometric'] = (wake.plot_wake, {'side':'isometric'})
        plot_logic_dict['wake_xy'] = (wake.plot_wake, {'side':'xy'})
        plot_logic_dict['wake_yz'] = (wake.plot_wake, {'side':'yz'})
        plot_logic_dict['wake_xz'] = (wake.plot_wake, {'side':'xz'})
        plot_logic_dict['circulation'] = (output.plot_circulation, None)
        plot_logic_dict['states'] = (variables.plot_states, None)
        plot_logic_dict['wake_states'] = (variables.plot_wake_states, None)
        for variable in list(variables_dict['x'].keys()) + integral_variables:
            plot_logic_dict['states:' + variable] = (variables.plot_states, {'individual_state':variable})
        plot_logic_dict['controls'] = (variables.plot_controls, None)
        for control in list(variables_dict['u'].keys()):
            plot_logic_dict['controls:' + control] = (variables.plot_controls, {'individual_control':control})
        plot_logic_dict['invariants'] = (variables.plot_invariants, None)
        plot_logic_dict['algebraic_variables'] = (variables.plot_algebraic_variables, None)
        plot_logic_dict['wake_lifted_variables'] = (variables.plot_wake_lifted, None)
        plot_logic_dict['lifted_variables'] = (variables.plot_lifted, None)
        plot_logic_dict['constraints'] = (output.plot_constraints, None)
        for output_top_name in list(outputs.keys()):
            plot_logic_dict['outputs:' + output_top_name] = (output.plot_outputs, {'output_top_name': output_top_name})

        self.__plot_logic_dict = plot_logic_dict
        self.__plot_dict['plot_logic_dict'] = plot_logic_dict

    def __produce_plot(self, flag, fig_name, cosmetics, fig_num = None):
        """
        Produce the plot for a given flag, fig_num and cosmetics.
        :param flag: string identifying the kind of plot that should be produced
        :param fig_num: number of the figure that the plot should be displayed in
        :param cosmetics: cosmetic options for the plot
        :return: updated fig_num
        """

        # map flag to function
        fig_name = self.__plot_dict['name'] + '_' + flag + '_' + fig_name

        if fig_num is not None:
            self.__plot_logic_dict[flag][1]['fig_num'] = fig_num

        tools.map_flag_to_function(flag, self.__plot_dict, cosmetics, fig_name, self.__plot_logic_dict)

        if fig_num is not None:
            del self.__plot_logic_dict[flag][1]['fig_num']

        # save figures
        if cosmetics['save_figs']:
            name_rep = self.__plot_dict['name']
            for char in ['(', ')', '_', ' ']:
                name_rep = name_rep.replace(char, '')

            directory = "./figures"
            directory_exists = os.path.isdir(directory)
            if not directory_exists:
                os.mkdir(directory)

            save_name = directory + '/' + name_rep + '_' + flag
            plt.savefig(save_name + '.eps', bbox_inches='tight', format='eps', dpi=1000)
            plt.savefig(save_name + '.pdf', bbox_inches='tight', format='pdf', dpi=1000)

        return None

    @property
    def plot_dict(self):
        return self.__plot_dict

    @plot_dict.setter
    def plot_dict(self, value):
        self.__plot_dict = value

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
       self.__options = value

    @property
    def plot_logic_dict(self):
        return self.__plot_logic_dict

    @plot_logic_dict.setter
    def plot_logic_dict(self, value):
        print('Cannot set plot_logic_dict object.')


class VisualizationSAM(Visualization):

    def __init__(self):
        super().__init__()
        self.__plot_dict_SAM: dict = None
        self.__options: dict = None

    def build(self, model, nlp, name, options):
        """
        Generate plot dictionary with all relevant plot information.
        :param model: system model
        :param nlp: NLP formulation
        :param visualization_options: visualization related options
        :return: None
        """

        self.__plot_dict = tools.calibrate_visualization(model, nlp, name, options)
        self.__plot_dict_SAM = tools.calibrate_visualization(model, nlp, name, options)
        # self.create_plot_logic_dict()
        self.__options = options

    def recalibrate(self, V_plot, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref, global_outputs):
        """ Recalibrate both the SAM and the RECONSTRUCTED plot dictionaries. """

        # in the original (SAM) dictionary, only the timegrid is different
        self.plot_dict_SAM = tools.recalibrate_visualization(V_plot, self.plot_dict_SAM, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref, global_outputs)

        # replace the interpolating grid with the SAM grid
        time_grid_ip_original: np.ndarray = self.plot_dict_SAM['time_grids']['ip']
        time_grid_xcoll_original: ca.DM = self.plot_dict_SAM['time_grids']['x_coll']
        originalTimeToSAMTime_f = originalTimeToSAMTime(self.options['nlp'],V_plot['theta', 't_f'])
        time_grid_SAM_eval = eval_time_grids_SAM(self.options['nlp'],V_plot['theta', 't_f'])
        time_grid_SAM_eval['ip'] = originalTimeToSAMTime_f.map(time_grid_ip_original.size)(time_grid_ip_original).full().flatten()

        # add the region indices to the SAM plot dictionary
        self.plot_dict_SAM['SAM_regions_x_coll'] = calculate_SAM_regionIndexArray(self.options['nlp'],
                                                                              V_plot,
                                                                              time_grid_xcoll_original.full().flatten())
        self.plot_dict_SAM['SAM_regions_ip'] = calculate_SAM_regionIndexArray(self.options['nlp'],
                                                                              V_plot,
                                                                              time_grid_ip_original)
        self.plot_dict_SAM['time_grids'] = time_grid_SAM_eval  # we do this AFTER we calculate the region indices

        # the plot dict is now the RECONSTRUCTED one
        self.plot_dict = self.create_reconstructed_plot_dict(V_plot, output_vals[1], global_outputs,integral_outputs_final)

    def create_reconstructed_plot_dict(self, V_plot, output_vals, global_outputs,integral_outputs_final) -> dict:
        """ Create the plot dictionary for the RECONSTRUCTED variables and outputs. """

        # extract information
        plot_dict = self.plot_dict  # get the existing plot dict, it already contains some information
        nlp_options = self.options['nlp']
        scaling = plot_dict['scaling']
        V_plot = struct_op.scaled_to_si(V_plot, scaling) # convert V_plot to SI units

        # reconstruct the full trajectory
        awelogger.logger.info('Reconstructing the full trajectory from the SAM solution..')
        V_reconstructed, time_grid_reconstructed, output_vals_reconstructed = reconstruct_full_from_SAM(
            nlpoptions=nlp_options, Vopt=V_plot, output_vals_opt=output_vals)

        # interpolate the reconstructed trajectory
        n_ip = self.options['visualization']['cosmetics']['interpolation']['N']
        awelogger.logger.info(f'Interpolating reconstruted trajectory with {n_ip} points  ..')
        funcs_ip = build_interpolate_functions_full_solution(V_reconstructed, time_grid_reconstructed, nlp_options,
                                                             output_vals_reconstructed)

        # evaluate states, controls, algebraic variables, outputs at the interpolated points
        t_ip = np.linspace(0, float(time_grid_reconstructed['x'][-1]), n_ip)
        x_ip = funcs_ip['x'].map(t_ip.size)(t_ip)
        x_ip_dict = dict_from_repeated_struct(plot_dict['variables_dict']['x'], x_ip)
        u_ip = funcs_ip['u'].map(t_ip.size)(t_ip)
        u_ip_dict = dict_from_repeated_struct(plot_dict['variables_dict']['u'], u_ip)
        z_ip = funcs_ip['z'].map(t_ip.size)(t_ip)
        z_ip_dict = dict_from_repeated_struct(plot_dict['variables_dict']['z'], z_ip)
        y_ip = funcs_ip['y'].map(t_ip.size)(t_ip)
        y_ip_dict = dict_from_repeated_struct(plot_dict['outputs_struct'], y_ip)

        # build the output dict
        awelogger.logger.info('Building plot ditionary for the reconstructed trajectory..')
        plot_dict['z'] = z_ip_dict
        plot_dict['x'] = x_ip_dict
        plot_dict['u'] = u_ip_dict
        plot_dict['outputs'] = y_ip_dict
        plot_dict['output_vals'] = [output_vals_reconstructed,output_vals_reconstructed] # TODO: this is not the intended functionality
        plot_dict['time_grids'] = time_grid_reconstructed
        plot_dict['time_grids']['ip'] = t_ip
        plot_dict['global_outputs'] = global_outputs
        plot_dict['V_plot'] = V_reconstructed
        plot_dict['integral_outputs_final'] = integral_outputs_final
        plot_dict['power_and_performance'] = diagnostics.compute_power_and_performance(plot_dict)

        awelogger.logger.info('... Done!')

        return plot_dict



    @property
    def plot_dict(self) -> dict:
        """ The interpolated RECONSTRUCTED trajectory and data it contains:

            - the same variables as the original plot_dict, expect:
            - the nlp variables V_plot now are the RECONSTRUCTED variables
            - the interpolated trajectories ('x', 'u', 'z', 'outputs') are the RECONSTRUCTED trajectories
            - the time grid is the RECONSTRUCTED time grid ('time_grids')
            - the raw output vals ('output_vals') are the RECONSTRUCTED output values, but both entries [0] and [1] are
              the same, since there are no initial reconstructed trajectories.

        Since this is the reconstructed trajectory from a SAM problem,
        the trajectory is only an approximation of a physical trajectory (!).
        """
        awelogger.logger.warning('`plot_dict` - You are accessing the RECONSTRUCTED results from a SAM problem. These '
                                 'results are only an approximation of a physical trajectory.')
        return self.__plot_dict

    @plot_dict.setter
    def plot_dict(self, value):
        self.__plot_dict = value

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
       self.__options = value

    @property
    def plot_dict_SAM(self):
        """ The plot dictionary for the original SAM problem and its outputs. It contains:

                - the same variables as the original plot_dict, expect:
                - the time grid is the SAM time grid ('time_grids')
                - the SAM regions are calculated and stored ('SAM_regions_x_coll', 'SAM_regions_ip')
        """
        return self.__plot_dict_SAM

    @plot_dict_SAM.setter
    def plot_dict_SAM(self, value):
        self.__plot_dict_SAM = value


def build_interpolate_functions_full_solution(V: cas.struct, tgrid: dict , nlpoptions: dict, output_vals: np.ndarray) -> Dict[str, ca.Function]:
    """ Build functions that interpolate the full solution from a given V structure and with nodes on timegrid['x'].

    Returns a dictionary of casadi functions that interpolate the state, control,algebraic variables and the outputs
    for a given time, i.e. x(t), u(t), z(t), y(t)

    :param V: the solution structure, containing 'x' (nx, n_k+1), 'u' (nu, n_k), 'z' (nz, n_k)
    :param tgrid: the time grid structure, containing 'x' (n_k+1)
    :param nlpoptions: the nlp options e.g. trial.options['nlp']
    :param output_vals: the output values, containing the structures of the model outputs e.g. plot_dict['output_vals'][1]
    """

    assert {'x', 'u', 'z'}.issubset(V.keys())

    # build micro-integration interpolation functions
    d_micro = nlpoptions['collocation']['d']
    coll_points = np.array(ca.collocation_points(d_micro,nlpoptions['collocation']['scheme']))
    coll_integrator = OthorgonalCollocation(coll_points)
    intpoly_x_f = coll_integrator.getPolyEvalFunction(V.getStruct('x')(0).shape, includeZero=True)
    intpoly_z_f = coll_integrator.getPolyEvalFunction(V.getStruct('z')(0).shape, includeZero=False)
    intpoly_outputs_f = coll_integrator.getPolyEvalFunction(output_vals[:, 0].shape, includeZero=True)

    # number of intervals & edges
    n_k = len(V['u'])
    assert tgrid['x'].shape[0] == n_k + 1, (f'The number of edges in the time grid'
                                            f' should be equal to n_k + 1 = {n_k + 1}, but is {tgrid["x"].shape[0]}')
    edges = tgrid['x'].full().flatten()

    # iterate over the intervals
    express_x = []
    express_u = []
    express_z = []
    express_y = []
    t = ca.SX.sym('t')
    for n in range(n_k):
        t_n = t - edges[n]  # remove the offset from the time
        delta_t = edges[n + 1] - edges[n]  # duration of the interval

        # build casadi expressions for the local interpolations of state, control and algebraic variables
        express_x.append(intpoly_x_f(t_n / delta_t, V['x', n], *V['coll_var', n, :, 'x']))
        express_z.append(intpoly_z_f(t_n / delta_t, *V['coll_var', n, :, 'z']))
        express_u.append(V['u', n])
        express_y.append(intpoly_outputs_f(t_n / delta_t, *[output_vals[:, n*(d_micro+1)+i] for i in range(d_micro+1)]))

    # shift the final edge a bit to avoid numerical issues
    edges[-1] = edges[-1] + 1e-6

    # combine into single function
    express_x = constructPiecewiseCasadiExpression(t, edges.tolist(), express_x)
    express_z = constructPiecewiseCasadiExpression(t, edges.tolist(), express_z)
    express_u = constructPiecewiseCasadiExpression(t, edges.tolist(), express_u)
    express_y = constructPiecewiseCasadiExpression(t, edges.tolist(), express_y)

    return {'x': ca.Function('interpolated_x', [t], [express_x]),
            'u': ca.Function('interpolated_u', [t], [express_u]),
            'z': ca.Function('interpolated_z', [t], [express_z]),
            'y': ca.Function('interpolated_y', [t], [express_y])}


def dict_from_repeated_struct(struct: ca.tools.struct, values: ca.DM) -> dict:
    """ Create a nested dictionary from values of a repeated
    casadi structure (n_struct, n_vals) that contains the values.

    Args:
        struct: casadi struct
        values: casadi DM of shape (n_struct, n_vals) with the values
    """

    assert struct.shape[0] == values.shape[0]

    # cast into repeated structure
    struct_repeated = struct.repeated(values)

    dict_out = {}
    for canon in struct.canonicalIndices():
        vals = ca.vertcat(*struct_repeated[(slice(None),) + canon]).full().flatten()
        assign_nested_dict(dict_out, canon, vals)

    return dict_out


def assign_nested_dict(dictionary: dict, keys: list, value):
    """ (from CHATGPT)
    Indexes a nested dictionary with a list of keys and assigns a value to the final key.

    Args:
        dictionary (dict): The nested dictionary to be indexed.
        keys (list): A list of keys to traverse the nested dictionary.
        value: The value to be assigned at the final key.
    """
    # Navigate through the dictionary using the keys
    for index,key in enumerate(keys[:-1]):
        dictionary = dictionary.setdefault(key, {})

    # Assign the value to the final key
    dictionary[keys[-1]] = value
