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
keep the variable construction separate,
so that non-sequential code can be used non-sequentially
python-3.5 / casadi-3.4.5
- authors: elena malz 2016
           rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import casadi.tools as cas
import awebox.tools.struct_operations as struct_op
import awebox.ocp.collocation as collocation
import awebox.tools.print_operations as print_op


def setup_nlp_v(nlp_options, model, nk, Collocation=None):

    # extract necessary inputs
    models_variables_dict = model.variables_dict

    # define interval struct entries for controls and states
    entry_tuple = (
        cas.entry('x', repeat = [nk+1], struct = models_variables_dict['x']),
        )

    # add additional variables according to provided options
    assert nlp_options['collocation']['u_param'] == 'zoh'
    entry_tuple += (
        cas.entry('u',  repeat = [nk],   struct = models_variables_dict['u']),
    )

    # add state derivative variables at interval
    entry_tuple += (
        cas.entry('xdot', repeat = [nk], struct= models_variables_dict['xdot']),
    )

    # add algebraic variables at shooting nodes for constraint evaluation
    entry_tuple += (cas.entry('z', repeat = [nk],   struct= models_variables_dict['z']),)

    # add collocation node variables
    if Collocation is None:
        message = 'a None instance of Collocation was passed to the NLP variable structure generator'
        raise Exception(message)

    d = nlp_options['collocation']['d'] # interpolating polynomial order
    coll_var = Collocation.get_collocation_variables_struct(models_variables_dict, 'zoh')
    entry_tuple += (cas.entry('coll_var', struct = coll_var, repeat= [nk,d]),)

    # add global entries
    # when the global variables are before the discretized variables, it leads to prettier kkt matrix spy plots
    # entry_list = [
    #     cas.entry('theta', struct = theta),
    #     cas.entry('phi',   struct = model.parameters_dict['phi']),
    #     cas.entry('xi',    struct = get_xi_struct()),
    #     entry_tuple
    # ]

    # generate structure
    V = cas.struct_symMX([entry_tuple])

    # print_op.print_variable_info('NLP', V)

    return V

def get_phase_fix_theta(variables_dict):

    entry_list = []
    for name in list(variables_dict['theta'].keys()):
        if name == 't_f':
            entry_list.append(cas.entry('t_f', shape = (2,1)))
        else:
            entry_list.append(cas.entry(name, shape = variables_dict['theta'][name].shape))

    theta = cas.struct_symSX(entry_list)

    return theta

def get_xi_struct():
    xi = cas.struct_symMX([(cas.entry('xi_0'), cas.entry('xi_f'))])

    return xi
