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
discretization code (direct collocation or multiple shooting)
creates nlp variables and outputs, and gets discretized constraints
python-3.5 / casadi-3.4.5
- authors: elena malz 2016
           rachel leuthold, jochem de schutter alu-fr 2017-21
'''

import casadi.tools as cas
import numpy as np

import awebox.ocp.constraints as constraints
import awebox.ocp.collocation as coll_module
import awebox.ocp.multiple_shooting as ms_module
import awebox.ocp.ocp_outputs as ocp_outputs
import awebox.ocp.var_struct_averageModel as var_struct

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

def construct_time_grids(nlp_options):

    assert nlp_options['phase_fix'] == 'single_reelout'
    # assert nlp_options['discretization'] == 'direct_collocation'

    time_grids = {}
    nk = nlp_options['n_k']
    # if nlp_options['discretization'] == 'direct_collocation':
    direct_collocation = True
    ms = False
    d = nlp_options['collocation']['d']
    scheme = nlp_options['collocation']['scheme']
    tau_root = cas.vertcat(cas.collocation_points(d, scheme))
    tcoll = []

    # elif nlp_options['discretization'] == 'multiple_shooting':
    #     direct_collocation = False
    #     ms = True
    #     tcoll = None

    # make symbolic time constants
    # tfsym[0]: duration of the reelooout phase
    # tfsym[1]: duration of the reelin phase
    tfsym = cas.SX.sym('tfsym',2)
    nk_reelout = nlp_options['n_reelout']
    nk_reelin  = nlp_options['n_reelin']


    t_switch = tfsym[0]
    time_grids['t_switch'] = cas.Function('tgrid_tswitch', [tfsym], [t_switch])

    # build time grid for interval nodes x
    tx_reelout = cas.vertcat(np.linspace(0,1,nk_reelout+1))*tfsym[0]
    tx_reelin =  tfsym[0] + cas.vertcat(np.linspace(0,1,nk_reelin+1))*tfsym[1]
    tx = cas.vertcat(tx_reelout, tx_reelin)
    # build time grid for interval collocation points

    # time grid for the controls
    tu_reelout = cas.vertcat(np.linspace(0,1,nk_reelout,endpoint=False))*tfsym[0]
    tu_reelin =  tfsym[0] + cas.vertcat(np.linspace(0,1,nk_reelin,endpoint=False))*tfsym[1]
    tu = cas.vertcat(tu_reelout, tu_reelin)

    # time grid for the collocation points
    h_reelout = tfsym[0]/nk_reelout # step size for the reelout phase
    h_reelin = tfsym[1]/nk_reelin # step size for the reelin phase
    tcoll_reelout = cas.vertcat(*[tx_reelout[k] + tau_root*h_reelout for k in range(nk_reelout)]) # (nk_reelout, d)
    tcoll_reelin = cas.vertcat(*[tx_reelin[k] + tau_root*h_reelin for k in range(nk_reelin)]) # (nk_reelout, d)
    tcoll = cas.vertcat(tcoll_reelout, tcoll_reelin)

    # time grid for the node points and the collocation points
    _root_with_zero = cas.vertcat(0.0, tau_root)
    tx_coll_reelout = cas.vertcat(*[tx_reelout[k] + _root_with_zero*h_reelout for k in range(nk_reelout)])  # (nk_reelout * (d+1))
    tx_coll_reelout = cas.vertcat(tx_coll_reelout, tfsym[0])  # append time of the last node point
    tx_coll_reelin = cas.vertcat(*[tx_reelin[k] + _root_with_zero*h_reelin for k in range(nk_reelin)])  # (nk_reelout* (d+1))
    tx_coll_reelin = cas.vertcat(tx_coll_reelin, tfsym[1])  # append time of the last node point
    tx_coll = cas.vertcat(tx_coll_reelout, tx_coll_reelin) # (nk_reelout* (d+1) + 1  + nk_reelin* (d+1) +1)

    # write out collocation grids
    time_grids['coll'] = cas.Function('tgrid_coll',[tfsym],[tcoll])
    time_grids['x_coll'] = cas.Function('tgrid_x_coll',[tfsym],[tx_coll])

    # write out interval grid
    time_grids['x'] = cas.Function('tgrid_x',[tfsym],[tx])
    time_grids['u'] = cas.Function('tgrid_u',[tfsym],[tu])


    return time_grids

def setup_nlp_cost():

    cost = cas.struct_symMX([(
        cas.entry('tracking'),
        cas.entry('u_regularisation'),
        cas.entry('xdot_regularisation'),
        cas.entry('gamma'),
        cas.entry('iota'),
        cas.entry('psi'),
        cas.entry('tau'),
        cas.entry('eta'),
        cas.entry('nu'),
        cas.entry('upsilon'),
        cas.entry('fictitious'),
        cas.entry('power'),
        cas.entry('power_derivative'),
        cas.entry('t_f'),
        cas.entry('theta_regularisation'),
        cas.entry('nominal_landing'),
        cas.entry('compromised_battery'),
        cas.entry('transition'),
        cas.entry('slack'),
        cas.entry('beta'),
        cas.entry('P_max')
    )])

    return cost


def setup_nlp_p_fix(V, model):

    # fixed system parameters
    p_fix = cas.struct_symMX([(
        cas.entry('ref', struct=V),     # tracking reference for cost function
        cas.entry('weights', struct=model.variables)  # weights for cost function
    )])

    return p_fix

def setup_nlp_p(V, model):

    cost = setup_nlp_cost()
    p_fix = setup_nlp_p_fix(V, model)

    # use_vortex_linearization = 'lin' in model.parameters_dict.keys()
    # if use_vortex_linearization:
    #     P = cas.struct_symMX([
    #         cas.entry('p', struct=p_fix),
    #         cas.entry('cost', struct=cost),
    #         cas.entry('theta0', struct=model.parameters_dict['theta0']),
    #         cas.entry('lin', struct=V)
    #     ])
    # else:
    P = cas.struct_symMX([
        cas.entry('p',      struct = p_fix),
        cas.entry('cost',   struct = cost),
        cas.entry('theta0', struct = model.parameters_dict['theta0'])
    ])

    return P

def setup_integral_output_structure(nlp_options, integral_outputs):

    nk = nlp_options['n_k']

    entry_tuple = ()

    # interval outputs
    entry_tuple += (
        cas.entry('int_out', repeat = [nk+1], struct = integral_outputs),
    )

    if nlp_options['discretization'] == 'direct_collocation':
        d  = nlp_options['collocation']['d']
        entry_tuple += (
            cas.entry('coll_int_out', repeat = [nk,d], struct = integral_outputs),
        )

    Integral_outputs_struct = cas.struct_symMX([entry_tuple])

    return Integral_outputs_struct

def setup_output_structure(nlp_options, model_outputs, global_outputs):

    # create outputs
    nk = nlp_options['n_k']

    entry_tuple =  ()
    if nlp_options['discretization'] == 'direct_collocation':

        # extract collocation parameters
        d  = nlp_options['collocation']['d']
        scheme = nlp_options['collocation']['scheme']

        # define outputs on interval and collocation nodes
        if nlp_options['collocation']['u_param'] == 'poly':
            entry_tuple += (
                cas.entry('coll_outputs', repeat = [nk,d], struct = model_outputs),
            )

        elif nlp_options['collocation']['u_param'] == 'zoh':
            entry_tuple += (
                cas.entry('outputs',      repeat = [nk],   struct = model_outputs),
                cas.entry('coll_outputs', repeat = [nk,d], struct = model_outputs),
            )

    elif nlp_options['discretization'] == 'multiple_shooting':

        # define outputs on interval nodes
        entry_tuple += (
            cas.entry('outputs', repeat = [nk], struct = model_outputs),
        )

    Outputs = cas.struct_symMX([entry_tuple]
                           + [cas.entry('final', struct=global_outputs)])

    return Outputs

def discretize(nlp_options, model, formulation):

    # -----------------------------------------------------------------------------
    # discretization setup
    # -----------------------------------------------------------------------------
    n_reelout = nlp_options['n_reelout'] # number of integration intervals in the reel out phase
    n_reelin = nlp_options['n_reelin'] # number of integration intervals in the reel in phase

    assert n_reelout == n_reelin == nlp_options['n_k']

    d = nlp_options['collocation']['d']
    scheme = nlp_options['collocation']['scheme']
    Collocation_reelout = coll_module.Collocation(n_reelout, d, scheme)
    Collocation_reelin = coll_module.Collocation(n_reelin, d, scheme)

    dae = None
    Multiple_shooting = None




    # create a top level structure that contains both structures for the reel out and reel in phase

    # check if phase fix and adjust theta accordingly
    assert nlp_options['phase_fix'] == 'single_reelout'
    model_variables_dict = model.variables_dict
    theta = var_struct.get_phase_fix_theta(model_variables_dict)


    # add global entries
    # when the global variables are before the discretized variables, it leads to prettier kkt matrix spy plots
    entry_list_global_Variables = [
        cas.entry('theta', struct = theta),
        cas.entry('phi',   struct = model.parameters_dict['phi']),
        cas.entry('xi',    struct = var_struct.get_xi_struct()),
    ]

    V_reelout = var_struct.setup_nlp_v(nlp_options, model, n_reelout, Collocation_reelout)
    V_reelin = var_struct.setup_nlp_v(nlp_options, model, n_reelin, Collocation_reelin)

    # put the three structures together into one
    V = cas.struct_symMX([cas.entry('V_reelout', struct = V_reelout),
                          cas.entry('V_reelin', struct = V_reelin)]+
                         entry_list_global_Variables)


    P = setup_nlp_p(V, model)

    Xdot_reelout = Collocation_reelout.get_xdot(nlp_options, V_reelout, model)
    Xdot_reelin = Collocation_reelin.get_xdot(nlp_options, V_reelin, model)

    [coll_outputs_RO,
    Integral_outputs_list_RO,
    Integral_constraint_list_RO] = Collocation_reelout.collocate_outputs_and_integrals(nlp_options, model, formulation, V_reelout, P, Xdot_reelout)

    [coll_outputs_RI,
    Integral_outputs_list_RI,
    Integral_constraint_list_RI] = Collocation_reelin.collocate_outputs_and_integrals(nlp_options, model, formulation, V_reelin, P, Xdot_reelin)


    ms_xf = None
    ms_z0 = None
    ms_vars = None
    ms_params = None


    #-------------------------------------------
    # DISCRETIZE VARIABLES, CREATE NLP PARAMETERS
    #-------------------------------------------

    # construct time grids for this nlp
    time_grids = construct_time_grids(nlp_options)


    # ---------------------------------------
    # PREPARE OUTPUTS STRUCTURE
    # ---------------------------------------
    mdl_outputs = model.outputs

    global_outputs, _ = ocp_outputs.collect_global_outputs(nlp_options, model, V)
    global_outputs_fun = cas.Function('global_outputs_fun', [V, P], [global_outputs.cat])

    #-------------------------------------------
    # COLLOCATE OUTPUTS
    #-------------------------------------------

    # prepare listing of outputs and constraints
    Outputs_list = []

    # Construct outputs
    for kdx in range(nk):

        if nlp_options['collocation']['u_param'] == 'zoh':
            Outputs_list.append(coll_outputs[:,kdx*(d+1)])

        # add outputs on collocation nodes
        for ddx in range(d):

            # compute outputs for this time interval
            if nlp_options['collocation']['u_param'] == 'zoh':
                Outputs_list.append(coll_outputs[:,kdx*(d+1)+ddx+1])
            elif nlp_options['collocation']['u_param'] == 'poly':
                Outputs_list.append(coll_outputs[:,kdx*(d)+ddx])

    Outputs_fun = cas.Function('Outputs_fun', [V, P], [cas.horzcat(*Outputs_list)])
    Outputs_struct = None

    # Create Integral outputs struct and function
    Integral_outputs_struct = setup_integral_output_structure(nlp_options, model.integral_outputs)
    Integral_outputs = Integral_outputs_struct(cas.vertcat(*Integral_outputs_list))
    Integral_outputs_fun = cas.Function('Integral_outputs_fun', [V, P], [cas.vertcat(*Integral_outputs_list)])

    Xdot_struct = Xdot
    Xdot_fun = cas.Function('Xdot_fun',[V],[Xdot])

    # -------------------------------------------
    # GET CONSTRAINTS
    # -------------------------------------------
    ocp_cstr_list, ocp_cstr_struct = constraints.get_constraints(nlp_options, V, P, Xdot, model, dae, formulation,
        Integral_constraint_list, Integral_outputs, Collocation, Multiple_shooting, ms_z0, ms_xf,
            ms_vars, ms_params, Outputs_struct, time_grids)

    return V, P, Xdot_struct, Xdot_fun, ocp_cstr_list, ocp_cstr_struct, Outputs_struct, Outputs_fun, Integral_outputs_struct, Integral_outputs_fun, time_grids, Collocation, Multiple_shooting, global_outputs, global_outputs_fun


