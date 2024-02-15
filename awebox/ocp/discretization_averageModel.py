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
import casadi
import casadi.tools as cas
import numpy as np

import awebox.ocp.constraints as constraints
import awebox.ocp.collocation as coll_module
import awebox.ocp.multiple_shooting as ms_module
import awebox.ocp.ocp_outputs as ocp_outputs
import awebox.ocp.var_struct_averageModel as var_struct

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op
from awebox.ocp import ocp_constraint

# todo: move this somewhere or replace with existing collocation functions
class OthorgonalCollocation:
    """
    Base Class for all RK Integration Methods, stores the Butcher Tableau of the method.
    """
    c: np.ndarray = None
    A = None
    b = None
    d = None

    def getButcher(self) -> (np.ndarray,np.ndarray,np.ndarray):
        """Returns the Butcher Tableau (c,A,b) of the Integrator"""
        if self.A is None:
            raise NotImplementedError
        return self.c, self.A, self.b

    @property
    def isExplicit(self):
        return np.allclose(self.A, np.tril(self.A))

    @property
    def isCollocationMethod(self):
        # check if there are double entries in the c vector
        return len(self.c) == len(np.unique(self.c))

    def __init__(self, collPoints: np.ndarray):
        assert collPoints.ndim == 1
        assert np.all(np.unique(collPoints, return_counts=True)[1] <= 1), 'CollPoints have to be distinct!'
        # assert np.all(collPoints <= 1) and np.all(0 <= collPoints), 'CollPoints must be between 0 and 1'

        self.d = collPoints.shape[-1]

        # create list of polynomials
        self._ls = []
        for j in range(self.d):
            l = np.poly1d([1])
            for r in range(self.d):
                if r != j:
                    l *= np.poly1d([1, -collPoints[r]]) / (collPoints[j] - collPoints[r])
            self._ls.append(l)


        self.c = collPoints
        self.b = np.array([np.polyint(l)(1) for l in self._ls])
        self.A = np.array([[np.polyint(l)(ci) for l in self._ls] for ci in self.c])

    @property
    def polynomials(self) -> list:
        """A list of the numpy polynomials that correspond to the lagrange polynomials"""
        return self._ls

    def __str__(self):
        return f"Oth. Coll. with {self.d} stages"

    def getPolyEvalFunction(self, shape: tuple, includeZero: bool = False, includeOne: bool = False, fixedValues: list = None) -> cas.Function:
        """
        Generates a casadi function that evaluates the polynomial at a given point t of the form

        p(t) = F(t, [x0], x1, ..., xd)

        where t is a scalar in [0,1] and x0, ..., xd are the collocation points of the provided shape.

        If fixed values for the nodes x0, ..., xd are provided, the function will be of the form

        p(t) = F(t)

        :param shape: the shape of the collocation nodes, can be matrices or vectors
        :param includeZero: if true, the collocation point at time 0 is included
        :param fixedValues: a list of fixed values for the nodes, if provided, the function will be of the form x(t) = F(t)
        """
        assert self.isCollocationMethod is True, "Can only reconstruct polynomial for collocation methods!"

        assert not(includeOne and includeZero), 'either includeOne or includeZero can be true, not both!'

        # append zero if needed
        if includeZero:
            collPoints = cas.DM(np.concatenate([[0],self.c]))
            d = self.d + 1
        elif includeOne:
            collPoints = cas.DM(np.concatenate([self.c,[1]]))
            d = self.d + 1
        else:
            collPoints = cas.DM(self.c)
            d = self.d

        nx = shape[0]*shape[1]
        t = cas.SX.sym('t')

        if fixedValues is None:
            # create symbolic variables for the nodes
            Xs = []
            for i in np.arange((0 if includeZero else 1),self.d+1):
                Xs.append(cas.SX.sym(f'x{i}', shape))

        else:
            assert len(fixedValues) == d, f"The number of fixed values ({len(fixedValues)}) must be equal to the number of collocation points ({d})!"
            assert all([v.shape == shape for v in fixedValues]), "The shape of the fixed values must be equal to the shape of the collocation points!"
            assert all([type(v) == cas.DM for v in fixedValues]), "The fixed values must be of type casadi.DM!"
            Xs = fixedValues

        # reshape input variables into a matrix of shape (nx, d)
        p_vals = cas.horzcat(*[X.reshape((nx, 1)) for X in Xs])

        # create list of polynomials
        _ls = []
        for j in range(d):
            l = 1
            for r in range(d):
                if r != j:
                    l *= (t -collPoints[r]) / (collPoints[j] - collPoints[r])
            _ls.append(l)

        # evaluate polynomials
        sum = cas.DM.zeros((nx, 1))
        for i in range(d):
            sum += p_vals[:, i] * _ls[i]

        # reshape the result into the original shape
        result = cas.reshape(sum,shape)

        if fixedValues is None:
            return cas.Function('polyEval', [t] + Xs, [result])
        else:
            return cas.Function('polyEval', [t], [result])

class ForwardEuler():
    c = np.array([0])
    A = np.array([[0]])
    b = np.array([1])
    d = c.shape[0]


class Lobatto3A_Order4:
    # (trapazoidal rule)
    c = np.array([0, 0.5, 1])
    A = np.array([[0, 0, 0],[5/24,1/3,-1/24], [1/6, 2/3, 1/6]])
    b = np.array([1/6, 2/3, 1/6])
    d = c.shape[0]

def construct_time_grids(nlp_options):

    assert nlp_options['phase_fix'] == 'single_reelout'
    # assert nlp_options['discretization'] == 'direct_collocation'

    time_grids = {}
    nk = nlp_options['n_k']
    if nlp_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        ms = False
        d = nlp_options['collocation']['d']
        scheme = nlp_options['collocation']['scheme']
        tau_root = cas.vertcat(cas.collocation_points(d, scheme))
        tcoll = []

    elif nlp_options['discretization'] == 'multiple_shooting':
        direct_collocation = False
        ms = True
        tcoll = None

    # make symbolic time constants
    if nlp_options['useAverageModel']:
        tfsym = cas.SX.sym('tfsym', nlp_options['d_SAM'] + 2)
        regions_indexes = struct_op.calculate_SAM_regions(nlp_options)
    elif nlp_options['phase_fix'] == 'single_reelout':
        tfsym = cas.SX.sym('tfsym',2)
        nk_reelout = round(nk * nlp_options['phase_fix_reelout'])

        t_switch = tfsym[0] * nk_reelout / nk
        time_grids['t_switch'] = cas.Function('tgrid_tswitch', [tfsym], [t_switch])

    else:
        tfsym = cas.SX.sym('tfsym',1)

    # initialize
    tx = []
    tu = []

    tcurrent = 0
    for k in range(nk):

        # speed of time of the specific interval
        tf_current = tfsym[struct_op.calculate_tf_index(nlp_options, k)]

        # add interval timings
        tx.append(tcurrent)
        tu.append(tcurrent)

        # add collocation timings
        if direct_collocation:
            for j in range(d):
                tcoll.append(tcurrent + tau_root[j] * tf_current/nk)

        # update current time
        tcurrent = tcurrent + tf_current/nk

    # add last interval time to tx for last integration node
    tx.append(tcurrent)

    tu = cas.vertcat(*tu)
    tx = cas.vertcat(*tx)
    tcoll = cas.vertcat(*tcoll)

    if direct_collocation:
        # reshape tcoll
        tcoll = tcoll.reshape((d,nk)).T
        tx_coll = cas.vertcat(cas.horzcat(tu, tcoll).T.reshape((nk*(d+1),1)),tx[-1])

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
    nk = nlp_options['n_k']

    # direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    # multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')
    #
    # if direct_collocation:
    d = nlp_options['collocation']['d']
    scheme = nlp_options['collocation']['scheme']
    Collocation = coll_module.Collocation(nk, d, scheme)

    dae = None
    Multiple_shooting = None

    V = var_struct.setup_nlp_v(nlp_options, model, Collocation)
    P = setup_nlp_p(V, model)

    Xdot = Collocation.get_xdot(nlp_options, V, model)
    [coll_outputs,
    Integral_outputs_list,
    Integral_constraint_list] = Collocation.collocate_outputs_and_integrals(nlp_options, model, formulation, V, P, Xdot)

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


    # ---------------------------------------------
    # modify the constraints for SAM
    # ---------------------------------------------
    SAM_cstrs_list = ocp_constraint.OcpConstraintList()
    # phase constraints to start and endpoint of the the reel-out phase




    SAM_cstrs_entry_list = []

    N_SAM = nlp_options['N_SAM']
    d_SAM = nlp_options['d_SAM']
    # assert d_SAM == 1, 'for now we only support d_SAM = 1'
    # macroIntegrator = Lobatto3A_Order4()
    macroIntegrator = OthorgonalCollocation(np.array(cas.collocation_points(d_SAM,nlp_options['SAM_MaInt_type'])))
    # macroIntegrator = ForwardEuler()
    c_macro,A_macro,b_macro = macroIntegrator.c,macroIntegrator.A,macroIntegrator.b
    assert d_SAM == c_macro.size

    tf_regions_indices = struct_op.calculate_SAM_regions(nlp_options)
    SAM_regions_indeces = tf_regions_indices[1:-1] # we are not intersted in the first region (pre-reelout) and the last region (reelin)

    # vz_phase = V['sam_misc','beta']

    # iterate micro-integrations
    n_SAM_microints = SAM_regions_indeces.__len__()


    for i in range(n_SAM_microints):

        n_first = SAM_regions_indeces[i][0] # first interval index of the region
        n_last = SAM_regions_indeces[i][-1] # last interval index of the region

        # 1. XMINUS: connect x_minus with start of the micro integration
        xminus = model.variables_dict['x'](V['x_micro_minus',i])
        micro_connect_xminus = cstr_op.Constraint(expr= xminus.cat - V['x', n_first],
                                      name= f'micro_connect_xminus_{i}',
                                      cstr_type='eq')
        SAM_cstrs_list.append(micro_connect_xminus)
        SAM_cstrs_entry_list.append(cas.entry(f'micro_connect_xminus_{i}', shape=xminus.shape))

        # 2. XPLUS: replace the continutiy constraint for the last collocation interval of the reelout phase
        xplus = model.variables_dict['x'](V['x_micro_plus', i])
        ocp_cstr_list.get_constraint_by_name(f'continuity_{n_last}').expr = xplus.cat - model.variables_dict['x'](Collocation.get_continuity_expression(V,n_last)).cat


        # 3. PHASE CONSTRAINT:
        # phase_cstr_end = cstr_op.Constraint(expr= V['x_micro_minus', i, 'dq10', 1] - V['x_micro_plus', i, 'dq10', 1],
        #                                       name=f'phase_end_{i}',
        #                                       cstr_type='eq')
        # SAM_cstrs_list.append(phase_cstr_end)
        # SAM_cstrs_entry_list.append(cas.entry(f'phase_end_{i}', shape=(1, 1)))

        # 4. SAM dynamics approximation - vcoll
        ada_vcoll_cstr = cstr_op.Constraint(expr= (xplus.cat - xminus.cat)*N_SAM - V['v_macro_coll', i],
                                              name=f'ada_vcoll_cstr_{i}',
                                              cstr_type='eq')
        SAM_cstrs_list.append(ada_vcoll_cstr)
        SAM_cstrs_entry_list.append(cas.entry(f'ada_vcoll_cstr_{i}', shape=xminus.shape))


        # 5. Connect to Macro integraiton point
        ada_type = nlp_options['SAM_ADAtype']
        assert ada_type in ['FD','BD','CD'], 'only FD, BD, CD are supported'
        ada_coeffs = {'FD': [1,-1, 0], 'BD':[0,-1,1], 'CD':[1,-2,1]}[ada_type]
        # if i==0:
        #     expr_connect = V['x_micro_minus', i] - V['x_macro_coll', i]
        # elif i==d_SAM-1:
        #     expr_connect = V['x_micro_plus', i] - V['x_macro_coll', i]
        # else: # somewhere in the middle
        #     expr_connect = V['x_micro_minus', i] +V['x_micro_plus', i] - 2* V['x_macro_coll', i]

        expr_connect = (ada_coeffs[0]*V['x_micro_minus', i]
                        + ada_coeffs[1]*V['x_macro_coll', i]
                        + ada_coeffs[2]*V['x_micro_plus', i])
        micro_connect_macro = cstr_op.Constraint(expr= expr_connect,
                                      name=f'micro_connect_macro_{i}',
                                      cstr_type='eq')
        SAM_cstrs_list.append(micro_connect_macro)
        SAM_cstrs_entry_list.append(cas.entry(f'micro_connect_macro_{i}', shape=xminus.shape))

    # MACRO INTEGRATION
    X_macro_start = model.variables_dict['x'](V['x_macro', 0])
    X_macro_end = model.variables_dict['x'](V['x_macro', -1])

    # START: connect X0_macro and the x_plus
    ocp_cstr_list.get_constraint_by_name(f'continuity_{tf_regions_indices[0][-1]}').expr = X_macro_start.cat - model.variables_dict['x'](
        Collocation.get_continuity_expression(V, tf_regions_indices[0][-1])).cat

    # Macro RK scheme
    for i in range(d_SAM):
        macro_rk_cstr = cstr_op.Constraint(expr=V['x_macro_coll',i] - (X_macro_start.cat + cas.horzcat(*V['v_macro_coll'])@A_macro[i,:].T),
                                              name=f'macro_rk_cstr_{i}',
                                              cstr_type='eq')
        SAM_cstrs_list.append(macro_rk_cstr)
        SAM_cstrs_entry_list.append(cas.entry(f'macro_rk_cstr_{i}', shape=xminus.shape))



    # END: connect x_plus with end of the reelout
    macro_end_cstr = cstr_op.Constraint(expr= X_macro_end.cat  - (X_macro_start.cat + cas.horzcat(*V['v_macro_coll'])@b_macro),
                                  name='macro_end_cstr',
                                  cstr_type='eq')
    SAM_cstrs_list.append(macro_end_cstr)
    SAM_cstrs_entry_list.append(cas.entry('macro_end_cstr', shape=xminus.shape))

    # connect endpoint of the macro-integration with start of the reelin phase
    macro_connect_reelin = cstr_op.Constraint(expr= X_macro_end.cat - V['x', tf_regions_indices[-2][-1] + 1],
                                  name='macro_connect_reelin',
                                  cstr_type='eq')
    SAM_cstrs_list.append(macro_connect_reelin)
    SAM_cstrs_entry_list.append(cas.entry('macro_connect_reelin', shape=xminus.shape))




    # overwrite the ocp_cstr_struct with new entries
    ocp_cstr_list.append(SAM_cstrs_list)
    ocp_cstr_entry_list = ocp_cstr_struct.entries + SAM_cstrs_entry_list
    ocp_cstr_struct = cas.struct_symMX(ocp_cstr_entry_list)

    return V, P, Xdot_struct, Xdot_fun, ocp_cstr_list, ocp_cstr_struct, Outputs_struct, Outputs_fun, Integral_outputs_struct, Integral_outputs_fun, time_grids, Collocation, Multiple_shooting, global_outputs, global_outputs_fun


