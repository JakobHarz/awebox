import pdb

import casadi.tools as cas

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

import awebox.mdl.lagr_dyn_dir.tools as tools

import awebox.mdl.lagr_dyn_dir.holonomics as holonomic_comp
import awebox.mdl.lagr_dyn_dir.mass as mass_comp
import awebox.mdl.lagr_dyn_dir.energy as energy_comp
import awebox.mdl.lagr_dyn_dir.forces as forces_comp

import numpy as np

def get_dynamics(options, atmos, wind, architecture, system_variables, system_gc, parameters, outputs, wake, scaling):

    parent_map = architecture.parent_map
    number_of_nodes = architecture.number_of_nodes

    # generalized coordinates, velocities and accelerations
    generalized_coordinates = {}
    generalized_coordinates['scaled'] = generate_generalized_coordinates(system_variables['scaled'], system_gc)
    generalized_coordinates['SI'] = generate_generalized_coordinates(system_variables['SI'], system_gc)

    # --------------------------------
    # tether and ground station masses
    # --------------------------------
    outputs = mass_comp.generate_mass_outputs(options, system_variables['SI'], outputs, parameters, architecture, scaling)

    # --------------------------------
    # lagrangian
    # --------------------------------

    outputs = energy_comp.energy_outputs(options, parameters, outputs, system_variables['SI'], architecture)
    e_kinetic = sum(outputs['e_kinetic'][nodes] for nodes in list(outputs['e_kinetic'].keys()))
    e_potential = sum(outputs['e_potential'][nodes] for nodes in list(outputs['e_potential'].keys()))

    holonomic_constraints, outputs, g, gdot, gddot = holonomic_comp.generate_holonomic_constraints(
        architecture,
        outputs,
        system_variables,
        parameters,
        options,
        scaling)

    # lagrangian function
    lag = e_kinetic - e_potential - holonomic_constraints

    # --------------------------------
    # generalized forces in the system
    # --------------------------------

    f_nodes, outputs = forces_comp.generate_f_nodes(options, atmos, wind, wake, system_variables['SI'], outputs, parameters, architecture)
    outputs = forces_comp.generate_tether_moments(options, system_variables['SI'], system_variables['scaled'], holonomic_constraints, outputs,
                                                  architecture)

    cstr_list = cstr_op.MdlConstraintList()

    # --------------------------------
    # translational dynamics
    # --------------------------------

    # lhs of lagrange equations
    dlagr_dqdot = cas.jacobian(lag, generalized_coordinates['scaled']['xgcdot'].cat).T
    dlagr_dqdot_dt = tools.time_derivative(dlagr_dqdot, system_variables['scaled'], architecture, scaling)

    dlagr_dq = cas.jacobian(lag, generalized_coordinates['scaled']['xgc'].cat).T

    xgcdot_scaling_factors = []
    xgc_scaling_factors = []
    for gc_name in generalized_coordinates['scaled']['xgc'].keys():
        scaling_factor_q = scaling['x', gc_name]
        xgc_scaling_factors = cas.vertcat(xgc_scaling_factors, scaling_factor_q)

        scaling_factor_dq = scaling['x', 'd' + gc_name]
        xgcdot_scaling_factors = cas.vertcat(xgcdot_scaling_factors, scaling_factor_dq)

    dlagr_dqdot_si_dt = cas.mtimes(cas.inv(cas.diag(xgcdot_scaling_factors)), dlagr_dqdot_dt)
    dlagr_dq_si = cas.mtimes(cas.inv(cas.diag(xgc_scaling_factors)), dlagr_dq)

    lagrangian_lhs_translation = dlagr_dqdot_si_dt - dlagr_dq_si

    # lagrangian momentum correction
    if options['tether']['use_wound_tether']:
        lagrangian_momentum_correction = 0.
    else:
        lagrangian_momentum_correction = momentum_correction(options, generalized_coordinates, system_variables,
                                                             outputs, architecture, scaling)

    # rhs of lagrange equations
    lagrangian_rhs_translation = cas.vertcat(
        *[f_nodes['f' + str(n) + str(parent_map[n])] for n in range(1, number_of_nodes)]) + \
                                 lagrangian_momentum_correction

    # scaling
    node_masses_scaling = mass_comp.generate_m_nodes_scaling(options, system_variables['SI'], parameters, architecture, scaling)
    forces_scaling = options['scaling']['z']['f_aero'] * (node_masses_scaling / parameters['theta0', 'geometry', 'm_k'])

    dynamics_translation_si = (lagrangian_lhs_translation - lagrangian_rhs_translation)
    dynamics_translation_scaled = cas.mtimes(cas.inv(cas.diag(forces_scaling)), dynamics_translation_si)
    dynamics_translation_cstr = cstr_op.Constraint(expr=dynamics_translation_scaled,
                                                   cstr_type='eq',
                                                   name='dynamics_translation')
    cstr_list.append(dynamics_translation_cstr)


    # ---------------------------
    # holonomic constraints
    # ---------------------------

    lagrangian_lhs_constraints = holonomic_comp.get_constraint_lhs(g, gdot, gddot, parameters)
    lagrangian_rhs_constraints = np.zeros(g.shape)
    holonomic_scaling = holonomic_comp.generate_holonomic_scaling(options, architecture, scaling, system_variables['SI'], parameters)

    print_op.warn_about_temporary_functionality_alteration()
    dynamics_constraints_si = (lagrangian_lhs_constraints - lagrangian_rhs_constraints)
    dynamics_constraints_scaled = cas.mtimes(cas.inv(cas.diag(holonomic_scaling)), dynamics_constraints_si)

    dynamics_constraint_cstr = cstr_op.Constraint(expr=dynamics_constraints_scaled,
                                                cstr_type='eq',
                                                name='dynamics_constraint')
    cstr_list.append(dynamics_constraint_cstr)


    # --------------------------------
    # rotational dynamics
    # --------------------------------

    kite_has_6dof = (int(options['kite_dof']) == 6)
    if kite_has_6dof:
        rotation_dynamics_cstr, outputs = generate_rotational_dynamics(options, system_variables, f_nodes, parameters, outputs, architecture)
        cstr_list.append(rotation_dynamics_cstr)

    # --------------------------------
    # trivial kinematics
    # --------------------------------

    for name in system_variables['SI']['xdot'].keys():

        name_in_x = name in system_variables['SI']['x'].keys()
        name_in_u = name in system_variables['SI']['u'].keys()

        if name_in_x or name_in_u:
            if name_in_x:
                si_diff = system_variables['SI']['xdot'][name] - system_variables['SI']['x'][name]
                scaled_diff = struct_op.var_si_to_scaled('x', name, si_diff, scaling)

            elif name_in_u:
                si_diff = system_variables['SI']['xdot'][name] - system_variables['SI']['u'][name]
                scaled_diff = struct_op.var_si_to_scaled('u', name, si_diff, scaling)

            else:
                message = 'something went wrong when defining trivial constraints'
                print_op.log_and_raise_error(message)

            trivial_dyn = cas.vertcat(*[scaled_diff])

            trivial_dyn_cstr = cstr_op.Constraint(expr=trivial_dyn,
                                                  cstr_type='eq',
                                                  name='trivial_' + name)
            cstr_list.append(trivial_dyn_cstr)

    return cstr_list, outputs



def momentum_correction(options, generalized_coordinates, system_variables, outputs, architecture, scaling):
    """Compute momentum correction for translational lagrangian dynamics of an open system.
    Here the system is "open" because the main tether mass is changing in time. During reel-out,
    momentum is injected in the system, and during reel-in, momentum is extracted.
    It is assumed that the tether mass is concentrated at the main tether node.

    See "Lagrangian Dynamics for Open Systems", R. L. Greene and J. J. Matese 1981, Eur. J. Phys. 2, 103.

    @return: lagrangian_momentum_correction - correction term that can directly be added to rhs of transl_dyn
    """

    # initialize
    xgcdot = generalized_coordinates['scaled']['xgcdot'].cat
    lagrangian_momentum_correction = cas.DM.zeros(xgcdot.shape)

    use_wound_tether = options['tether']['use_wound_tether']
    if not use_wound_tether:

        for node in range(1, architecture.number_of_nodes):
            label = str(node) + str(architecture.parent_map[node])
            mass = outputs['masses']['m_tether{}'.format(node)]
            mass_flow = tools.time_derivative(mass, system_variables['scaled'], architecture, scaling)

            # velocity of the mass particles leaving the system
            velocity = system_variables['SI']['x']['dq' + label]
            # see formula in reference
            lagrangian_momentum_correction += mass_flow * cas.mtimes(velocity.T, cas.jacobian(velocity,
                                                                                              xgcdot)).T

    return lagrangian_momentum_correction


def generate_rotational_dynamics(options, variables, f_nodes, parameters, outputs, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    j_inertia = parameters['theta0', 'geometry', 'j']

    x = variables['SI']['x']
    xdot = variables['SI']['xdot']

    cstr_list = cstr_op.MdlConstraintList()

    for kite in kite_nodes:
        parent = parent_map[kite]
        moment = f_nodes['m' + str(kite) + str(parent)]

        rlocal = cas.reshape(x['r' + str(kite) + str(parent)], (3, 3))
        drlocal = cas.reshape(xdot['dr' + str(kite) + str(parent)], (3, 3))

        omega = x['omega' + str(kite) + str(parent)]
        omega_skew = vect_op.skew(omega)
        domega = xdot['domega' + str(kite) + str(parent)]

        tether_moment = outputs['tether_moments']['n{}{}'.format(kite, parent)]

        # moment = J dot(omega) + omega x (J omega) + [tether moment which is zero if holonomic constraints do not depend on omega]
        J_dot_omega = cas.mtimes(j_inertia, domega)
        omega_cross_J_omega = vect_op.cross(omega, cas.mtimes(j_inertia, omega))
        omega_derivative = moment - (J_dot_omega + omega_cross_J_omega + tether_moment)
        m_scale = options['scaling']['z']['m_aero']
        rotational_2nd_law = omega_derivative / m_scale

        rotation_dynamics_cstr = cstr_op.Constraint(expr=rotational_2nd_law,
                                                    name='rotation_dynamics' + str(kite),
                                                    cstr_type='eq')
        cstr_list.append(rotation_dynamics_cstr)

        # Rdot = R omega_skew -> R ( kappa/2 (I - R.T R) + omega_skew )
        baumgarte = parameters['theta0', 'kappa_r']
        orthonormality = baumgarte / 2. * (cas.DM_eye(3) - cas.mtimes(rlocal.T, rlocal))
        ref_frame_deriv_matrix = drlocal - (cas.mtimes(rlocal, orthonormality + omega_skew))
        ref_frame_derivative = cas.reshape(ref_frame_deriv_matrix, (9, 1))

        ortho_cstr = cstr_op.Constraint(expr=ref_frame_derivative,
                                        name='ref_frame_deriv' + str(kite),
                                        cstr_type='eq')
        cstr_list.append(ortho_cstr)

    return cstr_list, outputs


def generate_generalized_coordinates(system_variables, system_gc):

    struct_flag = (not isinstance(system_variables, dict)) and ('[x,q10,0]' in system_variables.labels())

    if struct_flag == 1:
        generalized_coordinates = {}
        generalized_coordinates['xgc'] = cas.struct_SX(
            [cas.entry(name, expr=system_variables['x', name]) for name in system_gc])
        generalized_coordinates['xgcdot'] = cas.struct_SX(
            [cas.entry('d' + name, expr=system_variables['x', 'd' + name])
             for name in system_gc])
        # generalized_coordinates['xgcddot'] = cas.struct_SX(
        #     [cas.entry('dd' + name, expr=system_variables['xdot', 'dd' + name])
        #      for name in system_gc])
    else:
        generalized_coordinates = {}
        generalized_coordinates['xgc'] = cas.struct_SX(
            [cas.entry(name, expr=system_variables['x'][name]) for name in system_gc])
        generalized_coordinates['xgcdot'] = cas.struct_SX(
            [cas.entry('d' + name, expr=system_variables['x']['d' + name])
             for name in system_gc])
        # generalized_coordinates['xgcddot'] = cas.struct_SX(
        #     [cas.entry('dd' + name, expr=system_variables['xdot']['dd' + name])
        #      for name in system_gc])

    return generalized_coordinates