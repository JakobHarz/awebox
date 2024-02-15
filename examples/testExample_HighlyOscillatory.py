#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Rachel Leuthold
"""

import awebox as awe
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np


# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}
# options['user_options.system_model.architecture'] = {1:0}
# options = set_ampyx_ap2_settings(options)

from examples.paper_benchmarks import reference_options as ref
options = ref.set_reference_options(user='A')
options = ref.set_dual_kite_options(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 3

# indicate desired environment
# here: wind velocity profile according to power-law
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.


# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
n_k = 8 * options['user_options.trajectory.lift_mode.windings']
options['nlp.n_k'] = n_k
options['nlp.phase_fix_reelout'] = (options['user_options.trajectory.lift_mode.windings'] - 1)/options['user_options.trajectory.lift_mode.windings']
# options['nlp.phase_fix_reelout'] = 0.7
options['nlp.useAverageModel'] = True
options['nlp.cost.output_quadrature'] = False # use enery as a state, works better with SAM
options['nlp.SAM_MaInt_type']  = 'radau'
options['nlp.N_SAM'] = 10
options['nlp.d_SAM'] = options['user_options.trajectory.lift_mode.windings'] - 1
options['nlp.SAM_ADAtype'] = 'BD'
options['model.system_bounds.theta.t_f'] = [10, 40]  # [s]


options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
options['solver.linear_solver'] = 'ma57'

options['visualization.cosmetics.interpolation.N'] = 1000 # high plotting resolution
options['visualization.cosmetics.plot_bounds'] = True # high plotting resolution

options['solver.callback'] = True

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'Ampyx_AP2')
trial.build()
trial.optimize()

# draw some of the pre-coded plots for analysis

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3

print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

# # %%
# trial.plot(['states'])
# plt.gcf().tight_layout()

# %% Post-Processing

import casadi as ca
from awebox.ocp.discretization_averageModel import OthorgonalCollocation
from awebox.tools.struct_operations import calculate_SAM_regions

d_SAM = options['nlp.d_SAM']
N_SAM = options['nlp.N_SAM']

# Vopt = trial.optimization.V_opt
Vinit = trial.optimization.V_init
Vopt = trial.optimization.V_init

time_grid = trial.nlp.time_grids['x'](Vopt['theta','t_f']).full().flatten()


regions_indeces = calculate_SAM_regions(trial.nlp.options)
strobo_indeces = [region_indeces[0] for region_indeces in regions_indeces[1:]]
model = trial.model
macroIntegrator = OthorgonalCollocation(np.array(ca.collocation_points(d_SAM,options['nlp.SAM_MaInt_type'])))




t_f_opt = Vopt['theta','t_f']

X_macro_coll_opt = np.hstack(Vopt['x_macro_coll',:])
X_macro_startend_opt = np.hstack(Vopt['x_macro',:])
X_macro_coll = np.hstack([X_macro_startend_opt[:,[0]],X_macro_coll_opt,X_macro_startend_opt[:,[1]]])

macro_state_poly_eval_f = macroIntegrator.getPolyEvalFunction(trial.model.variables_dict['x'].shape,includeZero=True, fixedValues=[Vopt['x_macro',0],*Vopt['x_macro_coll',:]])
tau_eval = np.linspace(0,1,100)
X_macro_coll_eval = model.variables_dict['x'].repeated(macro_state_poly_eval_f.map(tau_eval.size)(tau_eval))

q10_opt_plot = np.vstack(plot_dict['x']['q10']) # the 'continous trajectory' of the connecting point
q21_opt_plot = np.vstack(plot_dict['x']['q21']) # the 'continous trajectory' of the kite
q31_opt_plot = np.vstack(plot_dict['x']['q31']) # the 'continous trajectory' of the kite
q10_opt = np.hstack(Vopt['x', :, 'q10']) # the actual optimal trajectory variables
q21_opt = np.hstack(Vopt['x', :, 'q21']) # the actual optimal trajectory variables
q31_opt = np.hstack(Vopt['x',:,'q31']) # the actual optimal trajectory variables

q21_opt_average = np.hstack(X_macro_coll_eval[:, 'q21',:])
q21_opt_average_strobo = np.hstack([*Vopt['x_macro',[0],'q21'],
                                    *Vopt['x_macro_coll',:,'q21'],
                                    *Vopt['x_macro',[1],'q21']])

q31_opt_average = np.hstack(X_macro_coll_eval[:, 'q31',:])
q31_opt_average_strobo = np.hstack([*Vopt['x_macro',[0],'q31'],
                                    *Vopt['x_macro_coll',:,'q31'],
                                    *Vopt['x_macro',[1],'q31']])


# reconstruct the full trajectory
coeff_fun = trial.visualization.plot_dict['Collocation']._Collocation__coeff_fun
zs_micro = []
for i in range(1,d_SAM+1):
    # interpolate the micro-collocation polynomial
    z_i = []
    for j in regions_indeces[i]:
        x_micro = Vopt['x', j] # start point of the collocaiton interval
        x_coll_micro = Vopt['coll_var', j, :, 'x'] # the collocation points
        poly_vars = ca.horzcat(x_micro, *x_coll_micro)

        # evaluate the micro collocation polynomial at 10 points
        for tau in np.linspace(0,1,10):
            val = poly_vars @ coeff_fun(tau)
            z_i.append(val)

    zs_micro.append(ca.horzcat(*z_i))
    # z_i = ca.horzcat()
    # zs_micro.append(z_i)
z_interpol_f = macroIntegrator.getPolyEvalFunction(shape=zs_micro[0].shape,includeZero=False, fixedValues=zs_micro)

strobos_eval = np.arange(N_SAM) + {'BD':1,'CD':0.5,'FD':0.0}[options['nlp.SAM_ADAtype']]
strobos_eval = strobos_eval*1/N_SAM

z_interpol_list = [z_interpol_f(strobo_eval) for strobo_eval in strobos_eval]
x_reconstructed = model.variables_dict['x'].repeated(ca.horzcat(*z_interpol_list))

q21_reconstruct = ca.horzcat(*x_reconstructed[:,'q21']).full()


# %% Plot state trajectories

plt.figure()

plot_states = ['q21','dq21','l_t','e']
for index, state_name in enumerate(plot_states):
    plt.subplot(2,2,index+1)
    state_traj = np.hstack(Vopt['x', :, state_name]).T
    plt.plot(time_grid,state_traj,label=state_name)

    # add phase switches
    plt.axvline(x=time_grid[regions_indeces[1][0]],color='k',linestyle='--')
    plt.axvline(x=time_grid[regions_indeces[-1][0]],color='k',linestyle='--')

    for region_indeces in regions_indeces[1:-1]:
        plt.axvline(x=time_grid[region_indeces[0]],color='b',linestyle='--')

    plt.xlabel('time [s]')

    plt.legend()
plt.tight_layout()
plt.show()

# %% plot the results
import matplotlib
import mpl_toolkits.mplot3d as a3

plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

_raw_vertices = np.array([[-1.2, 0, -0.4, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0]])
_raw_vertices = _raw_vertices - np.mean(_raw_vertices,axis=1).reshape((3,1))

def drawKite(pos, rot, wingspan, color='C0', alpha=1):

    rot = np.reshape(rot, (3,3)).T

    vtx = _raw_vertices * wingspan/2 # -np.array([[0.5], [0], [0]]) * sizeKite
    vtx = rot @ vtx + pos
    tri = a3.art3d.Poly3DCollection([vtx.T])
    tri.set_color(matplotlib.colors.to_rgba(color,alpha-0.1))
    tri.set_edgecolor(matplotlib.colors.to_rgba(color,alpha))
    # tri.set_alpha(alpha)
    # tri.set_edgealpha(alpha)
    ax.add_collection3d(tri)

nk_reelout = int(options['nlp.n_k']*options['nlp.phase_fix_reelout'])
nk_cut = round(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])




# ax.plot3D(q10_opt_plot[0, :], q10_opt_plot[1, :], q10_opt_plot[2, :], 'C0-', alpha=0.3)
# ax.plot3D(q21_opt_plot[0, :], q21_opt_plot[1, :], q21_opt_plot[2, :], 'C0-', alpha=0.3)
# ax.plot3D(q31_opt_plot[0, :], q31_opt_plot[1, :], q31_opt_plot[2, :], 'C0-', alpha=0.3)

for region_indeces, color in zip(regions_indeces[1:-1],[f'C{i}' for i in range(20)]):
    # ax.plot3D(q10_opt[0, region_indeces], q10_opt[1, region_indeces], q10_opt[2, region_indeces], '.', color = color, alpha=1, markersize=3)
    ax.plot3D(q21_opt[0, region_indeces], q21_opt[1, region_indeces], q21_opt[2, region_indeces], 'o-', color = color, alpha=1, markersize=3)
    # ax.plot3D(q31_opt[0, region_indeces], q31_opt[1, region_indeces], q31_opt[2, region_indeces], 's', color = color, alpha=1, markersize=2)

ax.plot3D(q21_opt[0, regions_indeces[-1]], q21_opt[1, regions_indeces[-1]], q21_opt[2, regions_indeces[-1]], 'C0.', alpha=0.3, markersize=6)
# ax.plot3D(q31_opt[0, regions_indeces[-1]], q31_opt[1, regions_indeces[-1]], q31_opt[2, regions_indeces[-1]], 'C0.', alpha=0.3, markersize=3)

ax.plot3D(q21_opt_average[0], q21_opt_average[1],q21_opt_average[2], 'b-', alpha=1)
ax.plot3D(q21_opt_average_strobo[0], q21_opt_average_strobo[1],q21_opt_average_strobo[2], 'bo', alpha=1)
# ax.plot3D(q31_opt_average[0], q31_opt_average[1],q31_opt_average[2], 'b-', alpha=1)
# ax.plot3D(q31_opt_average_strobo[0], q31_opt_average_strobo[1],q31_opt_average_strobo[2], 'bs', alpha=1)

# reconstructed trajectory
# ax.plot3D(q21_reconstruct[0], q21_reconstruct[1],q21_reconstruct[2], 'C1-', alpha=0.5)


# average trajectory
# ax.plot3D(X_macro_coll[0,:], X_macro_coll[1,:], X_macro_coll[2,:], 'b.-', alpha=1, markersize=6)


# plot important points
# ax.plot3D(q1_opt[0,0], q1_opt[1,0], q1_opt[2,0], 'C1o', alpha=1)
# ax.plot3D(q1_opt[0,strobo_indeces[-1]], q1_opt[1,strobo_indeces[-1]], q1_opt[2,strobo_indeces[-1]], 'C1o', alpha=1)



tether10 = np.hstack([q21_opt_plot[:, [-1]], np.zeros((3, 1))])
# tether21 = np.hstack([q2_opt[:, [-1]], q1_opt[:, [-1]]])
# tether31 = np.hstack([q3_opt[:, [-1]], q1_opt[:, [-1]]])
# ax.plot3D(tether21[0], tether21[1], tether21[2], '-',color='black')
# ax.plot3D(tether31[0], tether31[1], tether31[2], '-',color='black')
# ax.plot3D(tether10[0], tether10[1], tether10[2], '-',color='black')


#set bounds for nice view
meanpos = np.mean(q21_opt_plot[:], axis=1)

# bblenght = np.max(np.abs(q21_opt_plot - meanpos.reshape(3, 1)))
# ax.set_xlim3d(meanpos[0]-bblenght, meanpos[0]+bblenght)
# ax.set_ylim3d(meanpos[1]-bblenght, meanpos[1]+bblenght)
# ax.set_zlim3d(meanpos[2]-bblenght, meanpos[2]+bblenght)

# ax.quiver(meanpos[0]-bblenght/2,meanpos[1]-bblenght/2,meanpos[2]-bblenght,1,0,0,length=40, color='g')
# ax.text(meanpos[0]-bblenght/2, meanpos[1]-bblenght/2, meanpos[2]-bblenght, "Wind",'x', color='g',size=15)

ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')

# ax.legend()
# plt.axis('off')
ax.view_init(elev=23., azim=-45)

# plt.legend()
plt.tight_layout()
# plt.savefig('3DReelout.pdf')
plt.show()
