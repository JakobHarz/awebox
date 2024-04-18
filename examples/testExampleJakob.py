import awebox as awe
# from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
from examples.paper_benchmarks import reference_options as ref

from examples.paper_benchmarks import reference_options

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model

options = {}

# options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
# options = set_ampyx_ap2_settings(options)

options = ref.set_reference_options(user='A')
options = ref.set_dual_kite_options(options)

# # number of cycles
N = 5
#
# # indicate desired operation mode
# # here: lift-mode system with pumping-cycle operation, with a one winding trajectory
# options['user_options.trajectory.type'] = 'power_cycle'
# options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = N
#
# # indicate desired environment
# # here: wind velocity profile according to power-law
# options['params.wind.z_ref'] = 100.0
# options['params.wind.power_wind.exp_ref'] = 0.15
# options['user_options.wind.model'] = 'power'
# options['user_options.wind.u_ref'] = 10.
#
# # initializations
# options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
# options['solver.initialization.l_t'] = 1000.0
# options['solver.initialization.inclination_deg'] = 25.0
# options['solver.initialization.cone_deg'] = 20.0
# options['solver.initialization.theta.l_s'] = 100.0
# options['solver.initialization.groundspeed'] = 24.0
# options['solver.initialization.psi0_rad'] = 0.0
# options['solver.initialization.theta.diam_s'] = 4e-3/np.sqrt(2)
# options['solver.initialization.theta.diam_t'] = 4e-3
options['nlp.n_k'] = 15 * N + 30
# options['solver.mu_hippo'] = 1e-4
#
# # no collisions
# options['model.model_bounds.anticollision.include'] = True
# options['model.model_bounds.anticollision.safety_factor'] = 1
#
# # options['solver.cost.u_regularisation.0'] = 1e-1
#
#
options['model.system_bounds.theta.t_f'] = [40, 40 + 15*N]  # [s]
# options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
# options['nlp.phase_fix_reelout'] = 0.7
#
#
# # indicate numerical nlp details
# # here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
# # options['nlp.n_k'] = 40
# options['nlp.collocation.u_param'] = 'zoh'
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'
options['visualization.cosmetics.interpolation.N'] = 1000 # high plotting resolution

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'JakobsMWExample')
trial.build()
trial.optimize()


# %% draw some of the pre-coded plots for analysis
trial.plot(['states', 'controls', 'constraints','quad'])

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3


# %% Plot Solution

print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

plt.subplots(5, 1, sharex=True)
plt.subplot(511)
plt.plot(time, plot_dict['x']['l_t'][0], label = 'Tether Length')
plt.ylabel('[m]')
plt.legend()
plt.grid(True)

plt.subplot(512)
plt.plot(time, plot_dict['x']['dl_t'][0], label = 'Tether Reel-out Speed')
plt.ylabel('[m/s]')
plt.legend()
plt.hlines([20, -15], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)

plt.subplot(513)
plt.plot(time, outputs['aerodynamics']['airspeed2'][0], label = 'Airspeed 1')
plt.plot(time, outputs['aerodynamics']['airspeed2'][0], label = 'Airspeed 2')
plt.ylabel('[m/s]')
plt.legend()
plt.hlines([10, 32], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)

plt.subplot(514)
plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['alpha2'][0], label = 'Angle of Attack 1')
plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['alpha3'][0], label = 'Angle of Attack 2')
# plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['beta1'][1], label = 'Side-Slip Angle 1')
# plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['beta1'][2], label = 'Side-Slip Angle 2')
plt.ylabel('[deg]')
plt.legend()
plt.hlines([9, -6], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)

plt.subplot(515)
plt.plot(time, outputs['local_performance']['tether_force10'][0], label = 'Tether Force Magnitude')
plt.ylabel('[kN]')
plt.xlabel('t [s]')
plt.legend()
plt.hlines([50, 1800], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)
plt.show()

# %%  create a 3D plot of the reelout phase

def latexify():
    import matplotlib
    params_MPL_Tex = {
                'text.usetex': True,
                'font.family': 'serif',
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 10,
                "font.size": 10,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8
              }
    matplotlib.rcParams.update(params_MPL_Tex)
latexify()

# offset = 11
N_cycle = 137
# slice_cuttof = slice(0+offset,N_cycle*(N-1)+1+offset)
# slice_cuttof = slice(0+offset,N_cycle*(N-1)+1+offset)

slice_indeces = np.arange(0,N_cycle*(N-1)+1,N_cycle)

q1_opt = np.vstack(plot_dict['x']['q10'])[:,:]
q2_opt = np.vstack(plot_dict['x']['q21'])[:,:]
q3_opt = np.vstack(plot_dict['x']['q31'])[:,:]

r21_opt = np.vstack(plot_dict['x']['r21'])[:, :]
r31_opt = np.vstack(plot_dict['x']['r31'])[:, :]

q2_strobo = q2_opt[:, slice_indeces]
q3_strobo = q3_opt[:, slice_indeces]

t_plot = plot_dict['time_grids']['ip']
# t_strobo = t_plot[slice_indeces]
# average_x = np.polyval(np.polyfit(t_strobo, q3_strobo[0],3),t_plot)
# average_y = np.polyval(np.polyfit(t_strobo, q3_strobo[1],3),t_plot)
# average_z = np.polyval(np.polyfit(t_strobo, q3_strobo[2],3),t_plot)




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

ax.plot3D(q2_opt[0],q2_opt[1],q2_opt[2],'C0-', alpha=1)
ax.plot3D(q3_opt[0],q3_opt[1],q3_opt[2],'C1-', alpha=1)
# ax.plot3D(average_x,average_y,average_z,'C1--', alpha=1)
# ax.plot3D(q2_strobo[0],q2_strobo[1],q2_strobo[2],'C0o')
ax.plot3D(q3_strobo[0,0],q3_strobo[1,0],q3_strobo[2,0],'C1o')

plot_index = 370
drawKite(q2_opt[:, [plot_index]], r21_opt[:, [plot_index]], 30, 'C0',alpha=0.3)
drawKite(q3_opt[:, [plot_index]], r31_opt[:, [plot_index]], 30, 'C1',alpha=0.3)
ax.plot3D(q1_opt[0,plot_index], q1_opt[1,plot_index], q1_opt[2,plot_index], 'o',color='black',alpha=0.3)
tether10 = np.hstack([q1_opt[:, [plot_index]], np.zeros((3,1))])
tether21 = np.hstack([q2_opt[:, [plot_index]], q1_opt[:, [plot_index]]])
tether31 = np.hstack([q3_opt[:, [plot_index]], q1_opt[:, [plot_index]]])
ax.plot3D(tether21[0], tether21[1], tether21[2], '-',color='black',alpha=0.3)
ax.plot3D(tether31[0], tether31[1], tether31[2], '-',color='black',alpha=0.3)
ax.plot3D(tether10[0], tether10[1], tether10[2], '-',color='black',alpha=0.3)



#set bounds for nice view
meanpos = np.mean(q2_opt[:, [270]],axis=1)

bblenght = 120
ax.set_xlim3d(meanpos[0]-bblenght, meanpos[0]+bblenght)
ax.set_ylim3d(meanpos[1]-bblenght, meanpos[1]+bblenght)
ax.set_zlim3d(meanpos[2]-bblenght, meanpos[2]+bblenght)

ax.quiver(meanpos[0]-bblenght/2,meanpos[1]-bblenght/2,meanpos[2]-bblenght,1,0,0,length=40, color='g')
ax.text(meanpos[0]-bblenght/2, meanpos[1]-bblenght/2, meanpos[2]-bblenght, "Wind",'x', color='g',size=15)

ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')

# ax.legend()
# plt.axis('off')
ax.view_init(elev=18., azim=100)

# plt.legend()
# plt.tight_layout()
plt.savefig('figures/noSAM_3DReelout.pdf')
plt.show()



# %% plot a single state and slowly changing polynomial
import scipy



for state_name,subindex, disp_name, unit in [('q31',2,'p_z','m'),('dq31',0,'v_x','m/s')]:
    state_traj = np.vstack(plot_dict['x'][state_name])
    plt.figure(figsize=(5, 2))
    plt.plot(t_plot,state_traj[subindex],'C0-', label=f'${disp_name}(t)$')
    # plt.plot(t_plot,average_z,'C1--')
    # plt.plot(t_plot[0:N_cycle],q3_opt[2,0:N_cycle],'C1-',alpha=1)
    # plt.plot(t_strobo[0:2],q3_strobo[2,0:2],'C1o')
    # plt.plot(t_strobo[-1],q3_strobo[2,-1],'C1o')
    plt.grid(alpha=0.2)
    plt.xlabel(r'$t$ in s')
    plt.ylabel(f'${disp_name}$ in {unit}')
    plt.tight_layout(pad=0.2)
    plt.legend(loc='lower right')
    plt.savefig(f'figures/noSAM_singleState_{disp_name}.pdf')
    plt.show()