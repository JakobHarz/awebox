import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# %% Latexify the plots
def latexify():
    import matplotlib
    params_MPL_Tex = {
        'text.usetex': True,
        'font.family': 'serif',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    matplotlib.rcParams.update(params_MPL_Tex)


latexify()

# %% Load Data
filepath = '_export/20260714_1318_AWE_SAM_DUALKITE_N10_d3.npz'
data = np.load(filepath, allow_pickle=True)

data_SAM = data['SAM'].item()
data_REC = data['REC'].item()

WITH_MPC = 'MPC' in data.keys()
if WITH_MPC:
    data_MPC = data['MPC'].item()

d = data_SAM['d']
N = data_SAM['N']


# %% Compute and compare the produced powers
power_SAM = data_SAM['x']['e'][0][-1] / data_SAM['time'][-1]
power_REC = data_REC['x']['e'][0][-1] / data_REC['time'][-1]
if WITH_MPC:
    power_MPC = data_MPC['x']['e'][0][-1] / data_MPC['time'][-1]

print(f'Power produced by SAM: {power_SAM / 1000} kW')
print(f'Power produced by REC: {power_REC / 1000} kW')
if WITH_MPC:
    print(f'Power produced by MPC: {power_MPC / 1000} kW')


# %% draw kite function
_raw_vertices = np.array([[-1.2, 0, -0.4, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0]])
_raw_vertices = _raw_vertices - np.mean(_raw_vertices, axis=1).reshape((3, 1))


def drawKite(pos, rot, wingspan, color='C0', alpha=1):
    rot = np.reshape(rot, (3, 3)).T

    vtx = _raw_vertices * wingspan / 2  # -np.array([[0.5], [0], [0]]) * sizeKite

    vtx = rot @ vtx + pos
    tri = a3.art3d.Poly3DCollection([vtx.T])
    tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
    tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
    plt.gca().add_collection3d(tri)


def drawPlane(pos, rot, wingspan, color='C0', alpha=1, twoDimensions=False):
    rot = np.reshape(rot, (3, 3)).T @ np.diag([-1, 1, 1])

    normalization = 32.0 / (float(wingspan))  # size of triangles is 32, normalize with half of the wingspan
    y_offset = -16
    x_offset = -3.5 / 2
    z_offset = 0

    x = {}  # dictionary of x positions of the vertices
    y = {}  # dictionary of y positions of the vertices
    z = {}  # dictionary of z positions of the vertices

    x['fuselage'] = [0, 3.5, 8, 9, 9.5, 9.75, 9.5, 9, 8, 3.5, 0, 2, -6, -8, -9, -8, -6, -2]
    y['fuselage'] = [15, 15, 15, 15.25, 15.75, 16, 16.25, 16.75, 17, 17, 17, 17, 16.75, 16.5, 16, 15.5, 15.25, 15]
    z['fuselage'] = [-0.01] * 18

    x['wing'] = [0, 2, 3, 3.5, 3.5, 3, 2, 0]
    y['wing'] = [0, 0, 8, 13, 19, 24, 32, 32]
    z['wing'] = [0, 0, 0, 0, 0, 0, 0, 0]

    x['elev'] = [-8, -8, -9, -9]
    y['elev'] = [13.5, 18.5, 18.5, 13.5]
    z['elev'] = [-0.01] * 4

    for part in x.keys():
        x[part] = [(xx + x_offset) / normalization for xx in x[part]]
        y[part] = [(yy + y_offset) / normalization for yy in y[part]]
        z[part] = [(zz + z_offset) / normalization for zz in z[part]]
        verts = np.vstack([x[part], y[part], z[part]])
        verts_rot = rot @ verts + pos

        if part != 'wing':
            zorder = -1
        else:
            zorder = 0

        if not twoDimensions:
            # plot in three dimensions
            tri = a3.art3d.Poly3DCollection([verts_rot.T], linewidth=0.1, zorder=zorder)
            tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
            tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
            plt.gca().add_collection3d(tri)
        else:
            # plot in two dimensions
            polygon = Polygon(verts_rot[1:3, :].T, facecolor=color)
            polygon.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
            polygon.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
            plt.gca().add_patch(polygon)


# %% 3D PLOT: Figure 5 equivalent, both aircraft (kite 2 -> q21, kite 3 -> q31)
import mpl_toolkits.mplot3d as a3
import matplotlib

kite_states = {21: {'q': 'q21', 'r': 'r21'}, 31: {'q': 'q31', 'r': 'r31'}}

q21_REC = data_REC['x']['q21']
q31_REC = data_REC['x']['q31']

q21_opt = data_SAM['x']['q21']
q31_opt = data_SAM['x']['q31']
ip_regions_SAM = data_SAM['regions']

if WITH_MPC:
    q21_MPC = data_MPC['x']['q21']
    q31_MPC = data_MPC['x']['q31']

Q21_SAM = data_SAM['X']['q21']
Q31_SAM = data_SAM['X']['q31']
time_X = data_SAM['time_X']

for figure_type in ['SAM', 'REC', 'MPC']:
    plt.figure(figsize=(5.5, 4.5))
    ax = plt.axes(projection='3d')

    if figure_type == 'SAM':

        for q_opt, Q_SAM, color_reelin, color_avg, color_micro in [
            (q21_opt, Q21_SAM, 'C0', 'C1', 'C2'),
            (q31_opt, Q31_SAM, 'C0', 'C1', 'C2'),
        ]:
            # reel in
            ax.plot3D(q_opt[0][np.where(ip_regions_SAM == d)],
                      q_opt[1][np.where(ip_regions_SAM == d)],
                      q_opt[2][np.where(ip_regions_SAM == d)]
                      , '-', color=color_reelin,
                      alpha=1, markersize=3)

            # average
            ax.plot3D(Q_SAM[0], Q_SAM[1], Q_SAM[2], '-', color=color_avg, alpha=1)
            ax.plot3D(Q_SAM[0][0], Q_SAM[1][0], Q_SAM[2][0], '.', color=color_avg, alpha=1)
            ax.plot3D(Q_SAM[0][-1], Q_SAM[1][-1], Q_SAM[2][-1], '.', color=color_avg, alpha=1)

            for region_index in np.arange(0, d + 1):
                color = color_reelin if region_index == d else color_micro

                ax.plot3D(q_opt[0][np.where(ip_regions_SAM == region_index)],
                          q_opt[1][np.where(ip_regions_SAM == region_index)],
                          q_opt[2][np.where(ip_regions_SAM == region_index)]
                          , '-', color=color,
                              alpha=1, markersize=3)

    if figure_type == 'REC':
        ax.plot3D(q21_REC[0], q21_REC[1], q21_REC[2], 'C0-', alpha=0.5)
        ax.plot3D(q31_REC[0], q31_REC[1], q31_REC[2], 'C0-', alpha=0.5)

    if figure_type == 'MPC':
        ax.plot3D(q21_REC[0], q21_REC[1], q21_REC[2], 'C0-', alpha=0.2)
        ax.plot3D(q31_REC[0], q31_REC[1], q31_REC[2], 'C0-', alpha=0.2)

        final_index = q21_MPC[0].size // 4 + 20
        section_to_plot = slice(0, final_index)
        section_to_plot_mpc = slice(final_index, final_index + 30)

        for q_MPC, color_line, color_kite in [(q21_MPC, 'r', 'k'), (q31_MPC, 'r', 'k')]:
            ax.plot3D(q_MPC[0][section_to_plot], q_MPC[1][section_to_plot], q_MPC[2][section_to_plot],
                      color_line + '-', alpha=0.75)
            ax.plot3D(q_MPC[0][section_to_plot_mpc], q_MPC[1][section_to_plot_mpc], q_MPC[2][section_to_plot_mpc],
                      color_line + '--', alpha=0.75)

        # plot a kite at the end of the section, for both aircraft
        for kite_name, r_name, q_name in [('21', 'r21', 'q21'), ('31', 'r31', 'q31')]:
            r_k = np.vstack([data_MPC['x'][r_name][i][final_index] for i in range(data_MPC['x'][r_name].__len__())])
            q_k = np.vstack([data_MPC['x'][q_name][i][final_index] for i in range(data_MPC['x'][q_name].__len__())])
            drawPlane(q_k, r_k, wingspan=30, color='k')
            # draw a straight tether to the branch node q10
            q10_k = np.vstack([data_MPC['x']['q10'][i][final_index] for i in range(data_MPC['x']['q10'].__len__())])
            ax.plot3D([float(q10_k[0]), float(q_k[0])], [float(q10_k[1]), float(q_k[1])],
                      [float(q10_k[2]), float(q_k[2])], 'k-', alpha=0.5, linewidth=1)
        # main tether to the ground
        q10_k = np.vstack([data_MPC['x']['q10'][i][final_index] for i in range(data_MPC['x']['q10'].__len__())])
        ax.plot3D([0, float(q10_k[0])], [0, float(q10_k[1])], [0, float(q10_k[2])], 'k-', alpha=0.5, linewidth=1)

    # set bounds for nice view (based on both aircraft's reconstructed trajectories)
    q_REC_all = np.vstack([q21_REC[0], q21_REC[1], q21_REC[2], q31_REC[0], q31_REC[1], q31_REC[2]]).reshape(2, 3, -1)
    q_REC_all = np.hstack([q_REC_all[0], q_REC_all[1]])
    meanpos = np.mean(q_REC_all, axis=1) + np.array([0, -50, 30])

    bblenght = np.max(np.abs(q_REC_all - meanpos.reshape(3, 1))) / 1.5

    # ticks on the axis in 100m steps
    ax.set_xticks(np.arange(500, 1500, 100))
    ax.set_yticks(np.arange(-1000, 1000, 100))
    ax.set_zticks(np.arange(-1000, 1000, 100))

    ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
    ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
    ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)
    ax.set_box_aspect([1, 1, 1])

    pos_wind_arrow = np.array([meanpos[0] - bblenght / 2, meanpos[1] - bblenght, meanpos[2] + bblenght * 0.7])
    ax.quiver(pos_wind_arrow[0], pos_wind_arrow[1], pos_wind_arrow[2], 1, 0, 0, length=100, color='k')
    ax.text(pos_wind_arrow[0], pos_wind_arrow[1], pos_wind_arrow[2], "Wind", 'x', color='k', size=12)

    ax.set_xlabel(r'$x$ in m')
    ax.set_ylabel(r'$y$ in m')
    ax.set_zlabel(r'$z$ in m')

    ax.view_init(elev=3., azim=142)

    plt.tight_layout()
    plt.savefig(f'figures/3DReelout_DUALKITE_{figure_type}.pdf')
    plt.show()
