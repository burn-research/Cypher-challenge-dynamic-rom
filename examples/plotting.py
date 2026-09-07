#%%
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_code_submission"))
from utils import list_training_simulations, load_simulation

sim_name = 'sineSweep_f1_f80_A04'
path_train = '../bundle/data_directory/train'
state_train, phi_train = load_simulation(f"{path_train}/{sim_name}")
print(f"Loaded '{sim_name}': state {state_train.shape}, phi {phi_train.shape}")


#%%
import pyvista as pv
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_snapshot_hor(mesh, z, normal, origin, axis, feature, cmap='viridis', filename=''):
    mesh['z'] = z
    plane = mesh.ctp().slice(normal=normal, origin=origin, generate_triangles=True)

    vmin = np.min(plane['z'].min())
    vmax = np.max(plane['z'].max())

    x = plane.points*1000
    tri = plane.faces.reshape((-1, 4))[:, 1:]

    fig, ax = plt.subplots(figsize=(4, 2.3))
    levels = 64

    axis_labels = ['x', 'y', 'z']
    fig.subplots_adjust(bottom=0., top=.925, left=0, right=1., wspace=0.05, hspace=0.0)

    cs = ax.tricontourf(x[:, axis[1]], x[:, axis[0]], tri, plane['z'], cmap=cmap,
                        levels=levels, vmin=vmin, vmax=vmax)

    ax.set_aspect('equal')
    ax.set_xlabel(f'{axis_labels[axis[1]]} (mm)')
    ax.set_ylabel(f'{axis_labels[axis[0]]} (mm)')

    ax_bounds = ax.get_position().bounds
    cb_ax = fig.add_axes([ax_bounds[0], 0.95, ax_bounds[2], 0.05,])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cb_ax, orientation='horizontal', label=feature,
                        ticklocation='top')
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 4))
    cb_ax.yaxis.set_offset_position('left')

    if filename != '':
        fig.savefig(filename, transparent=False, dpi=600, bbox_inches='tight')

    plt.show()

grid = pv.read('../bundle/data_directory/grid.vtu')

n_cells = state_train.shape[0]
n_features = state_train.shape[1]
n_pred = state_train.shape[2]
features = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
dt = 2e-3
time = np.arange(0, n_pred, dtype=np.float32)*dt

f_plot = 'OH'
i_plot = features.index(f_plot)

i_test = 200

plot_snapshot_hor(grid, state_train[:, i_plot, i_test],
                  (0, 1, 0), (0, 0, 0), (0, 2), features[i_plot], cmap='inferno')

#%%

f_plot = 'T'
i_plot = features.index(f_plot)

grid["field"] = state_train[:, i_plot, 0]

# Configure vertical colorbar outside domain
sargs = dict(
    title=f"{f_plot}",
    vertical=True,
    position_x=0.85,
    position_y=0.15,
    height=0.7,
    width=0.08,
    title_font_size=12,
    label_font_size=10,
    shadow=False,
    n_labels=5,
)

plotter = pv.Plotter(window_size=[1200, 800], off_screen=True)
plotter.add_mesh(
    grid,
    scalars="field",
    cmap="viridis",
    clim=[state_train[:, i_plot, :].min(), state_train[:, i_plot, :].max()],
    scalar_bar_args=sargs,
)
plotter.add_text("t = 0.000 s", position="upper_left", font_size=12, name="time_label")

plotter.view_xz()                       # or view_xz() / view_yz() depending on your plane
plotter.enable_parallel_projection()    # no perspective distortion
plotter.reset_camera()
plotter.camera.zoom(0.85)               # leave margin so domain does not overlap vertical colorbar

plotter.open_gif("animation.gif", fps=15)

step = 5
for t in range(time.size//step):
    grid["field"] = state_train[:, i_plot, t*step]
    plotter.add_text(f"t = {time[t*step]:.3f} s", position="upper_left", font_size=12, name="time_label")
    plotter.write_frame()

plotter.close()
print("Saved animation.")
# %%

def integrate_Q(Q, grid):
    tmp = grid.compute_cell_sizes(
        length=False,
        area=True,
        volume=True
    )

    weights = tmp.cell_data["Volume"]
    Q_integrated = weights @ Q

    return Q_integrated

Q_train = state_train[:, features.index('mix:Q'), :]
Q_train_int = integrate_Q(Q_train, grid)

plt.plot(time, Q_train_int, label='Training')

plt.xlabel('Time step')
plt.ylabel('Integrated Q')
plt.legend()    
plt.show()
