"""
Plot comparison of integrated heat release Q for multiple ROM models vs ground truth.

For each model in MODEL_NAMES, loads the predicted state from:
    ../bundle/{model_name}_output_data/test/{SIM_NAME}/state_pred.zarr

Ground truth is loaded from:
    ../bundle/reference_data/test/{SIM_NAME}/state_full.zarr
    or reconstructed from the training split.

The grid for volume-weighted integration is read from:
    ../bundle/input_data/grid.vtu
"""

import os
import numpy as np
import zarr
import pyvista as pv
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS  –– edit these two lists to match your setup
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAMES = [
    #"sample",       # → ../bundle/sample_output_data/
    "group_2",
    "group_4",
]

SIM_NAME = "sine_f10_A05"   # always in the test split

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BUNDLE_DIR   = "../bundle"
INPUT_DIR    = os.path.join(BUNDLE_DIR, "reference_data")
GRID_PATH    = os.path.join(BUNDLE_DIR, "input_data/grid.vtu")

FEATURES     = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
Q_IDX        = FEATURES.index('mix:Q')

DT           = 2e-3   # s  (same as in the sample model)
DTYPE        = np.float32

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_zarr(path):
    """Load a zarr array to a float32 numpy array."""
    return np.asarray(zarr.open_array(path, mode='r')[:], dtype=DTYPE)


def integrate_Q(Q_cells, grid):
    """
    Volume-weighted integral of Q over all cells.

    Parameters
    ----------
    Q_cells : ndarray, shape (n_cells, n_timesteps)
    grid    : pyvista UnstructuredGrid

    Returns
    -------
    Q_int : ndarray, shape (n_timesteps,)
    """
    tmp     = grid.compute_cell_sizes(length=False, area=True, volume=True)
    weights = np.asarray(tmp.cell_data["Volume"], dtype=DTYPE)   # (n_cells,)
    return weights @ Q_cells   # (n_timesteps,)


def load_state_zarr(folder, is_ground_truth=False):
    """
    is_ground_truth = True:  load state_full.zarr from "{SIM_NAME}/state_full.zarr" folder 
    is_ground_truth = False: load state_pred.zarr from "{SIM_NAME}/state_pred.zarr" folder.
    
    Supports both shapes:
      - (n_cells, n_features, n_timesteps)  — predicted output
      - (n_features, n_cells, n_timesteps)  — some reference formats
    Returns ndarray of shape (n_cells, n_features, n_timesteps).
    """
    if is_ground_truth:
        path = os.path.join(folder, "state_full.zarr")
    else:
        path = os.path.join(folder, "state_pred.zarr")
    
    arr  = load_zarr(path)
    if arr.ndim == 3 and arr.shape[1] == len(FEATURES):
        # already (n_cells, n_features, n_timesteps)
        return arr
    elif arr.ndim == 3 and arr.shape[0] == len(FEATURES):
        # (n_features, n_cells, n_timesteps)  → transpose
        return arr.transpose(1, 0, 2)
    else:
        raise ValueError(f"Unexpected state shape {arr.shape} in {path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRID
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading grid from {GRID_PATH} …")
grid = pv.read(GRID_PATH)
print(f"  Grid cells : {grid.n_cells}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

gt_path = os.path.join(INPUT_DIR, "test", SIM_NAME)
gt_state = load_state_zarr(gt_path, is_ground_truth=True)
print(f"Ground truth state shape : {gt_state.shape}")

Q_gt      = gt_state[:, Q_IDX, :]          # (n_cells, n_timesteps)
Q_gt_int  = integrate_Q(Q_gt, grid)
n_steps   = Q_gt_int.size
time_axis = np.arange(n_steps, dtype=DTYPE) * DT

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EACH MODEL'S PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

model_results = {}   # { model_name: Q_int }

for model_name in MODEL_NAMES:
    pred_folder = os.path.join(
        BUNDLE_DIR, f"{model_name}_output_data", "test", SIM_NAME
    )
    if not os.path.isdir(pred_folder):
        print(f"[WARNING] Folder not found, skipping: {pred_folder}")
        continue

    print(f"Loading predictions for '{model_name}' …")
    pred_state  = load_state_zarr(pred_folder, is_ground_truth=False)
    print(f"  Predicted state shape : {pred_state.shape}")

    Q_pred     = pred_state[:, Q_IDX, :]   # (n_cells, n_timesteps)
    Q_pred_int = integrate_Q(Q_pred, grid)

    # Trim or pad to the same length as ground truth (safety check)
    n_pred = Q_pred_int.size
    if n_pred != n_steps:
        print(f"  [WARNING] Length mismatch: pred={n_pred}, gt={n_steps}. Trimming to min.")
        n_use       = min(n_pred, n_steps)
        Q_pred_int  = Q_pred_int[:n_use]

    model_results[model_name] = Q_pred_int

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(time_axis, Q_gt_int, color='black', linewidth=1.5,
        label='Ground truth', zorder=10)

colors = plt.cm.tab10.colors   # up to 10 distinct colours
for idx, (model_name, Q_int) in enumerate(model_results.items()):
    t_plot = time_axis[:len(Q_int)]
    ax.plot(t_plot, Q_int,
            color=colors[idx % len(colors)],
            linewidth=2.5,
            linestyle='-',
            label=model_name)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Integrated heat release Q  (J/s)', fontsize=12)
ax.set_title(f'Integrated Q — {SIM_NAME}', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.5)
plt.ylim((np.mean(Q_gt_int)-(np.max(Q_gt_int)-np.mean(Q_gt_int))*1.3, np.mean(Q_gt_int)+(np.max(Q_gt_int)-np.mean(Q_gt_int))*1.3))
fig.tight_layout()

out_path = f"example_Q_comparison_{SIM_NAME}.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")
plt.show()