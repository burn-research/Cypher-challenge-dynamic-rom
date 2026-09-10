#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions to load the CYPHER 2026 dynamic-ROM dataset.
Feel free to modify or ignore this file entirely -- it is only provided as
a convenience starting point for your own submission.
"""

import os
import random
import numpy as np
import zarr

def set_global_seed(seed: int):
    """Set seed for Python, NumPy, PyTorch and TensorFlow (if installed),
    for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def list_training_simulations(train_folder):
    """Return the sorted list of simulation sub-folder names in the training
    data folder, e.g. ['sweep_A0.2', 'sweep_A0.4']."""
    return sorted(
        d for d in os.listdir(train_folder)
        if os.path.isdir(os.path.join(train_folder, d))
    )


def load_simulation(sim_folder):
    """Load a full training simulation.

    Parameters
    ----------
    sim_folder : str
        Path to a simulation folder containing state.zarr and phi.zarr.

    Returns
    -------
    state : ndarray, shape (n_features * n_cells, n_timesteps)
    phi   : ndarray, shape (n_timesteps,)
    """
    state = np.asarray(
        zarr.open_array(os.path.join(sim_folder, 'state.zarr'), mode='r')[:],
        dtype=np.float32,
    )
    phi = np.asarray(
        zarr.open_array(os.path.join(sim_folder, 'phi.zarr'), mode='r')[:],
        dtype=np.float32,
    )
    return state, phi


def load_test_simulation(sim_folder):
    """Load a validation/test simulation, as seen by predict().
    Only the initial snapshot and the future forcing trajectory are
    available -- never the ground-truth state evolution.

    Returns
    -------
    initial_state : ndarray, shape (n_features * n_cells,)
    phi           : ndarray, shape (n_timesteps,) -- known future forcing signal
    """
    initial_state = np.asarray(
        zarr.open_array(os.path.join(sim_folder, 'initial_state.zarr'), mode='r')[:],
        dtype=np.float32,
    )
    phi = np.asarray(
        zarr.open_array(os.path.join(sim_folder, 'phi.zarr'), mode='r')[:],
        dtype=np.float32,
    )
    return initial_state, phi


def build_one_step_pairs(state, phi):
    """Turn a (state, phi) simulation into supervised one-step-ahead pairs,
    a common starting point for autoregressive forecasting models:

        input  = [state_t, phi_t, phi_{t+1}]
        target = state_{t+1}

    i.e. "given where I am now and how the forcing changes, predict the
    next state". Returned arrays have (n_timesteps - 1) rows.
    """
    X_state = state[:, :-1].T
    phi_t = phi[:-1].reshape(-1, 1)
    phi_tp1 = phi[1:].reshape(-1, 1)
    X = np.hstack([X_state, phi_t, phi_tp1])
    Y = state[:, 1:].T
    return X.astype(np.float32), Y.astype(np.float32)


def d_function(ar, bc):
        r = ar.size
        idx = np.tril_indices(r)
        ar_nl = np.outer(ar, ar)[idx]
        temp = np.concatenate([np.array([1], dtype=np.float32), ar, ar_nl, bc])
        return temp.astype(np.float32, copy=False)

def rhs(t, ar, operator, amplitude, frequency, sim_type):
        if sim_type == 'sine':
            phi = 1+amplitude*np.sin(2*np.pi*frequency*t)
        else:
            phi = 1+amplitude

        d = d_function(ar, [phi])
        a_dot = d @ operator.T
        
        return a_dot

def integrate_Q(Q, grid):
        tmp = grid.compute_cell_sizes(
            length=False,
            area=True,
            volume=True
        )

        weights = tmp.cell_data["Volume"]
        Q_integrated = weights @ Q

        return Q_integrated
