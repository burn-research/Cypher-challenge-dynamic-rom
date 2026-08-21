#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring utilities for the CYPHER 2026 dynamic ROM challenge.

Adapted from the CYPHER 2025 DNS challenge (Lorenzo Piu, ULB) and from the
NRMSE routine used internally by the CYPHER team for ROM evaluation. The
core idea (per-feature NRMSE, normalized by the standard deviation of the
ground truth, averaged across features) is preserved; the implementation
below is adapted to the (n_timesteps, n_features * n_cells) array layout
used in this challenge instead of a generic (n_samples, n_features) table.
"""

import os
import numpy as np


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def compute_nrmse_field(state_true, state_pred, n_features,
                         eps_mean=1e-7, eps_std=1e-7):
    """
    Compute the Normalized Root Mean Square Error (NRMSE) between a
    predicted and a true flow-state trajectory, per physical feature.

    Parameters
    ----------
    state_true : ndarray, shape (n_timesteps, n_features * n_cells)
        Ground-truth trajectory.
    state_pred : ndarray, shape (n_timesteps, n_features * n_cells)
        Predicted trajectory (same shape as state_true).
    n_features : int
        Number of physical features stacked along the column axis
        (e.g. 11 for p, U1, U3, rho, T, mix:Q, CH4, O2, H2O, CO2, OH).
    eps_mean, eps_std : float
        Features whose ground-truth mean and standard deviation are both
        below these thresholds (e.g. an all-zero species field) are
        excluded from the average, to avoid dividing by ~0.

    Returns
    -------
    nrmse_global : float
        NRMSE averaged over the retained features.
    nrmse_per_feature : ndarray, shape (n_kept_features,)
        NRMSE for each retained feature (in the original feature order,
        excluded features removed).
    n_kept_features : int
        Number of features actually used in the average.
    """
    state_true = np.asarray(state_true, dtype=np.float64)
    state_pred = np.asarray(state_pred, dtype=np.float64)

    if state_true.shape != state_pred.shape:
        raise ValueError(
            f"Shape mismatch between prediction {state_pred.shape} and "
            f"ground truth {state_true.shape}."
        )

    # shape attesa: [n_cells, n_features, n_timesteps]
    if state_true.ndim != 3:
        raise ValueError(
            f"Expected 3D array [n_cells, n_features, n_timesteps], "
            f"got shape {state_true.shape}."
        )
    n_cells, n_features_actual, n_timesteps = state_true.shape
    if n_features_actual != n_features:
        raise ValueError(
            f"Array has {n_features_actual} features along axis 1, "
            f"but n_features={n_features} was passed."
        )

    true_r = state_true.transpose(1, 0, 2)   # [n_features, n_cells, n_timesteps]
    pred_r = state_pred.transpose(1, 0, 2)

    mean = np.mean(true_r, axis=(1, 2))      # [n_features,]
    std  = np.std(true_r,  axis=(1, 2), ddof=1)

    mask_eliminate = (np.abs(mean) < eps_mean) & (std < eps_std)
    mask = ~mask_eliminate

    std_safe = std.copy()
    std_safe[std_safe < eps_std] = eps_std

    rmse_per_feature = np.sqrt(np.mean((true_r - pred_r) ** 2, axis=(1, 2)))
    nrmse_per_feature_all = rmse_per_feature / std_safe

    nrmse_per_feature = nrmse_per_feature_all[mask]

    if nrmse_per_feature.size == 0:
        raise ValueError("All features were excluded from the NRMSE "
                          "computation (constant ground truth?).")

    nrmse_global = float(np.mean(nrmse_per_feature))

    return nrmse_global, nrmse_per_feature, int(mask.sum())
