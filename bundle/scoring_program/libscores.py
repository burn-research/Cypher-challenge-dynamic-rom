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


def compute_transfer_function_metrics(state_true, phi, grid_cell_volumes,
                                       feature_index, dt):
    """
    Compute gain error and phase error of the volume-integrated heat-release
    (or any scalar feature) transfer function  H(f) = Q_hat(f) / phi_hat(f).

    Parameters
    ----------
    state_true : ndarray, shape (n_cells, n_features, n_timesteps)
        Ground-truth trajectory.
    state_pred : ndarray -- NOT passed here; call once for ref, once for pred.
        See ``compute_tf_gain_phase_errors`` for the paired version.
    phi : ndarray, shape (n_timesteps,)
        Forcing signal used as the transfer-function input.
    grid_cell_volumes : ndarray, shape (n_cells,)
        Cell volumes used for spatial integration of the scalar field.
    feature_index : int
        Column index (along axis 1) of the feature to integrate
        (e.g. index of 'mix:Q').
    dt : float
        Time step between snapshots [s].

    Returns
    -------
    H : ndarray, shape (n_timesteps,)  (complex)
        Transfer function in the frequency domain.
    Q_integrated : ndarray, shape (n_timesteps,)
        Volume-integrated scalar field in the time domain.
    """
    Q = state_true[:, feature_index, :]          # (n_cells, n_timesteps)
    Q_integrated = grid_cell_volumes @ Q         # (n_timesteps,)

    phi_hat = np.fft.fft(phi)
    Q_hat   = np.fft.fft(Q_integrated)

    # Avoid division by zero for frequencies where phi_hat ≈ 0
    eps = np.finfo(float).eps * np.abs(phi_hat).max() * 10
    safe_phi_hat = np.where(np.abs(phi_hat) < eps, eps, phi_hat)

    H = Q_hat / safe_phi_hat
    return H, Q_integrated


def compute_tf_gain_phase_errors(state_true, state_pred, phi,
                                  grid_cell_volumes, feature_index, dt):
    """
    Compute gain error and RMS phase error between the true and predicted
    transfer functions.

    Parameters
    ----------
    state_true, state_pred : ndarray, shape (n_cells, n_features, n_timesteps)
    phi : ndarray, shape (n_timesteps,)
    grid_cell_volumes : ndarray, shape (n_cells,)
    feature_index : int
    dt : float

    Returns
    -------
    gain_error : float
        Relative L2 error between |H_ref| and |H_pred|.
    phase_error : float
        RMS phase difference [rad] between H_pred and H_ref.
    """
    H_ref,  _ = compute_transfer_function_metrics(
        state_true, phi, grid_cell_volumes, feature_index, dt)
    H_pred, _ = compute_transfer_function_metrics(
        state_pred, phi, grid_cell_volumes, feature_index, dt)

    # Gain error: relative L2 norm of the gain difference
    gain_ref  = np.abs(H_ref)
    gain_pred = np.abs(H_pred)
    denom = np.linalg.norm(gain_ref)
    gain_error = (np.linalg.norm(gain_ref - gain_pred) / denom
                  if denom > 0 else float('inf'))

    # Phase error: RMS of the wrapped phase difference
    phase_diff  = np.angle(H_pred * np.conj(H_ref))
    phase_error = float(np.sqrt(np.mean(phase_diff ** 2)))

    return float(gain_error), phase_error


def error_time_function(inference_time, time_reference=10.0, k=1.0):
    """
    Logistic penalty on inference time.

    Maps inference_time to (0, 1):
      * values well below ``time_reference``  →  ~0   (fast: no penalty)
      * values well above ``time_reference``  →  ~1   (slow: full penalty)

    Parameters
    ----------
    inference_time : float
        Measured inference time [s].
    time_reference : float
        Inflection point of the logistic curve [s].  Default: 10 s.
    k : float
        Steepness of the transition.  Default: 1.

    Returns
    -------
    float in (0, 1)
    """
    return float(1.0 / (1.0 + np.exp(-k * (inference_time - time_reference))))


def compute_global_score(nrmse, gain_error, phase_error, inference_time,
                          time_reference=10.0, k=1.0,
                          w_nrmse=0.4, w_gain=0.3,
                          w_phase=0.2, w_time=0.1):
    """
    Compute the composite score combining NRMSE, gain error, phase error,
    and inference-time penalty.

    Score = w_nrmse * S_nrmse
          + w_gain  * S_gain
          + w_phase * S_phase
          + w_time  * S_time

    where each sub-score lies in [0, 1] and *higher is better*.

    Parameters
    ----------
    nrmse : float
        Global NRMSE (from ``compute_nrmse_field``).
    gain_error : float
        Relative gain error (from ``compute_tf_gain_phase_errors``).
    phase_error : float
        RMS phase error [rad] (from ``compute_tf_gain_phase_errors``).
    inference_time : float
        Inference time per snapshot [s].
    time_reference, k : float
        Parameters for the logistic time penalty.
    w_nrmse, w_gain, w_phase, w_time : float
        Weights (must sum to 1).

    Returns
    -------
    score_global : float
    sub_scores   : dict   {name: value}
    """
    score_nrmse = 1.0 / (1.0 + nrmse)
    score_gain  = 1.0 / (1.0 + gain_error)
    score_phase = 1.0 - phase_error / np.pi
    score_time  = 1.0 / (1.0 + error_time_function(inference_time,
                                                     time_reference, k))

    score_global = (w_nrmse * score_nrmse
                    + w_gain  * score_gain
                    + w_phase * score_phase
                    + w_time  * score_time)

    sub_scores = {
        'score_nrmse':  score_nrmse,
        'score_gain':   score_gain,
        'score_phase':  score_phase,
        'score_time':   score_time,
    }
    return float(score_global), sub_scores
