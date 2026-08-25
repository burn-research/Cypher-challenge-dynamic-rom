#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scoring program for the CYPHER 2026 dynamic ROM challenge.
# Written by the organizers (Tommaso Baffetti, Alberto Procacci, ULB, 2026).

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# THE ORGANIZERS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

import os
import json
from sys import argv
import numpy as np
import time
import math

from libscores import (
    compute_nrmse_field,
    compute_tf_gain_phase_errors,
    compute_global_score,
    mkdir,
)

# Default I/O directories (used only when running score.py with no arguments,
# e.g. for local testing):
root_dir = "./"
default_input_dir  = root_dir + "reference_data/valid"
default_output_dir = root_dir + "scoring_output"

verbose    = True
debug_mode = 0

# ----------------------------------------------------------------------------
# Transfer-function settings
# ----------------------------------------------------------------------------
# Index of the feature used for the volume-integrated transfer function.
# Must match the feature list in meta.json / your data convention.
# e.g.  features = ['p','U1','U3','rho','T','mix:Q','CH4','O2','H2O','CO2','OH']
#       'mix:Q' is at index 5.
Q_FEATURE_INDEX = 5

# Time step between snapshots [s]  – must match the simulation dt.
DT = 2e-3

# Logistic time-penalty parameters (see error_time_function in libscores.py)
TIME_REFERENCE = 10.0   # inflection point [s]
K_LOGISTIC     = 1.0    # steepness

# Composite score weights  (must sum to 1)
W_NRMSE = 0.4
W_GAIN  = 0.3
W_PHASE = 0.2
W_TIME  = 0.1

# Constant used if a score could not be computed
missing_score   = -0.999999
scoring_version = 2.0


def show_dir(run_dir):
    print('\n=== Listing run dir ===\n')
    run_dir = os.path.abspath(run_dir)

    def tree(dir_path, prefix=''):
        contents = sorted(os.listdir(dir_path))
        pointers = ['|-- '] * (len(contents) - 1) + ['`-- ']
        for pointer, name in zip(pointers, contents):
            path = os.path.join(dir_path, name)
            print(prefix + pointer + name)
            if os.path.isdir(path):
                extension = '|   ' if pointer == '|-- ' else '    '
                tree(path, prefix + extension)

    print(os.path.basename(run_dir) + '/')
    tree(run_dir)


# =============================== MAIN ========================================

if __name__ == "__main__":

    print('\n*****************************************')
    print('******* Scoring program version ' + str(scoring_version) + ' *******')
    print('*******************************************\n')

    t1 = time.time()

    if debug_mode > 0:
        print(f'Debugging mode: {debug_mode}')
        show_dir('../')
        print('')

    # ---- I/O directories ----------------------------------------------------
    if len(argv) == 1:
        input_dir  = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir  = argv[1]
        output_dir = argv[2]

    if verbose:
        print(f'Using {output_dir} as output directory')
        print(f'Using {input_dir} as input directory')

    results_dir       = os.path.join(input_dir, 'res')
    reference_data_dir = os.path.join(input_dir, 'ref')
    if len(argv) == 1:
        # Local debugging: input_dir IS the reference data folder;
        # results are expected in sample_output_data/ next to it.
        reference_data_dir = input_dir
        results_dir        = 'sample_output_data'

    # ---- Phase & metadata ---------------------------------------------------
    with open(os.path.join(reference_data_dir, 'phase'), 'r') as f:
        phase = f.read().strip()

    with open(os.path.join(reference_data_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
    n_features = meta['n_features']

    # ---- Grid cell volumes (needed for TF metrics) --------------------------
    # grid.vtu lives in the same phase folder as the reference data
    # (reference_data/valid/grid.vtu  or  reference_data/test/grid.vtu).
    cell_volumes = None
    grid_path = os.path.join(reference_data_dir, 'grid.vtu')
    if os.path.exists(grid_path):
        import pyvista as pv
        grid = pv.read(grid_path)
        tmp  = grid.compute_cell_sizes(length=False, area=True, volume=True)
        cell_volumes = tmp.cell_data['Volume']
        if verbose:
            print(f'Loaded grid from {grid_path} '
                  f'({len(cell_volumes)} cells)')
    else:
        raise RuntimeError(
            f"Grid file not found: {grid_path}\n"
            "Place grid.vtu in the reference_data/valid/ and "
            "reference_data/test/ folders."
        )

    if verbose:
        print(f'\nScoring phase: {phase}')
        print('Reading submitted solution results...')

    # ---- Training time (shared by all simulations) --------------------------
    training_time_file = os.path.join(results_dir, 'training_time.txt')
    with open(training_time_file, 'r') as f:
        training_time = float(f.read())

    # ---- Loop over every simulation in the reference data -------------------
    sim_names = sorted(
        d for d in os.listdir(reference_data_dir)
        if os.path.isdir(os.path.join(reference_data_dir, d))
    )
    if len(sim_names) == 0:
        raise RuntimeError(
            f"No simulation sub-folders found in {reference_data_dir}."
        )

    nrmse_list                      = []
    gain_error_list                 = []
    phase_error_list                = []
    inference_time_per_snapshot_list = []

    for sim_name in sim_names:
        ref_sim_dir = os.path.join(reference_data_dir, sim_name)
        res_sim_dir = os.path.join(results_dir, phase, sim_name)

        # ---- Ground-truth state ---------------------------------------------
        state_true = np.load(
            os.path.join(ref_sim_dir, "state_full.npz"))["data"]

        # ---- Predicted state ------------------------------------------------
        pred_path = os.path.join(res_sim_dir, 'state_pred.npz')
        if not os.path.exists(pred_path):
            raise RuntimeError(
                f"Missing prediction for simulation '{sim_name}' "
                f"(expected file: {pred_path}). Did your predict() method "
                "raise an error for this simulation?"
            )
        state_pred = np.load(pred_path)['data']

        # ---- Forcing signal (needed for TF metrics) -------------------------
        phi_path = os.path.join(ref_sim_dir, 'phi.npz')
        phi = np.load(phi_path)['data'] if os.path.exists(phi_path) else None

        # ---- Inference time -------------------------------------------------
        with open(os.path.join(res_sim_dir, 'inference_time.txt'), 'r') as f:
            inference_time = float(f.read())

        # ---- Shape check ----------------------------------------------------
        if state_pred.shape[2] != state_true.shape[2]:
            raise RuntimeError(
                f"Simulation '{sim_name}': prediction has "
                f"{state_pred.shape[2]} time steps, expected "
                f"{state_true.shape[2]}. Your predict() method must return "
                "a forecast covering the whole horizon given by phi.npz."
            )

        # ---- NRMSE ----------------------------------------------------------
        nrmse_sim, _, _ = compute_nrmse_field(state_true, state_pred,
                                               n_features)
        nrmse_list.append(nrmse_sim)

        # ---- Gain & phase errors --------------------------------------------
        if phi is not None:
            gain_err, phase_err = compute_tf_gain_phase_errors(
                state_true, state_pred, phi,
                cell_volumes, Q_FEATURE_INDEX, DT,
            )
        else:
            raise RuntimeError(
                f"Missing phi.npz for simulation '{sim_name}' "
                f"in {ref_sim_dir}."
            )
        gain_error_list.append(gain_err)
        phase_error_list.append(phase_err)

        # ---- Inference time per snapshot ------------------------------------
        n_timesteps = state_true.shape[2]
        inference_time_per_snapshot_list.append(inference_time / n_timesteps)

        if verbose:
            print(f'  - {sim_name}:\n'
                  f'        NRMSE          = {nrmse_sim:.5f}\n'
                  f'        gain_err       = {gain_err:.5f}\n'
                  f'        phase_err      = {phase_err:.5f} rad\n'
                  f'        inference_time = {inference_time / n_timesteps:.5e} sec/snapshot\n') 

    # ---- Aggregate over simulations -----------------------------------------
    nrmse                    = float(np.mean(nrmse_list))
    gain_error               = float(np.mean(gain_error_list))
    phase_error              = float(np.mean(phase_error_list))
    inference_time_per_snapshot = float(
        np.mean(inference_time_per_snapshot_list))

    # ---- Sanity checks on NRMSE ---------------------------------------------
    if math.isnan(nrmse):
        print("\n WARNING: NRMSE is NaN!")
        print(" Probably the prediction error exploded.")
        print(" Setting NRMSE = 10000.\n")
        nrmse = 10000.0

    elif nrmse > 10e8:
        print("\n WARNING: NRMSE is extremely high!")
        print(f" NRMSE = {nrmse}")
        print(" Setting NRMSE = 10000.\n")
        nrmse = 10000.0

    # ---- Composite score ----------------------------------------------------
    score_global, sub_scores = compute_global_score(
        nrmse               = nrmse,
        gain_error          = gain_error,
        phase_error         = phase_error,
        inference_time      = inference_time_per_snapshot,
        time_reference      = TIME_REFERENCE,
        k                   = K_LOGISTIC,
        w_nrmse             = W_NRMSE,
        w_gain              = W_GAIN,
        w_phase             = W_PHASE,
        w_time              = W_TIME,
    )

    # ---- Write results ------------------------------------------------------
    if verbose:
        print('\n====================================')
        print('======== COMPUTING METRICS =========')
        print('====================================\n')
        print('Creating output directory...')
    mkdir(output_dir)

    results = {
        'score':            score_global,
        'score_nrmse':      sub_scores['score_nrmse'],
        'score_gain':       sub_scores['score_gain'],
        'score_phase':      sub_scores['score_phase'],
        'score_time':       sub_scores['score_time'],
        'NRMSE':            nrmse,
        'gain_error':       gain_error,
        'phase_error':      phase_error,
        'time_inference':   inference_time_per_snapshot,
        'time_training':    training_time,
    }

    if verbose:
        print('Writing scoring metrics to file...')
        print(f'\n  NRMSE        : {nrmse:.5f}')
        print(f'  Gain error   : {gain_error:.5f}')
        print(f'  Phase error  : {phase_error:.5f} rad')
        print(f'  Infer. time  : {inference_time_per_snapshot:.3e} sec/snapshot\n')
        for k, v in sub_scores.items():
            print(f'  {k:15s}: {v:.4f}')
        print(f' -----------------------')
        print(f'\n  GLOBAL SCORE : {score_global:.4f}')

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as score_file:
        for key, value in results.items():
            score_file.write(f"{key}: {value}\n")

    t2 = time.time()
    overall_time_spent = t2 - t1

    if verbose:
        print("\n[+] Done")
        print("[+] Overall time spent %5.2f sec " % overall_time_spent)

    exit(0)