#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scoring program for the CYPHER 2026 dynamic ROM challenge.
# Adapted from the CYPHER 2025 DNS challenge (Lorenzo Piu, ULB).

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# THE ORGANIZERS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

import os
import json
from sys import argv
import numpy as np
import time
import math

from libscores import compute_nrmse_field, mkdir

# Default I/O directories (used only when running score.py with no arguments,
# e.g. for local testing):
root_dir = "./"
default_input_dir = root_dir + "reference_data/valid"
default_output_dir = root_dir + "scoring_output"

verbose = True
debug_mode = 0

# ----------------------------------------------------------------------------
# Multi-objective score coefficient: balances NRMSE against inference cost.
# score = NRMSE + beta * inference_time_per_snapshot
# Tune beta so that a "reasonably fast" model is not unfairly penalized, while
# a pathologically slow one is. Update this once you have a sense of the
# typical inference times of submitted models.
# ----------------------------------------------------------------------------
beta = 1e-2

# Constant used if a score could not be computed
missing_score = -0.999999

scoring_version = 1.0


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
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]

    if verbose:
        print(f'Using {output_dir} as output directory')
        print(f'Using {input_dir} as input directory')

    results_dir = os.path.join(input_dir, 'res')
    reference_data_dir = os.path.join(input_dir, 'ref')
    if len(argv) == 1:
        # local debugging convenience: input_dir IS the reference data folder,
        # and results are expected in sample_output_data/ next to it.
        reference_data_dir = input_dir
        results_dir = 'sample_output_data'

    with open(os.path.join(reference_data_dir, 'phase'), 'r') as f:
        phase = f.read().strip()

    with open(os.path.join(reference_data_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
    n_features = meta['n_features']

    if verbose:
        print(f'\nScoring phase: {phase}')
        print('Reading submitted solution results...')

    # ---- Read training time (shared by all simulations of this submission) --
    training_time_file = os.path.join(results_dir, 'training_time.txt')
    with open(training_time_file, 'r') as f:
        training_time = float(f.read())

    # ---- Loop over every simulation in the reference data for this phase ----
    sim_names = sorted(
        d for d in os.listdir(reference_data_dir)
        if os.path.isdir(os.path.join(reference_data_dir, d))
    )
    if len(sim_names) == 0:
        raise RuntimeError(
            f"No simulation sub-folders found in {reference_data_dir}."
        )

    nrmse_list = []
    inference_time_per_snapshot_list = []

    for sim_name in sim_names:
        ref_sim_dir = os.path.join(reference_data_dir, sim_name)
        res_sim_dir = os.path.join(results_dir, phase, sim_name)

        state_true = np.load(os.path.join(ref_sim_dir, "state_full.npz"))["data"]

        pred_path = os.path.join(res_sim_dir, 'state_pred.npz')
        if not os.path.exists(pred_path):
            raise RuntimeError(
                f"Missing prediction for simulation '{sim_name}' "
                f"(expected file: {pred_path}). Did your predict() method "
                "raise an error for this simulation?"
            )
        state_pred = np.load(pred_path)['data']

        with open(os.path.join(res_sim_dir, 'inference_time.txt'), 'r') as f:
            inference_time = float(f.read())

        # print(f' state pred shape: {state_pred.shape}')
        # print(f' state true shape: {state_true.shape}')
        if state_pred.shape[2] != state_true.shape[2]:
            raise RuntimeError(
                f"Simulation '{sim_name}': prediction has "
                f"{state_pred.shape[2]} time steps, expected "
                f"{state_true.shape[2]}. Your predict() method must return "
                "a forecast covering the whole horizon given by phi.npz."
            )

        nrmse_sim, _, _ = compute_nrmse_field(state_true, state_pred, n_features)
        nrmse_list.append(nrmse_sim)

        n_timesteps = state_true.shape[2]
        inference_time_per_snapshot_list.append(inference_time / n_timesteps)

        if verbose:
            print(f'  - {sim_name}: NRMSE = {nrmse_sim:.5f}')

    nrmse = float(np.mean(nrmse_list))
    inference_time_per_snapshot = float(np.mean(inference_time_per_snapshot_list))

    score = nrmse + beta * inference_time_per_snapshot

    if math.isnan(nrmse):
        print("\n WARNING: NRMSE is NaN!")
        print("\n Probably the prediction error exploded")
        print(" We set the default value of NRMSE = 1000...\n\n")
        nrmse = 1000
        score = nrmse + beta * inference_time_per_snapshot

    elif nrmse > 10e8:
        print("\n WARINING: NRMSE is extremely high!")
        print(f" NRMSE = {nrmse}")
        print(" We set the default value of NRMSE = 1000...\n\n")
        nrmse = 1000
        score = nrmse + beta * inference_time_per_snapshot

    if verbose:
        print('\n====================================')
        print('======== COMPUTING METRICS =========')
        print('====================================\n')
        print('Creating output directory...')
    mkdir(output_dir)

    results = {
        'score': score,
        'NRMSE': nrmse,
        'time_inference': inference_time_per_snapshot,
        'time_training': training_time,
    }

    if verbose:
        print('Writing scoring metrics to file...')
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as score_file:
        for key, value in results.items():
            score_file.write(f"{key}: {value}\n")

    t2 = time.time()
    overall_time_spent = t2 - t1

    if verbose:
        print("\n[+] Done")
        print("[+] Overall time spent %5.2f sec " % overall_time_spent)

    exit(0)
