#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ingestion program for the CYPHER 2026 dynamic ROM challenge.
# Adapted from the CYPHER 2025 DNS challenge (Lorenzo Piu, ULB, May 2025),
# itself adapted from the Codabench "iris" example bundle:
# https://github.com/codalab/competition-examples/tree/master/codabench/iris

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
# This is the "ingestion program" written by the organizers (Tommaso Baffetti, Alberto Procacci, ULB, 2026).
# It runs on the challenge platform for every submission.

# ----------------------------------------------------------------------------
# Input data structure (input_dir):
#
# input_data
# ├── grid.vtu                     shared unstructured grid, pyvista format
# ├── xyz.npz                      shared cells coordinates matrix (n_cells, 3)
# ├── train
# │   ├── sineSweep_A02
# │   │   ├── state.npz            shape (n_cells, n_features, n_timesteps)
# │   │   └── phi.npz              shape (n_timesteps,)
# │   └── sineSweep_A04
# │       ├── state.npz            shape (n_cells, n_features, n_timesteps)
# │       └── phi.npz              shape (n_timesteps,)
# ├── valid
# │   ├── sineSweep_A02
# │   │   ├── initial_state.npz    shape (n_cells, n_features)
# │   │   └── phi.npz              shape (n_timesteps,)
# │   └── sineSweep_A04
# │       ├── initial_state.npz    shape (n_cells, n_features)
# │       └── phi.npz              shape (n_timesteps,)
# └── test
#     ├── step_A03
#     │   ├── initial_state.npz    shape (n_cells, n_features)
#     │   └── phi.npz              shape (n_timesteps,)
#     ├── step_A05
#     │   ├── initial_state.npz    shape (n_cells, n_features)
#     │   └── phi.npz              shape (n_timesteps,)
#     ├── sine_f10_A03
#     │   ├── initial_state.npz    shape (n_cells, n_features)
#     │   └── phi.npz              shape (n_timesteps,)
#     ├── sine_f10_A05
#     │   ├── initial_state.npz    shape (n_cells, n_features)
#     │   └── phi.npz              shape (n_timesteps,)
#     ├── sine_f40_A03
#     │   ├── initial_state.npz    shape (n_cells, n_features)
#     │   └── phi.npz              shape (n_timesteps,)
#     └── sine_f40_A05
#         ├── initial_state.npz    shape (n_cells, n_features)
#         └── phi.npz              shape (n_timesteps,)

#
# The train/ simulations contain the full ground-truth state. The valid/ and
# test/ simulations contain ONLY the initial snapshot and the future forcing
# signal: the ground truth is never exposed to the ingestion program, it
# lives exclusively in reference_data/ and is read by the scoring program.
#
# Submitted model.py must define a class `model` with 3 methods:
# 1. preprocess(self, data_folder) -> D
#    data_folder is the path to the "train" folder above.
# 2. fit(self, D) -> None
#    trains the model in place using D.
# 3. predict(self, test_data_folder) -> numpy array, shape (n_cells, n_features, n_timesteps)
#    test_data_folder is the path to ONE valid/ or test/ simulation folder.
# ----------------------------------------------------------------------------

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# THE ORGANIZERS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR ANY PARTICULAR PURPOSE.

# =========================== BEGIN OPTIONS ===================================
verbose = True          # print progression messages
debug_mode = 0           # >0 prints extra debug information
save_previous_results = False   # keep previous local runs (useful when running locally)
show_versions = False    # print installed library versions

max_time = 21600         # time budget for preprocess+fit, in seconds (6 hours)
debug_time = 1000        # time budget used instead of max_time when debug_mode >= 1

root_dir = "./"
default_input_dir = root_dir + "input_data"
default_output_dir = root_dir + "sample_output_data"
default_program_dir = root_dir + "ingestion_program"
default_submission_dir = root_dir + "sample_code_submission"
default_data_dir = root_dir + "data_directory"

EVAL_PHASES = ["valid", "test"]   # both are always predicted; scoring picks the relevant one
# ============================ END OPTIONS =====================================

version = 1

import time
overall_start = time.time()
import os
import sys
from sys import argv, path
import datetime
import numpy as np

the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")


if __name__ == "__main__" and debug_mode < 4:

    # ---- I/O directories --------------------------------------------------
    if len(argv) == 1:
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir = default_program_dir
        submission_dir = default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])

    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)
        print('\nImporting modules...')

    path.append(program_dir)
    path.append(submission_dir)
    import data_io
    from data_io import vprint, show_version, cpdir, rmdir, mkdir
    from data_io import check_model_interface, process_training_time, time_limit
    from model import model

    if show_versions:
        show_version()

    if save_previous_results:
        data_io.mvdir(output_dir, output_dir + '_' + the_date)
    mkdir(output_dir)

    print('\n*****************************************')
    print('****** Ingestion program version ' + str(version) + ' ******')
    print('*******************************************\n')

    if debug_mode > 1:
        print(f'Debugging mode: {debug_mode}')
        data_io.show_dir('../')
        print('')

    time_budget = max_time if debug_mode < 1 else debug_time
    vprint(verbose, f"Time available for preprocessing + training: {time_budget} seconds")

    # ---- Copy input data locally to avoid touching the original folder ----
    vprint(verbose, "\nCopying input data directory...")
    if os.path.exists(default_data_dir):
        rmdir(default_data_dir)
    cpdir(input_dir, default_data_dir)

    # ---- Initialize submitted model ---------------------------------------
    vprint(verbose, '\nInitializing model...')
    M = model()
    vprint(verbose, 'Model initialized successfully')

    vprint(verbose, '\nChecking model structure...')
    check_model_interface(M, verbose=verbose)
    vprint(verbose, 'Model structure OK')

    vprint(verbose, '\n====================================')
    vprint(verbose, '========== MODEL TRAINING ==========')
    vprint(verbose, '====================================\n')

    train_dir = os.path.join(default_data_dir, 'train')

    with time_limit(time_budget):
        vprint(verbose, 'Preprocessing training data...')
        t1 = time.time()
        D = M.preprocess(train_dir)

        vprint(verbose, '\nBeginning model training...')
        M.fit(D)

    t2 = time.time()
    training_time = t2 - t1
    tt, unit = process_training_time(training_time)
    vprint(verbose, f'\nModel trained successfully in {tt:.2f} {unit}')

    with open(os.path.join(output_dir, "training_time.txt"), "w") as f:
        f.write(str(training_time))

    # ---- Inference on every validation and test simulation ----------------
    vprint(verbose, '\n===================================')
    vprint(verbose, '========== TESTING MODEL ==========')
    vprint(verbose, '===================================\n')

    for phase in EVAL_PHASES:
        phase_input_dir = os.path.join(default_data_dir, phase)
        if not os.path.isdir(phase_input_dir):
            vprint(verbose, f'No "{phase}" folder found in input data, skipping.')
            continue

        sim_names = sorted(
            d for d in os.listdir(phase_input_dir)
            if os.path.isdir(os.path.join(phase_input_dir, d))
        )

        for sim_name in sim_names:
            sim_dir = os.path.join(phase_input_dir, sim_name)
            vprint(verbose, f'\n[{phase}] Forecasting simulation "{sim_name}"...')

            t_start = time.time()
            state_pred = M.predict(sim_dir)
            t_end = time.time()
            inference_time = t_end - t_start

            state_pred = np.asarray(state_pred, dtype=np.float32)

            out_sim_dir = os.path.join(output_dir, phase, sim_name)
            mkdir(out_sim_dir)
            np.savez_compressed(os.path.join(out_sim_dir, 'state_pred.npz'), data=state_pred)
            with open(os.path.join(out_sim_dir, 'inference_time.txt'), 'w') as f:
                f.write(str(inference_time))

            tt, unit = process_training_time(inference_time)
            vprint(verbose,
                   f'  -> forecast shape {state_pred.shape}, done in {tt:.2f} {unit}')

    overall_time_spent = time.time() - overall_start

    vprint(verbose, "\n[+] Done")
    vprint(verbose, "[+] Overall time spent %5.2f sec " % overall_time_spent +
           ":: Overall time budget for training %5.2f sec" % time_budget)

    exit(0)
