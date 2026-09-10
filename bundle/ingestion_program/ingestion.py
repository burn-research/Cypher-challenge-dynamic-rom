#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ingestion program for the CYPHER 2026 dynamic ROM challenge.
# Written by the organizers (Tommaso Baffetti, Alberto Procacci, ULB, 2026).

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
# ├── grid.vtu                      shared unstructured grid, pyvista format
# ├── xyz.zarr                      shared cells coordinates array (n_cells, 3)
# ├── train
# │   ├── sineSweep_f1_f80_A02
# │   │   ├── state.zarr            shape (n_cells, n_features, n_timesteps)
# │   │   └── phi.zarr              shape (n_timesteps,)
# │   └── sineSweep_f1_f80_A04
# │       ├── state.zarr            shape (n_cells, n_features, n_timesteps)
# │       └── phi.zarr              shape (n_timesteps,)
# ├── valid
# │   ├── step_A03
# │   │   ├── initial_state.zarr    shape (n_cells, n_features)
# │   │   └── phi.zarr              shape (n_timesteps,)
# │   └── sine_f40_A05
# │       ├── initial_state.zarr    shape (n_cells, n_features)
# │       └── phi.zarr              shape (n_timesteps,)
# └── test
#     ├── step_A05
#     │   ├── initial_state.zarr    shape (n_cells, n_features)
#     │   └── phi.zarr              shape (n_timesteps,)
#     ├── sine_f10_A03
#     │   ├── initial_state.zarr    shape (n_cells, n_features)
#     │   └── phi.zarr              shape (n_timesteps,)
#     ├── sine_f10_A05
#     │   ├── initial_state.zarr    shape (n_cells, n_features)
#     │   └── phi.zarr              shape (n_timesteps,)
#     ├── sine_f40_A03
#     │   ├── initial_state.zarr    shape (n_cells, n_features)
#     │   └── phi.zarr              shape (n_timesteps,)
#     └── sine_f40_A05
#         ├── initial_state.zarr    shape (n_cells, n_features)
#         └── phi.zarr              shape (n_timesteps,)
#
# The train/ simulations contain the full ground-truth state (state.zarr).
# The valid/ and test/ simulations contain ONLY the initial snapshot
# (initial_state.zarr) and the future forcing signal (phi.zarr): the ground
# truth is never exposed to the ingestion program, it lives exclusively in
# reference_data/ and is read by the scoring program.
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
import gc
import os
import zarr

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
    from data_io import vprint, show_version, mkdir
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

    train_dir = os.path.join(input_dir, 'train')

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
        phase_input_dir = os.path.join(input_dir, phase)
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
            zarr.save(os.path.join(out_sim_dir, 'state_pred.zarr'), state_pred)
            with open(os.path.join(out_sim_dir, 'inference_time.txt'), 'w') as f:
                f.write(str(inference_time))

            tt, unit = process_training_time(inference_time)
            vprint(verbose,
                   f'  -> forecast shape {state_pred.shape}, done in {tt:.2f} {unit}')
            del state_pred
            gc.collect()

    overall_time_spent = time.time() - overall_start

    vprint(verbose, "\n[+] Done")
    vprint(verbose, "[+] Overall time spent %5.2f sec " % overall_time_spent +
           ":: Overall time budget for training %5.2f sec" % time_budget)
    
    gc.collect()

    if debug_mode > 1:
        # ── Contenuto e peso dell'output ─────────────────────────────────
        # print("\n[DEBUG] Output directory contents:")
        total_size = 0
        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                size = os.path.getsize(fpath)
                total_size += size
                #print(f"  {fpath}  ({size/1e6:.1f} MB)")
        print(f"[DEBUG] Total output size: {total_size/1e6:.1f} MB  ({total_size/1e9:.2f} GB)")

        # ── RAM disponibile via /proc/meminfo ────────────────────────────
        with open('/proc/meminfo', 'r') as f:
            meminfo = dict(line.split(':', 1) for line in f)
        mem_total = int(meminfo['MemTotal'].strip().split()[0]) / 1e6  # GB
        mem_avail = int(meminfo['MemAvailable'].strip().split()[0]) / 1e6
        mem_used  = mem_total - mem_avail
        print(f"[DEBUG] RAM used:      {mem_used:.2f} GB / {mem_total:.2f} GB")
        print(f"[DEBUG] RAM available: {mem_avail:.2f} GB")

        # ── Disco disponibile via os.statvfs ─────────────────────────────
        st = os.statvfs(output_dir)
        disk_free  = st.f_bavail * st.f_frsize / 1e9
        disk_total = st.f_blocks * st.f_frsize / 1e9
        disk_used  = disk_total - disk_free
        print(f"[DEBUG] Disk used: {disk_used:.1f} GB / {disk_total:.1f} GB, free: {disk_free:.1f} GB")

    exit(0)
