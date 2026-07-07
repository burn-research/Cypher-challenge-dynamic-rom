#!/usr/bin/env python
"""
Ingestion program - CYPHER 2026 Data-driven Dynamic ROM challenge.

Adapted from the CYPHER turbulence-closure bundle (Lorenzo Piu, ULB, 2025),
itself adapted from codalab/codabench official examples.

Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

This program runs on the challenge platform to test your submission.

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.

This is the "ingestion program" written by the organizers.
This program also runs on the challenge platform to test your code.

============================================================================
Expected input_data structure:

input_data/
├── train/
│   ├── SineSweep_f1_f80_A02/
│   │   ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
│   │   └── phi.npy      # shape (T,), forcing signal value at each timestep
│   └── SineSweep_f1_f80_A04/
│       ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
│       └── phi.npy      # shape (T,), forcing signal value at each timestep
│
└── test/
    ├── step_A03/
    │   ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
    │   └── phi.npy      # shape (T,), forcing signal value at each timestep
    ├── step_A05/
    │   ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
    │   └── phi.npy      # shape (T,), forcing signal value at each timestep
    ├── sine_f10_A03/
    │   ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
    │   └── phi.npy      # shape (T,), forcing signal value at each timestep
    ├── sine_f10_A05/
    │   ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
    │   └── phi.npy      # shape (T,), forcing signal value at each timestep
    ├── sine_f40_A03/
    │   ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
    │   └── phi.npy      # shape (T,), forcing signal value at each timestep
    └── sine_f40_A05/
        ├── states.npy   # shape (T, ny, nx, n_features), full trajectory
        └── phi.npy      # shape (T,), forcing signal value at each timestep

The feature axis order is fixed, see FEATURES below.

Submission:
The ingestion code accepts submissions that contain an UNTRAINED model, that 
can rely on the libraries Tensorflow, ScikitLearn, and pytorch, depending
on the participants' preferences.

Your model.py MUST define a class model() with three methods:
- preprocess(self, data_folder): data_folder is the path to input_data/train.
    It must loop over the case subfolders itself, load states.npy/phi.npy,
    and return an object D to be consumed by fit().
- fit(self, D): trains the model in place, no return value required.
- predict(self, case_folder): case_folder is the path to ONE case folder
    inside valid/ or test/ (e.g. input_data/valid/step_A0.3). It contains
    init_state.npy and phi.npy. It must return the FULL predicted
    trajectory, shape (T, ny, nx, n_features), with T = len(phi.npy).
============================================================================
"""

import time
overall_start = time.time()
import os
import sys
import json
from sys import argv, path
import numpy as np

FEATURES = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']

# =========================== BEGIN OPTIONS ==============================
verbose = True                  # print progress messages
debug_mode = 0                  # 0: normal run; >0 prints debug messages; 4: skip ingestion and run model.py directly
show_versions = False           # print installed library versions
save_previous_results = False   # if True, previous output_dir is renamed with a timestamp (useful locally)   

max_time = 21600                # 6 hours, max training time budget (seconds)
debug_time = 1000               # If debug >=1, you can decrease the maximum time (in sec) with the variable debug_time

# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "./"
default_input_dir = root_dir + "input_data"
default_output_dir = root_dir + "sample_output_data"
default_program_dir = root_dir + "ingestion_program"
default_submission_dir = root_dir + "sample_code_submission"
default_data_dir = root_dir + "data_directory"
# =========================== END OPTIONS ================================

version = 1

if __name__ == "__main__" and debug_mode < 4:

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
    from data_io import (vprint, show_version, cpdir, check_model_interface,
                          rmdir, mkdir, process_training_time, time_limit)
    from model import model

    if show_versions:
        show_version()

    if save_previous_results:
        the_date = time.strftime("%y-%m-%d-%H-%M")
        data_io.mvdir(output_dir, output_dir + '_' + the_date)
    mkdir(output_dir)

    print('\n*****************************************')
    print('****** Ingestion program version ' + str(version) + ' ******')
    print('*******************************************\n')

    time_budget = max_time if debug_mode < 1 else debug_time
    vprint(verbose, f"Time available for training: {time_budget} seconds")

    vprint(verbose, "\nCopying input data directory...")
    if os.path.exists(default_data_dir):
        rmdir(default_data_dir)
    cpdir(input_dir, default_data_dir)

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

    # ======== Inference on the valid and test splits ========
    for split in ['valid', 'test']:
        split_dir = os.path.join(default_data_dir, split)
        if not os.path.isdir(split_dir):
            vprint(verbose, f'\nNo "{split}" folder found in input data, skipping.')
            continue

        vprint(verbose, '\n===================================')
        vprint(verbose, f'====== INFERENCE ON "{split.upper()}" SPLIT ======')
        vprint(verbose, '===================================\n')

        case_names = sorted(d for d in os.listdir(split_dir)
                             if os.path.isdir(os.path.join(split_dir, d)))

        pred_out_dir = os.path.join(output_dir, 'predictions', split)
        mkdir(pred_out_dir)

        inference_times = {}
        for case_name in case_names:
            case_dir = os.path.join(split_dir, case_name)
            vprint(verbose, f'Running inference on case "{case_name}"...')
            t_start = time.time()
            pred = np.asarray(M.predict(case_dir))
            t_end = time.time()
            inference_times[case_name] = t_end - t_start

            np.save(os.path.join(pred_out_dir, f'{case_name}.npy'), pred)
            tt, unit = process_training_time(inference_times[case_name])
            vprint(verbose, f'  -> done in {tt:.2f} {unit}, output shape {pred.shape}')

        with open(os.path.join(output_dir, f"inference_time_{split}.json"), "w") as f:
            json.dump(inference_times, f, indent=2)

    overall_time_spent = time.time() - overall_start
    vprint(verbose, "\n[+] Done")
    vprint(verbose, "[+] Overall time spent %5.2f sec" % overall_time_spent +
           " :: Overall time budget for training %5.2f sec" % time_budget)

    exit(0)
