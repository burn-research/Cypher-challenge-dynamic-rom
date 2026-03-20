#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
from sys import argv
import aPrioriDNS as ap
import numpy as np
import time

import libscores
import yaml
from libscores import loss, mkdir
from utils import mute_print

# Default I/O directories:
root_dir = "./"
default_input_dir = root_dir + "reference_data/valid"
default_output_dir = root_dir + "scoring_output"

# Debug flag 
# 0: no debug, 
# 1: show all scores, 
# 2: also show version and listing of directories before and after, writes random numbers as score
debug_mode = 0

# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Multi-objective loss coefficient:
###################################
# Coefficient to tune to balance between MSE loss and inference time
beta = 1e6
beta_MSE = 1e3

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 2.0


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

def show_dir(run_dir):
    print('\n=== Listing run dir ===\n')
    run_dir = os.path.abspath(run_dir)
    
    def tree(dir_path, prefix=''):
        contents = sorted(os.listdir(dir_path))
        pointers = ['├── '] * (len(contents) - 1) + ['└── ']
        
        for pointer, name in zip(pointers, contents):
            path = os.path.join(dir_path, name)
            print(prefix + pointer + name)
            if os.path.isdir(path):
                extension = '│   ' if pointer == '├── ' else '    '
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
        print('\n====================================')
        print('======== LISTING DIRECTORIES =======')
        print('====================================\n')
        show_dir('../')
        print('')
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        # Create the output directory, if it does not already exist and open output files
    
    if verbose:
        print(f'Using {output_dir} as output directory')
        print(f'Using {input_dir} as input directory')
    
    results_dir = os.path.join(input_dir, 'res')
    reference_data_dir = os.path.join(input_dir, 'ref')
    if len(argv) == 1:
        reference_data_dir = input_dir
        results_dir = 'sample_output_data'
        
    with open(os.path.join(reference_data_dir, 'phase'), "r") as f:
        phase = f.read()
    
    if verbose:
        print('\nReading solution file...')
    
    with mute_print():
        field = ap.Field3D(reference_data_dir)
    if verbose:
        print('Reading submitted solution results...')
    alpha_t = ap.process_file(os.path.join(results_dir, 'alpha_t'+phase+'.dat'))
    with open(os.path.join(results_dir, 'inference_time'+phase+'.txt'), "r") as f:
        inference_time = float(f.read())
    with open(os.path.join(results_dir, 'training_time.txt'), "r") as f:
        training_time = float(f.read())
    inference_time_per_sample = inference_time/len(field.RHO.value)
    
    
    if verbose:
        print('\n====================================')
        print('======== COMPUTING METRICS =========')
        print('====================================\n')
    if verbose:
        print('Creating output directory...')
    mkdir(output_dir)
    if verbose:
        print('Computing MSE loss...')
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    C_grad = np.hstack([field.C_grad_X.reshape_column(), field.C_grad_Y.reshape_column(), field.C_grad_Z.reshape_column()])
    Tau    = np.hstack([field.TAU_C_X.reshape_column(), field.TAU_C_Y.reshape_column(), field.TAU_C_Z.reshape_column()])
    mse    = loss(alpha_t, field.RHO.value, C_grad, Tau)
    score  = beta_MSE*mse + beta*inference_time_per_sample
    
  
    # ======== Write the scores in the score file
    if verbose:
        print('Writing scoring metrics to file...')
    results = {
        'score': score,
        'MSE': mse,
        'time_inference': inference_time_per_sample*1e6,  # Scaling factor because only two decimal digits are shown
        'time_training': training_time
    }

    for key, value in results.items():
        score_file.write(f"{key}: {value}\n")

    score_file.close()
    
    t2 = time.time()
    overall_time_spent = t2-t1
    
    if verbose:
        print("\n[+] Done")
        print("[+] Overall time spent %5.2f sec " % overall_time_spent )

    
    exit(0)

