#!/usr/bin/env python

# Main contributor for this version: Lorenzo Piu, Université Libre de Bruxelles,
# May 2025

# The code was adapted from an existing bundle provided by codabench's developers
# at the official github page: 
# https://github.com/codalab/competition-examples/tree/master/codabench/iris

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.


# Input data structure:
# the input data folder has the following structure:
# input_data
# ├── test
# │   └── Phi0.5
# │       └── Filter6FavreGaussDS
# ├── train
# │   ├── Phi0.4
# │   │   ├── Filter4FavreGaussDS
# │   │   └── Filter8FavreGaussDS
# │   ├── Phi0.6
# │   │   ├── Filter4FavreGaussDS
# │   │   └── Filter8FavreGaussDS
# │   └── Phi0.7
# │       ├── Filter4FavreGaussDS
# │       └── Filter8FavreGaussDS
# └── valid
#     └── Phi0.4
#         └── Filter8FavreGaussDS
# training data contain both the input variables (temperature, pressure,
# progress variable, etc...) and the ground truth progress variable sub-filter
# fluxes (TAU_C_X, TAU_C_Y and TAU_C_Z), while the testing and validation data
# solely include the input data.

# Submission:
# The ingestion code accepts submissions that contain an UNTRAINED model, that 
# can rely on the libraries Tensorflow, ScikitLearn, and pytorch, depending
# on the participants' preferences.
# The compressed folder submitted must contain a file named model.py, that 
# contains a class model() constituting the predictive model. This class MUST
# have the following methods, that will be called in the ingestion program:
# 1. preprocess() ==> this method takes as input only the relative path pointing to
#    the directory "input_data" (see above). The method ingest the data in the 
#    folder, and gives as output an object, that contains the processed training
#    data and is needed by the next method
# 2. fit() ==> this method trains the model based on the input object that the
#    previous method gave as output. 
# 3. predict() ==> this method takes as input the relative path of a folder containing
#    the 


# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, October 2017

# =========================== BEGIN OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# This list is actually not updated
# 0: run the code normally, using the time budget of the tasks
# >0: prints additional information useful for debugging
debug_mode = 0

# Save previous results:
########################
# If set to true the old results are saved in a different output folder.
# useful if running the code locally
save_previous_results = False

# List libraries and python version:
####################################
# When set to True, shows all the available libraries in the docker with the 
# respective version
show_versions = False

# Time budget
#############
# max_time is the maximum time available to train the model in seconds.
# The code should keep track of time spent and NOT exceed the time limit
# If debug >=1, you can decrease the maximum time (in sec) with the 
# variable debug_time
max_time = 21600   # 6 hours
debug_time = 1000

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
# save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "./"
default_input_dir = root_dir + "input_data"
default_output_dir = root_dir + "sample_output_data"
default_program_dir = root_dir + "ingestion_program"
default_submission_dir = root_dir + "sample_code_submission"
default_data_dir = root_dir + 'data_directory'
test_dir = 'test/Phi0.5/Filter6FavreGaussDS'
valid_dir = 'valid/Phi0.4/Filter8FavreGaussDS'

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 1

# General purpose functions
import time
overall_start = time.time()         # <== Mark starting time
import os
import sys
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
import aPrioriDNS as ap


# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir

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
        print("Using data_dir: " + default_data_dir)
    if verbose:
        print('\nImporting modules...')

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)
    import data_io                       # general purpose input/output functions
    from data_io import vprint           # print only in verbose mode
    from data_io import show_version, cpdir, check_model_interface, rmdir
    if len(argv)==1: # When running in local add the relative import dir for debugging purposes
        sys.path.append('/Users/lolli/Desktop/ENCODING/Workdir/15-Cypher-challenge/CodaBench/sample_code_submission')
    from model import model
        
    if show_versions:
        show_version()

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)


    print('\n*****************************************')
    print('****** Ingestion program version ' + str(version) + ' ******')
    print('*******************************************\n')
    
    
    # ======== List directories
    if debug_mode > 1:
        print(f'Debugging mode: {debug_mode}')
        print('\n====================================')
        print('======== LISTING DIRECTORIES =========')
        print('====================================\n')
        data_io.show_dir('../')
        print('')

    # ======== Keeping track of time
    # When debugging assing infinite time for the task
    if debug_mode<1:
        time_budget = max_time  # <== HERE IS THE TIME BUDGET!
    else:
        time_budget = debug_time
        
    vprint(verbose, f"Time available for training: {time_budget} seconds" )
    
    # ======== Copy input folder to avoid modifications to the input data
    # This step could be redundant but does not seem to take excessive time
    vprint( verbose,  "\nCopying input data directory...")
    if os.path.exists(default_data_dir):
        rmdir(default_data_dir)
    cpdir(input_dir, default_data_dir)
    
    
    # ======== Loading model
    vprint(verbose, '\nInitializing model...')
    M = model()
    vprint(verbose, 'Model initialized successfully')
    
    vprint(verbose, '\nChecking model structure...')
    check_model_interface(M, verbose=verbose)
    vprint(verbose, 'Model structure OK')
    
    vprint(verbose, '\n====================================')
    vprint(verbose, '========== MODEL TRAINING ==========')
    vprint(verbose, '====================================\n')
    
    
    # ========= Running preprocessing and training
    with data_io.time_limit(time_budget):
        
        vprint(verbose, 'Preprocessing data...')
        t1 = time.time()
        D = M.preprocess(default_data_dir + '/train')
        
        vprint(verbose, '\nBeginning model training...')
        M.fit(D)
        if debug_mode > 2:
            time.sleep(10) # Add fictitious time to the training process
        
    t2 = time.time()
    training_time = t2-t1
    tt, unit = data_io.process_training_time(training_time)
    vprint(verbose, f'\nModel trained successfully in {tt:.2f} {unit}')
    
    vprint(verbose, '\n===================================')
    vprint(verbose, '========== TESTING MODEL ==========')
    vprint(verbose, '===================================\n')
    
    # Testing on validation data
    vprint(verbose, 'Starting inference on validation data...')
    t_start_valid = time.time()
    alpha_t_valid = M.predict(os.path.join(default_data_dir, valid_dir))
    t_end_valid = time.time()    
    inference_time_valid = t_end_valid-t_start_valid
    tt, unit = data_io.process_training_time(inference_time_valid)
    print(f'Model inference done in {tt:.2f} {unit}')
    ap.save_file(alpha_t_valid, file_name=os.path.join(output_dir, 'alpha_t_valid.dat'))
    with open(os.path.join(output_dir,"inference_time_valid.txt"), "w") as f:
        f.write(str(inference_time_valid))
        
    # Testing on testing data
    vprint(verbose, '\nStarting inference on testing data...')
    t_start_test = time.time()
    alpha_t_test = M.predict(os.path.join(default_data_dir, test_dir))
    t_end_test = time.time()
    inference_time_test = t_end_test-t_start_test
    tt, unit = data_io.process_training_time(inference_time_test)
    print(f'Model inference done in {tt:.2f} {unit}')
    ap.save_file(alpha_t_test, file_name=os.path.join(output_dir, 'alpha_t_test.dat'))
    with open(os.path.join(output_dir,"inference_time_test.txt"), "w") as f:
        f.write(str(inference_time_test))
    
    # Save training time
    with open(os.path.join(output_dir,"training_time.txt"), "w") as f:
        f.write(str(training_time))
    
    overall_time_spent = time.time() - overall_start

    vprint( verbose,  "\n[+] Done")
    vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget for training %5.2f sec" % time_budget)


    exit(0)