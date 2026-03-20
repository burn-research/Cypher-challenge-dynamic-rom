# Score library for NUMPY arrays
# ChaLearn AutoML challenge

# For regression:
# solution and prediction are vectors of numerical values of the same dimension

# For classification:
# solution = array(p,n) of 0,1 truth values, samples in lines, classes in columns
# prediction = array(p,n) of numerical scores between 0 and 1 (analogous to probabilities)

# Isabelle Guyon and Arthur Pesah, ChaLearn, August-November 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

import os
from sys import stderr
from sys import version

import numpy as np
import scipy as sp
from sklearn import metrics
from sklearn.preprocessing import *

swrite = stderr.write
from os import getcwd as pwd
#from pip import get_installed_distributions as lib # TODO: UPDATE
from glob import glob
import platform
import psutil

def loss(alpha_t, rho, C_grad, Tau):
    # Convert inputs to numpy arrays (if they aren't already)
    alpha_t = np.asarray(alpha_t)
    rho = np.asarray(rho)
    C_grad = np.asarray(C_grad)
    Tau = np.asarray(Tau)
    
    # Check if alpha_t and rho are 1D or 2D column vectors
    def is_vector(x):
        return (x.ndim == 1) or (x.ndim == 2 and x.shape[1] == 1)
 
    if not is_vector(alpha_t):
        raise ValueError("alpha_t must be a 1D array or a 2D column vector (shape (n,) or (n,1)).")
    if not is_vector(rho):
        raise ValueError("rho must be a 1D array or a 2D column vector (shape (n,) or (n,1)).")
 
    # Normalize shape to (n,) for comparison
    alpha_len = alpha_t.shape[0]
    rho_len = rho.shape[0]
 
    # Check if C_grad and Tau are 2D with 3 columns
    if C_grad.ndim != 2 or C_grad.shape[1] != 3:
        raise ValueError("C_grad must be a 2D array with 3 columns.")
    if Tau.ndim != 2 or Tau.shape[1] != 3:
        raise ValueError("Tau must be a 2D array with 3 columns.")
 
    # Check all lengths match
    if not (rho_len == alpha_len == C_grad.shape[0] == Tau.shape[0]):
        raise ValueError("alpha_t, rho, C_grad, and Tau must all have the same number of rows.")
    
    
    alpha_t = alpha_t.reshape(-1,1)
    rho   = rho.reshape(-1, 1)
    pred  = rho*alpha_t*C_grad
    mse   = np.average(((pred-Tau).flatten())**2)
    
    return mse
     

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
