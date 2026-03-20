#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:47:37 2025

@author: lorenzo piu
"""

import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import os
import random
import aPrioriDNS as ap
from utils import product, sample_indices, mute_print


class DataLoader():
    def __init__(self, data_folder, verbose=True):
        self.is_read = False
        self.phi_list =  ['0.4', '0.6', '0.7'] # ["0.4"] # ['0.4', '0.6', '0.7'] # List of equivalence ratios included in the training data
        self.delta_list = ['4', '8'] # ['8'] # ['4', '8']         # List of filter sizes included in the training data
        
        self.read(data_folder)

    
    def read(self, data_folder, verbose=True):
        if verbose:
            print(f'\nReading data in {data_folder}...')
        self.data_dict = dict()
        
        if self.is_read:
            print('The object already read a file directory. Use self.clean() to delete the existing data_dict')
            pass
        
        # Loop over the various fields
        for phi in self.phi_list:
            for delta in self.delta_list:
                dir_name = os.path.join(data_folder, f'Phi{phi}', f'Filter{delta}FavreGaussDS')
                if not os.path.exists(dir_name):
                    raise ValueError(f'Directory {dir_name} not found')
                # In the following the field is initialized using the aPriori package for simplicity,
                #  as the data formatting is coherent with Blastnet's data and hence compatible
                # with the library. The data can be read in many other ways, considering that
                # all the .dat files can be read with the following function: 
                # np.fromfile(file_path,dtype='<f4')
                with mute_print():
                    self.data_dict[f'Phi{phi}_Delta{delta}'] = ap.Field3D(dir_name)
        
        self.is_read = True
        
        self.compute_min_size()
        
        if verbose:
            print(f'Minimum field size is {self.min_size}')
     
    def compute_min_size(self):
        if not self.is_read:
            raise ValueError('Read a field before computing the minimum field size')
        sizes = list()
        for key in self.data_dict:
            field = self.data_dict[key]
            sizes.append(product(field.shape))
            
        self.min_size = min(sizes)
        
    def clean(self):
        if hasattr(self, 'data_dict'):
            delattr(self, 'data_dict')
        if self.is_read:
            self.is_read = False
    

class DataManager(DataLoader):
    def build_tensor(self, 
                      field,
                      attributes,
                      idx
                      ):
        
        # field = self.data_dict[key]
        
        dim0 = product(field.shape)
        dim1 = len(attributes)
        
        X = np.zeros([dim0, dim1], dtype=np.float32)
        
        for i, attr in enumerate(attributes):
            X[:, i] = getattr(field, attr).value
        
        return X[idx, :]
        
        
    def build_training_data(self, 
                            X_attributes,
                            Y_attributes,
                            ):
        
        self.X_attributes = X_attributes
        X = None
        
        self.Y_attributes = Y_attributes
        Y = None
        
        # ======= Builds X and Y tensors stacking all the fields
        for key in self.data_dict:
            
            field = self.data_dict[key]
            
            dim0 = product(field.shape)
            idx = sample_indices(self.min_size, dim0)
            
            X_temp = self.build_tensor(field, X_attributes, idx)
            if X is None:
                X = X_temp
            else:
                X = np.vstack([X, X_temp])
            del X_temp # release memory
            
            Y_temp = self.build_tensor(field, Y_attributes, idx)
            if Y is None:
                Y = Y_temp
            else:
                Y = np.vstack([Y, Y_temp])
            del Y_temp # release memory
        
        self.X = X
        self.Y = Y

        
