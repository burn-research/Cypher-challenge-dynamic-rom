#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:51:59 2025

@author: lorenzo piu
"""

import numpy as np
import os
import random
import sys
from contextlib import contextmanager



def set_global_seed(seed: int):
    """Set seed for Python, NumPy, PyTorch, and TensorFlow to ensure reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python built-in random module
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    
def product(values):
    """
    Calculate the product of all numeric values in a list.

    Parameters:
        values (list): A list of numeric values (int or float).

    Returns:
        float|int: The product of all the values in the list.

    Raises:
        TypeError: If the input is not a list or contains non-numeric elements.
    """
    if not isinstance(values, list):
        raise TypeError("Input must be a list.")

    result = 1
    for v in values:
        if not isinstance(v, (int, float)):
            raise TypeError(f"All elements must be int or float, got {type(v).__name__}.")
        result *= v

    return result


def sample_indices(n, m):
    """
    Return a list of n unique random integers from the range [0, m-1].

    Parameters:
        n (int): Number of integers to extract.
        m (int): Upper bound (exclusive) of the range to sample from.

    Returns:
        list[int]: A list of n unique integers from 0 to m-1.

    Raises:
        ValueError: If n > m or if n/m are not positive integers.
    """
    if not (isinstance(n, int) and isinstance(m, int)):
        raise TypeError("Both n and m must be integers.")
    if n > m:
        raise ValueError("Cannot sample more elements than the size of the range (n > m).")
    if n < 0 or m <= 0:
        raise ValueError("n must be non-negative and m must be positive.")
    
    return sorted(random.sample(range(m), n))


# Context manager to mute print statements
@contextmanager
def mute_print():
    # Backup the original stdout
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull
    try:
        yield  # Execute the block of code inside the context manager
    finally:
        # Restore the original stdout after the block runs
        sys.stdout = original_stdout

