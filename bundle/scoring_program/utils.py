#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:57:31 2025

@author: lorenzo piu
"""

import sys
import os
from contextlib import contextmanager

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

