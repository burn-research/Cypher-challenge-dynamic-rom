"""
Generic I/O utilities for the ingestion program
"""

import os
import shutil
import sys
import time
import platform
import signal
from contextlib import contextmanager
from sys import stderr

def vprint(verbose, msg):
    ''' Print to stdout, only if in verbose mode'''
    if verbose:
        print(msg)


def mkdir(d):
    ''' Create a new directory'''
    if not os.path.exists(d):
        os.makedirs(d)


def rmdir(d):
    ''' Remove a directory'''
    if os.path.exists(d):
        shutil.rmtree(d)


def mvdir(source, dest):
    ''' Move a directory'''
    if os.path.exists(source):
        if os.path.exists(dest):
            rmdir(dest)
        os.rename(source, dest)


def cpdir(source, dest):
    ''' Copy a directory'''
    if os.path.exists(dest):
        rmdir(dest)
    shutil.copytree(source, dest)

def show_dir(run_dir):
    print('\n=== Listing run dir ===\n')
    run_dir = os.path.abspath(run_dir)

    def tree(dir_path, prefix=''):
        contents = sorted(os.listdir(dir_path))
        pointers = ['├── '] * (len(contents) - 1) + ['└── '] if contents else []
        for pointer, name in zip(pointers, contents):
            path = os.path.join(dir_path, name)
            print(prefix + pointer + name)
            if os.path.isdir(path):
                extension = '│   ' if pointer == '├── ' else '    '
                tree(path, prefix + extension)

    print(os.path.basename(run_dir) + '/')
    tree(run_dir)

def show_version():
    # Python version and library versions
    print('\n=== VERSIONS ===\n\n')
    # Python version
    print(f"Python version: {sys.version}\n")
    print(f"Platform: {platform.platform()}\n")
    for lib in ["numpy", "torch", "tensorflow", "sklearn"]:
        try:
            mod = __import__(lib)
            print(f"{lib}: {getattr(mod, '__version__', 'unknown')}\n")
        except ImportError:
            print(f"{lib}: not installed\n")

def check_model_interface(M, verbose=True):
    """Checks that the submitted model exposes preprocess/fit/predict."""
    required_methods = ['preprocess', 'fit', 'predict']
    missing = [m for m in required_methods if not hasattr(M, m) or not callable(getattr(M, m))]
    if missing:
        raise AttributeError(
            "The submitted model.py is missing required method(s): " + ", ".join(missing)
        )
    if verbose:
        print("All required methods found: " + ", ".join(required_methods))

# ===================== Time handling utilities ========================

def process_training_time(t_seconds):
    if t_seconds < 60:
        return t_seconds, 's'
    elif t_seconds < 3600:
        return t_seconds / 60, 'min'
    else:
        return t_seconds / 3600, 'h'


@contextmanager
def time_limit(seconds):
    """Raises TimeoutError if the wrapped block runs longer than `seconds`."""
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds!")
    has_alarm = hasattr(signal, 'SIGALRM')
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(seconds))
    try:
        yield
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
