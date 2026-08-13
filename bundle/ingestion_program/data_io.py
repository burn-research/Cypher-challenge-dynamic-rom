#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General purpose I/O helper functions for the ingestion program.

Adapted from the CYPHER 2025 DNS challenge (Lorenzo Piu, ULB) for the
CYPHER 2026 dynamic reduced-order-modeling challenge. Edited by the
organizers (Tommaso Baffetti, Alberto Procacci, ULB, 2026).
"""

import os
import shutil
import signal
import sys
from contextlib import contextmanager


def vprint(verbose, message):
    """Print only if verbose is True."""
    if verbose:
        print(message)


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def rmdir(d):
    if os.path.exists(d):
        shutil.rmtree(d)


def cpdir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def mvdir(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)


def show_dir(run_dir):
    """Print a tree view of a directory, for debugging purposes."""
    print('\n=== Listing run dir ===\n')
    run_dir = os.path.abspath(run_dir)

    def tree(dir_path, prefix=''):
        contents = sorted(os.listdir(dir_path))
        pointers = ['|-- '] * (len(contents) - 1) + ['`-- ']
        for pointer, name in zip(pointers, contents):
            path = os.path.join(dir_path, name)
            print(prefix + pointer + name)
            if os.path.isdir(path):
                extension = '|   ' if pointer == '|-- ' else '    '
                tree(path, prefix + extension)

    print(os.path.basename(run_dir) + '/')
    tree(run_dir)


def show_version():
    """Print python version and the version of the main libraries used,
    if installed."""
    print(f'Python version: {sys.version}')
    for lib_name in ['numpy', 'scipy', 'sklearn', 'torch', 'tensorflow']:
        try:
            lib = __import__(lib_name)
            print(f'{lib_name}: {getattr(lib, "__version__", "unknown")}')
        except ImportError:
            print(f'{lib_name}: not installed')


def check_model_interface(M, verbose=True):
    """Check that the submitted model exposes the required interface:
    preprocess(), fit(), predict(). Raises an informative error otherwise."""
    required_methods = ['preprocess', 'fit', 'predict']
    missing = [m for m in required_methods if not hasattr(M, m)]
    if missing:
        raise AttributeError(
            "Your submitted model.py is missing the following required "
            f"method(s): {missing}. The 'model' class must implement "
            "preprocess(self, data_folder), fit(self, D) and "
            "predict(self, test_data_folder). See the Submission page for "
            "details."
        )
    if verbose:
        for m in required_methods:
            vprint(verbose, f'  - found method: {m}()')


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Enforce a time limit on a block of code using a context manager.

    This function uses Unix signals to interrupt execution after a specified
    number of seconds. If the time limit is exceeded, a TimeoutException is raised.

    Args:
        seconds (int): The maximum number of seconds the code block is allowed to run.

    Raises:
        TimeoutException: If the time limit is exceeded during execution.

    Note:
        - This works only on Unix-like systems (Linux/macOS).
        - It must be used in the main thread.
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds!")

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Fallback for platforms without SIGALRM (e.g. Windows): no enforcement.
        yield

def process_training_time(seconds):
    """Converts training time in seconds into a human-readable format.

    Depending on the total training time, this function converts the value
    to seconds, minutes, or hours for readability.

    Args:
        training_time (float): The training time in seconds.

    Returns:
        Tuple[float, str]: A tuple containing:
            - The converted time value (in seconds, minutes, or hours).
            - A string representing the time unit ('seconds', 'minutes', or 'hours').

    Example:
        >>> process_training_time(90)
        (1.5, 'minutes')
    """
    if seconds < 60:
        return seconds, 's'
    elif seconds < 3600:
        return seconds / 60, 'min'
    else:
        return seconds / 3600, 'h'
