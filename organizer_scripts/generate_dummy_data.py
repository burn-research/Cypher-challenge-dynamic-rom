#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a tiny SYNTHETIC dataset with the exact same folder structure and
array shapes as the real one, so you can sanity-check that
ingestion_program + scoring_program run end-to-end BEFORE plugging in your
real (and much heavier) simulation data, and before uploading anything to
Codabench.

This does not need pyvista and runs in a few seconds.
"""

import os
import json
import numpy as np

np.random.seed(0)

BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "..", "bundle")

FEATURES = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
N_FEATURES = len(FEATURES)
N_CELLS = 400          # tiny grid, just for a fast local smoke test
N_COLS = N_FEATURES * N_CELLS

TRAIN_SIMS = {"dummy_sineSweep_A02": 60, "dummy_sineSweep_A04": 60}
VALID_SIMS = {"dummy_step_A03": 30,      "dummy_sine_f40_A05": 30}
TEST_SIMS  = {"dummy_step_A03": 30,      "dummy_sine_f10_A03": 30,
              "dummy_step_A05": 30,      "dummy_sine_f10_A05": 30,
              "dummy_sine_f40_A03": 30,  "dummy_sine_f40_A05": 30}

def fake_trajectory(nt):
    """Smooth-ish random trajectory, just for shape/plumbing testing."""
    base = np.random.randn(N_COLS).astype(np.float32)
    drift = np.cumsum(np.random.randn(N_COLS, nt).astype(np.float32) * 0.01, axis=1)
    return base[:, None] + drift
    


def main():
    input_data_dir = os.path.join(BUNDLE_DIR, "input_data")
    reference_data_dir = os.path.join(BUNDLE_DIR, "reference_data")

    xyz = np.zeros((N_CELLS, 3), dtype=np.float32)
    xyz[:, 0] = np.linspace(0, 0.025, N_CELLS)
    np.save(os.path.join(input_data_dir, "xyz.npy"), xyz)

    meta = {"n_features": N_FEATURES, "features": FEATURES, "n_cells": N_CELLS}
    for phase in ["valid", "test"]:
        d = os.path.join(reference_data_dir, phase)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "phase"), "w") as f:
            f.write(phase)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)

    for name, nt in TRAIN_SIMS.items():
        d = os.path.join(input_data_dir, "train", name)
        os.makedirs(d, exist_ok=True)
        state = fake_trajectory(nt)
        phi = 1.0 + 0.3 * np.sin(np.linspace(0, 10, nt)).astype(np.float32)
        np.save(os.path.join(d, "state.npy"), state)
        np.save(os.path.join(d, "phi.npy"), phi)

    for split, sims in [("valid", VALID_SIMS), ("test", TEST_SIMS)]:
        for name, nt in sims.items():
            in_d = os.path.join(input_data_dir, split, name)
            os.makedirs(in_d, exist_ok=True)
            state = fake_trajectory(nt)
            phi = 1.0 + 0.3 * np.sin(np.linspace(0, 10, nt)).astype(np.float32)
            np.save(os.path.join(in_d, "initial_state.npy"), state[:, 0])
            np.save(os.path.join(in_d, "phi.npy"), phi)

            ref_d = os.path.join(reference_data_dir, split, name)
            os.makedirs(ref_d, exist_ok=True)
            np.save(os.path.join(ref_d, "state_full.npy"), state)

    print("Dummy dataset generated under bundle/input_data and "
          "bundle/reference_data.")


if __name__ == "__main__":
    main()
