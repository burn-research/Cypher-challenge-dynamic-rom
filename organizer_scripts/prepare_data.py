#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORGANIZER-ONLY script. Do NOT include this in the Codabench bundle, and do
NOT give it to participants: it is what turns your raw CFD dumps (.npy +
.vtu, one folder per simulation, arbitrary unstructured grid) into the
standardized files expected by bundle/ingestion_program and
bundle/scoring_program:

  bundle/input_data/grid.npy
  bundle/input_data/train/<sim_name>/state.npy, phi.npy
  bundle/input_data/valid/<sim_name>/initial_state.npy, phi.npy
  bundle/input_data/test/<sim_name>/initial_state.npy, phi.npy
  bundle/reference_data/valid/phase, meta.json, <sim_name>/state_full.npy
  bundle/reference_data/test/phase, meta.json, <sim_name>/state_full.npy

Run this ONCE locally, then copy/zip the resulting bundle/ folder (input_data
and reference_data included) as-is when you create the competition on
Codabench. This script itself must never be uploaded anywhere participants
can reach it, since it (indirectly) touches the test ground truth.

Requirements: numpy, pyvista  (pip install pyvista)

USAGE: python prepare_data.py
"""

import os
import shutil
import json
import numpy as np
import pyvista as pv

# =============================================================================
# 1) CONFIGURATION -- EDIT THIS SECTION FOR YOUR RAW DATA LAYOUT
# =============================================================================

# Where your raw simulation dumps live. Each entry needs:
#   name          : short id used as the folder name in the bundle
#   raw_data_path : path to the raw DataMatrix .npy (shape (n_features*n_cells_raw, nt))
#   raw_grid_path : path to the corresponding .vtu grid file
#   split         : "train", "valid" or "test"
#   signal        : "sweep", "step" or "sine"       (used to compute phi(t))
#   A             : forcing amplitude
#   f             : forcing frequency in Hz (sine only, ignored otherwise)
#   dt            : simulation time step in seconds
#   t0            : step/sweep start time (s), see U_code snippets

RAW_SIMULATIONS = [
    dict(name="sineSweep_f1_f80_A02", raw_data_path="/globalscratch/baffetti/hackaton/Data/sineSweep_f1_f80_A02.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="train", signal="sweep", A=0.2, f=None, dt=5e-4, t0=0.0),
    dict(name="sineSweep_f1_f80_A04", raw_data_path="/globalscratch/baffetti/hackaton/Data/sineSweep_f1_f80_A04.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="train", signal="sweep", A=0.4, f=None, dt=5e-4, t0=0.0),

    dict(name="step_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/step_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="valid", signal="step", A=0.3, f=None, dt=5e-4, t0=0.0),
    dict(name="sine_f40_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f40_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="valid", signal="sine", A=0.5, f=40.0, dt=5e-4, t0=0.0),

    dict(name="step_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/step_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="step", A=0.3, f=None, dt=5e-4, t0=0.0),
    dict(name="step_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/step_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="step", A=0.5, f=None, dt=5e-4, t0=0.0),
    dict(name="sine_f10_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f10_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.3, f=10.0, dt=5e-4, t0=0.0),
    dict(name="sine_f10_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f10_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.5, f=10.0, dt=5e-4, t0=0.0),
    dict(name="sine_f40_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f40_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.3, f=40.0, dt=5e-4, t0=0.0),
    dict(name="sine_f40_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f40_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.5, f=40.0, dt=5e-4, t0=0.0),
]

FEATURES = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
N_FEATURES = len(FEATURES)

# Uniform resampling grid (edit to match your domain / resolution needs)
X_MIN, X_MAX, DX = 0.0, 0.025, 0.025 / 64
Z_MIN, Z_MAX, DZ = 0.0, 0.1, 0.1 / (64 * 4)

BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "..", "bundle")


# =============================================================================
# 2) FORCING SIGNAL phi(t) -- translated from the OpenFOAM U_code snippets
# =============================================================================

def phi_step(t, A, t0=0.0):
    """phi = 1 before t0, 1+A after (step change)."""
    t = np.asarray(t, dtype=np.float64)
    return np.where(t < t0, 1.0, 1.0 + A)


def phi_sine(t, A, f, tau=0.0):
    """phi = 1 + A*sin(2*pi*f*(t - tau))."""
    t = np.asarray(t, dtype=np.float64)
    return 1.0 + A * np.sin(2 * np.pi * f * (t - tau))


def phi_sweep(t, A, t0=0.0, t1=1.0, t2=2.0):
    """Chirp: linear-in-log frequency sweep 1Hz -> 80Hz -> 1Hz, amplitude A."""
    t = np.asarray(t, dtype=np.float64)
    phi = np.ones_like(t)

    # segment 1: t0 <= t < t1, 1Hz -> 80Hz
    f0, f1, dur = 1.0, 80.0, (t1 - t0)
    k = np.log(f1 / f0) / dur
    mask1 = (t >= t0) & (t < t1)
    rel1 = t[mask1] - t0
    phase1 = 2 * np.pi * f0 * (np.exp(k * rel1) - 1.0) / k
    phi[mask1] = 1.0 + A * np.sin(phase1)

    # segment 2: t1 <= t <= t2, 80Hz -> 1Hz, phase-continuous with segment 1
    phase_at_t1 = 2 * np.pi * f0 * (np.exp(k * dur) - 1.0) / k
    f0_2, f1_2, dur2 = 80.0, 1.0, (t2 - t1)
    k2 = np.log(f1_2 / f0_2) / dur2
    mask2 = (t >= t1) & (t <= t2)
    rel2 = t[mask2] - t1
    phase2 = phase_at_t1 + 2 * np.pi * f0_2 * (np.exp(k2 * rel2) - 1.0) / k2
    phi[mask2] = 1.0 + A * np.sin(phase2)

    return phi


def compute_phi(signal, nt, dt, A, f=None, t0=0.0):
    t = np.arange(nt) * dt
    if signal == "step":
        return phi_step(t, A, t0=t0)
    elif signal == "sine":
        return phi_sine(t, A, f)
    elif signal == "sweep":
        return phi_sweep(t, A, t0=t0, t1=t0 + 1.0, t2=t0 + 2.0)
    else:
        raise ValueError(f"Unknown signal type '{signal}'")


# =============================================================================
# 3) RESAMPLING onto the shared uniform grid
# =============================================================================

def build_uniform_grid():
    x_samples = np.arange(X_MIN, X_MAX, DX)
    z_samples = np.arange(Z_MIN, Z_MAX, DZ)
    xg, zg = np.meshgrid(x_samples, z_samples)
    xyz_samples = np.zeros((xg.size, 3))
    xyz_samples[:, 0] = xg.flatten()
    xyz_samples[:, 2] = zg.flatten()
    return xyz_samples


def resample_simulation(raw_data_path, raw_grid_path, xyz_samples):
    """
    Resample one raw simulation (unstructured grid) onto the shared uniform
    grid. Mirrors the manual pipeline already used by the CYPHER team.

    Returns
    -------
    state : ndarray, shape (n_timesteps, N_FEATURES * n_cells_uniform)
    """
    DataMatrix = np.load(raw_data_path)
    grid = pv.read(raw_grid_path)
    xyz = grid.cell_centers().points
    n_cells_raw = xyz.shape[0]
    nt = DataMatrix.shape[1]

    samples = pv.PolyData(xyz_samples)
    n_cells_s = xyz_samples.shape[0]

    Data_samples = np.zeros((N_FEATURES * n_cells_s, nt), dtype=np.float32)
    for f_idx, feat in enumerate(FEATURES):
        for t in range(nt):
            field = DataMatrix[f_idx * n_cells_raw:(f_idx + 1) * n_cells_raw, t]
            grid['f'] = field
            sampled = samples.sample(grid)['f']
            Data_samples[f_idx * n_cells_s:(f_idx + 1) * n_cells_s, t] = sampled
        print(f'  feature {feat} resampled ({f_idx + 1}/{N_FEATURES})')

    return Data_samples


# =============================================================================
# 4) MAIN
# =============================================================================

def main():
    xyz_samples = build_uniform_grid()
    n_cells = xyz_samples.shape[0]
    print(f'Uniform grid: {n_cells} cells')

    input_data_dir = os.path.join(BUNDLE_DIR, "input_data")
    reference_data_dir = os.path.join(BUNDLE_DIR, "reference_data")

    np.save(os.path.join(input_data_dir, "grid.npy"), xyz_samples)

    meta = {"n_features": N_FEATURES, "features": FEATURES, "n_cells": n_cells}
    for phase in ["valid", "test"]:
        phase_ref_dir = os.path.join(reference_data_dir, phase)
        os.makedirs(phase_ref_dir, exist_ok=True)
        with open(os.path.join(phase_ref_dir, "phase"), "w") as f:
            f.write(phase)
        with open(os.path.join(phase_ref_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    for sim in RAW_SIMULATIONS:
        print(f"\n=== Processing {sim['name']} ({sim['split']}) ===")
        state = resample_simulation(sim["raw_data_path"], sim["raw_grid_path"],
                                     xyz_samples)
        nt = state.shape[1]
        phi = compute_phi(sim["signal"], nt, sim["dt"], sim["A"],
                           f=sim.get("f"), t0=sim.get("t0", 0.0))

        split = sim["split"]
        sim_input_dir = os.path.join(input_data_dir, split, sim["name"])
        os.makedirs(sim_input_dir, exist_ok=True)

        if split == "train":
            np.save(os.path.join(sim_input_dir, "state.npy"), state.astype(np.float32))
            np.save(os.path.join(sim_input_dir, "phi.npy"), phi.astype(np.float32))
        else:
            # participants only ever see the initial snapshot + full phi
            np.save(os.path.join(sim_input_dir, "initial_state.npy"),
                    state[:, 0].astype(np.float32))
            np.save(os.path.join(sim_input_dir, "phi.npy"), phi.astype(np.float32))

            # ground truth goes ONLY into reference_data, never into input_data
            ref_sim_dir = os.path.join(reference_data_dir, split, sim["name"])
            os.makedirs(ref_sim_dir, exist_ok=True)
            np.save(os.path.join(ref_sim_dir, "state_full.npy"),
                    state.astype(np.float32))

        print(f"  -> state {state.shape}, phi {phi.shape}, split={split}")

    print("\nDone. You can now zip bundle/ (input_data and reference_data "
          "included) and upload it to Codabench, or use it for local testing.")


if __name__ == "__main__":
    main()
