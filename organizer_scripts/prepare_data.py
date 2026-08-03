#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORGANIZER-ONLY script. Do NOT include this in the Codabench bundle, and do
NOT give it to participants: it is what turns your raw CFD dumps (.npy +
.vtu, one folder per simulation, arbitrary unstructured grid) into the
standardized files expected by bundle/ingestion_program and
bundle/scoring_program:

  bundle/input_data/xyz.npy, grid.vtu
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
         split="train", signal="sweep", A=0.2, f=None, dt=1e-3, t0=0.0),
    dict(name="sineSweep_f1_f80_A04", raw_data_path="/globalscratch/baffetti/hackaton/Data/sineSweep_f1_f80_A04.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="train", signal="sweep", A=0.4, f=None, dt=1e-3, t0=0.0),

    dict(name="step_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/step_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="valid", signal="step", A=0.3, f=None, dt=1e-3, t0=0.0),
    dict(name="sine_f40_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f40_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="valid", signal="sine", A=0.5, f=40.0, dt=1e-3, t0=0.0),

    dict(name="step_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/step_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="step", A=0.3, f=None, dt=1e-3, t0=0.0),
    dict(name="step_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/step_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="step", A=0.5, f=None, dt=1e-3, t0=0.0),
    dict(name="sine_f10_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f10_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.3, f=10.0, dt=1e-3, t0=0.0),
    dict(name="sine_f10_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f10_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.5, f=10.0, dt=1e-3, t0=0.0),
    dict(name="sine_f40_A03", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f40_A03.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.3, f=40.0, dt=1e-3, t0=0.0),
    dict(name="sine_f40_A05", raw_data_path="/globalscratch/baffetti/hackaton/Data/sine_f40_A05.npy",
         raw_grid_path="/globalscratch/baffetti/hackaton/Data/grid_0_0 6.vtu",
         split="test", signal="sine", A=0.5, f=40.0, dt=1e-3, t0=0.0),
]

FEATURES = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
N_FEATURES = len(FEATURES)

BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "..", "bundle")

RAW_DT = 5e-4
TARGET_DT = 20e-4
STRIDE = round(TARGET_DT / RAW_DT)   # = 2
assert abs(STRIDE * RAW_DT - TARGET_DT) < 1e-12, \
    "TARGET_DT must be an exact integer multiple of RAW_DT"

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

def npy_shape(path):
    """
    It returns the shape of a .npy file saved on disk WITHOUT loading it
    all into RAM (mmap_mode='r' reads only the header). Returns None if the
    file does not exist or is unreadable/corrupted (e.g. interrupted write).
    """
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path, mmap_mode='r')
        return arr.shape
    except Exception as e:
        print(f"  [WARNING] {path} exists but is not readable ({e}); it will be reconstructed.")
        return None

def npz_shape(path, key='data'):
    """
    Array `key` shape inside a .npz file, reading only the header (~128B),
    without decompressing the data. Returns None if the file/key does not exist or is corrupted."""
    if not os.path.exists(path):
        return None
    try:
        import zipfile
        from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
        with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as f:
            version = read_magic(f)
            shape, _, _ = (read_array_header_1_0(f) if version == (1, 0)
                            else read_array_header_2_0(f))
            return shape
    except Exception as e:
        print(f"  [WARNING] {path} exists but is not readable ({e}); it will be reconstructed.")
        return None
 
def outputs_ready(sim, input_data_dir, reference_data_dir):
    """
    True : IF all expected output files for this simulation already exist on disk
           AND have the correct shape -> we can skip resampling for this sim.
    False: otherwise.
    
    """
    split = sim["split"]
    sim_input_dir = os.path.join(input_data_dir, split, sim["name"])
 
    # nt: number of time steps in the simulation. We read it from the raw .npy
    # file header, because the resampling step does not change nt (it only
    # changes the number of cells).
    raw_shape = npy_shape(sim["raw_data_path"])
    if raw_shape is None:
        # we cannot read the raw .npy file -> we cannot verify the output shapes, better to (re)process
        return False
    n_rows, nt_raw = raw_shape
    nt_new = len(range(0, nt_raw, STRIDE))          # nt after temporal subsampling
    expected_shape = (n_rows, nt_new)

    if split == "train":
        state_shape = npz_shape(os.path.join(sim_input_dir, "state.npz"))
        phi_shape = npz_shape(os.path.join(sim_input_dir, "phi.npz"))
        return state_shape == expected_shape and phi_shape == (nt_new,)
    else:
        initial_shape = npz_shape(os.path.join(sim_input_dir, "initial_state.npz"))
        phi_shape = npz_shape(os.path.join(sim_input_dir, "phi.npz"))
        ref_sim_dir = os.path.join(reference_data_dir, split, sim["name"])
        full_shape = npz_shape(os.path.join(ref_sim_dir, "state_full.npz"))
        return (initial_shape == (n_rows,)
                and phi_shape == (nt_new,)
                and full_shape == expected_shape)

# =============================================================================
# 4) MAIN
# =============================================================================

def main():
    input_data_dir = os.path.join(BUNDLE_DIR, "input_data")
    reference_data_dir = os.path.join(BUNDLE_DIR, "reference_data")
    os.makedirs(input_data_dir, exist_ok=True)

    # --- grid + cells positions: shared by all simulations, so we can compute 
    #     it once and save it to disk (assuming the same .vtu file for all
    #     simulations, as in your current RAW_SIMULATIONS) ---
    raw_grid_path = RAW_SIMULATIONS[0]["raw_grid_path"]
    xyz_path = os.path.join(input_data_dir, "xyz.npz")
    grid_dest = os.path.join(input_data_dir, "grid.vtu")

    if not os.path.exists(xyz_path) or not os.path.exists(grid_dest):
        grid = pv.read(raw_grid_path)
        xyz = grid.cell_centers().points   # row i = position (x,y,z) of cell i
        np.savez_compressed(xyz_path, data=xyz.astype(np.float32))
        shutil.copy2(raw_grid_path, grid_dest)   # copy also the raw .vtu
        n_cells = xyz.shape[0]
        print(f"Grid: {n_cells} cells (raw, unstructured, no resampling)")
    else:
        n_cells = npz_shape(xyz_path)[0]
        print(f"Grid already present: {n_cells} cells")

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

        if outputs_ready(sim, input_data_dir, reference_data_dir):
            print("  -> output already exists and has correct shape, skip.")
            continue

        raw_shape = npy_shape(sim["raw_data_path"])
        nt_new = len(range(0, raw_shape[1], STRIDE))
        phi = compute_phi(sim["signal"], nt_new, sim["dt"], sim["A"],
                           f=sim.get("f"), t0=sim.get("t0", 0.0))

        split = sim["split"]
        sim_input_dir = os.path.join(input_data_dir, split, sim["name"])
        os.makedirs(sim_input_dir, exist_ok=True)

        if split == "train":
            # pass-through, solo sottocampionamento temporale (nessun resampling spaziale)
            state = np.load(sim["raw_data_path"])[:, ::STRIDE].astype(np.float32)
            np.savez_compressed(os.path.join(sim_input_dir, "state.npz"), data=state)
            np.savez_compressed(os.path.join(sim_input_dir, "phi.npz"), data=phi.astype(np.float32))
        else:
            # participants only ever see the initial snapshot + full phi
            DataMatrix = np.load(sim["raw_data_path"], mmap_mode='r')  # do not load all into RAM
            np.savez_compressed(os.path.join(sim_input_dir, "initial_state.npz"), data=np.array(DataMatrix[:, 0], dtype=np.float32))
            np.savez_compressed(os.path.join(sim_input_dir, "phi.npz"), data=phi.astype(np.float32))

            # ground truth goes ONLY into reference_data, never into input_data
            ref_sim_dir = os.path.join(reference_data_dir, split, sim["name"])
            os.makedirs(ref_sim_dir, exist_ok=True)
            state_full = np.load(sim["raw_data_path"])[:, ::STRIDE].astype(np.float32)
            np.savez_compressed(os.path.join(ref_sim_dir, "state_full.npz"), data=state_full)

        print(f"  -> raw nt={raw_shape[1]} -> subsampled nt={nt_new}, phi {phi.shape}, split={split}")

    print("\nDone. You can now zip bundle/ (input_data and reference_data "
          "included) and upload it to Codabench, or use it for local testing.")


if __name__ == "__main__":
    main()