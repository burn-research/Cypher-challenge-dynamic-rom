"""
Sample predictive model for the CYPHER 2026 dynamic ROM challenge.

Your submitted model.py must define a class named `model` implementing
3 methods: preprocess(), fit(), predict(). See the "Submission" page of the
competition for the exact contract.

This example implements a trivial PERSISTENCE baseline: it predicts that
the flow state never changes, i.e. every future snapshot is equal to the
initial one. It completely ignores the forcing signal phi(t) and will
obviously fail to capture the flame dynamics -- it exists only to show you
a minimal, working submission.

To build a real model you will typically want to:
  1. In preprocess(): load every training simulation (see utils.load_simulation),
     build supervised pairs (see utils.build_one_step_pairs, or design your
     own multi-step / sequence-to-sequence scheme), and fit a scaler.
  2. In fit(): train a model (MLP, RNN/LSTM, Neural ODE, operator network,
     ...) that maps [state_t, phi_t, phi_{t+1}] (or a longer history) to
     state_{t+1}. Remember: phi carries the information about which forcing
     regime the system is in (step / sine / sweep, amplitude, frequency) --
     without it, the model has no way to know what is about to happen.
  3. In predict(): starting from initial_state, roll your model forward
     autoregressively, feeding it the known future phi(t) values one step
     at a time, until you have produced a forecast for every time step
     given in phi.npz.
"""

import numpy as np
import os
import pyvista
import matplotlib
import torch
import tensorflow
import sklearn
import pandas

from utils import list_training_simulations, load_simulation


class model:
    def __init__(self):
        self.is_trained = False

    def preprocess(self, data_folder):
        """
        Parameters
        ----------
        data_folder : str
            Path to the training data folder (e.g. .../input_data/train),
            containing one sub-folder per training simulation.

        Returns
        -------
        D : any object of your choice, passed on to fit().
        """
        simulations = {}
        for sim_name in list_training_simulations(data_folder):
            state, phi = load_simulation(f"{data_folder}/{sim_name}")
            simulations[sim_name] = {"state": state, "phi": phi}
            print(f"Loaded '{sim_name}': state {state.shape}, phi {phi.shape}")

        return simulations

    def fit(self, D):
        """
        Train the model. This baseline has nothing to learn, but you would
        typically train your network / regressor here using the pairs built
        from D, e.g. with utils.build_one_step_pairs(state, phi).
        """
        self.is_trained = True
        print("Model 'trained' (persistence baseline: nothing to learn).")

    def predict(self, test_data_folder):
        """
        Parameters
        ----------
        test_data_folder : str
            Path to ONE validation/test simulation folder, containing
            initial_state.npz (shape (n_cells, n_features,)) and
            phi.npz (shape (n_timesteps,), the full known forcing signal).

        Returns
        -------
        state_pred : ndarray, shape (n_cells, n_features, n_timesteps)
            Forecast of the full flow state at every time step. Column 0 must
            equal initial_state.npz.
        """
        initial_state = np.load(f"{test_data_folder}/initial_state.npz")['data']
        phi = np.load(f"{test_data_folder}/phi.npz")['data']

        n_cells, n_features = initial_state.shape          # [N_cells, N_features]
        n_timesteps = phi.shape[0]
        state_pred = np.tile(initial_state[:, :, np.newaxis], (1, 1, n_timesteps))
        return state_pred
