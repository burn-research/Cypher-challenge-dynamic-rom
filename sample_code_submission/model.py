"""

Sample predictive model for the CYPHER 2026 dynamic ROM challenge.

Your submitted model.py must define a class named `model` implementing
3 methods: preprocess(), fit(), predict(). See the "Submission" page of the
competition for the exact contract.

This example uses a small MLPRegressor (scikit-learn) trained on one-step-ahead
pairs, then rolls forward autoregressively at prediction time.

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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from utils import list_training_simulations, load_simulation, build_one_step_pairs


class model:
    def __init__(self):
        self.is_trained = False
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.net = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            max_iter=50,
            random_state=0,
        )

    def preprocess(self, data_folder):
        X_list, Y_list = [], []
        for sim_name in list_training_simulations(data_folder):
            state, phi = load_simulation(f"{data_folder}/{sim_name}")
            X, Y = build_one_step_pairs(state, phi)
            X_list.append(X)
            Y_list.append(Y)
            print(f"Loaded '{sim_name}': X {X.shape}, Y {Y.shape}")

        X_all = np.vstack(X_list)
        Y_all = np.vstack(Y_list)
        return X_all, Y_all

    def fit(self, D):
        X_all, Y_all = D
        X_scaled = self.scaler_X.fit_transform(X_all)
        Y_scaled = self.scaler_Y.fit_transform(Y_all)

        print(f"Training on {X_scaled.shape[0]} samples...")
        self.net.fit(X_scaled, Y_scaled)
        self.is_trained = True
        print("Model trained.")

    def predict(self, test_data_folder):
        initial_state = np.load(f"{test_data_folder}/initial_state.npz")['data']
        phi = np.load(f"{test_data_folder}/phi.npz")['data']

        n_timesteps = phi.shape[0]
        n_features = initial_state.shape[0]
        state_pred = np.zeros((n_features, n_timesteps), dtype=np.float32)
        state_pred[:, 0] = initial_state

        current_state = initial_state
        for t in range(n_timesteps - 1):
            x = np.hstack([current_state, phi[t], phi[t + 1]]).reshape(1, -1)
            x_scaled = self.scaler_X.transform(x)
            y_scaled = self.net.predict(x_scaled)
            next_state = self.scaler_Y.inverse_transform(y_scaled).ravel()

            state_pred[:, t + 1] = next_state
            current_state = next_state

        return state_pred