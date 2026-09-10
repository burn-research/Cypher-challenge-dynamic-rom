"""
Sample predictive model for the CYPHER 2026 dynamic ROM challenge.

Your submitted model.py must define a class named `model` implementing
3 methods: preprocess(), fit(), predict(). See the "Submission" page of the
competition for the exact contract.

This example implements a simple operator inference model.

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
import pyvista as pv
from utils import list_training_simulations, load_simulation, load_test_simulation, d_function, rhs, integrate_Q
from sklearn.utils.extmath import randomized_svd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import zarr

DTYPE = np.float32

class model:
    def __init__(self):
        self.is_trained = False

    def _compute_derivative(self, A):
        """
        Compute the time derivative of A using finite differences.
        """
        A_dot = np.gradient(A, self.dt, axis=0)
        return A_dot

    def _flatten_data(self, data_tensor):
        """
        Flatten a 3D tensor (n_cells, n_features, n_timesteps) into a 2D matrix (n_cells*n_features, n_timesteps).
        """
        n_cells = data_tensor.shape[0]
        n_features = data_tensor.shape[1]
        n_timesteps = data_tensor.shape[2]
        data = np.zeros((n_cells*n_features, n_timesteps), dtype=DTYPE)
        for i in range(n_features):
            data[i*n_cells:(i+1)*n_cells, :] = data_tensor[:, i, :]
        return data
    
    def _unflatten_data(self, data):
        """
        Unflatten a 2D matrix (n_cells*n_features, n_timesteps) back into a 3D tensor (n_cells, n_features, n_timesteps).
        """
        n_cells = self.n_cells
        n_features = self.n_features
        n_timesteps = data.shape[1]
        return data.reshape(n_features, n_cells, n_timesteps).transpose(1, 0, 2)


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
        # Set quantities
        self.n_single = 1001
        self.n_train = 2*self.n_single
        self.dt = DTYPE(2e-3)
        self.time_single = np.arange(0, self.n_single, dtype=DTYPE)*self.dt
        self.time_train = np.concatenate([self.time_single, self.time_single])
        self.features = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']

        # Read training data
        data_list = []
        phi_list = []
        for sim_name in list_training_simulations(data_folder):
            state, phi = load_simulation(f"{data_folder}/{sim_name}")
            data_list.append(state)
            phi_list.append(phi)
            print(f"Loaded '{sim_name}': state {state.shape}, phi {phi.shape}")
        
        # Concatenate all training data into a single tensor
        data_tensor = np.concatenate(data_list, axis=2)
        del data_list

        self.n_cells = data_tensor.shape[0]
        self.n_features = data_tensor.shape[1]

        data = self._flatten_data(data_tensor)
        del data_tensor
        phi_all = np.concatenate(phi_list).astype(DTYPE, copy=False)[np.newaxis, :]
        del phi_list

        # Center and scale the data
        self.data_mu = np.mean(data, axis=1)
        data0 = data - self.data_mu[:, np.newaxis]
        del data
        self.data_sigma = np.zeros((self.n_features,), dtype=DTYPE)
        for i in range(self.n_features):
            self.data_sigma[i] = np.std(data0[i*self.n_cells:(i+1)*self.n_cells, :])
            data0[i*self.n_cells:(i+1)*self.n_cells, :] /= self.data_sigma[i]

        # Apply dimensionality reduction using randomized SVD
        self.n_components = 4
        self.U, self.s, Vt = randomized_svd(data0, n_components=self.n_components, random_state=0)
        At = np.diag(self.s) @ Vt
        A = At.T
        A_dot = self._compute_derivative(A)

        # Scale the reduced coordinates and their derivatives
        self.A_scale = np.std(A, axis=0)
        A_scaled = A / self.A_scale
        A_dot_scaled = A_dot / self.A_scale

        # Assemble the regression matrix for operator inference
        r_nl = self.n_components*(1+self.n_components)//2
        drp = 1+self.n_components+r_nl+1
        X = np.zeros((self.n_train, drp), dtype=DTYPE)

        for i in range(self.n_train):
            X[i, :] = d_function(A_scaled[i, :], phi_all[:, i])

        del phi_all
        
        # The training data for operator inference consists of the scaled derivatives and the regression matrix
        D = (A_dot_scaled, X)

        return D

    def fit(self, D):
        """
        Train the model. It simply computes the least-squares solution to the operator inference problem. 
        """
        self.is_trained = True
        self.operator = np.linalg.lstsq(D[1], D[0], rcond=None)[0].T.astype(DTYPE, copy=False)
        
    def predict(self, test_data_folder):
        """
        Parameters
        ----------
        test_data_folder : str
            Path to ONE validation/test simulation folder, containing
            initial_state.zarr (shape (n_cells, n_features,)) and
            phi.zarr (shape (n_timesteps,), the full known forcing signal).

        Returns
        -------
        state_pred : ndarray, shape (n_features * n_cells, n_timesteps)
            Forecast of the full flow state at every time step. Column 0 must
            equal initial_state.zarr.
        """
        # Load the initial state and forcing signal for the test simulation
        initial_state, phi_test = load_test_simulation(test_data_folder)
        initial_state = initial_state[:, :, np.newaxis]
        
        # Preprocess the initial state to match the training data format
        data_test = self._flatten_data(initial_state)
        data_test0 = data_test - self.data_mu[:, np.newaxis]
        for i in range(self.n_features):
            data_test0[i*self.n_cells:(i+1)*self.n_cells, :] /= self.data_sigma[i]

        # Project the initial state onto the reduced basis and scale it
        A_test = data_test0.T @ self.U
        A_test_scaled = A_test / self.A_scale

        time_test = np.arange(0, phi_test.size, dtype=DTYPE) * self.dt
        
        # Determine the type of simulation (sine or step) based on the folder name
        sim_type = 'sine' if 'sine' in test_data_folder else 'step'
        if sim_type == 'sine':
            temp = test_data_folder.split('/')[-1].split('_')
            amplitude = DTYPE(float(temp[2][-1])/10)
            frequency = DTYPE(float(temp[1][1:]))
        else:
            temp = test_data_folder.split('/')[-1].split('_')
            amplitude = DTYPE(float(temp[1][-1])/10)
            frequency = DTYPE(0)
        
        # Integrate the learned operator to predict the evolution of the reduced coordinates
        sol_val = solve_ivp(rhs, [time_test[0], time_test[-1]], A_test_scaled[0], args=(self.operator, amplitude, frequency, sim_type), t_eval=time_test, method='BDF')
        if sol_val.status != 0:
            print(f'failed integration')
            A_scaled_pred = np.zeros((time_test.size, self.n_components), dtype=DTYPE)
        else:    
            A_scaled_pred = sol_val.y.T.astype(DTYPE, copy=False)
        
        # Transform the predicted reduced coordinates back to the original state space
        self.A_pred = A_scaled_pred * self.A_scale
        state0_pred = self.U @ self.A_pred.T 
        
        # Unflatten the predicted state and rescale it to match the original data distribution
        state0_pred = self._unflatten_data(state0_pred)
        state_mu = self._unflatten_data(self.data_mu[:, np.newaxis])
        state_pred = state0_pred*self.data_sigma[np.newaxis, :, np.newaxis] + state_mu

        return state_pred.astype(DTYPE, copy=False)


if __name__ == "__main__":
    path_train = '../500_input_zarr/train'
    test_model = model()
    D = test_model.preprocess(path_train)
    test_model.fit(D)

    sim_name = 'sine_f40_A05'
    path_validation = f'../500_input_zarr/valid/{sim_name}'
    state_pred = test_model.predict(path_validation)
    print(f"Predicted state shape: {state_pred.shape}")

    # path_test = f'../500_valid_zarr/{sim_name}'
    # state_test = np.asarray(zarr.open_array(os.path.join(path_test, 'state_full.zarr'), mode='r')[:], dtype=np.float32)
    # print(f"Test state shape: {state_test.shape}")

    # grid = pv.read('../500_input_zarr/grid.vtu')

    # Q_test = state_test[:, test_model.features.index('mix:Q'), :]
    # Q_pred = state_pred[:, test_model.features.index('mix:Q'), :]

    # Q_test_int = integrate_Q(Q_test, grid)
    # Q_pred_int = integrate_Q(Q_pred, grid)

    # n_test = Q_test_int.size
    # time_test = np.arange(n_test)*test_model.dt
    # plt.plot(time_test, Q_test_int, label='Test')
    # plt.plot(time_test, Q_pred_int, label='Predicted')
    # plt.xlabel('Time step')
    # plt.ylabel('Integrated Q')
    # plt.legend()    
    # plt.show()