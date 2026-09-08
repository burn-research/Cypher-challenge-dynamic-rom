This folder is intentionally empty in the GitHub repository.

It must be populated by running `organizer_scripts/prepare_data.py` on your
raw simulation dumps (see the top-level README.md, section "Preparing the
data"). After running it, this folder will contain:

    input_data/

      xyz.zarr/                (cell centre coordinates, shape (n_cells, 3))
      grid.vtu                 (shared unstructured grid, pyvista format)

      train/
        <sim_name>/
          state.zarr/          (full state trajectory, shape (n_cells, n_features, n_timesteps))
          phi.zarr/            (forcing signal, shape (n_timesteps,))

      valid/
        <sim_name>/
          initial_state.zarr/  (initial snapshot, shape (n_cells, n_features))
          phi.zarr/            (known future forcing signal, shape (n_timesteps,))

      test/
        <sim_name>/
          initial_state.zarr/  (initial snapshot, shape (n_cells, n_features))
          phi.zarr/            (known future forcing signal, shape (n_timesteps,))

Only the populated bundle/ folder (this one, with real data) should be
zipped and uploaded to Codabench -- never commit real simulation data to a
public GitHub repository if the "test" split must stay hidden from
participants.