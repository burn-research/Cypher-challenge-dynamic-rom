This folder is intentionally empty in the GitHub repository.

It must be populated by running `organizer_scripts/prepare_data.py`, which
will create:

    reference_data/

      valid/
        cell_volumes.zarr/   (cell volumes for spatial integration, shape (n_cells,))
        phase                (contains the word "valid")
        meta.json            ({"n_features": 11, "features": [...], "n_cells": ...})
        <sim_name>/
          state_full.zarr/   (ground truth trajectory, shape (n_cells, n_features, n_timesteps))
          phi.zarr/          (forcing signal, shape (n_timesteps,))

      test/
        cell_volumes.zarr/   (cell volumes for spatial integration, shape (n_cells,))
        phase                (contains the word "test")
        meta.json            ({"n_features": 11, "features": [...], "n_cells": ...})
        <sim_name>/
          state_full.zarr/   (ground truth trajectory, shape (n_cells, n_features, n_timesteps))
          phi.zarr/          (forcing signal, shape (n_timesteps,))

THIS FOLDER MUST NEVER BE MADE PUBLIC OR SHARED WITH PARTICIPANTS. It is
only read by scoring_program/score.py, which runs on the Codabench backend.
Do not commit a populated version of this folder to a public GitHub repo.