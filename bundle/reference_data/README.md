This folder is intentionally empty in the GitHub repository.

It must be populated by running `organizer_scripts/prepare_data.py`, which
will create:

    reference_data/
      valid/
        phase                (contains the word "valid")
        meta.json             ({"n_features": 11, "features": [...], "n_cells": ...})
        <sim_name>/state_full.npy    (ground truth, one per validation simulation)
      test/
        phase                (contains the word "test")
        meta.json
        <sim_name>/state_full.npy    (ground truth, one per test simulation)

THIS FOLDER MUST NEVER BE MADE PUBLIC OR SHARED WITH PARTICIPANTS. It is
only read by scoring_program/score.py, which runs on the Codabench backend.
Do not commit a populated version of this folder to a public GitHub repo.
