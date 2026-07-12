This folder is intentionally empty in the GitHub repository.

It must be populated by running `organizer_scripts/prepare_data.py` on your
raw simulation dumps (see the top-level README.md, section "Preparing the
data"). After running it, this folder will contain:

    input_data/
      grid.npy
      train/<sim_name>/state.npy, phi.npy
      valid/<sim_name>/initial_state.npy, phi.npy
      test/<sim_name>/initial_state.npy, phi.npy

Only the populated bundle/ folder (this one, with real data) should be
zipped and uploaded to Codabench -- never commit real simulation data to a
public GitHub repository if the "test" split must stay hidden from
participants.
