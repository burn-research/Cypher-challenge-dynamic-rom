# CYPHER 2026 hackathon -- Codabench bundle (organizer package)

Data challenge on reduced-order modeling (ROM) for an externally forced 2D
laminar flames, organized within the CYPHER COST Action. System
identification task: forecast the full flow-state evolution given an
initial snapshot and a known time-varying forcing signal.

This package is the **organizer's working copy**. It contains both the
Codabench bundle itself (`bundle/`) and organizer-only tooling that must
never be shared with participants (`organizer_scripts/`).

```
.
├── bundle/                     <- THIS is what gets zipped and uploaded to Codabench
│   ├── competition.yaml
│   ├── pages/                  competition web pages (overview, data, submission, ...)
│   ├── ingestion_program/      runs a participant's model.py on the server
│   ├── scoring_program/        computes NRMSE + score from the ingestion output
│   ├── input_data/             training data + (label-free) valid/test inputs -- EMPTY until you run prepare_data.py
│   └── reference_data/         hidden ground truth used only by scoring -- EMPTY until you run prepare_data.py, NEVER share this
├── sample_code_submission/     example participant submission (persistence baseline)
├── sample_code_submission.zip  the same, zipped, ready to upload as the "starting kit"
└── organizer_scripts/          organizer-only tools, NOT part of the bundle
    ├── prepare_data.py         raw .npy/.vtu -> bundle/input_data + bundle/reference_data
    └── generate_dummy_data.py  generates a tiny fake dataset for local smoke-testing
```

---

## 0. Concepts you asked about

**What is a "bundle"?**
A bundle is just the specific folder structure Codabench expects to define a
competition: a `competition.yaml` describing phases/leaderboard, HTML pages,
an `ingestion_program` (runs on the server for every submission), a
`scoring_program` (computes the score from the ingestion output), and the
data folders (`input_data`, `reference_data`). When you "create a
competition" on Codabench, you zip the **contents** of `bundle/` (not the
`bundle` folder itself, i.e. `competition.yaml` must be at the root of the
zip) and upload that single zip file. That's it -- a bundle is nothing more
than this zip.

**What is a "docker image" and why do you need one?**
Codabench doesn't run your ingestion/scoring code on its own bare server: it
spins up a Docker container (a lightweight, self-contained virtual Linux
environment with a fixed set of pre-installed software) for every
submission, runs your `ingestion_program` and `scoring_program` inside it,
then throws the container away. The `docker_image:` field in
`competition.yaml` tells Codabench *which* pre-built environment to use --
i.e. which Python version and which libraries (numpy, torch, tensorflow,
scikit-learn, ...) are already available inside the container before your
code even starts running.

You do **not** need to build your own image right now: this bundle reuses
Lorenzo's existing image (`lorenzopiu1/cypher-codabench-image:v4`), which
already contains PyTorch, TensorFlow and scikit-learn -- everything a
participant needs, and everything our own `ingestion_program`/
`scoring_program` need (which is just numpy). Building a custom image is
only necessary if you need a Python package that isn't already inside it;
see section 4 below for how to do that if/when it happens.

**Where do the "input_data" and "reference_data" folders come from, and who
sees what?**
- `input_data` is what the ingestion program (i.e. the participant's
  submitted code) is allowed to read: the training simulations (state +
  forcing signal), and for validation/test simulations, ONLY the initial
  snapshot and the future forcing signal -- never the true future state.
- `reference_data` is the hidden ground truth (the full state trajectories
  for validation/test), read only by `scoring_program`, which runs
  separately from ingestion and whose output the participant never sees the
  internals of. This is where your "test data participants must never see"
  physically lives, and it is exactly what you must never commit to a
  public repository or hand to anyone outside the organizing team.

---

## 1. Preparing the data (your "passo 2")

Your raw data (`.npy` DataMatrix + `.vtu` grid, one folder per simulation)
never goes into the bundle directly -- it must first be converted into the
standardized format the ingestion/scoring programs expect. That's what
`organizer_scripts/prepare_data.py` does, using the exact resampling logic
you already had, generalized to loop over every simulation:

1. Open `organizer_scripts/prepare_data.py` and edit the `RAW_SIMULATIONS`
   list at the top: for each of your 2 training + 6 test/valid simulations,
   set the path to the raw `.npy` and `.vtu` files, the split
   (`"train"` / `"valid"` / `"test"`), the signal type
   (`"sweep"` / `"step"` / `"sine"`) and its parameters (A, f). The
   forcing signal &phi;(t) is computed analytically from these parameters
   using the exact formulas in the OpenFoam U_code -- you do
   not need a separate BC dataset.
2. Decide the train/valid/test split. Right now the script assumes:
   - `train`: the 2 sine-sweep simulations (A=0.2, A=0.4) -- given to
     participants in full (state + phi).
   - `valid` (Development-phase feedback): `step_A0.3` and `sine10_A0.3`
     -- participants only ever see the initial snapshot + phi.
   - `test` (Final-phase, hidden): `step_A0.5`, `sine10_A0.5`,
     `sine40_A0.3`, `sine40_A0.5`.
   This is a reasonable default (one step-type + one sine-type case for
   dev-phase feedback, the harder/larger-amplitude and higher-frequency
   cases held out for the final score) but it's your call -- move
   simulations between `valid`/`test` in the list if you'd rather split
   them differently.
3. `pip install pyvista` (only needed for this script, not for the bundle
   itself), then run:
   ```
   cd organizer_scripts
   python3 prepare_data.py
   ```
   This fills in `bundle/input_data/` and `bundle/reference_data/`.

**Important:** `bundle/reference_data/` (and the `test/` part of
`bundle/input_data/`) must stay private. Do not push a populated version of
these two folders to a public GitHub repo. Keep the populated `bundle/`
folder locally / in a private location, and only zip+upload it directly to
Codabench.

---

## 2. Running everything locally first (your "passo 3")

Before touching Codabench at all, you can (and should) run the exact same
programs Codabench will run, on your own machine. This is the fastest way
to catch bugs.

**A. Quick smoke test with fake data (no real data needed, seconds to run):**
```
cd organizer_scripts
python3 generate_dummy_data.py     # fills bundle/input_data + bundle/reference_data with tiny fake arrays
cd ../bundle
python3 ingestion_program/ingestion.py input_data sample_output_data ingestion_program ../sample_code_submission
```
This runs the example baseline submission exactly like Codabench would,
producing `bundle/sample_output_data/`. Then simulate what the scoring
program receives (Codabench packages ingestion output as `res/` and the
hidden ground truth as `ref/` inside one folder):
```
mkdir -p /tmp/score_input/res /tmp/score_input/ref
cp -r sample_output_data/* /tmp/score_input/res/
cp -r reference_data/valid/* /tmp/score_input/ref/     # or reference_data/test for the final phase
python3 scoring_program/score.py /tmp/score_input /tmp/score_output
cat /tmp/score_output/scores.txt
```
If this runs without errors and prints a `scores.txt` with `score`, `NRMSE`,
`time_inference`, `time_training`, the whole pipeline (ingestion + scoring +
the model interface) is wired correctly. I ran exactly this before handing
you the zip, so it works as shipped.

**B. Same thing with your real data:** once you've run `prepare_data.py`
(step 1), repeat the same two commands (skip `generate_dummy_data.py`) --
now you're testing the actual challenge with the actual baseline model, and
with any model you or a colleague writes against the same `model.py`
interface, before ever uploading anything.

**C. Testing a participant-like submission:** put any `model.py` (+
optional helper files) with the required 3-method interface in its own
folder and pass that folder as the last argument to `ingestion.py` instead
of `../sample_code_submission`.

---

## 3. Building the sample_code_submission.zip

Whenever you change `sample_code_submission/`, re-zip it (this is the
"starting kit" participants download from the Files tab):
```
cd sample_code_submission
zip -r ../sample_code_submission.zip .
```

---

## 4. Uploading to Codabench

1. Create a Codabench account / log in, go to "My Competitions" -> "Create
   Competition" -> "Upload".
2. Zip the **contents** of `bundle/` (after running `prepare_data.py`), so
   that `competition.yaml` sits at the root of the zip -- not
   `bundle/competition.yaml` one level down. E.g.:
   ```
   cd bundle
   zip -r ../cypher_2026_bundle.zip . -x "*.DS_Store"
   ```
3. Upload `cypher_2026_bundle.zip` on the competition creation page.
4. Also upload `sample_code_submission.zip` on the "Files" tab so
   participants can download the starting kit.
5. Before opening registration: submit `sample_code_submission.zip` yourself
   as a test participant, on both phases, and check the leaderboard shows
   sensible NRMSE/score values. This exercises the real Docker image on the
   real Codabench backend, which is the one thing you can't fully test
   locally.
6. Fill in the placeholders left in the bundle: `pages/organizing.html`
   (organizing committee), the `start`/`end` dates in `competition.yaml`
   (currently placeholders), and re-check `pages/terms.html` matches the
   actual hackathon rules (team size, submission limits, etc.).

Reference: Codabench's own docs/wiki -- https://github.com/codalab/codabench/wiki

---

## 5. About the Docker image (only if you need extra packages)

You don't need to do this yet. If, once the hackathon is closer, you find
participants need a package that isn't in `lorenzopiu1/cypher-codabench-image:v4`,
the general recipe is:
1. Write a `Dockerfile`:
   ```dockerfile
   FROM lorenzopiu1/cypher-codabench-image:v4
   RUN pip install --no-cache-dir <your-extra-package>
   ```
2. Build and push it to a Docker Hub account you control:
   ```
   docker build -t <your-dockerhub-username>/cypher-2026-image:v1 .
   docker push <your-dockerhub-username>/cypher-2026-image:v1
   ```
3. Update `docker_image:` in `competition.yaml` to point to your new image,
   re-zip the bundle, and re-upload / update the competition.

You will need Docker Desktop (or `docker` CLI) and a free Docker Hub
account for this -- but again, only if/when the current image turns out to
be missing something.

---

## 6. What changed vs. the CYPHER 2025 (DNS) bundle you started from

The previous bundle (Lorenzo Piu's) was a *different* challenge: a-priori
prediction of a sub-filter turbulent diffusivity (`alpha_t`) from static,
already-labeled 3D DNS snapshots, using the `aPrioriDNS` library for I/O and
a single train/valid/test simulation each. This one is a genuinely
different task -- autoregressive forecasting of an 11-field 2D flow state
under a known time-varying forcing, evaluated over multiple held-out
forcing signals -- so essentially every script was rewritten:
`ingestion_program`, `scoring_program`, `model.py`/`data_manager.py`, the
metric (MSE of a sub-filter flux -> multi-simulation, multi-feature NRMSE
of the whole forecast), the data layout, and all HTML pages. Only the
overall bundle *shape* (which files Codabench expects, and where) and the
general 3-method `model` interface convention were kept, since those are
Codabench requirements, not challenge-specific choices.
