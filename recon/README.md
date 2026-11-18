# Reconstruction pipeline

List-mode proton CT reconstruction. The pipeline takes the phase-space output
of a Geant4 simulation (one ROOT file per projection angle with the entry and
exit tracker measurements of each proton), builds a sparse system matrix from
the most-likely path of every proton, and reconstructs the relative stopping
power image with FISTA and a TV or directional TV (dTV) penalty.

Heavy numerical work (MLP evaluation, matrix construction, matrix-vector
products over the compressed blocks) lives in the separate
[`pct-tools`](https://pypi.org/project/pct-tools/) package. The code here
drives it and holds the physics choices.

## Stages

```
ROOT files ──1──> protons/df_<angle>.pkl ──2──> b/b_<angle>.vec
                          │
                          └──3──> matrix-<pixel_w>mm/<angle>-<chunk>.mat ──4──> step size ──5──> recon/
```

| # | Script | What it does |
| --- | --- | --- |
| 1 | `preprocess/run.py` | Reads a ROOT file, matches entry and exit measurements (`matching.py`), converts energy loss to WEPL with a lookup table (`utils.py`), optionally degrades the measurements to realistic detector resolutions, subsamples to the target dose, intersects each path with the phantom hull (`hull.py`), applies the 3 sigma data cuts on WEPL and angle (`cuts.py`) and keeps the protons whose MLP stays within the reconstructed slices. Output: one pickled dataframe per projection angle (and chunk). |
| 2 | `preprocess/b_vec.py` | Writes the WEPL of each kept proton as the measurement vector `b`, in the same order and chunking as the matrix. Stage 3 writes the same vectors alongside the matrix, so this script is mainly a tool to check or rebuild individual vectors when debugging. |
| 3 | `matrix/process_blocks.py` | Evaluates the MLP of each proton in 3D (`matrix/paths.py`), rotates it into the image frame and stores the exact chord lengths through the pixel grid as compressed sparse blocks. |
| 4 | `lipschitz/run.py` | Power iteration for the largest eigenvalue of `A^T A`, which sets the FISTA step size. Results are recorded by hand in `lipschitz/lipschitz_constants.yml` and `lipschitz/step_sizes.yml`; `lipschitz/convert.py` prints one from the other. These files hold the values used for the thesis; `run.py` prints the entries to paste, and step sizes are 1/L rounded down to four significant digits. |
| 5 | `algo/fista.py` | FISTA with the proximal operator of TV or dTV (`algo/regularisation.py`, built on the CCPi regularisation toolkit), non-negativity and a circular zero mask outside the phantom (`algo/utils.py`). Writes the iterates and metrics to `OUTPUT_DIR`. |

`utils/subsample_matrix_vector_pair.py` derives lower-dose datasets from the
5 mGy matrix and vector pair by random subsampling of rows.
`utils/recompress.py` recompresses existing matrix blocks with LZ4HC; it is
only needed to repair blocks whose compression went wrong, or to try other
compression settings.

## Filtered back-projection

Filtered back-projection (FBP) reconstructions from the preprocessed proton
data, used as a baseline and as the optional initial image (`X_INIT=fbp`) of
the iterative reconstruction.

* `fbp.py` bins the WEPL of the preprocessed protons into a sinogram
  and reconstructs it with [ODL](https://odlgroup.github.io/odl/)
  (scikit-image backend). It writes `fbp_<dose>mGy_<width>pix.npy` into
  `--output-dir`. `recon/algo/fista.py` looks for that file in `FBP_DIR`,
  which the job scripts set to the `fbp/` directory of the processed dataset.

ODL is not needed by the rest of the project, so it is listed in the
development requirements (`dev-requirements.txt`) rather than in
`requirements.txt`. With those installed, from the repository root, for
example:

```
python recon/fbp.py --phantom ctp528-ideal --width 880 \
    --pixel-size 0.25 --dose 1 -n 720 \
    --protons-dir <data>/ctp528-ideal/5mGy/720proj/processed-3sigma/protons \
    --output-dir <data>/ctp528-ideal/1mGy/720proj/processed-3sigma/fbp
```

`--dose` is the target dose and `--actual-dose` (5 mGy by default) the dose of
the proton dataframes; the script draws the corresponding fraction of the
protons of each projection at random.

## Configuration

`config.py` defines `ReconConfig`, a frozen dataclass of the scalar settings,
and `load_config()`, which fills it from a `.env` file found by `python-dotenv`
and from environment variables (the environment wins). Nothing is read at
import time: `fista.py` calls `load_config()` in its `main()` and writes the
resolved values to `config.yml` in the output directory. The other stages take
their inputs on the command line (`WEPL_LUT_PATH` is the one variable
`preprocess/run.py` also reads from the environment). See `.env.example` at
the repository root. The variables in `ReconConfig`:

| Variable | Meaning |
| --- | --- |
| `SETUP` | `<phantom>-<dataset>`; phantom in `gammex`, `ctp528`, `spiral`, dataset in `ideal`, `realistic`. Phantom radius, detector distance and image size derive from it (`geometry.py`). |
| `N_PROJECTIONS`, `TARGET_DOSE`, `N_SIGMA` | Acquisition and data-cut parameters. Dose is in mGy. |
| `PIXEL_W`, `N_SLICES`, `SLICE_THICKNESS` | Image grid in mm. The image width in pixels follows from the phantom radius. |
| `MATRIX_DIR`, `VECTOR_DIR`, `FBP_DIR`, `OUTPUT_DIR`, `REF_DIR` | Data locations. `REF_DIR` holds the x-ray reference images for dTV (`phantoms/xray_refs`); `load_xray_ref` in `algo/fista.py` loads the `realistic_slice_filtered` one unless its `variant` argument (an `XrayReference`, `algo/utils.py`) says otherwise (see `phantoms/README.md`). |
| `REGULARISER`, `ALPHA`, `ETA`, `PROX_TOLERANCE`, `PROX_MAX_LOOPS` | `None`, `TV` or `dTV`; regularisation weight; dTV smoothing parameter; inner proximal loop settings. |
| `ITER_START`, `ITER_END`, `X_INIT`, `DUMP_MODULO` | Iteration range, initial image (`zero` or `fbp`), how often iterates are written. |
| `N_THREADS` | Threads used by `pct-tools`. Blosc threading is disabled in the scripts because the matrix products are parallelised explicitly. |

## Output layout

Each reconstruction is written to

```
OUTPUT_DIR/fista_exact_init_<X_INIT>_<width>pix_<dose>mGy_<n>proj_<penalty>_iter<N>/
```

where `<penalty>` is `no_reg`, `TV_iso_alpha_<alpha>` or
`dTV_iso_alpha_<alpha>_eta_<eta>`. The reference image is not part of the
name. The directory holds the
iterates `yk_<i>` (Blosc2 compressed) every `DUMP_MODULO` iterations, the final
`yk_final` and a `metrics_final` array with one row per iteration (cost,
squared residuals and regularisation term). The analysis scripts rely on this
naming.

## Running locally

With a `.env` file in place, a reconstruction is simply

```bash
PYTHONPATH=. python recon/algo/fista.py
```

The other stages take their inputs as command-line arguments; run them with
`--help`. `scripts/README.md` documents how they were chained on the cluster.
