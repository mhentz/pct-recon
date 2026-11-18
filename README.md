# Proton CT reconstruction with TV and directional TV regularisation

Code developed for a PhD on iterative list-mode proton computed tomography
(pCT) reconstruction. Proton tracks simulated with Geant4 are converted into a
sparse system matrix using the most-likely path of each proton, and relative
stopping power images are reconstructed with FISTA using total variation (TV)
or directional total variation (dTV) penalties, the latter guided by an x-ray
CT image of the same object. The analysis compares image quality (accuracy,
spatial resolution, SSIM) across doses, projection counts and regularisers.

## Repository layout

| Directory | Contents |
| --- | --- |
| `recon/` | The reconstruction pipeline: preprocessing, system matrix, Lipschitz constant, FISTA. See `recon/README.md`. |
| `scripts/` | Sun Grid Engine job scripts and submission wrappers used to run the pipeline on a cluster. See `scripts/README.md`. |
| `phantoms/` | Digital phantoms, their RSP ground truths and the simulated x-ray references used by dTV. See `phantoms/README.md`. |
| `analysis/` | Scripts producing the figures and tables (L-curves, SSIM, accuracy, MTF). See `analysis/README.md`. |
| `tests/` | Unit tests of `recon/` and `phantoms/`. |

Heavy numerical routines (MLP evaluation, matrix construction and storage,
matrix-vector products) are in the companion package
[`pct-tools`](https://pypi.org/project/pct-tools/), installed from PyPI.

The arrays in `phantoms/` (about 100 MB) are stored with
[Git LFS](https://git-lfs.com). Run `git lfs install` once before cloning;
without it the `.npy` files are text pointers and cannot be loaded.

## Preparing the environment

This project requires Python 3.12. Packages can be installed using `uv`, for
example

```bash
uv python install 3.12.10
uv venv --python 3.12.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

If this version of Python is not available via `uv`, update it with
`uv self update` and try again. `requirements.txt` is compiled from
`requirements.in` with `uv pip compile`; `dev-requirements.in` adds `pytest`, `pytest-mock` and ODL, which the tests and `recon/fbp.py` need; install `dev-requirements.txt` instead to use them.

### Install the regularisers

We use the
[CCPi-Regularisation-Toolkit](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit)
to compute the proximal operators of the TV and dTV norms. [A Python package is available
via conda](https://anaconda.org/ccpi/ccpi-regulariser) but if that is not an option for you, you
can use the instructions below to build the package and install it in your Python environment.

```bash
git clone https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit
```

You can now build the library. As previously mentioned, I use
[uv](https://astral.sh/blog/uv) to manage my Python environments but this also
should work with pip. Modify the CMake variables as required to match your
system and the installation you desire. The CMake commands have been tested with CMake
version 3.27.3 and may not work with earlier versions.

```bash
cd CCPi-Regularisation-Toolkit
cmake -S . -B ./build_proj \
    -DBUILD_MATLAB_WRAPPER=OFF \
    -DBUILD_PYTHON_WRAPPER=ON \
    -DBUILD_CUDA=OFF \
    -DPython_EXECUTABLE="$(pwd)/../.venv/bin/python" \
    -DCMAKE_INSTALL_PREFIX=./install
cmake --build ./build_proj --target cilreg --parallel 8
cmake --install ./build_proj
cd ..
uv pip install CCPi-Regularisation-Toolkit/src/Python
```

Only the `cilreg` library is built here, because the toolkit's own Python target calls `pip`, which a uv environment does not provide. On macOS, CMake may not find Homebrew's OpenMP unless `-DOpenMP_ROOT="$(brew --prefix libomp)"` is added to the first command.

The installation was successful if the following command exits without errors

```bash
python -c "from ccpi.filters.regularisers import FGP_TV, FGP_dTV"
```

which is also what `pytest tests/` checks.

## Running a reconstruction

1. Copy `.env.example` to `.env` and point the data directories at your
   preprocessed dataset.
2. Run the stages described in `recon/README.md`: preprocess the ROOT files,
   build the b-vectors and the system matrix, compute the step size, then

   ```bash
   PYTHONPATH=. python recon/algo/fista.py
   ```

The simulated datasets are not part of this repository; they were produced
with [pct-sim](https://github.com/mhentz/pct-sim). The three phantoms
(`gammex`, `ctp528`, `spiral`) were each simulated in an `ideal` variant
(perfect trackers, detector 10 mm from the phantom) and a `realistic` variant
(detector resolutions applied, detector 150 mm away), at 5 mGy with 720
projections; lower doses and 45-projection datasets were derived by
subsampling.

## Development

Formatting and linting are configured in `pyproject.toml` (ruff, line length
110) and run through `pre-commit`:

```bash
uv pip install pre-commit
pre-commit install
```

## How to cite

If you use this software, please cite the thesis it was developed for:

M. Hentz, *Iterative Reconstruction using Structural Priors in Proton Computed Tomography*, PhD thesis, University College London, 2026.

Software metadata are in [CITATION.cff](CITATION.cff).

## Acknowledgements

The proximal operators of the TV and dTV penalties are computed with the
[CCPi-Regularisation-Toolkit](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit)
(Apache License 2.0). If you use the regularisers, please also cite.

D. Kazantsev, E. Pasca, M. J. Turner and P. J. Withers, *CCPi-Regularisation
toolkit for computed tomographic image reconstruction with proximal splitting
algorithms*, SoftwareX 9 (2019) 317–323,
[doi:10.1016/j.softx.2019.04.003](https://doi.org/10.1016/j.softx.2019.04.003).

## Licence

This project is released under the MIT License; see [LICENSE](LICENSE).
