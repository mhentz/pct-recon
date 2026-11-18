# Analysis

Scripts that turn reconstruction output into the figures and tables of the
thesis. Most of them take the reconstruction directory layout produced by
`recon/algo/fista.py` as input (see `recon/README.md`) and the ground truths in
`phantoms/ground_truths/`.

Run them from the repository root with `PYTHONPATH=.` so that `analysis.*` and
`recon.*` can be imported. Scripts that use `typer` print their arguments with
`--help`.

The scripts locate reconstruction output through `RECON_DATA_DIR`
(`analysis/paths.py`), read from the environment or from `.env` at the
repository root; see `.env.example`. Below that root they expect
`<phantom>-<dataset>/<dose>mGy/<n_proj>proj/recon/`.

## Preparing reconstruction output

| Script | Purpose |
| --- | --- |
| `preprocess.py` | For every TV run of one setup, dose and projection count, find the iterate with the smallest error and collect the residuals, regularisation values and minimum-error images over alpha into `tv_*.npy` files in the `recon/` directory, where `L_curve/main.py` reads them |
| `preprocess_with_images.py` | For one run, write the minimum-error iterate, its central slice, the error norms and the residual and regularisation histories into a `processed/` directory next to the run, with images of the slice and of its error against the ground truth |
| `ideal_alpha.py` | Lookup of the regularisation parameter chosen per setup, dose and projection count from the L-curve analysis |

## Parameter selection

| Directory | Purpose |
| --- | --- |
| `L_curve/` | L-curves (residual norm against regularisation term) over alpha for TV reconstructions; `main.py` plots the three phantoms for one setup type, dose and projection count and marks the corner point |
| `ssim-dtv-eta/` | SSIM of dTV reconstructions as a function of the smoothing parameter eta, one panel per phantom with a bar for each dose and projection count |

## Image quality metrics

| Directory | Purpose |
| --- | --- |
| `accuracy/` | RSP accuracy in the inserts of the Gammex phantom (`constants.py` holds the ROI positions and reference RSP values) |
| `flat_regions/` | Mean and standard deviation of the RSP in a uniform region of each phantom. `main.py` has the commands `overlay` (the regions on the ground truths), `plot` (the regions in the TV and dTV reconstructions) and `extras` (profiles and error maps); `place_roi.py` is the interactive tool used to choose the regions in `constants.py` |
| `mtf/` | Spatial resolution from the edge spread function of the spiral phantom inserts, reported as k10 (the frequency at 10 % MTF). `compute_mtf_all_inserts.py` fits every insert, `aggregate_k10.py` builds per-ring and overall statistics, `compute_mtf_overall_mean_k10.py` prints the overall values as table rows, and `k10_bars_triangle.py`, `central_insert_esf_plot.py` and the two `central_slice_mtf_plot*.py` scripts plot them |

## Figures

`plotting/` holds the scripts for the image figures: reconstructions and error
maps of the three phantoms (`tiles.py`), and the comparison, profiles and
close-ups of the line-pair inserts of the Catphan phantom
(`catphan_comparison_tv_and_dtv.py`, `catphan_profiles.py`,
`catphan_insert_tiles_720_45_proj.py`).

The scripts in the table below save their figures to a `figures/` directory
next to themselves.

## What is where in the thesis

The table lists the figures and tables of chapters 3 to 5 and appendices A to
E by their numbers in the thesis. `<type>` is `ideal` (chapter 4, appendices A
and B, Figures E.1 to E.4) or `realistic` (chapter 5, appendices C and D,
Figures E.5 to E.8); where a dose and projection count appear, the figures
for 5.0 mGy are in the chapters and the others in the appendices. Commands are given from the
repository root with `PYTHONPATH=.` set.

| Thesis output | Script and command |
| --- | --- |
| Figures 4.1, 5.1, A.1 to A.5 and C.1 to C.5 (L-curves) | `python analysis/L_curve/main.py <type> <dose> <n>`, after `preprocess.py` |
| Figures 4.2, 4.4, 5.2, 5.4, A.6 to A.10, B.1 to B.5, C.6 to C.10 and D.1 to D.5 (reconstructions and error images) | `python analysis/plotting/tiles.py <dose> <n> <type> <TV\|dTV>` |
| Figures 4.3 and 5.3 (SSIM against eta) | `python analysis/ssim-dtv-eta/ssim_dtv_bars_column.py --setup-type <type>` |
| Figures 4.5 and 5.5 (ESF of the central insert) | `python analysis/mtf/central_insert_esf_plot.py <type> TV` |
| Figures 4.6 and 5.6 (MTF of the central insert) | `python analysis/mtf/central_slice_mtf_plot_ideal_alpha.py TV <type> 5.0 720`, saved as `mtf-<type>-tv-5.0mGy-720proj.png` |
| Figures 4.7 and 5.7 (k10 against distance from the centre) | `python analysis/mtf/k10_bars_triangle.py <type>` |
| Figures 4.8 and 5.8 (Catphan, TV against dTV) | `python analysis/plotting/catphan_comparison_tv_and_dtv.py <type> ideal` |
| Figures 4.9 and 5.9 (line-pair profiles) | `python analysis/plotting/catphan_profiles.py <type> 5.0 720` |
| Figures 4.10 and 5.10 (line-pair close-ups) | `python analysis/plotting/catphan_insert_tiles_720_45_proj.py <type>` |
| Figures 4.11, 4.12, 5.11, 5.12 and E.1 to E.8 (flat regions) | `python analysis/flat_regions/main.py plot <type> <dose> <n>` |
| Figure 3.5 (flat regions on the ground truths) | `python analysis/flat_regions/main.py overlay` |
| Tables 4.1 and 5.1 (Lipschitz constants and step sizes) | `recon/lipschitz/run.py`; the values are recorded in `recon/lipschitz/lipschitz_constants.yml` and `step_sizes.yml` |
| Tables 4.2 and 5.2 (regularisation parameters) | `analysis/ideal_alpha.py`. For the ideal Gammex data set with 45 projections at 1.0 mGy Table 4.2 prints 1.58e-3 and `ideal_alpha.py` has 2.5e-3; the thesis figure for that case (A.9) is titled 2.5e-3 and Table 4.4 is reproduced with 2.5e-3 |
| Tables 4.3, 4.4, 5.3 and 5.4 (RSP accuracy) | `python analysis/accuracy/accuracy.py <type> <TV\|dTV> <n>` |
| Tables 4.5 and 5.5 (ESF fit parameters) | Printed by `central_insert_esf_plot.py` |
| Tables 4.6 and 5.6 (mean k10) | `python analysis/mtf/compute_mtf_overall_mean_k10.py <type>`, after `compute_mtf_all_inserts.py` and `aggregate_k10.py` |
| Table 3.4 (theoretical RSP) | `phantoms/constants.py` (`RSP_MAP`). Nine of the twelve values equal the thesis table to three decimals; AP6, CB2-30% and CB2-50% are printed as 0.948, 1.280 and 1.450 and are 0.947, 1.278 and 1.449 here |

## Conventions

* `setup` is `<phantom>-<dataset>`, for example `spiral-realistic`.
* Doses are 5.0, 1.0 and 0.2 mGy; projection counts are 720 and 45.
* Regularisation parameters are passed as strings such as `1.58e-3` so that
  they match the reconstruction directory names.
