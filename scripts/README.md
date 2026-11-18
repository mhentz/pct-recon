# HPC job scripts

Everything in the pipeline was run on a Sun Grid Engine cluster. The `.job`
files are the `qsub` job scripts, one per pipeline stage, and the `submit_*.sh`
wrappers fill in their environment variables and submit them. Each job script
lists the variables it expects at the top.

The scripts assume this layout on the cluster:

```
~/pct-recon                                   this repository, with a .venv
~/Scratch/data/simulations/<setup>-<n>proj-5mGy/PhaseSpace   Geant4 ROOT output
~/Scratch/data/recon/<setup>/<dose>mGy/<n>proj/
    processed-3sigma/protons/               preprocessed proton dataframes
    processed-3sigma/b/                     WEPL (b) vectors
    processed-3sigma/matrix-<pixel_w>mm/    system matrix blocks
    processed-3sigma/fbp/                   FBP initial images
    recon/                                  reconstruction output
```

## Pipeline stages

| Stage | Job script | Submit wrapper | Runs |
| --- | --- | --- | --- |
| 1. Preprocess protons | `preprocess.job` | `submit_preprocess.sh <setup> <n_projections> <n_chunks> <batch_size>` | `recon/preprocess/run.py` |
| 2. Build b-vectors | – | `process_bvec.sh <setup> <n_projections> <n_input_chunks> <n_output_chunks>` | `recon/preprocess/b_vec.py` |
| 3. System matrix | `matrix.job` | `submit_matrix.sh <ci|cr|gi|gr|si|sr> <n_projections> <pixel_w> <image_w> <n_input_chunks> <n_output_chunks> <batch_size>` | `recon/matrix/process_blocks.py` |
| 4. Lipschitz constant | `lipschitz.job` | `submit_lipschitz.sh <setup> <dose> <n_projections> <pixel_w> <image_w> <n_iterations>` | `recon/lipschitz/run.py` |
| 5. Reconstruction | `fista.job` | `submit_fista.sh <setup> <dose> <n_proj> <pixel_w> <reg> <alpha> <eta> <n_iter> <zero|fbp>` | `recon/algo/fista.py` |

`recon/matrix/process_blocks.py` (stage 3) already writes the b-vectors
alongside the matrix; stage 2 is kept for checking or rebuilding individual
vectors.

The two-letter setup codes are the initials of phantom and dataset
(`ci` = ctp528-ideal, `gr` = gammex-realistic, and so on).

`submit_batch_*.sh` and `submit_multiple_fista_{tv,dtv}.sh` loop over the
parameter grids (doses, projection counts, alpha and eta values) used in the
thesis.

## Lower-dose datasets

Lower doses were obtained by subsampling the 5 mGy matrix and vector pairs
rather than by re-simulating:

* `subsample.job` / `submit_subsample.sh <setup> <n_projections> <pixel_w> <ratio>`
  runs `recon/utils/subsample_matrix_vector_pair.py`.

## Housekeeping

* `find_unfinished_fista.sh`, `find_unfinished_matrix.sh` scan job logs for
  runs that did not finish, and `resubmit_*.sh` resubmit them.
  `find_finished_fista.sh` lists the FISTA logs of runs that did finish.
* `progtar.sh`, `batch_progtar.sh`, `find_outdated_tarballs.sh` archive
  processed data from scratch space to long-term storage and find archives that
  are older than the data they hold.
* `total_slice_dose.sh` sums the dose scored by the Geant4 simulation in the
  reconstructed slice.
* `run_fista.sh` runs a short reconstruction interactively with the same
  environment as `fista.job`; useful for debugging.
* `recompress.job` / `submit_recompress.sh` recompress existing matrix blocks
  with LZ4HC (`recon/utils/recompress.py`). This is not part of the pipeline:
  it is only needed to repair blocks whose compression went wrong, or to try
  other compression settings.

Job scripts use `#$ -cwd` and `${HOME}` paths, so they should run from any
account as long as the repository is cloned to `~/pct-recon`.
