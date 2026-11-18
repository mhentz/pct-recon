# Phantoms, ground truths and x-ray references

Scripts that build the digital phantoms used throughout the project and the
reference images needed by the analysis and by directional TV (dTV)
reconstructions.

| Script | Produces |
| --- | --- |
| `ground_truth_catphan.py` | Catphan CTP528 line-pair module: RSP ground truth, x-ray attenuation ground truth and realistic x-ray references |
| `ground_truth_gammex.py` | Gammex 467 tissue-characterisation phantom (16 inserts, 165 mm radius) |
| `ground_truth_spiral.py` | Spiral phantom of 21 aluminium inserts (6 mm) in water, used for the MTF (k10) analysis |
| `analyse_xray_ref_noise.py` | Reports the noise level in homogeneous regions of a realistic x-ray reference |
| `constants.py` | Attenuation coefficients (100 keV, NIST tables) and relative stopping powers (200 and 250 MeV) of the materials; most RSP values come from the `rsp` binary of [pct-sim](https://github.com/mhentz/pct-sim) (see its [Relative stopping powers](https://github.com/mhentz/pct-sim/blob/main/docs/phantoms.md#relative-stopping-powers) docs) |
| `utils.py` | Drawing helpers and `generate_realistic_ct_image`, which simulates a noisy x-ray CT acquisition (Radon transform, pre-log Poisson noise, filtered back-projection) |

Each `ground_truth_*.py` script writes into `ground_truths/` and `xray_refs/`
under `--output-dir`, which defaults to this directory. Run them as modules from the repository root:

```bash
python -m phantoms.ground_truth_spiral
```

The file naming convention is `<phantom>_gt_<width>pix.npy` for RSP ground
truths and `<phantom>_xray_gt_<width>pix.npy` for noise-free attenuation images.
Image widths are 880 px for the 100 mm phantoms at 0.25 mm per pixel and 700 px
for the Gammex phantom at 0.5 mm per pixel.

There are two kinds of x-ray reference for dTV:

- Deteriorated references, `<phantom>_xray_ref_<width>pix_<sigma>sig_<noise>n.npy`,
  made by `deteriorate_image`: the noise-free attenuation image blurred with a
  Gaussian of `<sigma>` pixels, plus Gaussian noise of standard deviation
  `<noise>` times the image maximum inside the phantom. Negative noise values
  are set to zero, so the noise only raises the attenuation. The committed
  files use a 1-pixel blur and 2 % noise. Of the scripts, only
  `ground_truth_catphan.py` writes one, with a 0.5-pixel blur, into
  `--output-dir` itself.
- Realistic references,
  `<phantom>_xray_ref_realistic[_slice[_filtered]]_<width>pix.npy`, made by
  `generate_realistic_ct_image` from a simulated x-ray CT acquisition. The
  attenuation image is projected over ⌊π/2 × width⌋ views, Poisson noise is
  applied to the counts before the logarithm, and the image is reconstructed by
  filtered back-projection with a cosine filter. This is repeated twelve times
  with independent noise. The plain `realistic` file holds the last single
  reconstruction, `_slice` the mean of the twelve, which mimics a thick slice,
  and `_slice_filtered` that mean after Gaussian smoothing of 0.7
  (Catphan), 0.65 (spiral) or 0.8 (Gammex) pixels. The committed `_slice`
  and `_slice_filtered` files were made with an earlier version that left
  out the first reconstruction and counted the eleventh twice, so their
  noise is about 8 % higher than that of a plain mean of twelve.

`load_xray_ref` in `recon/algo/fista.py` loads the `_slice_filtered` reference
by default; its `variant` argument takes an `XrayReference`
(`recon/algo/utils.py`) to select one of the other realistic references or the
committed deteriorated one.

The committed arrays are the files used in the thesis. Re-running the scripts
should reproduce the ground truths up to floating-point rounding (differences of
the order of 1e-15 at material boundaries, with scikit-image 0.25.2). The
simulated x-ray references are seeded with `--random-seed`, but the thesis files
were generated before seeding was in place, so new references will be
statistically similar to them rather than identical.
