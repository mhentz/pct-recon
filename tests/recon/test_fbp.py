"""Tests of the sinogram binning in ``recon/fbp.py``; the ODL reconstruction is not run."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recon.fbp import bin_wepl, compute_sinogram, format_dose, mask_outside_phantom

XBINS = np.arange(-4.0, 5.0, 2.0)  # four detector bins of 2 mm
YBINS = np.arange(-1.5, 2.5, 1.0)  # three slices of 1 mm


def make_protons(n_per_bin: int, wepl: float) -> pd.DataFrame:
    """Return protons spread evenly over every detector bin and slice, all with the same WEPL."""
    x_centres = (XBINS[:-1] + XBINS[1:]) / 2
    y_centres = (YBINS[:-1] + YBINS[1:]) / 2
    x, y = np.meshgrid(x_centres, y_centres, indexing="ij")
    return pd.DataFrame(
        {
            "x_mid": np.repeat(x.ravel(), n_per_bin),
            "y_mid": np.repeat(y.ravel(), n_per_bin),
            "wepl": wepl,
        },
    )


def test_bin_wepl_gives_the_mean_and_count_per_bin():
    protons = pd.DataFrame(
        {
            "x_mid": [-3.0, -3.0, 1.0, 3.5],
            "y_mid": [0.0, 0.2, -1.0, 1.0],
            "wepl": [10.0, 20.0, 5.0, 7.0],
        },
    )

    mean_wepl, count = bin_wepl(protons, XBINS, YBINS)

    assert mean_wepl.shape == count.shape == (4, 3)
    assert mean_wepl[0, 1] == pytest.approx(15.0)
    assert mean_wepl[2, 0] == pytest.approx(5.0)
    assert mean_wepl[3, 2] == pytest.approx(7.0)
    assert count.sum() == 4
    assert count[0, 1] == 2
    assert mean_wepl[1].tolist() == [0.0, 0.0, 0.0]


def test_sinogram_has_one_flat_row_per_projection(tmp_path: Path):
    # Projection k holds protons of WEPL k + 1 in every bin, so each row of the sinogram is flat.
    n_projections = 4
    for k, angle in enumerate(np.arange(0, 360, 360 / n_projections)):
        make_protons(n_per_bin=6, wepl=k + 1.0).to_pickle(tmp_path / f"df_{angle}.pkl")

    sinogram, count = compute_sinogram(tmp_path, n_projections, 1.0, XBINS, YBINS)

    assert sinogram.shape == count.shape == (4, 4, 3)
    for k in range(n_projections):
        assert sinogram[k] == pytest.approx(np.full((4, 3), k + 1.0))
    assert count == pytest.approx(np.full((4, 4, 3), 6))


def test_sinogram_at_a_lower_dose_uses_a_fraction_of_the_protons(tmp_path: Path):
    n_projections = 2
    for angle in (0.0, 180.0):
        make_protons(n_per_bin=50, wepl=3.0).to_pickle(tmp_path / f"df_{angle}.pkl")

    _, count = compute_sinogram(tmp_path, n_projections, 0.2, XBINS, YBINS)

    assert count.sum(axis=(1, 2)).tolist() == [120, 120]


@pytest.mark.parametrize(("dose", "expected"), [(5, "5"), (5.0, "5"), (1.0, "1"), (0.2, "0.2")])
def test_format_dose_matches_the_name_the_reconstruction_loads(dose: float, expected: str):
    assert format_dose(dose) == expected


def test_mask_keeps_the_disk_of_the_phantom_plus_margin():
    im = np.ones((10, 10))

    masked = mask_outside_phantom(im, im_radius=3, margin=1)

    i, j = np.indices((10, 10))
    inside = (i - 4.5) ** 2 + (j - 4.5) ** 2 <= 4**2
    np.testing.assert_array_equal(masked, inside.astype(float))
