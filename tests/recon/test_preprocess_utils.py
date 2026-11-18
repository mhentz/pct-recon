"""Tests of ``recon/preprocess/utils.py``: WEPL lookup, angles and binning."""

import numpy as np
import pandas as pd
import pytest

from recon.preprocess.utils import compute_angles, compute_wepl, get_xy_bins


def test_compute_wepl_with_linear_lut():
    # Entry i of the LUT is 3 i; mid-bin energies 4.5 and 1.5 bins round up to entries 5 and 2.
    lut = 3.0 * np.arange(10)
    e_out = pd.Series([1.5e-4, 4.5e-4])

    wepl = compute_wepl(lut, pd.Series([4.5e-4, 4.5e-4]), e_out)

    assert isinstance(wepl, np.ndarray)
    np.testing.assert_allclose(wepl, [3 * 5 - 3 * 2, 0.0])


def test_compute_wepl_with_one_beam_energy():
    lut = 3.0 * np.arange(10)
    e_out = pd.Series([0.5e-4, 2.5e-4, 7.5e-4])

    wepl = compute_wepl(lut, 7.5e-4, e_out)

    np.testing.assert_allclose(wepl, [24 - 3, 24 - 9, 0.0])


def test_compute_angles_against_arctan2_by_hand():
    protons = pd.DataFrame(
        {
            "px_in": [0.0, 1.0],
            "py_in": [1.0, 0.0],
            "pz_in": [1.0, 1.0],
            "px_out": [-1.0, 0.0],
            "py_out": [0.0, 1.0],
            "pz_out": [np.sqrt(3), np.sqrt(3)],
        },
    )

    result = compute_angles(protons)

    np.testing.assert_allclose(result["ang_xz_in"], [0.0, np.pi / 4])
    np.testing.assert_allclose(result["ang_yz_in"], [np.pi / 4, 0.0])
    np.testing.assert_allclose(result["ang_xz_out"], [-np.pi / 6, 0.0])
    np.testing.assert_allclose(result["ang_yz_out"], [0.0, np.pi / 6])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.5, 0),  # inside the first bin
        (0.0, 1),  # an inner edge belongs to the bin on its right
        (1.99, 2),  # inside the last bin
        (-1.5, -1),  # below the first edge
        (2.0, 3),  # the last edge is outside the last bin
    ],
)
def test_get_xy_bins_edges(value, expected):
    edges = [-1.0, 0.0, 1.0, 2.0]

    x_bin, y_bin = get_xy_bins([value], [value], edges, edges)

    assert x_bin.tolist() == [expected]
    assert y_bin.tolist() == [expected]
