"""Analytic tests of the hull intersections in ``recon/preprocess/hull.py``.

The entry detector is at z = 0, the exit detector at z = 2 c_z, and the hull is the circle of
radius r centred at (z, x) = (c_z, 0).
"""

import numpy as np
import pandas as pd
import pytest

from recon.preprocess.hull import compute_hull_intersection

C_Z = 100.0
R = 50.0


def make_protons(
    x_in: list[float],
    ang_xz_in: list[float],
    x_out: list[float],
    ang_xz_out: list[float],
) -> pd.DataFrame:
    n = len(x_in)
    return pd.DataFrame(
        {
            "x_in": x_in,
            "y_in": np.zeros(n),
            "ang_xz_in": ang_xz_in,
            "ang_yz_in": np.zeros(n),
            "x_out": x_out,
            "y_out": np.zeros(n),
            "ang_xz_out": ang_xz_out,
            "ang_yz_out": np.zeros(n),
        },
    )


def test_straight_through_the_centre():
    protons = make_protons([0.0], [0.0], [0.0], [0.0])

    passed, missed = compute_hull_intersection(protons, C_Z, R)

    assert missed.empty
    row = passed.iloc[0]
    assert (row["z_hull_in"], row["x_hull_in"]) == pytest.approx((C_Z - R, 0.0))
    assert (row["z_hull_out"], row["x_hull_out"]) == pytest.approx((C_Z + R, 0.0))


def test_tangent_path_touches_the_hull_at_one_point():
    protons = make_protons([R], [0.0], [R], [0.0])

    passed, missed = compute_hull_intersection(protons, C_Z, R)

    assert missed.empty
    row = passed.iloc[0]
    assert (row["z_hull_in"], row["x_hull_in"]) == pytest.approx((C_Z, R))
    assert (row["z_hull_out"], row["x_hull_out"]) == pytest.approx((C_Z, R))


def test_oblique_path_through_the_centre():
    # The 45 degree line x = z - c_z enters at x_in = -c_z and leaves at x_out = +c_z; it crosses
    # the circle at a distance r from the centre along the direction (1, 1) / sqrt(2).
    protons = make_protons([-C_Z], [np.pi / 4], [C_Z], [np.pi / 4])
    d = R / np.sqrt(2)

    passed, missed = compute_hull_intersection(protons, C_Z, R)

    assert missed.empty
    row = passed.iloc[0]
    assert (row["z_hull_in"], row["x_hull_in"]) == pytest.approx((C_Z - d, -d))
    assert (row["z_hull_out"], row["x_hull_out"]) == pytest.approx((C_Z + d, d))


def test_y_follows_the_yz_slope_to_the_hull():
    protons = make_protons([0.0], [0.0], [0.0], [0.0])
    protons["y_in"] = 1.0
    protons["ang_yz_in"] = np.arctan(0.1)
    protons["y_out"] = 2.0
    protons["ang_yz_out"] = np.arctan(-0.1)

    passed, _ = compute_hull_intersection(protons, C_Z, R)

    row = passed.iloc[0]
    assert row["y_hull_in"] == pytest.approx(1.0 + 0.1 * (C_Z - R))
    assert row["y_hull_out"] == pytest.approx(2.0 - 0.1 * (C_Z + R - 2 * C_Z))


def test_missed_hull_falls_back_to_the_midplane():
    # The second proton passes outside the circle; its hull points are its straight-line
    # positions at z = c_z, from either detector.
    protons = make_protons([0.0, R + 1], [0.0, np.arctan(0.01)], [0.0, R + 3], [0.0, np.arctan(0.02)])

    passed, missed = compute_hull_intersection(protons, C_Z, R)

    assert passed.index.tolist() == [0]
    assert missed.index.tolist() == [1]
    row = missed.loc[1]
    assert row["z_hull_in"] == C_Z
    assert row["z_hull_out"] == C_Z
    assert row["x_hull_in"] == pytest.approx(R + 1 + 0.01 * C_Z)
    assert row["x_hull_out"] == pytest.approx(R + 3 - 0.02 * C_Z)


def test_proton_that_misses_on_one_side_only_is_missed():
    # Enters through the centre but its exit line passes outside the circle.
    protons = make_protons([0.0], [0.0], [R + 1], [0.0])

    passed, missed = compute_hull_intersection(protons, C_Z, R)

    assert passed.empty
    assert missed.index.tolist() == [0]
