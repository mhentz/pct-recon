"""Tests of the proton paths in ``recon/matrix/paths.py``.

The entry detector is at z = 0, the exit detector at z = 2 c_z, and the hull is entered at
``z_hull_in`` and left at ``z_hull_out``. The protons used here travel on straight lines, for
which the most likely path is that line.
"""

import numpy as np
import pandas as pd
import pytest

from recon.matrix.paths import (
    compute_paths,
    compute_paths_outside_hull,
    construct_mlp_parameters,
    rotate_paths_in_xz_plane,
)
from recon.utils import Axis

PHANTOM = "ctp528"
C_Z = 100.0
STEP = 0.5
Z_HULL_IN = 50.0
Z_HULL_OUT = 150.0


def make_straight_proton(
    x_in: float,
    slope_xz: float,
    y_in: float = 0.0,
    slope_yz: float = 0.0,
    z_hull_in: float = Z_HULL_IN,
    z_hull_out: float = Z_HULL_OUT,
) -> dict[str, float]:
    return {
        "x_in": x_in,
        "y_in": y_in,
        "ang_xz_in": np.arctan(slope_xz),
        "ang_yz_in": np.arctan(slope_yz),
        "x_out": x_in + slope_xz * 2 * C_Z,
        "y_out": y_in + slope_yz * 2 * C_Z,
        "ang_xz_out": np.arctan(slope_xz),
        "ang_yz_out": np.arctan(slope_yz),
        "x_hull_in": x_in + slope_xz * z_hull_in,
        "y_hull_in": y_in + slope_yz * z_hull_in,
        "z_hull_in": z_hull_in,
        "x_hull_out": x_in + slope_xz * z_hull_out,
        "y_hull_out": y_in + slope_yz * z_hull_out,
        "z_hull_out": z_hull_out,
    }


@pytest.mark.parametrize(
    ("projection_angle", "expected_z", "expected_x"),
    [
        (0.0, C_Z + 2.0, 6.0),
        (90.0, C_Z + 1.0, 3.0),
        (180.0, C_Z - 2.0, 4.0),
        (360.0, C_Z + 2.0, 6.0),
    ],
)
def test_rotation_about_the_centre(projection_angle: float, expected_z: float, expected_x: float):
    # The point lies 2 mm downstream of the centre of rotation and 1 mm to its side in x.
    c_x = 5.0
    z = np.array([C_Z + 2.0])
    x = np.array([c_x + 1.0])

    z_rot, x_rot = rotate_paths_in_xz_plane(z, x, projection_angle, c_x, C_Z)

    assert z_rot == pytest.approx([expected_z])
    assert x_rot == pytest.approx([expected_x])


def test_rotation_keeps_the_distance_to_the_centre():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 2 * C_Z, 50)
    x = rng.uniform(-40, 40, 50)

    z_rot, x_rot = rotate_paths_in_xz_plane(z, x, 37.0, 0.0, C_Z)

    assert np.hypot(z_rot - C_Z, x_rot) == pytest.approx(np.hypot(z - C_Z, x))


def test_rotation_broadcasts_one_z_grid_over_all_protons():
    z = np.linspace(0, 2 * C_Z, 5)
    x = np.zeros((3, 5))

    z_rot, x_rot = rotate_paths_in_xz_plane(z, x, 90.0, 0.0, C_Z)

    assert z_rot.shape == (3, 5)
    assert x_rot.shape == (3, 5)


@pytest.mark.parametrize("axis", [Axis.X, Axis.Y])
def test_mlp_parameters_column_order(axis: Axis):
    a = axis.value
    protons = pd.DataFrame(
        {
            f"{a}_hull_in": [1.0, 5.0],
            f"ang_{a}z_in": [2.0, 6.0],
            f"{a}_hull_out": [3.0, 7.0],
            f"ang_{a}z_out": [4.0, 8.0],
        },
    )

    parameters = construct_mlp_parameters(protons, axis)

    assert parameters.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_paths_outside_the_hull_are_the_detector_tangents():
    # Entry line x = 1 + 0.1 z from the entry detector, exit line x = 9 - 0.2 (z_end - z)
    # into the exit detector at z_end = 10. The three samples inside the hull are left alone.
    proton = pd.Series(
        {"x_in": 1.0, "ang_xz_in": np.arctan(0.1), "x_out": 9.0, "ang_xz_out": np.arctan(0.2)},
    )
    z = np.arange(0.0, 11.0)
    transverse_path = np.full(len(z), np.nan)

    path = compute_paths_outside_hull(proton, z, transverse_path, Axis.X, 4, 7)

    assert path[:4] == pytest.approx([1.0, 1.1, 1.2, 1.3])
    assert np.isnan(path[4:7]).all()
    assert path[7:] == pytest.approx([8.4, 8.6, 8.8, 9.0])


def test_parallel_protons_keep_their_transverse_position():
    protons = pd.DataFrame(
        [make_straight_proton(0.0, 0.0), make_straight_proton(10.0, 0.0, y_in=-1.0)],
    )

    x, y, z = compute_paths(PHANTOM, protons, 0.0, C_Z, STEP, n_threads=1)

    n_z = round(2 * C_Z / STEP) + 1
    assert x.shape == y.shape == z.shape == (2, n_z)
    assert z == pytest.approx(np.tile(np.arange(n_z) * STEP, (2, 1)))
    assert x[0] == pytest.approx(np.zeros(n_z), abs=1e-6)
    assert x[1] == pytest.approx(np.full(n_z, 10.0))
    assert y[0] == pytest.approx(np.zeros(n_z), abs=1e-6)
    assert y[1] == pytest.approx(np.full(n_z, -1.0))


def test_paths_start_and_end_at_the_detector_positions():
    protons = pd.DataFrame([make_straight_proton(3.0, 0.02, y_in=1.0, slope_yz=-0.01)])

    x, y, _ = compute_paths(PHANTOM, protons, 0.0, C_Z, STEP, n_threads=1)

    assert (x[0, 0], y[0, 0]) == pytest.approx((3.0, 1.0))
    assert (x[0, -1], y[0, -1]) == pytest.approx((3.0 + 0.02 * 2 * C_Z, 1.0 - 0.01 * 2 * C_Z))


@pytest.mark.parametrize(
    ("z_hull_in", "z_hull_out"),
    [(50.0, 150.0), (50.3, 149.8), (50.49, 149.99)],
)
def test_oblique_straight_proton_stays_on_its_line(z_hull_in: float, z_hull_out: float):
    # The path inside the hull is evaluated at evenly spaced depths between the hull entry and
    # exit and stored on the z grid, so a sample can sit up to two steps away from the depth it
    # was evaluated at. The path is therefore on the line only to within two steps times the
    # slope.
    slope_xz, slope_yz = 0.05, -0.02
    protons = pd.DataFrame(
        [make_straight_proton(3.0, slope_xz, 1.0, slope_yz, z_hull_in, z_hull_out)],
    )

    x, y, z = compute_paths(PHANTOM, protons, 0.0, C_Z, STEP, n_threads=1)

    assert np.abs(x - (3.0 + slope_xz * z)).max() <= 2 * STEP * abs(slope_xz) + 1e-6
    assert np.abs(y - (1.0 + slope_yz * z)).max() <= 2 * STEP * abs(slope_yz) + 1e-6


def test_paths_are_rotated_by_the_projection_angle():
    # A proton along the beam axis, displaced by 10 mm in x. At 90 degrees the z grid is
    # rotated onto x (decreasing) and the path lies at z = c_z + 10.
    protons = pd.DataFrame([make_straight_proton(10.0, 0.0, y_in=2.0)])

    x, y, z = compute_paths(PHANTOM, protons, 90.0, C_Z, STEP, n_threads=1)

    n_z = round(2 * C_Z / STEP) + 1
    assert z[0] == pytest.approx(np.full(n_z, C_Z + 10.0))
    assert x[0] == pytest.approx(C_Z - np.arange(n_z) * STEP)
    assert y[0] == pytest.approx(np.full(n_z, 2.0))


def test_proton_that_misses_the_hull_follows_its_detector_tangents():
    # Hull entry and exit coincide, so no depth lies inside the hull. The path is the entry
    # tangent up to that depth and the exit tangent after it.
    proton = make_straight_proton(60.0, 0.0, z_hull_in=C_Z, z_hull_out=C_Z)
    proton["ang_xz_out"] = np.arctan(0.1)
    proton["x_out"] = 70.0
    protons = pd.DataFrame([proton])

    x, _, z = compute_paths(PHANTOM, protons, 0.0, C_Z, STEP, n_threads=1)

    upstream = z[0] < C_Z
    assert x[0, upstream] == pytest.approx(np.full(upstream.sum(), 60.0))
    assert x[0, ~upstream] == pytest.approx(70.0 - 0.1 * (2 * C_Z - z[0, ~upstream]))


def test_step_size_that_does_not_divide_the_detector_distance_is_rejected():
    protons = pd.DataFrame([make_straight_proton(0.0, 0.0)])

    with pytest.raises(ValueError, match="z_end"):
        compute_paths(PHANTOM, protons, 0.0, C_Z, 0.3, n_threads=1)
