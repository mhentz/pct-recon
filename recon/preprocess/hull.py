"""Compute the intersections of proton paths with the circular phantom hull."""

import numpy as np
import pandas as pd


def compute_hull_entry_points(
    protons: pd.DataFrame,
    c_z: float,
    r: float,
) -> tuple[pd.DataFrame, tuple[np.ndarray, ...]]:
    """Compute hull intersection points for incoming protons.

    Args:
        protons: List-mode proton data.
        c_z: Centre of rotation in image coordinates.
        r: Radius of phantom.

    Returns:
        A dataframe with added columns for the intersection points in x, y, z and an array of indices
        of all the protons that entered the hull.
    """
    x, y, z = np.zeros((3, len(protons)))

    # project onto hull from entry detector
    # retrieve (x, y) positions on entrance detector and corresponding directions
    x_in = protons["x_in"].to_numpy()
    y_in = protons["y_in"].to_numpy()
    m_x = np.tan(protons["ang_xz_in"].to_numpy())  # slope of path entering image space
    m_y = np.tan(protons["ang_yz_in"].to_numpy())

    # quadratic coefficients for equation used to find intersection points
    a = 1 + m_x**2
    b = 2 * (m_x * x_in - c_z)
    c = c_z**2 + x_in**2 - r**2

    # filter out imaginary solutions: they don't intersect the phantom
    idx_in_hull = np.asarray(b**2 - 4 * a * c >= 0).nonzero()
    idx_not_in_hull = np.asarray(b**2 - 4 * a * c < 0).nonzero()

    a = a[idx_in_hull]
    b = b[idx_in_hull]
    c = c[idx_in_hull]

    # solve for z-coordinate at intersection using quadratic equation
    z[idx_in_hull] = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # plug into equation of incoming line to compute (x, y)-coords at intersection
    x[idx_in_hull] = m_x[idx_in_hull] * z[idx_in_hull] + x_in[idx_in_hull]
    y[idx_in_hull] = m_y[idx_in_hull] * z[idx_in_hull] + y_in[idx_in_hull]

    # for protons not in hull:
    # - set intersection z-coordinate to cz
    # - set intersection (x, y)-coords to their projection onto midplane at z = cz
    z[idx_not_in_hull] = c_z
    x[idx_not_in_hull] = m_x[idx_not_in_hull] * c_z + x_in[idx_not_in_hull]
    y[idx_not_in_hull] = m_y[idx_not_in_hull] * c_z + y_in[idx_not_in_hull]

    # save incoming hull intersection points in DataFrame
    protons["x_hull_in"] = x
    protons["y_hull_in"] = y
    protons["z_hull_in"] = z

    return protons, idx_in_hull


def compute_hull_exit_points(
    protons: pd.DataFrame,
    c_z: float,
    r: float,
) -> tuple[pd.DataFrame, tuple[np.ndarray, ...]]:
    """Compute hull intersection points for exiting protons.

    Args:
        protons: List-mode proton data.
        c_z: Centre of rotation in image coordinates.
        r: Radius of phantom.

    Returns:
        A dataframe with added columns for the intersection points in x, y, z and an array of indices
        of all the protons that exited the hull.
    """
    x, y, z = np.zeros((3, len(protons)))

    # project onto hull from exit detector
    # retrieve (x, y) positions on exit detector and corresponding directions
    x_out = protons["x_out"].to_numpy()
    y_out = protons["y_out"].to_numpy()
    m_x = np.tan(protons["ang_xz_out"].to_numpy())  # slope of path leaving image space
    m_y = np.tan(protons["ang_yz_out"].to_numpy())

    # quadratic coefficients for equation used to find intersection points
    a = 1 + m_x**2
    b = 2 * (m_x * x_out - 2 * c_z * m_x**2 - c_z)
    c = c_z**2 + (2 * c_z * m_x - x_out) ** 2 - r**2

    # filter out imaginary solutions: the protons don't intersect the phantom
    idx_in_hull = np.asarray(b**2 - 4 * a * c >= 0).nonzero()
    idx_not_in_hull = np.asarray(b**2 - 4 * a * c < 0).nonzero()

    a = a[idx_in_hull]
    b = b[idx_in_hull]
    c = c[idx_in_hull]

    # solve for z-coordinate at intersection using quadratic equation
    z[idx_in_hull] = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # plug into equation of outgoing line to compute (x, y)-coords at intersection
    x[idx_in_hull] = m_x[idx_in_hull] * (z[idx_in_hull] - 2 * c_z) + x_out[idx_in_hull]
    y[idx_in_hull] = m_y[idx_in_hull] * (z[idx_in_hull] - 2 * c_z) + y_out[idx_in_hull]

    # for protons not in hull:
    # - set intersection z-coordinate to c_z
    # - set intersection (x, y)-coords to their projection onto the midplane at z = c_z
    z[idx_not_in_hull] = c_z
    x[idx_not_in_hull] = m_x[idx_not_in_hull] * (c_z - 2 * c_z) + x_out[idx_not_in_hull]
    y[idx_not_in_hull] = m_y[idx_not_in_hull] * (c_z - 2 * c_z) + y_out[idx_not_in_hull]

    # save outgoing hull intersection points in DataFrame
    protons["x_hull_out"] = x
    protons["y_hull_out"] = y
    protons["z_hull_out"] = z

    return protons, idx_in_hull


def compute_hull_intersection(
    protons: pd.DataFrame,
    c_z: float,
    r: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute hull intersections with projected proton paths.

    Determines the (z, x)-value pairs where the proton path enters and exits the
    circular phantom. Protons that do not intersect it are separated from those
    that do (a tangent path, touching the circle at one point, counts as
    intersecting).

    The hull of the phantom is assumed to be a circle centred at (cz, 0) with
    radius r described by the equation (z - c_z)**2 + x**2 - r**2 = 0.

    The intersection point is found by substituting the equation of the straight
    entry path, x = mz + x_in, into the circle equation and solving for z using
    the quadratic formula, and substituting back into the line equation to find
    the corresponding x.

    Real solutions are deemed to intersect the phantom and vice versa for
    imaginary solutions.

    A proton counts as missing the hull if its entry or its exit line does not
    intersect the circle. Its hull points are set to the straight-line positions
    at the midplane z = c_z, projected from the entry and the exit detector.

    Note:
        The coordinate system is the same as in the Geant4 simulations, and hence
        the datasets. (z, x) maps to (x, y) in the "traditional" sense, i.e. z is
        the longitudinal position in image space while x is the transverse position.

    Args:
        protons: Dataframe of matched protons.
        c_z: Centre of rotation in the z-axis (protons' direction of travel).
        r: Phantom radius.

    Returns:
        The protons whose entry and exit lines both intersect the hull, and the
        protons that missed it, each with the hull entry and exit points added.
    """
    protons = protons.reset_index(drop=True)  # reset to match a, b, c arrays indexing
    index = protons.index.to_numpy()

    protons, idx_entered_hull = compute_hull_entry_points(protons, c_z, r)
    protons, idx_exited_hull = compute_hull_exit_points(protons, c_z, r)

    idx_passed_through_hull = np.intersect1d(idx_entered_hull, idx_exited_hull)
    idx_missed_hull = np.setdiff1d(index, idx_passed_through_hull)

    protons_passed_through_hull = protons.loc[idx_passed_through_hull]
    protons_missed_hull = protons.loc[idx_missed_hull]

    return protons_passed_through_hull, protons_missed_hull
