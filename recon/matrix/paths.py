"""Compute the paths of protons through the phantom with the most likely path model."""

from math import floor

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from pct_tools.mlp import compute_mlp, get_coefficients

from recon.utils import Axis


def construct_mlp_parameters(protons: pd.DataFrame, axis: Axis) -> np.ndarray:
    """Collect the hull positions and angles of each proton along one axis.

    Args:
        protons: Protons with hull entry and exit positions and angles.
        axis: Axis along which the parameters are taken.

    Returns:
        An array with one row per proton and four columns: the entry position,
        the entry angle, the exit position and the exit angle.
    """
    parameters = np.array(
        [
            protons[f"{axis.value}_hull_in"],
            protons[f"ang_{axis.value}z_in"],
            protons[f"{axis.value}_hull_out"],
            protons[f"ang_{axis.value}z_out"],
        ],
    )
    return parameters.T


def compute_paths_outside_hull(
    proton: pd.Series,
    z: np.ndarray,
    transverse_path: np.ndarray,
    axis: Axis,
    idx_hull_entry: int,
    idx_hull_exit: int,
) -> np.ndarray:
    """Compute the transverse path of a proton outside the hull.

    Args:
        proton: Proton with entry and exit positions and angles.
        z: Positions along the z axis.
        transverse_path: Array to fill with the transverse positions.
        axis: Transverse axis, X or Y.
        idx_hull_entry: Index in z where the path enters the hull.
        idx_hull_exit: Index in z where the path exits the hull.

    Returns:
        The transverse path array, with the segments outside the hull filled in.
    """
    m_in = np.tan(proton[f"ang_{axis.value}z_in"])
    m_out = np.tan(proton[f"ang_{axis.value}z_out"])

    transverse_path[:idx_hull_entry] = m_in * z[:idx_hull_entry] + proton[f"{axis.value}_in"]
    transverse_path[idx_hull_exit:] = m_out * (z[idx_hull_exit:] - z[-1]) + proton[f"{axis.value}_out"]
    return transverse_path


def rotate_paths_in_xz_plane(
    z: np.ndarray,
    x: np.ndarray,
    projection_angle: float,
    c_x: float,
    c_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate path coordinates in the xz plane about the centre of rotation.

    Args:
        z: Positions along the z axis.
        x: Positions along the x axis.
        projection_angle: Projection angle in degrees.
        c_x: x coordinate of the centre of rotation.
        c_z: z coordinate of the centre of rotation.

    Returns:
        The rotated z and x coordinates.
    """
    angle = -np.deg2rad(projection_angle)

    c = np.cos(angle)
    s = np.sin(angle)

    z_rot = (z - c_z) * c - (x - c_x) * s + c_z
    x_rot = (z - c_z) * s + (x - c_x) * c + c_x
    return z_rot, x_rot


def compute_paths(
    phantom: str,
    protons: pd.DataFrame,
    projection_angle: float,
    centre_of_rotation: float,
    step_size: float,
    *,
    n_threads: int = 4,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the 3D most likely paths of the protons for one projection.

    Outside the hull the path follows the directions measured at the detectors. Inside the
    hull the most likely path is evaluated at evenly spaced depths from the hull entry to the
    hull exit, and stored at the depths of the z grid that lie between them. The two sets of
    depths do not coincide: a stored sample can sit up to one step from the depth it was
    evaluated at near the entry, and up to two steps near the exit. The transverse position
    is therefore off by at most two steps times the slope of the path, which for scattering
    angles of a few tens of mrad and a step of half a pixel is a few percent of a pixel.

    Args:
        phantom: Name of the phantom.
        protons: Protons with hull entry and exit measurements.
        projection_angle: Projection angle in degrees.
        centre_of_rotation: Position of the centre of rotation along z in mm.
        step_size: Step size along z in mm.
        n_threads: Number of threads used by the MLP computation.

    Returns:
        The x and y path arrays and the z positions, with x and z rotated about the
        centre of rotation.
    """
    logger.info(f"Processing 3D MLPs for {len(protons)} protons")

    coefficients = get_coefficients(phantom)
    mlp_parameters_x = construct_mlp_parameters(protons, Axis.X)
    mlp_parameters_y = construct_mlp_parameters(protons, Axis.Y)

    z_end = 2 * centre_of_rotation
    z = np.arange(0, z_end + step_size, step=step_size)

    if z[-1] != z_end:
        msg = "Something is wrong as z[-1] != z_end for some reason."
        raise ValueError(msg)

    x = np.zeros((len(protons), len(z)))
    y = np.zeros((len(protons), len(z)))

    n_missed_hull = 0

    for i in range(len(protons)):
        if i % 20000 == 0:
            logger.info(f"Processing proton {i}/{len(protons)}")

        proton = protons.iloc[i]

        # indices in the z-array where the path enters and exits the hull
        idx_hull_entry = floor(proton["z_hull_in"] / step_size)
        idx_hull_exit = floor(proton["z_hull_out"] / step_size)

        x[i] = compute_paths_outside_hull(proton, z, x[i], Axis.X, idx_hull_entry, idx_hull_exit)
        y[i] = compute_paths_outside_hull(proton, z, y[i], Axis.Y, idx_hull_entry, idx_hull_exit)

        n_steps_in_phantom = len(z[idx_hull_entry:idx_hull_exit])

        if n_steps_in_phantom == 0:
            n_missed_hull += 1
            logger.debug(f"Proton {proton.name} missed the hull. Assuming straight lines for MLP")
            continue

        mlp_z = np.linspace(proton["z_hull_in"], proton["z_hull_out"], n_steps_in_phantom, endpoint=True)

        mlp_z = mlp_z.astype(np.float32)
        mlp_x = compute_mlp(mlp_z, *mlp_parameters_x[i], coefficients, n_threads=n_threads)
        mlp_y = compute_mlp(mlp_z, *mlp_parameters_y[i], coefficients, n_threads=n_threads)

        x[i, idx_hull_entry:idx_hull_exit] = mlp_x
        y[i, idx_hull_entry:idx_hull_exit] = mlp_y

    if n_missed_hull:
        logger.warning(
            f"{n_missed_hull}/{len(protons)} protons missed the hull. Assuming straight lines for MLP",
        )

    c_x, c_z = (0, centre_of_rotation)  # phantom centre in lab coordinates [mm]

    z, x = rotate_paths_in_xz_plane(z, x, projection_angle, c_x, c_z)
    logger.info("Computed MLPs")

    return x, y, z
