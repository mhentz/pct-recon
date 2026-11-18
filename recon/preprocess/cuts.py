"""Apply data cuts to the protons based on the statistics of their mid-plane positions."""

from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import binned_statistic_2d

from recon.preprocess.utils import get_xy_bins


def compute_mean_and_stddev(
    x: np.ndarray,
    y: np.ndarray,
    vals: np.ndarray,
    bins: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean and standard deviation of values in 2D bins.

    Args:
        x: x coordinates of the points.
        y: y coordinates of the points.
        vals: Values to average in each bin.
        bins: Bin edges along x and y.

    Returns:
        The mean in each bin, with empty bins set to zero, and the standard deviation in each bin.
    """
    mean = binned_statistic_2d(x, y, vals, statistic="mean", bins=bins)[0]
    mean = np.nan_to_num(mean, nan=0.0)
    std = binned_statistic_2d(x, y, vals, statistic="std", bins=bins)[0]
    return mean, std


def compute_positions_at_midplane(protons: pd.DataFrame, c_z: float) -> pd.DataFrame:
    """Add the x and y positions of each proton at the mid-plane.

    Args:
        protons: Protons with hull entry and exit positions.
        c_z: z coordinate of the mid-plane.

    Returns:
        The protons with the ``x_mid`` and ``y_mid`` columns added.
    """
    dx = protons["x_hull_out"] - protons["x_hull_in"]
    dy = protons["y_hull_out"] - protons["y_hull_in"]
    dz = protons["z_hull_out"] - protons["z_hull_in"]

    protons["x_mid"] = dx / dz * (c_z - protons["z_hull_in"]) + protons["x_hull_in"]
    protons["y_mid"] = dy / dz * (c_z - protons["z_hull_in"]) + protons["y_hull_in"]
    return protons


def compute_proton_stats(
    protons: pd.DataFrame,
    c_z: float,
    im_halfspace_xz: float,
    im_halfspace_yz: float,
    xbinsize: float = 2.0,
    ybinsize: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[np.ndarray]]:
    """Compute the mid-plane positions, bin statistics and relative angles of the protons.

    Args:
        protons: Protons with hull entry and exit measurements.
        c_z: z coordinate of the mid-plane.
        im_halfspace_xz: Half-width of the image in the xz plane.
        im_halfspace_yz: Half-height of the image in the yz plane.
        xbinsize: Bin size along x.
        ybinsize: Bin size along y.

    Returns:
        The protons inside the image, a dictionary of binned statistics and the bin edges.
    """
    x_edges = np.arange(-im_halfspace_xz, im_halfspace_xz + xbinsize, xbinsize)
    y_edges = np.arange(-im_halfspace_yz, im_halfspace_yz + ybinsize, ybinsize)

    protons = compute_positions_at_midplane(protons, c_z)

    # remove protons that are outside the reconstruction volume at mid-plane intersection to avoid
    # out of bounds exception when doing binning for statistic later
    #   shape of imaging space:
    #       - 220 x 220 x 3 mm for Catphan and spiral
    #       - 350 x 350 x 3 mm for Gammex
    # the upper edge is excluded because np.digitize would put it one bin past the end
    protons = protons[(protons["x_mid"] >= -im_halfspace_xz) & (protons["x_mid"] < im_halfspace_xz)]
    protons = protons[(protons["y_mid"] >= -im_halfspace_yz) & (protons["y_mid"] < im_halfspace_yz)]

    logger.info("Compute variances of WEPL and angles...")
    x = protons["x_mid"]
    y = protons["y_mid"]
    wepl = protons["wepl"]
    rel_ang_xz = protons["ang_xz_out"] - protons["ang_xz_in"]
    rel_ang_yz = protons["ang_yz_out"] - protons["ang_yz_in"]

    mean_wepl, std_wepl = compute_mean_and_stddev(x, y, wepl, [x_edges, y_edges])
    mean_ang_xz, std_ang_xz = compute_mean_and_stddev(x, y, rel_ang_xz, [x_edges, y_edges])
    mean_ang_yz, std_ang_yz = compute_mean_and_stddev(x, y, rel_ang_yz, [x_edges, y_edges])

    stats = {
        "mean_wepl": mean_wepl,
        "std_wepl": std_wepl,
        "mean_rel_ang_xz": mean_ang_xz,
        "std_rel_ang_xz": std_ang_xz,
        "mean_rel_ang_yz": mean_ang_yz,
        "std_rel_ang_yz": std_ang_yz,
    }

    bin_edges = [x_edges, y_edges]

    protons["rel_ang_xz"] = rel_ang_xz
    protons["rel_ang_yz"] = rel_ang_yz
    return protons, stats, bin_edges


class FilterQuantity(StrEnum):
    """Quantities used to filter the protons by their binned statistics."""

    WEPL = "wepl"
    ANGLE_XZ = "rel_ang_xz"
    ANGLE_YZ = "rel_ang_yz"


def filter_by_statistic(
    protons: pd.DataFrame,
    quantity: FilterQuantity,
    n_sigma: float,
    stats: Mapping[str, np.ndarray],
    bin_locs: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Find the protons whose quantity lies outside n_sigma of the bin mean.

    The comparison is strict, as in the thesis reconstructions, so a bin with zero spread
    (for example a single proton) loses all its protons.

    Args:
        protons: Protons to filter.
        quantity: Quantity to compare with the bin statistics.
        n_sigma: Number of standard deviations allowed around the bin mean.
        stats: Binned means and standard deviations, keyed by statistic name.
        bin_locs: Bin index of each proton along x and y.

    Returns:
        The index labels of the protons to remove.
    """
    x_bin_locs, y_bin_locs = bin_locs

    val = protons[quantity.value]
    mean = stats[f"mean_{quantity.value}"][x_bin_locs, y_bin_locs]
    std = stats[f"std_{quantity.value}"][x_bin_locs, y_bin_locs]
    condition = np.abs(val - mean) < n_sigma * std

    return protons[~condition].index


def apply_data_cuts(
    protons: pd.DataFrame,
    n_sigma: float,
    stats: Mapping[str, np.ndarray],
    bin_edges: Sequence[np.ndarray],
) -> pd.DataFrame:
    """Remove the protons that fail the statistical cuts.

    Args:
        protons: Protons with the ``x_mid`` and ``y_mid`` columns.
        n_sigma: Number of standard deviations allowed around the bin mean.
        stats: Binned means and standard deviations, keyed by statistic name.
        bin_edges: Bin edges along x and y.

    Returns:
        The protons that pass all cuts.
    """
    bin_locs = get_xy_bins(protons["x_mid"], protons["y_mid"], *bin_edges)

    idx_protons_to_remove = []
    for quantity in FilterQuantity:
        idx_protons_to_remove.extend(filter_by_statistic(protons, quantity, n_sigma, stats, bin_locs))
    idx_protons_to_remove = np.unique(idx_protons_to_remove)

    return protons.loc[~protons.index.isin(np.unique(idx_protons_to_remove))]
