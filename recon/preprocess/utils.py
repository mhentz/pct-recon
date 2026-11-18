"""Utilities for building the proton dataframes and their angles and bins."""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from recon.utils import Axis, DetectorChoice


def compute_angles(protons: pd.DataFrame) -> pd.DataFrame:
    """Compute angles of protons in xz and yz planes."""
    protons["ang_xz_in"] = get_proton_angle(protons, DetectorChoice.ENTRY, Axis.X)
    protons["ang_yz_in"] = get_proton_angle(protons, DetectorChoice.ENTRY, Axis.Y)
    protons["ang_xz_out"] = get_proton_angle(protons, DetectorChoice.EXIT, Axis.X)
    protons["ang_yz_out"] = get_proton_angle(protons, DetectorChoice.EXIT, Axis.Y)
    return protons


def compute_wepl(
    lut: np.ndarray,
    e_in: float | pd.Series,
    e_out: pd.Series,
    binsize: float = 0.1 / 1e3,
) -> np.ndarray:
    """Compute WEPL from energy loss in the detector.

    Args:
        lut: Lookup table whose entry ``i`` belongs to the energy ``i * binsize``.
        e_in: Energy of each proton on entry, or one beam energy for all.
        e_out: Energy of each proton on exit.
        binsize: Energy step between consecutive lookup table entries.

    Returns:
        The WEPL of each proton: the lookup table value at the entry energy minus the value at
        the exit energy, each energy rounded up to the next table entry.
    """
    return lut[np.ceil(e_in / binsize).astype(int)] - lut[np.ceil(e_out / binsize).astype(int)]


def get_proton_angle(protons: pd.DataFrame, detector: DetectorChoice, axis: Axis) -> pd.Series:
    """Compute the angle of each proton to the z axis, projected onto the xz or yz plane.

    Args:
        protons: Protons with direction components ``p<axis>_<detector>`` and ``pz_<detector>``.
        detector: Entry or exit detector.
        axis: Transverse axis, X or Y, that spans the plane with z.

    Returns:
        The angle in radians, ``arctan2(p_axis, p_z)``.
    """
    return np.arctan2(protons[f"p{axis.value}_{detector.value}"], protons[f"pz_{detector.value}"])


def clean_raw_df(
    df: pd.DataFrame,
    vars_to_save: Sequence[str],
    rename_vars: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Keep the requested variables of a raw dataframe and optionally rename them.

    Args:
        df: Raw dataframe.
        vars_to_save: Names of the variables to keep.
        rename_vars: Mapping from the old to the new column names.

    Returns:
        The selected and renamed dataframe, with the original index saved in ``old_index``.
    """
    df = df[vars_to_save]
    df["old_index"] = df.index

    if rename_vars:
        df = df.rename(columns=rename_vars)
    return df


def get_xy_bins(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    xbins: npt.ArrayLike,
    ybins: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the zero-based bin index of each (x, y) point.

    Args:
        x: x coordinates of the points.
        y: y coordinates of the points.
        xbins: Bin edges along x.
        ybins: Bin edges along y.

    Returns:
        The bin index along x and the bin index along y of each point.
    """
    # Subtract 1 so that the first bin has index 0
    x_mid_bin = np.digitize(np.asarray(x, dtype=float), np.asarray(xbins, dtype=float)) - 1
    y_mid_bin = np.digitize(np.asarray(y, dtype=float), np.asarray(ybins, dtype=float)) - 1

    return x_mid_bin, y_mid_bin
