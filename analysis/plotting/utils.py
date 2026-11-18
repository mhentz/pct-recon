"""Helpers for locating reconstructions and building colour normalisations for plots."""

from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np

from analysis.paths import RECON_DATA_DIR


def glob_path(dirpath: Path | str, pattern: str) -> Path:
    """Return the last path in a directory that matches a glob pattern.

    Args:
        dirpath: Directory to search.
        pattern: Glob pattern to match within ``dirpath``.

    Returns:
        Last matching path in sorted order.

    Raises:
        FileNotFoundError: If no path matches the pattern.
    """
    if isinstance(dirpath, str):
        dirpath = Path(dirpath)
    paths = sorted(dirpath.glob(pattern))
    if not paths:
        msg = f"No directories found matching pattern: {pattern}"
        raise FileNotFoundError(msg)
    return paths[-1]


def get_path(
    setup: str,
    dose: float,
    n_projections: int,
    alpha: str,
    regulariser: Literal["TV", "dTV"],
) -> Path:
    """Return the processed output directory of a reconstruction.

    Args:
        setup: Setup name, e.g. ``"gammex-ideal"``.
        dose: Dose in mGy.
        n_projections: Number of projections.
        alpha: Regularisation weight as a string.
        regulariser: Regulariser, ``"TV"`` or ``"dTV"``.

    Returns:
        ``processed`` directory of the last matching reconstruction.

    Raises:
        ValueError: If ``regulariser`` is neither ``"TV"`` nor ``"dTV"``.
    """
    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    image_width = 700 if setup.startswith("gammex") else 880

    pattern = (
        "fista_exact_init_zero_"
        f"{image_width}pix_{dose:.1f}mGy_{n_projections}proj_{regulariser}_iso_alpha_{alpha}_"
    )

    if regulariser == "TV":
        pattern += "iter*"
    elif regulariser == "dTV":
        pattern += "eta_2e-1_iter*"
    else:
        msg = "regulariser must be either 'TV' or 'dTV'"
        raise ValueError(msg)

    return (
        glob_path(
            dirpath=f"{RECON_DATA_DIR}/{setup}/{dose_str}mGy/{n_projections}proj/recon/",
            pattern=pattern,
        )
        / "processed"
    )


def load_image(path: Path) -> tuple[np.ndarray, int]:
    """Load the minimum-error slice and its iteration index from a processed directory.

    Args:
        path: Processed directory containing ``min_error_index.txt`` and ``yk_min_slice.npy``.

    Returns:
        Minimum-error slice and the index of its iteration.
    """
    i = int((path / "min_error_index.txt").read_text().strip())
    xi = np.load(path / "yk_min_slice.npy")

    return xi, i


def divnorm(max_val: float) -> mpl.colors.TwoSlopeNorm:
    """Return a symmetric diverging colour normalisation from -max_val to max_val.

    Args:
        max_val: Largest absolute value on the colour scale.

    Returns:
        Two-slope colour normalisation centred on zero.
    """
    return mpl.colors.TwoSlopeNorm(vmin=-max_val - 1e-6, vcenter=0, vmax=max_val + 1e-6)


def round_to_nearest_tick(x: float) -> float:
    """Round a value to the nearest colour-bar tick from 0.4, 0.6, 0.8 and 1.0.

    Args:
        x: Value to round.

    Returns:
        Nearest tick value.
    """
    targets = np.array([0.4, 0.6, 0.8, 1.0])
    return targets[np.argmin(np.abs(targets - x))]
