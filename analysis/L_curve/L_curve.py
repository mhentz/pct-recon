"""Collect L-curve points at the minimum-error iterate for each regularisation weight."""  # noqa: N999

from math import sqrt
from pathlib import Path

import numpy as np
from blosc2 import load_array
from loguru import logger

from analysis.paths import RECON_DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]


def TV(x: np.ndarray) -> float:
    """Compute the isotropic total variation of a 2D or 3D array.

    Args:
        x: Image or volume array.

    Returns:
        Sum of the gradient magnitudes over all voxels.
    """
    if x.ndim < 2:  # noqa: PLR2004
        msg = f"Require 2D or 3D numpy arrays. The provided input has {x.ndim=}."
        raise ValueError(msg)

    grad_x = np.zeros(x.shape)
    grad_y = np.zeros(x.shape)
    grad_z = np.zeros(x.shape)

    grad_x[:, :, :-1] = np.diff(x, axis=2)
    grad_y[:, :-1, :] = np.diff(x, axis=1)
    grad_z[:-1, :, :] = np.diff(x, axis=0)

    grad = np.stack([grad_x, grad_y, grad_z])
    return np.linalg.norm(grad, axis=0).sum()


def get_regularisation_values(metrics: np.ndarray) -> np.ndarray:
    """Return the regularisation term from each row of a metrics array.

    Args:
        metrics: Array of per-iteration metrics, one row per iteration.

    Returns:
        Last column of the metrics array.
    """
    return metrics[:, -1]


def get_squared_residuals(metrics: np.ndarray) -> np.ndarray:
    """Return the squared residual norm from each row of a metrics array.

    Args:
        metrics: Array of per-iteration metrics, one row per iteration.

    Returns:
        Second column of the metrics array.
    """
    return metrics[:, 1]


def curvature(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Compute the discrete curvature of an (x, y) curve at one point.

    The curvature is taken from the circumradius of the triangle formed by the
    point and its two neighbours.

    Args:
        x: x-coordinates of the curve.
        y: y-coordinates of the curve.
        index: Index of the point at which to evaluate the curvature.

    Returns:
        Curvature at ``index``, or 0.0 at the first and last points.
    """
    if index < 1:
        return 0.0

    if index >= len(x) - 1:
        return 0.0

    x0, y0 = x[index - 1], y[index - 1]
    x1, y1 = x[index], y[index]
    x2, y2 = x[index + 1], y[index + 1]

    a = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    b = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    c = sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

    s = (a + b + c) / 2
    area = sqrt(abs(s * (s - a) * (s - b) * (s - c)))

    return 4 * area / (a * b * c)


alphas = [
    "1e-5",
    "2.5e-5",
    "5e-5",
    "1e-4",
    "2.5e-4",
    "5e-4",
    "7.1e-4",
    "1e-3",
    "1.58e-3",
    "2.5e-3",
    "3.54e-3",
    "5e-3",
    "1e-2",
    "2.5e-2",
    "5e-2",
    "1e-1",
    "5e-1",
]

setup = "ctp528-ideal"
# setup = "gammex-ideal"
# setup = "spiral-ideal"

# setup = "ctp528-ideal"
# setup = "gammex-ideal"
# setup = "spiral-realistic"

image_w = 880 if setup != "gammex-ideal" else 700

grid_size = (3, image_w, image_w)

im_gt_2d = np.load(
    f"{REPO_ROOT}/phantoms/ground_truths/{setup.split('-', maxsplit=1)[0]}_gt_{image_w}pix.npy"
)
im_gt = np.stack([im_gt_2d] * grid_size[0])

x_init = "zero"
# x_init = "fbp"

reg_values = []
res_values = []
err_values = []

images = []
err_images = []

n_projections = 720
# n_projections = 360
# n_projections = 180
# n_projections = 90
# n_projections = 45

dose = 5
# dose = 1
# dose = 0.2

# alphas = [alpha for alpha, toggle in alpha_toggle[n_projections][x_init].items() if toggle]
# alpha_to_i = {alpha: i for i, alpha in enumerate(alphas)}

yk_min_list = []
yk_list = []

for alpha in alphas:
    logger.info(f"Processing alpha: {alpha}")

    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    filename = (
        f"{RECON_DATA_DIR}/ctp528-ideal/{dose_str}mGy/{n_projections}proj/recon/"
        f"fista_exact_init_{x_init}_880pix_{dose:.1f}mGy_{n_projections}proj_TV_iso_alpha_{alpha}_iter200/yk_final.bl2"
    )

    if not Path(filename).exists():
        logger.warning(f"File for {alpha} does not exist, skipping.")
        continue

    yk = load_array(filename)

    error = (yk[1:, 1, ...] - im_gt[0]).astype(np.float64)
    error_norm: np.ndarray = np.linalg.norm(error, axis=(1, 2))
    min_error_index = error_norm[2:].argmin()

    images.append(yk[min_error_index])
    err_images.append(error[min_error_index])
    err_values.append(error_norm[min_error_index])

    yk_min = yk[min_error_index]
    tv_min = TV(yk_min)

    yk_min_list.append(yk_min[1])
    yk_list.append(yk[:, 1, ...])

    metrics = load_array(
        f"{RECON_DATA_DIR}/ctp528-ideal/5mGy/{n_projections}proj/recon/"
        f"fista_exact_init_{x_init}_880pix_5.0mGy_{n_projections}proj_TV_iso_alpha_{alpha}_iter200/metrics_final.bl2",
    )

    res_norm_sq = get_squared_residuals(metrics)[min_error_index - 1]
    reg_val = get_regularisation_values(metrics)[min_error_index - 1] / float(alpha)

    reg_values.append(reg_val)
    res_values.append(res_norm_sq)
