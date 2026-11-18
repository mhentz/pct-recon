"""Report the percentage error of the mean ROI value for each Gammex insert."""

from collections import defaultdict
from enum import Enum, StrEnum

import numpy as np
from loguru import logger
from typer import run

from analysis.accuracy.constants import MATERIALS_MAP, ROI_LOCATIONS, RSP_MAP
from analysis.ideal_alpha import get_ideal_alpha
from analysis.plotting.utils import get_path


class SetupType(StrEnum):
    """Dataset."""

    IDEAL = "ideal"
    REALISTIC = "realistic"


class Regulariser(StrEnum):
    """Regulariser."""

    TV = "TV"
    DTV = "dTV"


class NumProjections(int, Enum):
    """Number of projections."""

    N45 = 45
    N720 = 720


def extract_pixel_values_in_roi(
    image: np.ndarray, location: tuple[float, float], radius: float, roi_shape: tuple[int, int] = (80, 80)
) -> tuple[np.ndarray, np.ndarray]:
    """Get a circular region of interest (ROI) from an image.

    Args:
        image: The image to get the ROI from.
        location: The location of the centre of the ROI.
        radius: The radius of the circular ROI.
        roi_shape: The shape of the square ROI the circle is drawn from.

    Returns:
        The ROI pixel values as a 1D numpy array, and the square ROI they were taken from.

    """
    x, y = location
    half_size_x = int(roi_shape[0] / 2)
    half_size_y = int(roi_shape[1] / 2)

    roi = image[y - half_size_y : y + half_size_y, x - half_size_x : x + half_size_x]

    pixels_in_roi = []

    ni = roi_shape[1]
    nj = roi_shape[0]

    ci = ni / 2
    cj = nj / 2

    for i in range(ni):
        for j in range(nj):
            if (i + 0.5 - ci) ** 2 + (j + 0.5 - cj) ** 2 <= radius**2:
                pixels_in_roi.append(roi[i, j])  # noqa: PERF401

    return np.array(pixels_in_roi), roi


DOSES = [5.0, 1.0, 0.2]
INSERT_LABELS = [4, 14, 7, 9, 11, 5, 8, 0, 2, 6, 3, 1, 10, 12, 13, 15]


def compute_accuracy(setup_type: SetupType, reg: Regulariser, n_proj: NumProjections) -> None:
    """Print the percentage RSP error of every insert at each dose, and the mean absolute errors.

    Args:
        setup_type: Dataset, ``ideal`` or ``realistic``.
        reg: Regulariser, ``TV`` or ``dTV``.
        n_proj: Number of projections.
    """
    setup = f"gammex-{setup_type.value}"
    n_projections = int(n_proj.value)

    logger.debug(f"Computing accuracy for {setup} {reg.value} {n_projections}")
    abs_err = np.zeros(len(DOSES))
    abs_err_star = np.zeros(len(DOSES))
    abs_err_starstar = np.zeros(len(DOSES))

    # insert_labels = [14, 7, 11, 5, 8, 0, 2, 6, 3, 1, 12, 13, 15]  # without LN300, LN450
    # insert_labels = [4, 14, 7, 9, 11, 5, 8, 0, 2, 6, 3, 1, 10, 12, 13, 15]

    # print(get_path(setup_type, reg, n_proj, 5.0))
    # images = {
    #     5.0: np.load(get_path(setup_type, reg, n_proj, 5.0)),
    #     1.0: np.load(get_path(setup_type, reg, n_proj, 1.0)),
    #     0.1: np.load(get_path(setup_type, reg, n_proj, 0.1)),
    # }

    images: dict[float, np.ndarray] = {}
    for dose in DOSES:
        alpha = get_ideal_alpha(setup, dose, n_projections)
        images[dose] = np.load(get_path(setup, dose, n_projections, alpha, reg.value) / "yk_min_slice.npy")

    percent_errors = defaultdict(dict)

    for label in INSERT_LABELS:
        material = MATERIALS_MAP[label]
        print(f"| {material:14} & ", end="")

        for j, dose in enumerate(DOSES):
            # for j, dose in enumerate([5.0]):
            image = np.flipud(images[dose])

            roi_values, _ = extract_pixel_values_in_roi(image, ROI_LOCATIONS[label], radius=7 / 0.5)
            percent_error = (roi_values.mean() - RSP_MAP[material]) / RSP_MAP[material] * 100

            print(f"{percent_error:5.2f} & ", end="")

            abs_err[j] += abs(percent_error)
            if label not in [4, 9, 10]:
                abs_err_star[j] += abs(percent_error)
            if label != 9:  # noqa: PLR2004
                abs_err_starstar[j] += abs(percent_error)

            percent_errors[dose][material] = percent_error

        print("\n", end="")

    abs_err /= len(INSERT_LABELS)
    abs_err_star /= len(INSERT_LABELS) - 3
    abs_err_starstar /= len(INSERT_LABELS) - 1

    print(f"| MAPE           & {abs_err[0]:5.2f} & {abs_err[1]:5.2f} & {abs_err[2]:5.2f}")

    print(f"| MAPE*          & {abs_err_star[0]:5.2f} & {abs_err_star[1]:5.2f} & {abs_err_star[2]:5.2f} ")
    print(
        "| MAPE**         & "
        f"{abs_err_starstar[0]:5.2f} & "
        f"{abs_err_starstar[1]:5.2f} & "
        f"{abs_err_starstar[2]:5.2f} "
    )
    print()
    print(
        f"Mean absolute error: {np.mean(abs_err[0]):.2f}%   "
        f"{np.mean(abs_err[1]):.2f}%   "
        f"{np.mean(abs_err[2]):.2f}% "
    )

    def get_min(insert_to_val: dict) -> tuple[str, float]:
        min_val = np.min(np.abs(list(insert_to_val.values())))
        min_index = np.argmin(np.abs(list(insert_to_val.values())))
        min_material = list(insert_to_val.keys())[min_index]
        return min_material, min_val

    # min_5 = np.min(np.abs(list(percent_errors[5.0].values())))
    # min_5_index = np.argmin(np.abs(list(percent_errors[5.0].values())))
    # min_5_material = list(percent_errors[5.0].keys())[min_5_index]

    min_5_material, min_5 = get_min(percent_errors[5.0])
    min_1_material, min_1 = get_min(percent_errors[1.0])
    min_02_material, min_02 = get_min(percent_errors[0.2])

    print(f"Min at 5.0 mGy: {min_5_material}: {min_5:.2f}%")
    print(f"Min at 1.0 mGy: {min_1_material}: {min_1:.2f}%")
    print(f"Min at 0.2 mGy: {min_02_material}: {min_02:.2f}%")


if __name__ == "__main__":
    # setup_type = "ideal"
    # # setup_type = "realistic"

    # reg = "TV"
    # # reg = "dTV"

    # n_proj = 720
    # # n_proj = 45

    # compute_accuracy(setup_type, reg, n_proj)

    run(compute_accuracy)
