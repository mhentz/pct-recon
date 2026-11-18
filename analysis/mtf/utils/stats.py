"""Statistical utility functions for MTF analysis.

This module provides functions for calculating statistics on histograms
and determining background/foreground regions in PCT and XCT images.
"""

from typing import Literal

import numpy as np

PCT_BG_MIN = 0.75
PCT_BG_MAX = 1.75
XCT_BG_MIN = 0.1
XCT_BG_MAX = 0.3
BG_THRESHOLD_MIN = 0.2


def calculate_mean(hist: np.ndarray, bins: np.ndarray, bin_selection: slice) -> float:
    """Compute the mean of a histogram over a selection of bins.

    Args:
        hist: Histogram counts.
        bins: Bin values.
        bin_selection: Slice of the bins to include.

    Returns:
        Count-weighted mean of the selected bins.
    """
    return np.sum(hist[bin_selection] * bins[bin_selection]) / np.sum(hist[bin_selection])


def determine_background_foreground_locations(
    bins: np.ndarray,
    image_type: Literal["pct", "xct"] = "pct",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices of the background and foreground bins of a histogram.

    Args:
        bins: Bin values.
        image_type: Modality, ``"pct"`` or ``"xct"``, which sets the background range.

    Returns:
        Indices of the background bins and indices of the foreground bins.

    Raises:
        NotImplementedError: If ``image_type`` is not supported.
    """
    if image_type == "pct":
        bg_bins = np.where((bins > PCT_BG_MIN) & (bins < PCT_BG_MAX))[0]  # bg_fg_threshold = 1.75
        fg_bins = np.where(bins > PCT_BG_MAX)[0]
    elif image_type == "xct":
        bg_bins = np.where((bins > XCT_BG_MIN) & (bins < XCT_BG_MAX))[0]  # bg_fg_threshold = 0.3
        fg_bins = np.where(bins > XCT_BG_MAX)[0]
    else:
        msg = f"image_type {image_type} not implemented"
        raise NotImplementedError(msg)

    return bg_bins, fg_bins


def compute_background_foreground_variance(img: np.ndarray, threshold: float = 1.7) -> tuple[float, float]:
    """Compute the variance of the background and foreground pixels of an image.

    Args:
        img: Input image.
        threshold: Intensity separating background (below) from foreground (at or above).

    Returns:
        Variance of the background pixels and variance of the foreground pixels.
    """
    bg_variance = np.var(img[(img > BG_THRESHOLD_MIN) & (img < threshold)])
    fg_variance = np.var(img[img >= threshold])

    return bg_variance, fg_variance
