"""MTF method based on Richard et al. and Takenaga et al."""

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv
from skimage.exposure import rescale_intensity
from skimage.feature import canny
from skimage.measure import CircleModel, ransac


def adjust_for_background(roi: np.ndarray) -> np.ndarray:
    """Fit a plane to the background and remove it from the ROI.

    Args:
        roi: 2D array containing the image of the disk.

    Returns:
        The input ROI with the fitted plane subtracted.

    """
    h, w = roi.shape

    rows, cols = np.mgrid[0:h, 0:w]
    A = np.c_[cols.ravel(), rows.ravel(), np.ones(h * w)]
    coef, *_ = np.linalg.lstsq(A, roi.ravel(), rcond=None)  # plane: a*x + b*y + c
    plane = (A @ coef).reshape(h, w)

    return roi - plane


def circle_score(edges: np.ndarray, centre: tuple[float, float], disk_radius: float) -> float:
    """Compute a score for the fitted circle.

    The score is based on the inlier ratio, coverage of the circle, and median residuals.
    A good fit has a high inlier ratio, high coverage, and low median residuals. We weight
    down the coverage a bit to prefer fits with more inliers and low median residuals.

    Args:
        edges: 2D array containing the edges.
        centre: Tuple of (xc, yc) coordinates of the circle centre.
        disk_radius: Radius of the fitted circle.

    Returns:
        The score for the circle defined by the parameters provided.

    """
    xc, yc = centre
    rows, cols = np.nonzero(edges)
    d = np.hypot(cols - xc, rows - yc)
    residuals = np.abs(d - disk_radius)

    residual_threshold = 3
    inliers = residuals <= residual_threshold
    n_inliers = inliers.sum()  # sum over booleans
    n_edge_pixels = len(edges[edges])

    inlier_ratio = n_inliers / n_edge_pixels if n_edge_pixels > 0 else 0

    median_residuals = np.median(residuals[inliers]) if n_inliers > 0 else np.inf
    median_residuals_norm = median_residuals / residual_threshold

    angles = np.arctan2(rows - yc, cols - xc)
    angles_sorted = np.sort((angles + 2 * np.pi) % (2 * np.pi))  # shift [-pi, pi] to [0, 2pi]

    if angles_sorted.size > 0:
        diffs = np.diff(np.r_[angles_sorted, angles_sorted[0] + 2 * np.pi])
        coverage = 1.0 - diffs.max() / (2 * np.pi)
    else:
        coverage = 0.0

    return inlier_ratio + 0.7 * coverage - median_residuals_norm


def compute_centre_coordinates(
    roi: np.ndarray,
    canny_sigma: float,
    residual_threshold: float = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Find centre coordinates of an image of a disk.

    This function finds edges in the image using a Canny edge detector and fits the edges
    to a circle using RANSAC to determine the centre coordinates and radius.

    Args:
        roi: 2D array containing the image of the disk.
        canny_sigma: Standard deviation for the Gaussian filter used in Canny edge detection.
        residual_threshold: Maximum distance from the circle edge for a data point
            to be classified as an inlier.
        seed: Random seed for RANSAC. If None, a random seed will be used.

    Returns:
        edges, (xc, yc), disk_radius

    """
    if seed is None:
        logger.warning("No seed set for RANSAC. Random seed will be used.")

    edges = canny(roi, sigma=canny_sigma)
    edge_points = np.column_stack(np.nonzero(edges))

    model, _ = ransac(
        edge_points[:, ::-1],  # (row, col) → (x, y)
        CircleModel,
        min_samples=3,
        residual_threshold=residual_threshold,
        max_trials=5_000,
        rng=seed,
    )
    xc, yc, disk_radius = model.params
    return edges, (xc, yc), disk_radius


def find_best_centre_coordinates(
    roi: np.ndarray,
    canny_sigmas: list[float] | None = None,
    ransac_seed: int | None = None,
    ransac_residual_threshold: float = 2,
    relative_score_threshold: float = 0.05,
) -> tuple[tuple[float, float], float]:
    """Compute the centre coordinates and radius of the disk in the ROI.

    This function tries multiple Canny edge detection sigmas and selects the smallest
    sigma that that deviates from the next result by less than `relative_score_threshold`
    (default: 5%) in terms of the circle score.

    Args:
        roi: The input image ROI.
        canny_sigmas: List of standard deviations for the Gaussian filter used in Canny edge detection.
            Defaults to [1.0, 1.5, 2.0, 3.0, 4.0, 5.0] if not provided.
        ransac_seed: The random seed for RANSAC used in ROI disk centre estimation. If no seed is
            provided, a random seed will be used.
        ransac_residual_threshold: Maximum distance measured in pixel units from the circle edge
            for a data point to be classified as an inlier.
        relative_score_threshold: The relative change in score required to continue iterating
            over different Canny sigmas.

    Returns:
        A tuple of ((x_centre, y_centre), disk_radius).
    """
    if not canny_sigmas:
        canny_sigmas = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    prev_score = None
    prev_centre: tuple[float, float] = (0, 0)
    prev_disk_radius: float = 0

    for canny_sigma in canny_sigmas:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")

                edges, centre, disk_radius = compute_centre_coordinates(
                    roi,
                    canny_sigma=canny_sigma,
                    residual_threshold=ransac_residual_threshold,
                    seed=ransac_seed,
                )
        except ValueError as error:
            logger.debug(f"Canny sigma={canny_sigma:.1f} failed, returning previous value")
            if prev_score is not None:
                return prev_centre, prev_disk_radius
            msg = "No valid circle fit found for this ROI"
            raise RuntimeError(msg) from error
        except UserWarning:
            continue

        score = circle_score(edges, centre, disk_radius)
        prev_score_str = f"{prev_score:.3f}" if prev_score is not None else "None"

        if prev_score is None:
            logger.debug(
                f"Canny sigma={canny_sigma:.1f}, prev_score={prev_score_str}, score={score:.3f}, "
                f"centre=({centre[0]:.2f}, {centre[1]:.2f}), radius={disk_radius:.2f}",
            )
        else:
            logger.debug(
                f"Canny sigma={canny_sigma:.1f}, prev_score={prev_score_str}, score={score:.3f}, "
                f"diff={abs(prev_score - score):.5f}, "
                f"0.05 * prev_score={relative_score_threshold * prev_score:.3f}, "
                f"centre=({centre[0]:.2f}, {centre[1]:.2f}), radius={disk_radius:.2f}",
            )

        if prev_score is None or abs(prev_score - score) > relative_score_threshold * prev_score:
            prev_score = score
            prev_centre, prev_disk_radius = centre, disk_radius
            continue

        return prev_centre, prev_disk_radius

    return centre, disk_radius


def compute_raw_esf(
    roi: np.ndarray,
    edge_band_px: int = 8,
    ransac_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect the signed distance of each ROI pixel from the disk edge.

    Args:
        roi: Image ROI containing a disk.
        edge_band_px: Half-width in pixels of the band around the edge that is kept.
        ransac_seed: Random seed for the RANSAC disk fit.

    Returns:
        Signed distances of the pixels inside the edge band, and their intensities.
    """
    centre, disk_radius = find_best_centre_coordinates(roi, ransac_seed=ransac_seed)

    col_centre, row_centre = centre[0], centre[1]
    h, w = roi.shape

    row_indices, col_indices = np.mgrid[0:h, 0:w]
    distance_from_edge = (
        np.sqrt((row_indices - row_centre) ** 2 + (col_indices - col_centre) ** 2) - disk_radius
    )
    mask = np.abs(distance_from_edge) <= edge_band_px
    distance_from_edge = distance_from_edge[mask].ravel()
    rsp_value = roi[mask].ravel()

    return distance_from_edge, rsp_value


def compute_binned_esf(
    roi: np.ndarray,
    oversample: int = 4,
    minimum_bin_count: int = 10,
    edge_band_px: int = 8,
    ransac_seed: int | None = None,
    binning_mode: Literal["mean", "median"] = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the binned edge spread function (ESF) of the given ROI.

    Args:
        roi: The input image ROI.
        oversample: The oversampling factor.
        minimum_bin_count: The minimum number of points required in a bin to compute the ESF value.
        edge_band_px: The width of the edge band in pixels. Only pixels within this distance from the
            disk edge will be considered for the ESF computation.
        ransac_seed: The random seed for RANSAC used in ROI disk centre estimation. If no seed is
            provided, a random seed will be used.
        binning_mode: The method to use for binning the ESF values. Either "mean" or "median" and
            defaults to "mean".

    Returns:
        Tuple of numpy arrays (esf_vals, esf_var_vals, bin_positions).

    """
    distance_from_edge, rsp_value = compute_raw_esf(roi, edge_band_px=edge_band_px, ransac_seed=ransac_seed)

    bin_w = 1.0 / max(1, int(oversample))
    bins = np.arange(-edge_band_px, edge_band_px + bin_w, bin_w)
    idx = np.digitize(distance_from_edge, bins) - 1

    esf = np.full(len(bins), np.nan)
    esf_var = np.full(len(bins), np.nan)

    for k in range(len(bins)):
        rsp_values_in_bin = rsp_value[idx == k]

        if rsp_values_in_bin.size != 0:
            if binning_mode == "median":
                esf[k] = np.median(rsp_values_in_bin)
            elif binning_mode == "mean":
                esf[k] = np.mean(rsp_values_in_bin)
            else:
                msg = f"Invalid binning_mode: {binning_mode}. Use 'mean' or 'median'."
                raise ValueError(msg)

            esf_var[k] = np.var(rsp_values_in_bin)

        if rsp_values_in_bin.size < minimum_bin_count:
            logger.debug(f"bin {k} (d={bins[k]:.2f}): only {rsp_values_in_bin.size} points, skipping")

    mask_exclude_nans = ~np.isnan(esf)
    bin_positions = bins[mask_exclude_nans]
    esf_vals = esf[mask_exclude_nans]
    esf_var_vals = esf_var[mask_exclude_nans]

    order = np.argsort(bin_positions)
    bin_positions = bin_positions[order]
    esf_vals = esf_vals[order]
    esf_var_vals = esf_var_vals[order]

    return esf_vals, esf_var_vals, bin_positions


def erf_esf(x: np.ndarray, offset: float, k: float, x0: float, sigma: float) -> np.ndarray:
    """Error function model for the edge spread function (ESF).

    Args:
        x: The input distances.
        offset: The vertical offset of the ESF.
        k: The amplitude of the ESF step.
        x0: The centre point of the ESF. Corresponds to the radius of the disk in the ROI.
        sigma: The Gaussian standard deviation of the approximation to the ESF. This controls
            the width of the edge transition and hence the steepness/sharpness of the edge.

    Returns:
        Array of ESF values at the given distances `x`.

    """
    arg = (x - x0) / (np.sqrt(2) * sigma)
    return offset + 0.5 * k * (1 + erf(arg))


def erf_lsf(x: np.ndarray, k: float, x0: float, sigma: float) -> np.ndarray:
    """Error function model for the line-spread function (LSF).

    This is the analytical derivative with respect to the distance `x` of the error function
    model for the ESF. The expected result is a Gaussian curve.

    Args:
        x: The input distances.
        k: The amplitude of the ESF step.
        x0: The centre point of the ESF. Corresponds to the radius of the disk in the ROI.
        sigma: The Gaussian standard deviation of the approximation to the ESF. This controls
            the width of the edge transition and hence the steepness/sharpness of the edge.

    Returns:
        Array of LSF values at the given distances `x`.

    """
    arg = (x - x0) / (np.sqrt(2) * sigma)
    prefactor = k / (np.sqrt(2 * np.pi) * sigma)
    return prefactor * np.exp(-(arg**2))


def fit_esf_erf(
    distances: np.ndarray,
    esf: np.ndarray,
    esf_var: np.ndarray,
    *,
    bounds: tuple[list[float], list[float]] | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the error function model to the given ESF data.

    Args:
        distances: The input distances.
        esf: The input ESF values.
        esf_var: The variance of the ESF values.
        bounds: Optional minimum and maximum bounds for the fit parameters. The order of the
            parameters is [offset, k, x0, sigma]. The element in the tuple is the minimum bounds,
            the second element is the maximum bounds. Defaults to ([-2, -5, -10, -5], [2, 5, 10, 5])
            if not provided.
        input_dir: Set this path if you want to load intermediary results from csv files.
        output_dir: Set this path if you want to store intermediary results to csv files.

    Returns:
        Tuple of numpy arrays (popt, pcov):
            popt: The optimal fit parameters [offset, k, x0, sigma].
            pcov: The covariance matrix of the fit parameters.

    """
    if input_dir and (input_dir / "popt.csv").exists() and (input_dir / "pcov.csv").exists():
        logger.debug(f"Loading fit parameters from {input_dir}")
        popt = pd.read_csv(input_dir / "popt.csv")["popt"].to_numpy()
        pcov = pd.read_csv(input_dir / "pcov.csv")["pcov"].to_numpy().reshape(4, 4)
        return popt, pcov

    def _x_at_level(distances: np.ndarray, y: np.ndarray, level: float, sign: float) -> float:
        """Estimate the x position at which the ESF reaches the given y level.

        Used for the initial guesses of the fit parameters: the ESF is made monotonic
        and x(level) is then found by inverse interpolation.

        Args:
            distances: The input distances, must be ascending and finite.
            y: The input y values (ESF values), must be finite.
            level: The target y level to find the corresponding x value for.
            sign: The sign of the edge step, used to determine whether to look for
                an increasing or decreasing edge. A positive sign indicates an increasing edge,
                a negative sign indicates a decreasing edge.

        Returns:
            The estimated x position for the given y level. If the level is outside the
            observed range, the nearest x value is returned.

        """
        if sign >= 0:
            y_mono = np.maximum.accumulate(y)
            keep = np.r_[True, np.diff(y_mono) > 0]  # compress plateaus
            yk, dk = y_mono[keep], distances[keep]
        else:
            y_mono = np.minimum.accumulate(y)
            keep = np.r_[True, np.diff(y_mono) < 0]
            yk, dk = y_mono[keep][::-1], distances[keep][::-1]  # reverse → increasing for interp

        if yk.size < 2:  # noqa: PLR2004
            # monotonic ESF is flat: take the point nearest to the level
            i = int(np.argmin(np.abs(y - level)))
            return float(distances[i])

        lvl = float(np.clip(level, yk[0], yk[-1]))  # clamp to observed range
        return float(np.interp(lvl, yk, dk))

    def _initial_guesses(distances: np.ndarray, esf: np.ndarray) -> tuple[float, float, float, float]:
        """Estimate initial guesses for the fit parameters.

        Args:
            distances: The input distances.
            esf: The input ESF values.

        Returns:
            Tuple of initial guesses (offset, k, x0, sigma).

        """
        y = esf
        n = y.size
        w = max(3, n // 10)  # ~10% tails

        # plateaus from the medians of the two tails
        L_left = float(np.median(y[:w]))
        L_right = float(np.median(y[-w:]))

        offset = L_left
        k = L_right - L_left

        # fallback if the step is degenerate (edge not fully in frame, etc.)
        if not np.isfinite(k) or np.isclose(k, 0.0, atol=1e-12):
            lo, hi = np.percentile(y, [5, 95])
            offset = float(lo)
            k = float(hi - lo)

        lvl10 = offset + 0.10 * k
        lvl50 = offset + 0.50 * k
        lvl90 = offset + 0.90 * k

        sign = 1.0 if k >= 0 else -1.0

        x10 = _x_at_level(distances, esf, lvl10, sign)
        x50 = _x_at_level(distances, esf, lvl50, sign)
        x90 = _x_at_level(distances, esf, lvl90, sign)

        # 10-90% width for an ERF: w10_90 = 2*sqrt(2)*erfinv(0.8) * sigma
        w10_90 = max(1e-12, abs(x90 - x10))
        sigma = w10_90 / (2 * np.sqrt(2) * erfinv(0.8))

        x0 = x50  # for ERF, x50 ≈ x0
        return float(offset), float(k), float(x0), float(sigma)

    offset_init, k_init, x0_init, sigma_init = _initial_guesses(distances, esf)
    p0 = np.array([offset_init, k_init, x0_init, sigma_init], float)

    bounds = bounds or ([-2, -5, -10, -5], [2, 5, 10, 5])
    popt, pcov = curve_fit(
        erf_esf,
        distances,
        esf,
        p0=p0,
        sigma=np.sqrt(esf_var),
        absolute_sigma=True,
        bounds=bounds,
    )

    if output_dir and output_dir.exists():
        logger.debug(f"Storing fit parameters to {output_dir}")

        pd.DataFrame({"popt": popt}).to_csv(output_dir / "popt.csv", index=False)
        pd.DataFrame({"pcov": pcov.flatten()}).to_csv(output_dir / "pcov.csv", index=False)

    return popt, pcov


def calculate_k10_from_fit_params(
    sigma_fit: float,
    sigma_fit_var: float,
    pixel_size: float,
    *,
    output_dir: Path | None = None,
    write_k10_to_file: bool = False,
) -> tuple[float, float]:
    """Calculate the k10 value from the fit parameters.

    The point at which the MTF drops to 10% is found analytically by differentiating the
    error function model for the ESF and solving for the frequency at which the MTF is
    0.1. The resulting formula is:
        k10 = sqrt(2 * ln(10)) / (2 * pi * sigma * pixel_size)

    The variance of k10 is calculated using standard error propagation. Since
    dk10/dsigma = -k10 / sigma, this gives:
        sigma_k10^2 = k10^2 * sigma_fit_var / sigma_fit^2

    Args:
        sigma_fit: The fitted sigma parameter from the ESF fit.
        sigma_fit_var: The variance of the fitted sigma parameter from the ESF fit.
        pixel_size: The pixel size in mm.
        output_dir: Set this path if you want to store intermediary results to csv files.
        write_k10_to_file: Whether to write the k10 value to a text file in the output directory.

    Returns:
        Tuple of (k10, sigma_k10).

    """
    k10 = np.sqrt(2 * np.log(10)) / (2 * np.pi * sigma_fit * pixel_size)
    sigma_k10 = np.sqrt(k10**2 * sigma_fit_var / sigma_fit**2)

    logger.debug(f"k10 = {k10:.3f} ± {sigma_k10:.3f}")

    if write_k10_to_file and output_dir is not None and output_dir.exists():
        (output_dir / "k10.txt").write_text(f"k10 = {k10:.3f} ± {sigma_k10:.3f} lp/mm\n")

    if output_dir and output_dir.exists():
        logger.debug(f"Storing k10 and sigma to {output_dir}")
        pd.DataFrame({"k10": [k10], "sigma_k10": [sigma_k10]}).to_csv(output_dir / "k10.csv", index=False)

    return k10, sigma_k10


def compute_mtf_from_lsf(
    distances: np.ndarray,
    popt: np.ndarray,
    ft_padding_factor: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the modulation transfer function from the given parameters.

    The Fourier transform is applied to the LSF to yield the OTF. The absolute value of the OTF
    is taken and normalised to get the MTF.

    Args:
        distances: The input distances.
        popt: The optimal fit parameters from the ESF fit. The order of the parameters is
            [offset, k, x0, sigma].
        ft_padding_factor: The factor by which to pad the input array for the FFT. This
            increases the frequency resolution of the output MTF.

    Returns:
        Tuple of (mtf, cycles_per_pixel).

    """

    def _uniform_grid(distances: np.ndarray, factor: float = 1.0) -> tuple[np.ndarray, float]:
        """Resample the distances to a uniform grid.

        The FFT needs uniform samples. The spacing is the median spacing of the input
        distances divided by the given factor, but at least 1e-3.

        Args:
            distances: The input distances.
            factor: The oversampling factor for the uniform grid.

        Returns:
            Tuple of (resampled_distances, sample_spacing).

        """
        distances_median = float(np.median(np.diff(distances)))
        sample_spacing = max(distances_median / factor, 1e-3)  # avoid zero/too small spacing
        resampled_distances = np.arange(distances[0], distances[-1] + sample_spacing / 2, sample_spacing)
        return resampled_distances, sample_spacing

    rebinned_distances, sample_spacing = _uniform_grid(distances, 10)
    lsf_array = erf_lsf(rebinned_distances, *popt[1:])
    n_samples = len(lsf_array)

    n_fft = int(2 ** np.ceil(np.log2(ft_padding_factor * n_samples)))  # power-of-two for speed

    otf = np.fft.rfft(lsf_array, n=n_fft)
    mtf = np.abs(otf / np.sum(lsf_array))
    cycles_per_pixel = np.fft.rfftfreq(n_fft, d=sample_spacing)

    return mtf, cycles_per_pixel


def construct_esf(
    roi: np.ndarray | None = None,
    oversample: int = 4,
    minimum_bin_count: int = 10,
    edge_band_px: int = 8,
    ransac_seed: int | None = None,
    input_file: Path | None = None,
    output_file: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ESF from the given ROI.

    Args:
        roi: The input image ROI.
        oversample: The oversampling factor used when constructing the ESF.
        minimum_bin_count: The minimum number of points required in a bin to compute the ESF.
        edge_band_px: The width of the edge band in pixels for the disk centre estimation.
        ransac_seed: The random seed for RANSAC.
        input_file: Set this path if you want to load intermediary results from csv files.
        output_file: Set this path if you want to store intermediary results to csv files.

    Returns:
        Tuple of numpy arrays (esf_binned, esf_var_binned, bin_positions).
    """
    if input_file and input_file.exists():
        logger.debug(f"Loading binned arrays from csv files in {input_file}")

        binned_arrays_df = pd.read_csv(input_file)
        esf_binned = binned_arrays_df["esf_binned"].to_numpy()
        esf_var_binned = binned_arrays_df["esf_var_binned"].to_numpy()
        bin_positions = binned_arrays_df["bin_positions"].to_numpy()
    else:
        if roi is None:
            msg = "ROI must be provided if input_dir is not set."
            raise ValueError(msg)

        roi = adjust_for_background(roi)
        roi = rescale_intensity(roi, in_range="image", out_range=(0, 1)).astype(np.float32)
        logger.debug(f"median roi: {np.median(roi):.6f}")

        esf_binned, esf_var_binned, bin_positions = compute_binned_esf(
            roi,
            oversample=oversample,
            minimum_bin_count=minimum_bin_count,
            edge_band_px=edge_band_px,
            ransac_seed=ransac_seed,
        )

        if output_file and output_file.parent.exists():
            logger.debug(f"Storing binned arrays to {output_file}")
            pd.DataFrame(
                {"esf_binned": esf_binned, "esf_var_binned": esf_var_binned, "bin_positions": bin_positions},
            ).to_csv(output_file, index=False)

    return esf_binned, esf_var_binned, bin_positions


def fit_esf_and_compute_k10(
    roi: np.ndarray,
    pixel_size: float,
    oversample: int = 4,
    minimum_bin_count: int = 10,
    edge_band_px: int = 8,
    ransac_seed: int | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    write_k10_to_file: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the k10 of a given ROI.

    Args:
        roi: The input image ROI.
        pixel_size: The pixel size in mm.
        oversample: The oversampling factor used when constructing the ESF.
        minimum_bin_count: The minimum number of points required in a bin to compute the ESF.
        edge_band_px: The width of the edge band in pixels for the disk centre estimation.
        ransac_seed: The random seed for RANSAC used in ROI disk centre estimation.
        input_dir: Set this path if you want to load intermediary results from csv files.
        output_dir: Set this path if you want to store intermediary results to csv files.
        write_k10_to_file: Whether to write the k10 value to a text file in the output directory.

    Returns:
        Tuple of (popt, bin_positions, k10, sigma_k10).
    """
    esf_binned, esf_var_binned, bin_positions = construct_esf(
        roi,
        oversample=oversample,
        minimum_bin_count=minimum_bin_count,
        edge_band_px=edge_band_px,
        ransac_seed=ransac_seed,
        input_file=input_dir / "binned_esf.csv" if input_dir else None,
        output_file=output_dir / "binned_esf.csv" if output_dir else None,
    )
    popt, pcov = fit_esf_erf(
        bin_positions,
        esf_binned,
        esf_var_binned,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    k10, sigma_k10 = calculate_k10_from_fit_params(
        popt[-1],
        pcov[-1, -1],
        pixel_size,
        output_dir=output_dir,
        write_k10_to_file=write_k10_to_file,
    )

    logger.debug(f"k10 = {k10:.3f} ± {sigma_k10:.3f}")
    return popt, bin_positions, k10, sigma_k10


def compute_mtf(
    roi: np.ndarray,
    pixel_size: float,
    oversample: int = 4,
    minimum_bin_count: int = 10,
    edge_band_px: int = 8,
    ransac_seed: int | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    write_k10_to_file: bool = False,
    log_fit_params: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the MTF and k10 from the given ROI.

    Args:
        roi: The input image ROI.
        pixel_size: The pixel size in mm.
        oversample: The oversampling factor used when constructing the ESF.
        minimum_bin_count: The minimum number of points required in a bin to compute the ESF.
        edge_band_px: The width of the edge band in pixels for the disk centre estimation.
        ransac_seed: The random seed for RANSAC used in ROI disk centre estimation.
        input_dir: Set this path if you want to load intermediary results from csv files.
        output_dir: Set this path if you want to store intermediary results to csv files.
        write_k10_to_file: Whether to write the k10 value to a text file in the output directory.
        log_fit_params: Whether to log the fitted parameters.

    Returns:
        Tuple of (mtf, cycles_per_mm, k10, sigma_k10).

    """
    popt, bin_positions, k10, sigma_k10 = fit_esf_and_compute_k10(
        roi,
        pixel_size,
        oversample=oversample,
        minimum_bin_count=minimum_bin_count,
        edge_band_px=edge_band_px,
        ransac_seed=ransac_seed,
        input_dir=input_dir,
        output_dir=output_dir,
        write_k10_to_file=write_k10_to_file,
    )

    if log_fit_params:
        logger.info(
            f"Fitted parameters: offset={popt[0]:.4f}, k={popt[1]:.4f}, x0={popt[2]:.4f}, sigma={popt[3]:.4f}"
        )

    mtf, cycles_per_pixel = compute_mtf_from_lsf(bin_positions, popt)
    cycles_per_mm = cycles_per_pixel / pixel_size

    return mtf, cycles_per_mm, k10, sigma_k10
