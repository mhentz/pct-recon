"""Fit error-function models to edge spread functions and derive k10 values."""

import numpy as np
from loguru import logger
from scipy.optimize import curve_fit
from scipy.special import erf


def cdf(x: np.ndarray, x0: float, sigma: float, k: float, offset: float) -> np.ndarray:
    """Evaluate a scaled and offset error-function CDF.

    Args:
        x: Positions at which to evaluate the CDF.
        x0: Centre of the edge.
        sigma: Width of the edge.
        k: Height of the step.
        offset: Baseline value.

    Returns:
        CDF values at ``x``.
    """
    arg = (x - x0) / (np.sqrt(2) * sigma)
    return offset + k * 0.5 * (1 + erf(arg))


def fit_esf(
    esf: np.ndarray,
    x: np.ndarray,
    sigma_init: float = 0.1,
    sigma: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """Fit an error-function CDF to an edge spread function and derive k10.

    Args:
        esf: Edge spread function values.
        x: Positions of the ESF samples.
        sigma_init: Initial guess for the edge width, sign-adjusted for up or down steps.
        sigma: Per-point uncertainties of the ESF. The fit is unweighted if not given.

    Returns:
        k10 value, its uncertainty, and the fitted parameters.
    """
    # figure out if it's up-step or down-step to set correct sign
    first_els = np.mean(esf[:5])
    last_els = np.mean(esf[-5:])
    factor = (last_els - first_els) / np.abs(last_els - first_els)

    p0 = [6, factor * sigma_init, 1, 1]
    bounds = ([0, -5, 0, 0], [14, 0, 10, 10])

    if sigma:
        popt, pcov = curve_fit(cdf, x, esf, p0=p0, bounds=bounds, sigma=sigma, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(cdf, x, esf, p0=p0, bounds=bounds)

    stddev = np.sqrt(np.diag(pcov))

    sigma_fit = np.abs(popt[1])
    std_sigma_fit = stddev[1]

    k10 = np.sqrt(2 * np.log(10)) / (2 * np.pi * sigma_fit)
    sigma_k10 = (k10 / sigma_fit) * std_sigma_fit

    logger.debug(f"initial guess: [x0, sigma, k, offset]\n {p0}")
    logger.debug(f"popt:\n {popt}")
    logger.debug(f"pcov:\n {pcov}")
    logger.debug(f"popt std:\n {stddev}")
    logger.debug(f"k10 = {k10:.3f} ± {1.96 * sigma_k10:.3f} lp/mm")

    return k10, sigma_k10, popt
