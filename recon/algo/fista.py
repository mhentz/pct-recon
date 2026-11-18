"""Run FISTA reconstructions of the proton CT image with a chosen regulariser."""

from __future__ import annotations

import os
from glob import glob
from math import floor, log10
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from blosc2 import save_array
from loguru import logger
from pct_tools.recon import gradient_descent, squared_residuals

from recon.algo.regularisation import (
    DirectionalTVRegulariser,
    Isotropy,
    NonNegativity,
    NoRegulariser,
    Regulariser,
    TVRegulariser,
)
from recon.algo.utils import RegulariserChoice, XrayReference, compute_zero_mask
from recon.config import ConfigError, ReconConfig, load_config

if TYPE_CHECKING:
    from collections.abc import Sequence

BLOSC_COMPRESSION_PARAMS = {"clevel": 5, "nthreads": 1}

# disable blosc's multi-threading since we handle the multi-threading ourselves
os.environ["BLOSC_NTHREADS"] = "1"
os.environ["BLOSC_NOLOCK"] = "1"


def format_alpha_string(alpha: str | float) -> str:
    """Format a regularisation weight as a compact string, such as ``2e-3``.

    Args:
        alpha: Regularisation weight, as a string or float.

    Returns:
        The weight with one or two significant decimals in the mantissa.

    Raises:
        TypeError: If alpha is neither a string nor a float.
        ValueError: If alpha is negative.
    """
    if type(alpha) is not str and type(alpha) is not float:
        msg = "alpha must be a string or float"
        raise TypeError(msg)

    if float(alpha) < 0:
        msg = "alpha must be a positive number"
        raise ValueError(msg)

    alpha = float(alpha)
    exponent = floor(log10(alpha))
    mantissa: float = alpha / 10**exponent

    if mantissa.is_integer():
        return f"{mantissa:.0f}e{exponent}"

    if len(str(mantissa)) == 3:  # noqa: PLR2004
        return f"{mantissa:.1f}e{exponent}"

    if len(str(mantissa)) == 4:  # noqa: PLR2004
        return f"{mantissa:.2f}e{exponent}"

    return str(alpha).replace("e-0", "e-")


def do_grad_update(
    x: np.ndarray[Any, np.dtype[np.float32]],
    step_size: float,
    matrix_filenames: Sequence[str],
    b_filenames: Sequence[str],
    n_threads: int = 1,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply one gradient descent update to the image estimate.

    Args:
        x: Current image estimate.
        step_size: Gradient descent step size.
        matrix_filenames: Paths to the system matrix chunks.
        b_filenames: Paths to the projection vector chunks.
        n_threads: Number of threads used for the update.

    Returns:
        The updated image, with the same shape as ``x``.
    """
    logger.debug("Compute gradient update")

    original_shape = x.shape
    x = gradient_descent(
        x.flatten(),
        step_size,
        matrix_filenames,
        b_filenames,
        verbosity_level=0,
        n_threads=n_threads,
    )
    return x.reshape(*original_shape)


def do_fista_step(
    x_k: np.ndarray,
    y_k: np.ndarray,
    t_k: float,
    step_size: float,
    regulariser: Regulariser,
    alpha: float,
    matrix_filenames: list[str],
    b_filenames: list[str],
    verbosity_level: int = 0,
    n_threads: int = 1,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.float32]],
    float,
    float,
    float,
    float,
]:
    """Run one FISTA iteration and return the updated iterates and cost.

    Args:
        x_k: Current iterate.
        y_k: Extrapolated point.
        t_k: Current momentum coefficient.
        step_size: Gradient descent step size.
        regulariser: Regulariser applied in the proximal step.
        alpha: Weight of the regularisation penalty in the cost.
        matrix_filenames: Paths to the system matrix chunks.
        b_filenames: Paths to the projection vector chunks.
        verbosity_level: Verbosity level passed to the residual computation.
        n_threads: Number of threads used by the matrix-vector products.

    Returns:
        A tuple of the new iterate, the new extrapolated point, the new momentum
        coefficient, the cost, the squared residuals and the penalty term.
    """
    logger.debug("Compute FISTA iteration")

    y_k = do_grad_update(y_k, step_size, matrix_filenames, b_filenames, n_threads=n_threads)

    x_kp1, info = regulariser.prox(y_k)

    if info is not None:
        logger.debug(f"Proximal step info: {int(info[0])} iter, {info[1]:.3e} tol")

    t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
    y_kp1 = x_kp1 + (t_k - 1) / t_kp1 * (x_kp1 - x_k)

    logger.debug("Compute cost(y_k)")
    regularisation_penalty_term = alpha * regulariser(x_kp1)

    logger.debug("Compute squared residuals")
    residuals_squared_xk = squared_residuals(
        x=x_kp1.flatten(),
        matrix_filenames=matrix_filenames,
        vector_filenames=b_filenames,
        verbosity_level=verbosity_level,
        n_threads=n_threads,
    )
    logger.debug("Squared residuals computed")
    cost = 0.5 * residuals_squared_xk + regularisation_penalty_term

    logger.debug(f"residuals squared (x_k) = {residuals_squared_xk:.6e}")
    logger.debug(f"regularisation_penalty_term (x_k) = {regularisation_penalty_term:.6e}")
    logger.debug(f"cost(x_k) = {cost:.3f} = {cost:.6e}")

    return x_kp1, y_kp1, t_kp1, cost, residuals_squared_xk, regularisation_penalty_term


def load_step_size(config: ReconConfig) -> float:
    """Look up the step size recorded for this setup in ``recon/lipschitz/step_sizes.yml``.

    The step size is the reciprocal of the Lipschitz constant measured for the setup,
    dose, number of projections and image width.
    """
    with (Path(__file__).parent / ".." / "lipschitz" / "step_sizes.yml").open() as f:
        step_sizes = yaml.safe_load(f)
    return step_sizes[config.setup][config.target_dose][config.n_projections][config.image_w]


def load_initial_image(config: ReconConfig) -> np.ndarray:
    """Return the initial image: zeros, or the FBP reconstruction of the same dose and width.

    Raises:
        ConfigError: If ``X_INIT=fbp`` and ``FBP_DIR`` is not set.
    """
    if config.x_init == "zero":
        return np.zeros(config.image_shape)
    if config.fbp_dir is None:
        msg = "FBP_DIR must be set when X_INIT=fbp"
        raise ConfigError(msg)
    target_dose = int(config.target_dose) if config.target_dose.is_integer() else config.target_dose
    x0 = np.load(config.fbp_dir / f"fbp_{target_dose}mGy_{config.image_w}pix.npy")
    if x0.ndim != 3:  # noqa: PLR2004
        x0 = np.stack([x0, x0, x0])
    return x0


def collect_chunks(matrix_dir: Path, vector_dir: Path, n_projections: int) -> tuple[list[str], list[str]]:
    """Return the matrix and b-vector files of every chunk, in projection order.

    A projection at angle ``a`` is stored as ``<matrix_dir>/<a>.mat`` or, when it
    was split, as ``<matrix_dir>/<a>_<chunk>.mat``; the matching b-vector is
    ``<vector_dir>/b_<same stem>.vec``.

    Args:
        matrix_dir: Directory holding the system matrix chunks.
        vector_dir: Directory holding the b-vector chunks.
        n_projections: Number of projections over a full rotation.

    Returns:
        The matrix filenames and the b-vector filenames, paired by position.
    """
    proj_list = np.arange(0, 360, 360 / n_projections).astype(float)
    chunks = [
        os.path.basename(filename).removesuffix(".mat")
        for proj in proj_list
        for filename in sorted(glob(f"{matrix_dir}/{proj}*.mat"))
    ]
    matrix_filenames = [f"{matrix_dir}/{chunk}.mat" for chunk in chunks]
    b_filenames = [f"{vector_dir}/b_{chunk}.vec" for chunk in chunks]
    return matrix_filenames, b_filenames


def load_xray_ref(
    ref_dir: Path,
    phantom: str,
    image_w: int,
    variant: XrayReference = XrayReference.REALISTIC_SLICE_FILTERED,
) -> np.ndarray:
    """Load the x-ray reference image for dTV and stack it to three slices.

    Args:
        ref_dir: Directory holding the reference images (``phantoms/xray_refs``).
        phantom: Phantom name.
        image_w: Image width in pixels.
        variant: Which reference to load: one of the realistic references or the deteriorated
            one (1-pixel blur, 2 % noise).

    Returns:
        The reference as a contiguous ``float32`` array of shape ``(3, image_w, image_w)``.
    """
    if variant == XrayReference.DETERIORATED:
        path = ref_dir / f"{phantom}_xray_ref_{image_w}pix_{variant}.npy"
    else:
        path = ref_dir / f"{phantom}_xray_ref_{variant}_{image_w}pix.npy"
    xray_ref = np.ascontiguousarray(np.load(path)).astype(np.float32)
    if xray_ref.ndim != 3:  # noqa: PLR2004
        xray_ref = np.stack([xray_ref, xray_ref, xray_ref])
    return xray_ref


def build_regulariser(
    config: ReconConfig,
    zero_mask: tuple[list[int], list[int], list[int]] | None,
) -> tuple[Regulariser, str]:
    """Build the regulariser named in the config.

    Args:
        config: The reconstruction settings.
        zero_mask: Voxel indices outside the phantom, set to zero after each proximal step,
            or ``None`` to leave them.

    Returns:
        The regulariser and the short description of it used in the output directory name.

    Raises:
        ConfigError: If ``ALPHA`` is missing or not positive for TV and dTV, or ``ETA`` or
            ``REF_DIR`` is missing for dTV.
    """
    if config.regulariser == RegulariserChoice.NONE:
        return NoRegulariser(non_negativity=NonNegativity.ENABLED, zero_mask=zero_mask), "no_reg"

    if config.alpha is None or config.alpha <= 0:
        msg = f"ALPHA must be a positive number when REGULARISER={config.regulariser}, got {config.alpha}"
        raise ConfigError(msg)
    alpha_string = format_alpha_string(config.alpha)

    if config.regulariser == RegulariserChoice.TV:
        regulariser = TVRegulariser(
            alpha=config.alpha,
            prox_max_loops=config.prox_max_loops,
            prox_tolerance=config.prox_tolerance,
            isotropy=Isotropy.ISOTROPIC,
            non_negativity=NonNegativity.ENABLED,
            zero_mask=zero_mask,
        )
        return regulariser, f"TV_iso_alpha_{alpha_string}"

    if config.eta is None or config.ref_dir is None:
        msg = "ETA and REF_DIR must be set when REGULARISER=dTV"
        raise ConfigError(msg)
    regulariser = DirectionalTVRegulariser(
        alpha=config.alpha,
        prox_max_loops=config.prox_max_loops,
        prox_tolerance=config.prox_tolerance,
        isotropy=Isotropy.ISOTROPIC,
        non_negativity=NonNegativity.ENABLED,
        zero_mask=zero_mask,
        xray_ref=load_xray_ref(config.ref_dir, config.phantom, config.image_w),
        eta=config.eta,
        edge_height_method="absolute",
    )
    return regulariser, f"dTV_iso_alpha_{alpha_string}_eta_{config.eta:.0e}".replace("e-0", "e-")


def output_dir_name(config: ReconConfig, prox_description: str) -> str:
    """Return the name of the run's output directory, which encodes its parameters.

    Args:
        config: The reconstruction settings.
        prox_description: The regulariser description from :func:`build_regulariser`.

    Returns:
        A name such as ``fista_exact_init_zero_880pix_5.0mGy_720proj_TV_iso_alpha_2e-3_iter200``.
    """
    name = (
        f"fista_exact_init_{config.x_init}_{config.image_w}pix_{config.target_dose}mGy"
        f"_{config.n_projections}proj_{prox_description}_iter{config.iter_end}"
    )
    return name.replace("e-0", "e-").replace("e+0", "e+")


def dump_iterates(
    output_dir: Path, suffix: str, yk_list: list[np.ndarray], metrics_list: list[list[float]]
) -> None:
    """Write the iterates and the metrics so far as ``yk_<suffix>.bl2`` and ``metrics_<suffix>.bl2``.

    Args:
        output_dir: Directory of the run.
        suffix: Iteration number, or ``final``.
        yk_list: The ``y_k`` iterates so far, including the initial image.
        metrics_list: Per-iteration ``[cost, squared residuals, penalty]``.
    """
    for name, values in (("yk", yk_list), ("metrics", metrics_list)):
        save_array(
            np.array(values).astype(np.float32),
            str(output_dir / f"{name}_{suffix}.bl2"),
            # blosc2 annotates **kwargs as dict, so the string mode is flagged
            mode="w",  # ty: ignore[invalid-argument-type]
            cparams=BLOSC_COMPRESSION_PARAMS,
        )


def log_settings(config: ReconConfig, step_size: float) -> None:
    """Log the settings of the run."""
    logger.info(f"Number of threads: {config.n_threads}")
    logger.info(f"Setup: {config.setup}")
    logger.info(f"Target dose [mGy]: {config.target_dose}")
    logger.info(f"Number of projections: {config.n_projections}")
    logger.info(f"Sigma used in data cuts: {config.n_sigma}")
    logger.info(f"Reading matrices from: {config.matrix_dir}")
    logger.info(f"Reading b-vectors from: {config.vector_dir}")
    logger.info(f"Image xz halfspace [mm]: {config.image_halfspace_xz}")
    logger.info(f"Image width [pixels]: {config.image_w}")
    logger.info(f"Phantom radius [mm]: {config.radius}")
    logger.info(f"Initial guess: {config.x_init}")
    logger.info(f"Iterations: {config.iter_start} to {config.iter_end}")
    logger.info(f"Regulariser: {config.regulariser}")
    logger.info(f"Regularisation parameter alpha: {config.alpha}")
    logger.info(f"dTV edge parameter eta: {config.eta}")
    logger.info(f"Proximal step tolerance: {config.prox_tolerance}")
    logger.info(f"Proximal step maximum iterations: {config.prox_max_loops}")
    logger.info(f"Step size: {step_size:.6e}")


def main(config: ReconConfig) -> None:
    """Run one reconstruction described by ``config`` and write its output.

    Raises:
        ConfigError: If a setting needed by the chosen regulariser or initial image is missing.
    """
    if config.matrix_dir is None or config.vector_dir is None or config.output_dir is None:
        msg = "MATRIX_DIR, VECTOR_DIR and OUTPUT_DIR must be set"
        raise ConfigError(msg)

    step_size = load_step_size(config)
    log_settings(config, step_size)

    logger.info("Initialise variables...")
    x0 = load_initial_image(config)
    x_k = x0.copy()
    y_k = x0.copy()
    t_k = 1

    yk_list = [y_k]
    metrics_list: list[list[float]] = []

    matrix_filenames, b_filenames = collect_chunks(config.matrix_dir, config.vector_dir, config.n_projections)
    logger.info(f"{len(matrix_filenames)=}")
    logger.info(f"Matrix filenames: '{matrix_filenames[0]}' ...")
    logger.info(f"Vector filenames: '{b_filenames[0]}' ...")

    zero_mask = compute_zero_mask(y_k.shape, int(config.radius / config.pixel_w), 1)
    regulariser, prox_description = build_regulariser(config, zero_mask)
    alpha = config.alpha if config.alpha is not None else 0.0

    output_dir = config.output_dir / output_dir_name(config, prox_description)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing output data to: {output_dir}")
    with (output_dir / "config.yml").open("w") as f:
        yaml.safe_dump({**config.as_dict(), "step_size": step_size}, f, sort_keys=False)

    logger.info("---> START RECONSTRUCTION <---")
    recon_start_time = time()

    cost = np.inf

    for i in range(config.iter_start, config.iter_end + 1):
        logger.info("-------------------------------------------------")
        logger.info(f"ITERATION {i}")
        iter_start_time = time()

        x_k, y_k, t_k, cost, residuals_squared_xk, penalty_term = do_fista_step(
            x_k,
            y_k,
            t_k,
            step_size,
            regulariser,
            alpha,
            matrix_filenames,
            b_filenames,
            n_threads=config.n_threads,
        )

        logger.info("Store metrics")
        yk_list.append(y_k.astype(np.float32))
        metrics_list.append([cost, residuals_squared_xk, penalty_term])

        logger.info(f"--> End of iteration {i}... took {time() - iter_start_time:.1f} s")

        total_recon_time = time() - recon_start_time
        logger.info(
            f"Total time elapsed: {total_recon_time / 60:.1f} minutes or "
            f"{total_recon_time // 3600:.0f} hours and "
            f"{total_recon_time % 3600 // 60:.0f} minutes",
        )

        if 0 < i < config.iter_end - 1 and i % config.dump_modulo == 0:
            logger.info("Dumping y_k and metrics...")
            dump_iterates(output_dir, str(i), yk_list, metrics_list)

    logger.info("-------------------------------------------------")
    logger.info("Dump final data...")
    dump_iterates(output_dir, "final", yk_list, metrics_list)

    logger.info("END OF RECONSTRUCTION")


if __name__ == "__main__":
    main(load_config())
