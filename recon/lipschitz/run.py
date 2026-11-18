"""Estimate the Lipschitz constant of a system matrix by power iteration."""

import os
from glob import glob
from typing import Annotated

import numpy as np
import typer
from loguru import logger
from pct_tools.lipschitz import compute_ATAx

from recon.lipschitz.convert import format_step_size

os.environ["BLOSC_NTHREADS"] = "1"
os.environ["BLOSC_NOLOCK"] = "1"


def compute_lipschitz(
    matrix_dir: str,
    n_projections: int,
    image_w: int,
    n_slices: int,
    n_iterations: Annotated[int, typer.Option()] = 20,
    n_threads: Annotated[int, typer.Option()] = 8,
) -> None:
    """Compute the Lipschitz constant of a system matrix.

    Args:
        n_projections: Number of projections.
        image_w: Width of the image in pixels.
        n_slices: Number of slices.
        n_iterations: Number of iterations for convergence.
        matrix_dir: Directory containing the system matrices.
        n_threads: Number of threads used when computing ATAx.
    """
    logger.info(f"{n_projections=}")
    logger.info(f"{image_w=}")
    logger.info(f"{n_slices=}")
    logger.info(f"{n_iterations=}")
    logger.info(f"{matrix_dir=}")

    rng = np.random.default_rng(0)
    v_0 = rng.random(image_w**2 * n_slices).astype(np.float32)
    v_k = (v_0 / np.linalg.norm(v_0)).astype(np.float32)

    proj_list = np.arange(0, 360, 360 / n_projections).astype(float)

    matrix_filenames = sorted(
        {filename for proj in proj_list for filename in glob(f"{matrix_dir}/{proj:.1f}*.mat")},
    )

    logger.info(
        f"Projection angles: {', '.join(proj_list[:5].astype(str))}, ..., "
        f"{', '.join(proj_list[-5:].astype(str))}",
    )
    logger.info(f"Matrix filename format: '{matrix_filenames[0]}'")

    tol = 1e-6
    for i in range(n_iterations):
        logger.info(f"Iteration {i + 1}/{n_iterations}")
        v_km1 = v_k.copy()

        logger.info("Compute ATAx")
        z = compute_ATAx(v_k, matrix_filenames, n_threads=n_threads)

        logger.info("Normalise resulting vector")
        v_k = (z / np.linalg.norm(z)).astype(np.float32)

        epsilon = np.linalg.norm(v_k - v_km1)
        logger.info(f"Discrepancy: {epsilon=:.6e}")

        if epsilon < tol:
            logger.info(f"Converged in {i} iterations.")
            break

    logger.info("Compute eigenvalue corresponding to final vector")

    ATAvk = compute_ATAx(x=v_k, filenames=matrix_filenames, n_threads=n_threads)

    mu = float(np.linalg.norm(ATAvk))
    mu_2 = v_k.dot(ATAvk) / v_k.dot(v_k)

    logger.info(f"Lipschitz = {mu:.6f}, {mu_2:.6f}")

    print("Entries to paste under the setup and dose in lipschitz_constants.yml and step_sizes.yml:")
    print(f"    {n_projections}:\n      {image_w}: {mu:.6f}")
    print(f"    {n_projections}:\n      {image_w}: {format_step_size(mu)}")


if __name__ == "__main__":
    typer.run(compute_lipschitz)
