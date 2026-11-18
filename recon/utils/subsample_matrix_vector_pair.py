"""Derive a lower-dose dataset by subsampling the rows of each matrix and b-vector pair."""

from pathlib import Path

import numpy as np
import typer
from loguru import logger
from pct_tools.utils import subsample_matrix_vector_pair


def subsample(
    input_dir: Path,
    output_dir: Path,
    n_projections: int,
    pixel_width: float,
    ratio: float,
    compression_level: int = 9,
) -> None:
    """Subsample every matrix and b-vector pair of a dataset to a fraction of its protons.

    The matrices are read from ``<input_dir>/matrix-<pixel_width>mm`` and the b-vectors from
    ``<input_dir>/b``; the subsampled pairs are written to the same locations under
    ``output_dir``.

    Args:
        input_dir: Directory of the full-dose dataset.
        output_dir: Directory in which the subsampled dataset is written.
        n_projections: Number of projections over a full rotation.
        pixel_width: Width of a pixel in mm; selects the matrix directory.
        ratio: Fraction of the rows (protons) to keep, in [0, 1].
        compression_level: Compression level of the stored matrices.
    """
    logger.debug(f"{input_dir=}")
    logger.debug(f"{output_dir=}")

    (output_dir / f"matrix-{pixel_width}mm").mkdir(parents=True, exist_ok=True)
    (output_dir / "b").mkdir(parents=True, exist_ok=True)

    proj_list = np.arange(0, 360, 360 / n_projections).astype(float)

    matrix_dir = input_dir / f"matrix-{pixel_width}mm"
    chunks = [path.stem for proj in proj_list for path in matrix_dir.glob(f"{proj}*.mat")]

    for chunk in chunks:
        logger.debug(f"chunk {chunk}")
        subsample_matrix_vector_pair(
            str(input_dir),
            chunk,
            str(output_dir),
            ratio,
            pixel_width,
            compression_level=compression_level,
        )


if __name__ == "__main__":
    typer.run(subsample)
