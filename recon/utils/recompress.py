"""Recompress matrices."""

import os
from glob import glob

import numpy as np
import typer
from loguru import logger
from pct_tools.utils import recompress_matrix


def main(
    input_dir: str,
    output_dir: str,
    n_projections: int,
    start_index: int,
    stop_index: int,
    compression_level: int = 9,
) -> None:
    """Recompress matrices with LZ4HC."""
    logger.info("Starting recompression of matrices with LZ4HC")

    proj_angles = np.arange(start_index, stop_index, 360 / n_projections).tolist()

    chunks = [
        os.path.basename(filename).replace(".mat", "")
        for proj in proj_angles
        for filename in glob(f"{input_dir}/{proj}*.mat")
    ]

    matrix_input_filenames = [f"{input_dir}/{chunk}.mat" for chunk in chunks]
    matrix_output_filenames = [f"{output_dir}/{chunk}.mat" for chunk in chunks]

    for input_filename, output_filename in zip(matrix_input_filenames, matrix_output_filenames, strict=False):
        logger.info(f"Recompressing matrix {input_filename} with LZ4HC")
        recompress_matrix(
            filename=input_filename,
            output=output_filename,
            compression_level=compression_level,
        )

    logger.info("Done")


if __name__ == "__main__":
    typer.run(main)
