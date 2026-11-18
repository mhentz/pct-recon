#!/usr/bin/env python3
"""Construct the b-vectors from the processed proton dataframes."""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from pct_tools.io import store_vector

from recon.preprocess.io import read_pickled_df

pd.options.display.width = 0


def construct_bvec(
    n_projections: int,
    n_input_chunks: int,
    n_output_chunks: int,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Write the b-vectors of each projection angle to the output directory.

    Args:
        n_projections: Number of projections over a full rotation.
        n_input_chunks: Number of chunks each proton dataframe is split into.
        n_output_chunks: Number of chunks each b-vector is split into.
        input_dir: Directory containing the processed proton dataframes.
        output_dir: Directory in which the b-vectors are written.
    """
    logger.debug(f"{n_projections=}")
    logger.debug(f"{n_input_chunks=}")
    logger.debug(f"{n_output_chunks=}")
    logger.debug(f"{input_dir=}")
    logger.debug(f"{output_dir=}")

    proj_angles = np.arange(0, 360, 360 / n_projections)

    logger.info("Construct b-vectors from processed dataframes")
    vector_output_dir = Path(output_dir)
    vector_output_dir.mkdir(parents=True, exist_ok=True)

    for proj_angle in proj_angles:
        if proj_angle % 200 == 0:
            logger.info(f"projection angle {proj_angle}")

        protons = read_pickled_df(input_dir, proj_angle, n_input_chunks)

        proton_index_list = list(range(len(protons)))
        chunked_indexes_arr = np.array_split(np.arange(len(proton_index_list)), n_output_chunks)

        for chunk_i, indices in enumerate(chunked_indexes_arr):
            protons_chunk: pd.DataFrame = protons.iloc[indices]

            if n_output_chunks == 1:
                vector_output_path = vector_output_dir / f"b_{proj_angle}.vec"
            else:
                vector_output_path = vector_output_dir / f"b_{proj_angle}-{chunk_i}.vec"

            store_vector(protons_chunk["wepl"].to_numpy(), str(vector_output_path))

    logger.info("Constructed b vectors")


if __name__ == "__main__":
    typer.run(construct_bvec)
