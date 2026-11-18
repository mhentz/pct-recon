"""Build the system matrices and b-vectors for each projection angle."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger
from pct_tools.io import ShuffleMethod, store_vector
from pct_tools.matrix import compute_and_store_sparse_matrix

from recon.geometry import (
    get_centre_of_rotation,
    get_detector_distance,
    get_image_halfspace_xz,
    get_image_halfspace_yz,
    get_image_w,
)
from recon.matrix.paths import compute_paths
from recon.preprocess.io import read_pickled_df


def process_matrix(
    setup: str,
    start_index: int,
    stop_index: int,
    n_projections: int,
    pixel_width: float,
    input_dir: Path,
    output_dir: Path,
    img_shape: Annotated[tuple[int, int, int] | None, typer.Option()] = None,
    slice_thickness: Annotated[float, typer.Option()] = 1.0,
    verbosity_level: Annotated[int, typer.Option()] = 0,
    compression_level: Annotated[int, typer.Option()] = 9,
    shuffle_method: Annotated[ShuffleMethod, typer.Option()] = ShuffleMethod.BLOSC_SHUFFLE,
    n_threads: Annotated[int, typer.Option()] = 4,
    n_input_chunks: Annotated[int, typer.Option()] = 1,
    n_output_chunks: Annotated[int, typer.Option()] = 1,
) -> None:
    """Compute and store the system matrices and b-vectors for a range of projections.

    Args:
        setup: Setup name, of the form ``phantom-dataset``.
        start_index: First projection angle in degrees.
        stop_index: Projection angle in degrees at which to stop, exclusive.
        n_projections: Number of projections over a full rotation.
        pixel_width: Width of a pixel in mm.
        input_dir: Directory containing the processed proton dataframes.
        output_dir: Directory in which the matrices are written.
        img_shape: Shape of the image as (slices, height, width). By default three slices of
            the width that follows from the phantom radius and ``pixel_width``.
        slice_thickness: Thickness of a slice in mm.
        verbosity_level: Verbosity level of the matrix computation.
        compression_level: Compression level of the stored matrices.
        shuffle_method: Shuffle method used when storing the matrices.
        n_threads: Number of threads used by the MLP and matrix computation.
        n_input_chunks: Number of chunks each proton dataframe is split into.
        n_output_chunks: Number of chunks each projection is split into.
    """
    phantom = setup.strip().split("-")[0]
    dataset = setup.strip().split("-")[1]

    if img_shape is None:
        img_shape = (3, get_image_w(phantom, pixel_width), get_image_w(phantom, pixel_width))

    logger.info(f"{setup=}")
    logger.info(f"{start_index=}")
    logger.info(f"{stop_index=}")
    logger.info(f"{n_input_chunks=}")
    logger.info(f"{n_output_chunks=}")
    logger.info(f"{n_projections=}")
    logger.info(f"{pixel_width=}")
    logger.info(f"{slice_thickness=}")
    logger.info(f"{input_dir=}")
    logger.info(f"{output_dir=}")
    logger.info(f"{img_shape=}")
    logger.info(f"{verbosity_level=}")
    logger.info(f"{compression_level=}")
    logger.info(f"{n_threads=}")

    proj_angles = np.arange(start_index, stop_index, 360 / n_projections).tolist()

    logger.info(f"Compute matrices from proj {start_index} to {stop_index} in steps of {360 / n_projections}")

    for proj_angle in proj_angles:
        logger.info(f"Compute matrix for projection angle {proj_angle}")

        protons = read_pickled_df(input_dir, proj_angle, n_input_chunks)

        proton_index_list = list(range(len(protons)))
        chunked_indexes_arr = np.array_split(np.arange(len(proton_index_list)), n_output_chunks)

        logger.info(
            f"Processing matrix for {len(protons)} protons in {len(chunked_indexes_arr)} chunks of "
            f"{', '.join([str(len(chunk)) for chunk in chunked_indexes_arr])} protons respectively.",
        )

        img_origin = (
            get_image_halfspace_xz(phantom),
            -get_image_halfspace_yz(img_shape[0], slice_thickness),
            get_detector_distance(phantom, dataset) - get_detector_distance(phantom, "ideal"),
        )

        for chunk_i, indices in enumerate(chunked_indexes_arr):
            logger.info(f"Processing chunk {chunk_i + 1}/{len(chunked_indexes_arr)}")

            logger.info("Compute 3D MLPs")
            protons_chunk = protons.iloc[indices]

            x, y, z = compute_paths(
                phantom,
                protons_chunk,
                proj_angle,
                get_centre_of_rotation(phantom, dataset),
                pixel_width / 2,
                n_threads=n_threads,
            )

            logger.info("Store b-vector")
            vector_output_dir: Path = output_dir.parent / "b"
            vector_output_dir.mkdir(parents=True, exist_ok=True)

            if n_output_chunks > 1:
                vector_output_path = vector_output_dir / f"b_{proj_angle}-{chunk_i}.vec"
            else:
                vector_output_path = vector_output_dir / f"b_{proj_angle}.vec"

            store_vector(protons_chunk["wepl"].to_numpy(), str(vector_output_path))
            logger.info(f"Vector written to {vector_output_path}")

            logger.info("Compute matrix")
            output_dir.mkdir(parents=True, exist_ok=True)

            if n_output_chunks > 1:
                output_path = output_dir / f"{proj_angle}-{chunk_i}.mat"
            else:
                output_path = output_dir / f"{proj_angle}.mat"

            compute_and_store_sparse_matrix(
                str(output_path),
                x.astype(np.float32),
                y.astype(np.float32),
                z.astype(np.float32),
                img_shape,
                img_origin,
                pixel_width,
                slice_thickness,
                verbosity_level,
                n_threads,
                compression_level,
                shuffle_method,
            )
            logger.info(f"Matrix written to {output_path}")

    logger.info("Done")


if __name__ == "__main__":
    typer.run(process_matrix)
