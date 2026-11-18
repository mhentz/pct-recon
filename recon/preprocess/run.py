"""Preprocess the raw list-mode proton data into the inputs of the reconstruction."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from pct_tools.mlp import compute_mlp, get_coefficients

from recon.geometry import (
    get_centre_of_rotation,
    get_image_halfspace_xz,
    get_image_halfspace_yz,
    get_radius,
)
from recon.preprocess.cuts import apply_data_cuts, compute_proton_stats
from recon.preprocess.hull import compute_hull_intersection
from recon.preprocess.io import read_root_file
from recon.preprocess.matching import match_protons
from recon.preprocess.utils import compute_angles, compute_wepl

pd.options.display.width = 0


def convert_energy_to_wepl(
    protons: pd.DataFrame,
    lookup_table: np.ndarray,
    beam_energy: float | None = None,
) -> pd.DataFrame:
    """Convert proton energy to WEPL using a lookup table."""
    initial_energy = beam_energy or protons["energy_in"]
    protons["wepl"] = compute_wepl(lookup_table, initial_energy, protons["energy_out"])
    return protons


def degrade_measurements(protons: pd.DataFrame, phantom: str, random_seed: int = 0) -> pd.DataFrame:
    """Degrade position, angle, and WEPL measurements using normal distributions.

    The standard deviations for position assume a resolution of 70 microns on the entry and exit
    trackers.

    The angular resolution is partly determined by the amount of scattering inside the
    detector which is energy dependent: for 200 MeV protons we have 2.3 mrad on entry and 5 mrad
    on exit, while for 250 MeV protons we have 2.2 mrad on entry and 3.5 mrad on exit.

    The WEPL resolution is approximately 0.13% of the WEPL measurement.

    Args:
        protons: Dataframe of proton measurements.
        phantom: Phantom name.
        random_seed: Seed of the random number generator.

    Returns:
        Dataframe of degraded proton measurements on entry and exit detectors.
    """
    rng = np.random.default_rng(random_seed)
    protons["x_in"] += rng.normal(0, 0.07, size=len(protons))
    protons["y_in"] += rng.normal(0, 0.07, size=len(protons))
    protons["x_out"] += rng.normal(0, 0.07, size=len(protons))
    protons["y_out"] += rng.normal(0, 0.07, size=len(protons))

    if phantom == "gammex":
        protons["ang_xz_in"] += rng.normal(0, 0.0022, size=len(protons))
        protons["ang_yz_in"] += rng.normal(0, 0.0022, size=len(protons))
        protons["ang_xz_out"] += rng.normal(0, 0.0035, size=len(protons))
        protons["ang_yz_out"] += rng.normal(0, 0.0035, size=len(protons))
    else:
        protons["ang_xz_in"] += rng.normal(0, 0.0023, size=len(protons))
        protons["ang_yz_in"] += rng.normal(0, 0.0023, size=len(protons))
        protons["ang_xz_out"] += rng.normal(0, 0.005, size=len(protons))
        protons["ang_yz_out"] += rng.normal(0, 0.005, size=len(protons))

    protons["wepl"] *= rng.normal(1, 0.0013, size=len(protons))
    return protons


def find_protons_in_central_slice(
    phantom: str,
    protons: pd.DataFrame,
    step_size: float = 0.1,
    im_halfspace_yz: float = 1.5,
    *,
    n_threads: int = 8,
) -> pd.DataFrame:
    """Filter out protons that left the central slices.

    Also remove protons that have travelled no more than the width of one pixel through the phantom.
    """
    logger.info(f"Filtering out protons outside the image space {-im_halfspace_yz} < y < {im_halfspace_yz}")
    n_protons_total = len(protons)

    protons = protons[
        (protons["y_hull_in"].abs() <= im_halfspace_yz)
        & (protons["y_hull_out"].abs() <= im_halfspace_yz)
        & (protons["z_hull_out"] - protons["z_hull_in"] > step_size)
    ]
    logger.info(
        f"{len(protons)}/{n_protons_total} protons were inside the image space "
        f"{-im_halfspace_yz} < y < {im_halfspace_yz} when they reached the phantom hull",
    )

    mlp_parameters = np.array(
        [
            protons["y_hull_in"],
            protons["ang_yz_in"],
            protons["y_hull_out"],
            protons["ang_yz_out"],
        ],
    )

    logger.info("Compute MLPs individually")
    coefficients = get_coefficients(phantom)
    y_list = []
    for i in range(len(protons)):
        if i % 10000 == 0:
            logger.info(f"Computing proton {i}")
        else:
            logger.debug(f"Computing proton {i}")

        z = np.arange(
            protons.iloc[i]["z_hull_in"],
            protons.iloc[i]["z_hull_out"] + step_size,
            step=step_size,
        )
        y = compute_mlp(z.astype(np.float32), *mlp_parameters[..., i], coefficients, n_threads=n_threads)
        y_list.append(y.squeeze())

    logger.info("Computed MLPs")

    idx_in = np.nonzero([(np.abs(y_i) <= im_halfspace_yz).all() for y_i in y_list])
    idx_out = np.nonzero([(np.abs(y_i) > im_halfspace_yz).any() for y_i in y_list])

    protons_in_slice = protons.iloc[idx_in]
    protons_outside_slice = protons.iloc[idx_out]

    logger.info(
        f"{len(protons_in_slice)}/{len(protons)} protons stayed in the image space while "
        f"{len(protons_outside_slice)}/{len(protons)} protons exited it",
    )
    return protons_in_slice


def process_projection(
    phantom: str,
    projection_angle: float,
    phantom_radius: float,
    centre_of_rotation: float,
    img_halfspace_xz: float,
    img_halfspace_yz: float,
    target_dose: float,
    actual_dose: float,
    input_dir: str,
    output_dir: str,
    wepl_lut_path: str,
    degrade: bool = False,  # noqa: FBT001, FBT002
    chunk: int | None = None,
    step_size: float = 0.1,
    n_sigma: float = 3,
    n_threads: int = 8,
) -> None:
    """Preprocess the data of one projection and write the result to a pickle file.

    Args:
        phantom: Name of the phantom.
        projection_angle: Projection angle in degrees.
        phantom_radius: Radius of the phantom in mm.
        centre_of_rotation: Position of the centre of rotation in mm.
        img_halfspace_xz: Half-width of the image in the xz plane.
        img_halfspace_yz: Half-height of the image in the yz plane.
        target_dose: Dose to which the protons are subsampled.
        actual_dose: Dose at which the data were acquired.
        input_dir: Directory containing the ROOT files.
        output_dir: Directory in which the pickle file is written.
        wepl_lut_path: Path to the WEPL lookup table.
        degrade: Whether to degrade the measurements with detector resolution.
        chunk: Index of the chunk of protons, if the projection is split.
        step_size: Step size along z in mm.
        n_sigma: Number of standard deviations used in the data cuts.
        n_threads: Number of threads used by the MLP computation.
    """
    with Path(wepl_lut_path).open(newline="", encoding="utf-8") as f:
        raw_lookup_table = next(iter(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)))
        lookup_table = np.array([np.float32(x) for x in raw_lookup_table if x != ""])

    projection = f"{projection_angle:.0f}" if projection_angle.is_integer() else f"{projection_angle:.1f}"

    if chunk is not None:
        projection += f"-{chunk}"

    logger.info(f"Processing projection angle {projection}...")
    protons_in, protons_out = read_root_file(input_dir, projection)
    protons = match_protons(protons_in, protons_out)
    protons = compute_angles(protons)
    protons = convert_energy_to_wepl(protons, lookup_table)

    if degrade:
        logger.info("Degrading measurements")
        protons = degrade_measurements(protons, phantom, random_seed=int(10 * projection_angle))

    logger.info(f"Number of protons before subsampling: {len(protons)}")
    subsampling_ratio = target_dose / actual_dose
    protons = protons.sample(n=int(len(protons) * subsampling_ratio), random_state=0).reset_index(drop=True)
    logger.info(f"Number of protons after subsampling with fraction {subsampling_ratio:.4f}: {len(protons)}")

    protons, protons_missed_hull = compute_hull_intersection(protons, centre_of_rotation, phantom_radius)
    logger.info(f"Number of protons that entered AND exited the hull {len(protons)}")
    logger.info(f"Number of protons that missed the hull {len(protons_missed_hull)}")

    protons, stats, bin_edges = compute_proton_stats(
        protons,
        centre_of_rotation,
        img_halfspace_xz,
        img_halfspace_yz + 0.5,
    )
    protons = apply_data_cuts(protons, n_sigma, stats, bin_edges)
    logger.info(f"Number of protons after data cuts {len(protons)}")

    protons = find_protons_in_central_slice(
        phantom,
        protons,
        step_size,
        img_halfspace_yz,
        n_threads=n_threads,
    )

    os.makedirs(output_dir, exist_ok=True)
    if chunk is not None:
        output_filename = f"{output_dir}/df_{projection_angle:.1f}-{chunk}.pkl"
    else:
        output_filename = f"{output_dir}/df_{projection_angle:.1f}.pkl"

    protons.to_pickle(output_filename)
    logger.info(f"File written to {output_filename}")


def preprocess_protons(
    setup: str,
    n_projections: int,
    actual_dose: float,
    target_dose: float,
    start_index: float,
    end_index: float,
    n_chunks: Annotated[int, typer.Argument()] = 1,
    *,
    input_dir: Annotated[str, typer.Argument()],
    output_dir: Annotated[str, typer.Argument()],
    wepl_lut_path: Annotated[str, typer.Option(envvar="WEPL_LUT_PATH")],
    n_slices: Annotated[int, typer.Argument()] = 3,
    slice_thickness: Annotated[float, typer.Argument()] = 1.0,
    n_sigma: Annotated[float, typer.Argument()] = 3.0,
    n_threads: Annotated[int, typer.Option()] = 8,
    degrade: Annotated[bool, typer.Option("--degrade", "-d")] = False,
) -> None:
    """Preprocess the raw data of a range of projections.

    Args:
        setup: Setup name, of the form ``phantom-dataset``.
        n_projections: Number of projections over a full rotation.
        actual_dose: Dose at which the data were acquired.
        target_dose: Dose to which the protons are subsampled.
        start_index: First projection angle in degrees.
        end_index: Projection angle in degrees at which to stop, exclusive.
        n_chunks: Number of chunks each projection is split into.
        input_dir: Directory containing the ROOT files.
        output_dir: Directory in which the pickle files are written.
        wepl_lut_path: Path to the WEPL lookup table (``--wepl-lut-path`` or ``WEPL_LUT_PATH``).
        n_slices: Number of slices in the volume.
        slice_thickness: Thickness of each slice in mm.
        n_sigma: Number of standard deviations used in the data cuts.
        n_threads: Number of threads used by the MLP computation.
        degrade: Whether to degrade the measurements with detector resolution.
    """
    logger.info(
        f"Processing raw data for setup {setup} with {n_projections} projections "
        f"at {actual_dose:.1f} mGy adjusted to {target_dose:.1f}",
    )
    logger.info(f"{input_dir=}")
    logger.info(f"{output_dir=}")

    proj_list = np.arange(start_index, end_index, 360 / n_projections)

    phantom = setup.strip().split("-")[0]
    dataset = setup.strip().split("-")[1]

    logger.info(f"Process raw list-mode data from {start_index} to {end_index}")
    for proj_angle in proj_list:
        if n_chunks > 1:
            for chunk in range(n_chunks):
                process_projection(
                    phantom=phantom,
                    projection_angle=proj_angle,
                    phantom_radius=get_radius(phantom),
                    centre_of_rotation=get_centre_of_rotation(phantom, dataset),
                    img_halfspace_xz=get_image_halfspace_xz(phantom),
                    img_halfspace_yz=get_image_halfspace_yz(n_slices, slice_thickness),
                    target_dose=target_dose,
                    actual_dose=actual_dose,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    wepl_lut_path=wepl_lut_path,
                    degrade=degrade,
                    chunk=chunk,
                    n_sigma=n_sigma,
                    n_threads=n_threads,
                )
        else:
            process_projection(
                phantom=phantom,
                projection_angle=proj_angle,
                phantom_radius=get_radius(phantom),
                centre_of_rotation=get_centre_of_rotation(phantom, dataset),
                img_halfspace_xz=get_image_halfspace_xz(phantom),
                img_halfspace_yz=get_image_halfspace_yz(n_slices, slice_thickness),
                target_dose=target_dose,
                actual_dose=actual_dose,
                input_dir=input_dir,
                output_dir=output_dir,
                wepl_lut_path=wepl_lut_path,
                degrade=degrade,
                n_sigma=n_sigma,
                n_threads=n_threads,
            )

    logger.info("Processed raw list-mode data")


if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    typer.run(preprocess_protons)
