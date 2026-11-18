"""Compute and store k10 and sigma_k10 for all inserts in an image."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger

from analysis.mtf.mtf import fit_esf_and_compute_k10

RADIUS_4 = [
    (slice(262 * 2, 297 * 2), slice(36 * 2, 70 * 2)),
    (slice(65 * 2, 100 * 2), slice(94 * 2, 130 * 2)),
    (slice(57 * 2, 90 * 2), slice(300 * 2, 334 * 2)),
    (slice(250 * 2, 287 * 2), slice(370 * 2, 407 * 2)),
    (slice(380 * 2, 415 * 2), slice(205 * 2, 245 * 2)),
]

RADIUS_3 = [
    (slice(205 * 2, 245 * 2), slice(70 * 2, 110 * 2)),
    (slice(75 * 2, 113 * 2), slice(155 * 2, 195 * 2)),
    (slice(117 * 2, 155 * 2), slice(303 * 2, 345 * 2)),
    (slice(275 * 2, 310 * 2), slice(312 * 2, 352 * 2)),
    (slice(330 * 2, 370 * 2), slice(165 * 2, 205 * 2)),
]

RADIUS_2 = [
    (slice(180 * 2, 215 * 2), slice(115 * 2, 150 * 2)),
    (slice(113 * 2, 150 * 2), slice(198 * 2, 235 * 2)),
    (slice(170 * 2, 205 * 2), slice(285 * 2, 319 * 2)),
    (slice(270 * 2, 305 * 2), slice(257 * 2, 295 * 2)),
    (slice(277 * 2, 313 * 2), slice(152 * 2, 190 * 2)),
]

RADIUS_1 = [
    (slice(180 * 2, 215 * 2), slice(160 * 2, 200 * 2)),
    (slice(160 * 2, 195 * 2), slice(210 * 2, 248 * 2)),
    (slice(200 * 2, 236 * 2), slice(242 * 2, 285 * 2)),
    (slice(243 * 2, 277 * 2), slice(217 * 2, 254 * 2)),
    (slice(230 * 2, 265 * 2), slice(166 * 2, 204 * 2)),
]

RADIUS_0 = [(slice(200 * 2, 240 * 2), slice(200 * 2, 240 * 2))]

ROI_LIST = [*RADIUS_0, *RADIUS_1, *RADIUS_2, *RADIUS_3, *RADIUS_4]


def main(path: Annotated[Path, typer.Argument()], *, skip_processed: bool = True) -> None:
    """Compute and store k10 and sigma_k10 for all inserts in an image.

    Args:
        path: The path to the image file.
        skip_processed: Whether to skip processing if the results already exist.

    """
    mtf_dir = path.parent / "mtf"

    if not mtf_dir.exists():
        mtf_dir.mkdir(parents=False)

    if skip_processed and (mtf_dir / "k10_values.csv").exists():
        logger.info(f"Skipping {path}, already processed.")
        return

    logger.info(f"Processing {path}")
    img = np.load(path)

    k10_list = []
    k10_sigma_list = []

    for i, roi_location in enumerate(ROI_LIST):
        logger.debug(f"Processing ROI {i} / {len(ROI_LIST)} at {roi_location}")
        _, _, k10, sigma_k10 = fit_esf_and_compute_k10(img[roi_location], pixel_size=0.25, ransac_seed=0)
        k10_list.append(k10)
        k10_sigma_list.append(sigma_k10)

    pd.DataFrame({"k10": k10_list, "sigma_k10": k10_sigma_list}).to_csv(
        mtf_dir / "k10_values.csv",
        index=False,
    )


if __name__ == "__main__":
    # path = Path(
    #     f"{RECON_DATA_DIR}/spiral-ideal/5mGy/720proj/recon/"
    #     "fista_exact_init_zero_880pix_5.0mGy_720proj_TV_iso_alpha_2.5e-3_iter200/yk_min_slice.npy"
    # )

    # main(
    #     Path(
    #         f"{RECON_DATA_DIR}/spiral-ideal/5mGy/45proj/recon/"
    #         "fista_exact_init_zero_880pix_5.0mGy_45proj_dTV_iso_alpha_1.58e-3_eta_2e-1_iter200/"
    #         "processed/yk_min_slice.npy"
    #     )
    # )

    typer.run(main)
