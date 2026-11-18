"""Generate ground truth and reference images for the spiral phantom."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

from phantoms import PHANTOMS_DIR
from phantoms.constants import MU_AIR, MU_AL, MU_WATER, RSP_AIR, RSP_AL, RSP_WATER
from phantoms.utils import draw_circle_in_image, generate_realistic_ct_image, get_circle_image


def calculate_insert_positions(theta_init: float) -> list[tuple[int, int]]:
    """Return the insert positions along a spiral arm starting at the given angle.

    Args:
        theta_init: Starting angle of the arm in radians.

    Returns:
        Integer (x, y) positions in mm, relative to the phantom centre.
    """
    positions = []
    dr = 22.5
    radius = 0
    theta = theta_init
    while radius < (100 - dr):
        radius += dr
        x = int(radius * np.cos(theta))
        y = int(radius * np.sin(theta))
        positions.append((x, y))
        theta -= 0.3
    return positions


RADIUS = 100
CENTRE_IN_MM = RADIUS + 10

PIXEL_W = 0.25
SCALED_PIXEL_WIDTH = PIXEL_W / 10


def main(
    output_dir: Annotated[Path, typer.Option()] = PHANTOMS_DIR,
    random_seed: Annotated[int, typer.Option()] = 1,
) -> None:
    """Write the spiral ground truths and x-ray references.

    Args:
        output_dir: Directory under which ``ground_truths/`` and ``xray_refs/`` are written.
        random_seed: Seed of the noise in the simulated x-ray CT acquisition.
    """
    gt_dir = output_dir / "ground_truths"
    ref_dir = output_dir / "xray_refs"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    im_rsp = get_circle_image(RADIUS, SCALED_PIXEL_WIDTH, RSP_AIR, RSP_WATER, circle_to_edge_distance=10)
    im_xct = get_circle_image(RADIUS, SCALED_PIXEL_WIDTH, MU_AIR, MU_WATER, circle_to_edge_distance=10)

    inserts = [(0, 0)]
    for i in range(5):
        circles = calculate_insert_positions(2.75 - i * 2 / 5 * np.pi)
        inserts += circles

    for x, y in inserts:
        shifted_x = x + CENTRE_IN_MM
        shifted_y = y + CENTRE_IN_MM
        im_rsp = draw_circle_in_image(im_rsp, (shifted_x, shifted_y), 6, SCALED_PIXEL_WIDTH, RSP_AL)
        im_xct = draw_circle_in_image(im_xct, (shifted_x, shifted_y), 6, SCALED_PIXEL_WIDTH, MU_AL)

    im_rsp_rescaled = rescale(np.flipud(im_rsp), 0.1, anti_aliasing=True)
    im_xct_rescaled = rescale(np.flipud(im_xct), 0.1, anti_aliasing=True)

    IMAGE_WIDTH = int(2 * CENTRE_IN_MM / PIXEL_W)

    xct_recon_slice, xct_recon = generate_realistic_ct_image(
        im_xct_rescaled.copy(),
        theta=np.linspace(0, 180, int(np.pi / 2 * IMAGE_WIDTH), endpoint=False),
        image_w=im_xct_rescaled.shape[0],
        pixel_w=PIXEL_W,
        initial_intensity=7e9,
        t_exposure=0.3,
        random_seed=random_seed,
    )

    np.save(gt_dir / f"spiral_gt_{IMAGE_WIDTH}pix.npy", im_rsp_rescaled)
    np.save(gt_dir / f"spiral_xray_gt_{IMAGE_WIDTH}pix.npy", im_xct_rescaled)
    np.save(ref_dir / f"spiral_xray_ref_realistic_{IMAGE_WIDTH}pix.npy", xct_recon)
    np.save(ref_dir / f"spiral_xray_ref_realistic_slice_{IMAGE_WIDTH}pix.npy", xct_recon_slice)

    scaling = 0.25 / PIXEL_W
    xct_recon_slice_filtered = gaussian_filter(xct_recon_slice, sigma=0.65 * scaling, mode="mirror")
    np.save(
        ref_dir / f"spiral_xray_ref_realistic_slice_filtered_{IMAGE_WIDTH}pix.npy",
        xct_recon_slice_filtered,
    )


if __name__ == "__main__":
    typer.run(main)
