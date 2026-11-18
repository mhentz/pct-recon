"""Generate ground truth and reference images for the Gammex phantom."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

from phantoms import PHANTOMS_DIR
from phantoms.constants import MATERIALS_MAP, MU_AIR, MU_MAP, MU_WATER, RSP_AIR, RSP_MAP, RSP_WATER
from phantoms.utils import draw_circle_in_image, generate_realistic_ct_image, get_circle_image

RADIUS = 165
CENTRE_IN_MM = RADIUS + 10

PIXEL_W = 0.5
SCALED_PIXEL_WIDTH = PIXEL_W / 10


def main(
    output_dir: Annotated[Path, typer.Option()] = PHANTOMS_DIR,
    random_seed: Annotated[int, typer.Option()] = 1,
) -> None:
    """Write the Gammex ground truths and x-ray references.

    Args:
        output_dir: Directory under which ``ground_truths/`` and ``xray_refs/`` are written.
        random_seed: Seed of the noise in the simulated x-ray CT acquisition.
    """
    gt_dir = output_dir / "ground_truths"
    ref_dir = output_dir / "xray_refs"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    theta_inside = np.arange(0, 2 * np.pi, 2 * np.pi / 8)
    theta_outside = np.arange(2 * np.pi / 16, 2 * np.pi * 17 / 16, 2 * np.pi / 8)

    rods_dist_inside = 12 / 36 * 165  # mm
    rods_dist_outside = 24 / 36 * 165  # mm

    rods_r = 14  # mm

    xy_dict = {}
    count = 0
    for t in theta_inside:
        x = int(rods_dist_inside * np.cos(-t)) + CENTRE_IN_MM
        y = int(rods_dist_inside * np.sin(-t)) + CENTRE_IN_MM
        print(count, x, y)

        # xy_dict['in'][count] = (x, y)
        xy_dict[count] = (x, y)
        count += 1

    for t in theta_outside:
        x = int(rods_dist_outside * np.cos(-t)) + CENTRE_IN_MM
        y = int(rods_dist_outside * np.sin(-t)) + CENTRE_IN_MM
        print(count, x, y)

        # xy_dict['out'][count] = (x, y)
        xy_dict[count] = (x, y)
        count += 1

    im_rsp = get_circle_image(RADIUS, SCALED_PIXEL_WIDTH, RSP_AIR, RSP_WATER, circle_to_edge_distance=10)
    im_xct = get_circle_image(RADIUS, SCALED_PIXEL_WIDTH, MU_AIR, MU_WATER, circle_to_edge_distance=10)

    for i, xy in xy_dict.items():
        rsp_val = RSP_MAP[MATERIALS_MAP[i]]
        im_rsp = draw_circle_in_image(im_rsp, xy, rods_r, SCALED_PIXEL_WIDTH, rsp_val)

        mu_val = MU_MAP[MATERIALS_MAP[i]]
        im_xct = draw_circle_in_image(im_xct, xy, rods_r, SCALED_PIXEL_WIDTH, mu_val)

    im_rsp = np.fliplr(np.flipud(im_rsp))
    im_xct = np.fliplr(np.flipud(im_xct))

    im_rsp_rescaled = rescale(im_rsp, 0.1, anti_aliasing=True)
    im_xct_rescaled = rescale(im_xct, 0.1, anti_aliasing=True)

    IMAGE_WIDTH = int(2 * CENTRE_IN_MM / PIXEL_W)

    xct_recon_slice, xct_recon = generate_realistic_ct_image(
        im_xct_rescaled.copy(),
        theta=np.linspace(0, 180, int(np.pi / 2 * IMAGE_WIDTH), endpoint=False),
        image_w=im_xct_rescaled.shape[0],
        pixel_w=PIXEL_W,
        initial_intensity=1.5e9,
        t_exposure=0.3,
        random_seed=random_seed,
    )

    np.save(gt_dir / f"gammex_gt_{IMAGE_WIDTH}pix.npy", im_rsp_rescaled)
    np.save(gt_dir / f"gammex_xray_gt_{IMAGE_WIDTH}pix.npy", im_xct_rescaled)
    # np.save(f"gammex_xray_ref_{IMAGE_WIDTH}pix_{sigma:.1f}sig_0.02n.npy", deteriorated_im_xct)
    np.save(ref_dir / f"gammex_xray_ref_realistic_{IMAGE_WIDTH}pix.npy", xct_recon)
    np.save(ref_dir / f"gammex_xray_ref_realistic_slice_{IMAGE_WIDTH}pix.npy", xct_recon_slice)

    scaling = 0.5 / PIXEL_W
    xct_recon_slice_filtered = gaussian_filter(xct_recon_slice, sigma=0.8 * scaling, mode="mirror")
    np.save(
        ref_dir / f"gammex_xray_ref_realistic_slice_filtered_{IMAGE_WIDTH}pix.npy",
        xct_recon_slice_filtered,
    )


if __name__ == "__main__":
    typer.run(main)
