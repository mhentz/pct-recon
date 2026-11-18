"""Generate ground truth and reference images for the CTP528 phantom."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon
from skimage.transform import rescale

from phantoms import PHANTOMS_DIR
from phantoms.constants import MU_AIR, MU_AL, MU_WATER, RSP_AIR, RSP_AL, RSP_WATER
from phantoms.utils import deteriorate_image, generate_realistic_ct_image, get_circle_image


def rotate_point(
    point: tuple[float, float],
    pivot: tuple[float, float],
    angle_in_deg: float,
) -> tuple[int, int]:
    """Rotate a point about a pivot by an angle in degrees.

    Args:
        point: The (x, y) coordinates of the point.
        pivot: The (x, y) coordinates of the rotation centre.
        angle_in_deg: The rotation angle in degrees, counter-clockwise.

    Returns:
        The rotated (x, y) coordinates, truncated to integers.
    """
    x, y = point
    cx, cy = pivot
    angle = np.deg2rad(angle_in_deg)

    x -= cx
    y -= cy

    x_new = x * np.cos(angle) - y * np.sin(angle)
    y_new = x * np.sin(angle) + y * np.cos(angle)

    x = x_new + cx
    y = y_new + cy

    return int(x), int(y)


def rotate_box(box: np.ndarray, pivot: tuple[float, float], angle_in_deg: float) -> np.ndarray:
    """Rotate each vertex of a box about a pivot.

    Args:
        box: The vertices of the box to rotate.
        pivot: The (x, y) coordinates of the rotation centre.
        angle_in_deg: The rotation angle in degrees. The sign is inverted before rotating.

    Returns:
        Array of the rotated vertices.
    """
    new_box = [rotate_point(v, pivot, -angle_in_deg) for v in box]

    return np.array(new_box)


def get_rect_vertices(location: tuple[float, float], dimensions: tuple[float, float]) -> np.ndarray:
    """Return the corners of an axis-aligned rectangle as integer coordinates.

    Args:
        location: The (x, y) coordinates of the rectangle centre.
        dimensions: The (width, height) of the rectangle.

    Returns:
        Array of shape (4, 2) with the corners in top-left, top-right, bottom-right and
        bottom-left order.
    """
    x, y = location
    w, h = dimensions

    top_left = (np.around(x - w / 2), np.around(y + h / 2))
    top_right = (np.around(x + w / 2), np.around(y + h / 2))
    bottom_right = (np.around(x + w / 2), np.around(y - h / 2))
    bottom_left = (np.around(x - w / 2), np.around(y - h / 2))

    #     return np.array([tl_rot, tr_rot, br_rot, bl_rot])
    return np.array([top_left, top_right, bottom_right, bottom_left]).astype(np.int32)


def draw_box(im: np.ndarray, rows: np.ndarray, cols: np.ndarray, value: float) -> np.ndarray:
    """Set the pixels inside the polygon with vertices (rows, cols) to value, in place."""
    im[polygon(rows, cols)] = value
    return im


def draw_inserts(im: np.ndarray, insert_value: float, centre: float, pixel_w: float) -> np.ndarray:
    """Draw the line-pair inserts of the CTP528 phantom into an image.

    Args:
        im: The image to draw into.
        insert_value: The value written inside the inserts.
        centre: The centre of the phantom in pixels.
        pixel_w: The pixel width in mm.

    Returns:
        The image, rotated and flipped to match the phantom orientation.
    """
    # Convert lp/cm to angle [deg]
    ctp528_angles_coefficients = [-3.8654e-6, 1.9470e-4, -2.3863e-3, -3.6253e-2, 1.3968, -3.0373e1, 1.9902e2]

    for i in np.arange(1, 21):
        gap_size = 5 / i  # mm
        angle = 0.0
        xx = 1.0
        Ns = len(ctp528_angles_coefficients)

        for j in range(Ns):
            angle += xx * ctp528_angles_coefficients[Ns - 1 - j]
            xx *= i

        placement_rot_angle = 270 + angle
        insert_rot_angle = 270 - angle

        if i == 1:
            n_lp = 2
        elif i == 2:  # noqa: PLR2004
            n_lp = 3
        elif i < 6:  # noqa: PLR2004
            n_lp = 4
        else:
            n_lp = 5

        for j in range(n_lp):
            x_container = centre + 47.5 / pixel_w
            y_container = centre

            # w_container = 8 / pixel_w
            # h_container = (
            #     (n_lp - 0.5) * gap_size / pixel_w * 2
            # )  # multiply by 2 to get full width from half width

            h_insert = 8 / pixel_w
            w_insert = gap_size / pixel_w

            x_container, y_container = rotate_point(
                (x_container, y_container),
                (centre, centre),
                placement_rot_angle,
            )

            x_insert = (
                x_container
                - (n_lp - 0.5) * gap_size / pixel_w
                + gap_size / pixel_w / 2
                + 2 * j * gap_size / pixel_w
            )
            y_insert = y_container

            box = get_rect_vertices((x_insert, y_insert), (w_insert, h_insert))
            box = rotate_box(box, (x_container, y_container), insert_rot_angle + 90)

            im = draw_box(im, box[:, 0], box[:, 1], insert_value)

    return np.flipud(np.rot90(im))


RADIUS = 100
CENTRE_IN_MM = RADIUS + 10

PIXEL_W = 0.25
SCALED_PIXEL_WIDTH = PIXEL_W / 10


def main(
    output_dir: Annotated[Path, typer.Option()] = PHANTOMS_DIR,
    random_seed: Annotated[int, typer.Option()] = 1,
) -> None:
    """Write the CTP528 ground truths, x-ray references and preview images.

    Args:
        output_dir: Directory under which ``ground_truths/`` and ``xray_refs/`` are written; the
            other references and the PNG previews go into the directory itself.
        random_seed: Seed of the noise in the deteriorated reference and in the simulated x-ray
            CT acquisition.
    """
    gt_dir = output_dir / "ground_truths"
    ref_dir = output_dir / "xray_refs"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    sigma = 0.5

    im_rsp = get_circle_image(RADIUS, SCALED_PIXEL_WIDTH, RSP_AIR, RSP_WATER, circle_to_edge_distance=10)
    im_xct = get_circle_image(RADIUS, SCALED_PIXEL_WIDTH, MU_AIR, MU_WATER, circle_to_edge_distance=10)

    im_rsp = draw_inserts(im_rsp, RSP_AL, CENTRE_IN_MM / SCALED_PIXEL_WIDTH, SCALED_PIXEL_WIDTH)
    im_xct = draw_inserts(im_xct, MU_AL, CENTRE_IN_MM / SCALED_PIXEL_WIDTH, SCALED_PIXEL_WIDTH)

    im_rsp_rescaled = rescale(im_rsp, 0.1, anti_aliasing=True)
    im_xct_rescaled = rescale(im_xct, 0.1, anti_aliasing=True)

    deteriorated_im_xct = deteriorate_image(im_xct_rescaled.copy(), sigma, 0.02, random_seed=random_seed)

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

    print(im_rsp.shape)
    print(im_rsp_rescaled.shape)

    np.save(gt_dir / f"ctp528_gt_{IMAGE_WIDTH}pix.npy", im_rsp_rescaled)
    np.save(gt_dir / f"ctp528_xray_gt_{IMAGE_WIDTH}pix.npy", im_xct_rescaled)
    np.save(output_dir / f"ctp528_xray_ref_{IMAGE_WIDTH}pix_{sigma:.1f}sig_0.02n.npy", deteriorated_im_xct)
    np.save(ref_dir / f"ctp528_xray_ref_realistic_{IMAGE_WIDTH}pix.npy", xct_recon)
    np.save(ref_dir / f"ctp528_xray_ref_realistic_slice_{IMAGE_WIDTH}pix.npy", xct_recon_slice)

    scaling = 0.25 / PIXEL_W
    xct_recon_slice_filtered = gaussian_filter(xct_recon_slice, sigma=0.7 * scaling, mode="mirror")
    np.save(
        ref_dir / f"ctp528_xray_ref_realistic_slice_filtered_{IMAGE_WIDTH}pix.npy",
        xct_recon_slice_filtered,
    )

    fig, ax = plt.subplots()
    ax.imshow(im_xct_rescaled, cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        output_dir / f"ctp528_xray_gt_{IMAGE_WIDTH}pix.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(deteriorated_im_xct, cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        output_dir / f"ctp528_xray_ref_{IMAGE_WIDTH}pix_{sigma:.1f}sig_0.02n.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(xct_recon, cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        output_dir / f"ctp528_xray_ref_realistic_{IMAGE_WIDTH}pix.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
