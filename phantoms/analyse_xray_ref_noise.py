"""Report the noise statistics of a realistic x-ray reference image in uniform regions."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from phantoms import PHANTOMS_DIR


def main(
    setup: str,
    image_w: int,
    xray_ref_dir: Annotated[Path, typer.Option()] = PHANTOMS_DIR / "xray_refs",
) -> None:
    """Print the mean and relative standard deviation of uniform regions in a reference image.

    Args:
        setup: Phantom name, one of "gammex", "spiral" or "ctp528".
        image_w: Width of the reference image in pixels.
        xray_ref_dir: Directory containing the x-ray references.

    Raises:
        ValueError: If the setup name is not recognised.
    """
    xray_ref_real_slice_filtered = np.load(
        xray_ref_dir / f"{setup}_xray_ref_realistic_slice_filtered_{image_w}pix.npy",
    )

    if setup == "gammex":
        size = 200
        roi_size = 100
        ul = (slice(120, 120 + roi_size), slice(120, 120 + roi_size))
        ur = (slice(120, 120 + roi_size), slice(480, 480 + roi_size))
        ll = (slice(480, 480 + roi_size), slice(120, 120 + roi_size))
        lr = (slice(480, 480 + roi_size), slice(480, 480 + roi_size))
    elif setup == "spiral":
        size = 230
        roi_size = 115
        ul = (slice(220, 220 + roi_size), slice(180, 180 + roi_size))
        ur = (slice(100, 100 + roi_size), slice(450, 450 + roi_size))
        ll = (slice(550, 550 + roi_size), slice(180, 180 + roi_size))
        lr = (slice(630, 630 + roi_size), slice(490, 490 + roi_size))

    if setup in ["gammex", "spiral"]:
        noise_image_slice = np.zeros((size, size))

        noise_image_slice[:roi_size, :roi_size] = xray_ref_real_slice_filtered[ul]
        noise_image_slice[:roi_size:, roi_size:] = xray_ref_real_slice_filtered[ur]
        noise_image_slice[roi_size:, :roi_size] = xray_ref_real_slice_filtered[ll]
        noise_image_slice[roi_size:, roi_size:] = xray_ref_real_slice_filtered[lr]

    elif setup == "ctp528":
        centre = image_w // 2
        size = 230

        centre_slice = slice(centre - size // 2, centre + size // 2)
        noise_image_slice = xray_ref_real_slice_filtered[centre_slice, centre_slice]

    else:
        msg = f"Unknown setup: {setup}"
        raise ValueError(msg)

    im_std = noise_image_slice.flatten().std(ddof=1)
    im_mean = noise_image_slice.flatten().mean()

    rel_std_pc = im_std / im_mean * 100

    logger.info(f"Std dev: {im_std:.3f}")
    logger.info(f"Mean: {im_mean:.3f}")
    logger.info(f"Relative std dev: {rel_std_pc:.3f}%")


if __name__ == "__main__":
    typer.run(main)
