"""Plot MTF for central slices."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

from analysis.mtf.mtf import compute_mtf

REPO_ROOT = Path(__file__).resolve().parents[2]

CENTRAL_ROI = (slice(200 * 2, 240 * 2), slice(200 * 2, 240 * 2))

plt.rcParams["figure.dpi"] = 200
plt.style.use(f"{REPO_ROOT}/analysis/mtf/style.txt")


def plot_central_slice_mtf(
    freqs: np.ndarray,
    mtf: np.ndarray,
    k10: float,
    outname: str | Path | None = None,
    outname_alt: str | Path | None = None,
    pixel_size: float | None = None,
    *,
    xmax: float,
) -> None:
    """Plot the MTF of a central slice with its k10 marker and save it.

    Args:
        freqs: Spatial frequencies. Divided by ``pixel_size`` in place if it is given.
        mtf: MTF values at ``freqs``.
        k10: Spatial frequency at which the MTF falls to 0.1.
        outname: Path of the primary output figure, if any.
        outname_alt: Path of a second output figure, if any.
        pixel_size: Pixel size in mm, used to convert frequencies to lp/mm.
        xmax: Upper limit of the frequency axis.
    """
    _, ax = plt.subplots(1, 1, figsize=(4.5, 2.7))

    if pixel_size:
        freqs /= pixel_size

    ax.plot(freqs, mtf, color="black", label="MTF")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.05)

    ax.set_xlabel("$k$ [lp/mm]")
    ax.set_ylabel(r"MTF$(k)$/MTF(0)")

    ax.hlines(0.1, 0, xmax, color="black", ls="dotted", lw=1, label=r"MTF$(k)$/MTF(0) = 0.1")
    ax.vlines(k10, 0, 0.5, color="black", ls="dashed", lw=1, label=f"$k_{{10}} = {k10:.3f}$ lp/mm")
    ax.legend(loc="upper right")

    if outname:
        logger.debug(f"Saving plot to {outname}")
        plt.savefig(outname, dpi=300, bbox_inches="tight")

    if outname_alt:
        logger.debug(f"Saving plot to {outname_alt}")
        plt.savefig(outname_alt, dpi=300, bbox_inches="tight")


def main(
    path: Annotated[Path, typer.Argument()],
    alternative_output_dir: Annotated[Path | None, typer.Argument()] = None,
    xmax: Annotated[float | None, typer.Argument()] = None,
    *,
    use_cache: bool = True,
    store_results: bool = False,
    skip_processed: bool = True,
) -> None:
    """Compute and plot the central-slice MTF of one reconstruction.

    Args:
        path: Path of the processed image array.
        alternative_output_dir: Second directory to save the plot in, if any.
        xmax: Upper limit of the frequency axis. Chosen automatically if not given.
        use_cache: Reuse cached MTF results if they exist.
        store_results: Store the computed MTF results next to the image.
        skip_processed: Skip the image if its plot already exists.
    """
    if skip_processed and (path.parent / "mtf" / "central_slice_mtf.png").exists():
        logger.info(f"Skipping {path}, already processed.")
        return

    mtf_dir = path.parent / "mtf"
    mtf_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {path}")
    img = np.load(path)

    mtf, cycles_per_mm, k10, _ = compute_mtf(
        img[CENTRAL_ROI],
        pixel_size=0.25,
        ransac_seed=0,
        input_dir=mtf_dir if use_cache else None,
        output_dir=mtf_dir if store_results else None,
        write_k10_to_file=store_results,
    )

    if not xmax:
        if k10 < 1.5:  # noqa: PLR2004
            xmax = 1.5
        elif k10 >= 1.5:  # noqa: PLR2004
            xmax = 3.0

    plot_central_slice_mtf(
        cycles_per_mm,
        mtf,
        k10,
        outname=mtf_dir / "central_slice_mtf.png",
        outname_alt=(
            alternative_output_dir / f"central_slice_mtf_{path.parent.parent.name}.png"
            if alternative_output_dir
            else None
        ),
        xmax=xmax,  # ty: ignore[invalid-argument-type]
    )
    logger.info(f"k10 = {k10:.3f} lp/mm")
    # plt.show()


if __name__ == "__main__":
    # path = Path(
    #     f"{RECON_DATA_DIR}/spiral-ideal/5mGy/720proj/recon/"
    #     "fista_exact_init_zero_880pix_5.0mGy_720proj_TV_iso_alpha_2.5e-3_iter200/yk_min_slice.npy"
    # )

    typer.run(main)
