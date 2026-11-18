"""Plot the MTF of the central insert of the spiral phantom at the ideal regularisation weight."""

from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

from analysis.ideal_alpha import get_ideal_alpha
from analysis.mtf.mtf import compute_mtf
from analysis.paths import RECON_DATA_DIR

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
    """Plot an MTF curve and mark k10 on it.

    Args:
        freqs: Spatial frequencies.
        mtf: MTF values at those frequencies.
        k10: Frequency at which the MTF falls to 10 %.
        outname: Path to save the plot to, if any.
        outname_alt: Second path to save the plot to, if any.
        pixel_size: If given, the frequencies are divided by it.
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
    reg: Annotated[Literal["TV", "dTV"], typer.Argument()],
    setup: Annotated[Literal["ideal", "realistic"], typer.Argument()],
    dose: Annotated[float, typer.Argument()],
    n_proj: Annotated[int, typer.Argument()],
    xmax: Annotated[float | None, typer.Option()] = None,
    *,
    log_fit_params: Annotated[bool, typer.Option()] = False,
) -> None:
    """Compute and plot the MTF of the central insert of one spiral-phantom reconstruction.

    Args:
        reg: Regulariser, ``"TV"`` or ``"dTV"``.
        setup: Dataset, ``"ideal"`` or ``"realistic"``.
        dose: Dose in mGy.
        n_proj: Number of projections.
        xmax: Upper limit of the frequency axis. Chosen from k10 if not given.
        log_fit_params: Log the parameters of the erf fit.
    """
    alpha = get_ideal_alpha(setup=f"spiral-{setup}", dose=dose, n_projections=n_proj)

    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"
    base_path = Path(f"{RECON_DATA_DIR}/spiral-{setup}/{dose_str}mGy/{n_proj}proj/recon/")

    suffix = "_eta_2e-1" if reg == "dTV" else ""
    img_dir = max(
        base_path.glob(
            f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_{reg}_iso_alpha_{alpha}{suffix}_iter*"
        )
    )

    logger.info(f"Processing {img_dir}")
    img = np.load(img_dir / "processed/yk_min_slice.npy")

    mtf, cycles_per_mm, k10, _ = compute_mtf(
        img[CENTRAL_ROI],
        pixel_size=0.25,
        ransac_seed=0,
        log_fit_params=log_fit_params,
    )

    if not xmax:
        xmax = 1.5 if k10 < 1.5 else 3.0  # noqa: PLR2004

    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_central_slice_mtf(
        cycles_per_mm,
        mtf,
        k10,
        outname=figures_dir / f"mtf-{setup}-{reg.lower()}-{dose:.1f}mGy-{n_proj}proj.png",
        xmax=xmax,
    )
    logger.info(f"k10 = {k10:.3f} lp/mm")
    # plt.show()


if __name__ == "__main__":
    typer.run(main)
