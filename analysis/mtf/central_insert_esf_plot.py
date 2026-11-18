"""Plot the edge spread function of the central insert of the spiral phantom and its erf fit."""

from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.ideal_alpha import get_ideal_alpha
from analysis.mtf.mtf import (
    adjust_for_background,
    calculate_k10_from_fit_params,
    compute_raw_esf,
    construct_esf,
    erf_esf,
    fit_esf_erf,
    rescale_intensity,
)
from analysis.paths import RECON_DATA_DIR

plt.rcParams["figure.figsize"] = [5, 3]
plt.rcParams["figure.dpi"] = 200

central_roi = (slice(200 * 2, 240 * 2), slice(200 * 2, 240 * 2))


def main(  # noqa: PLR0915
    setup: Annotated[Literal["ideal", "realistic"], typer.Argument()],
    reg: Annotated[Literal["TV", "dTV"], typer.Argument()],
    dose: Annotated[float, typer.Argument()] = 5.0,
    n_projections: Annotated[int, typer.Argument()] = 720,
) -> None:
    """Plot the central insert next to its edge spread function and print the fit parameters."""
    alpha = get_ideal_alpha(setup=f"spiral-{setup}", dose=dose, n_projections=n_projections)

    logger.info("Using ideal alpha {}", alpha)

    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    base_path = Path(f"{RECON_DATA_DIR}/spiral-{setup}/{dose_str}mGy/{n_projections}proj/recon")

    suffix = "_eta_2e-1" if reg == "dTV" else ""
    img_dir = max(
        base_path.glob(
            f"fista_exact_init_zero_880pix_{dose}mGy_{n_projections}proj_{reg}_iso_alpha_{alpha}{suffix}_iter*"
        )
    )

    img = np.load(img_dir / "processed" / "yk_min_slice.npy")
    raw_roi = img[central_roi]

    # Generate data for ESF plots
    roi = adjust_for_background(raw_roi)
    roi = rescale_intensity(roi, in_range="image", out_range=(0, 1)).astype(np.float32)

    distances, esf_raw = compute_raw_esf(roi, ransac_seed=0)
    esf_binned, esf_var_binned, bin_positions = construct_esf(roi, ransac_seed=0)
    popt, pcov = fit_esf_erf(bin_positions, esf_binned, esf_var_binned)
    esf_fit = erf_esf(bin_positions, *popt)

    k10, sigma_k10 = calculate_k10_from_fit_params(popt[-1], pcov[-1, -1], pixel_size=0.25)

    # Plot insert and raw ESF overlayed with fitted ESF
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.5), dpi=300)  # , gridspec_kw={"width_ratios": [0.75, 1]})

    im = ax[0].imshow(raw_roi, cmap="gray", vmin=0)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("top", size="5%", pad=0.05, frameon=False)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", format="%.1f")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xlim(1, 2.1)

    ax[1].plot(bin_positions, esf_fit, color="orange", alpha=0.8, lw=1.5, label=r"$\Phi_{\rm fit}$")
    ax[1].scatter(distances, esf_raw, alpha=0.25, s=0.1, color="black")
    ax[1].scatter(bin_positions[0], esf_binned[0], alpha=1, s=0.1, color="black", label="Raw ESF")

    # Set labels
    ax[0].set_xlabel("Central insert")
    ax[1].set_xlabel("Distance from edge [mm]")
    ax[1].set_ylabel("Normalised ESF", labelpad=-0.1)

    # Remove spines
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["left"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)

    # Set ticks
    im_ticks = np.arange(0, 81, 20) - 1
    im_ticks[0] = 0
    im_ticklabels = np.arange(0, 81, 20)
    im_ticklabels[0] = 1

    _ = ax[0].set_xticks(im_ticks)
    _ = ax[0].set_xticklabels(im_ticklabels)
    _ = ax[0].set_yticks(im_ticks)
    _ = ax[0].set_yticklabels(im_ticklabels)

    _ = ax[1].set_xticks(np.arange(-8, 9, 2))

    # Insert legend
    ax[1].legend(
        loc="upper right",
        bbox_to_anchor=(1.02, 1.07),
        scatterpoints=3,
        scatteryoffsets=[0.5],
        markerscale=1.5,
        handlelength=2,
    )

    # Adjust layout and save figure
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(
        figures_dir / f"esf-{setup}-{reg.lower()}-{dose:.1f}mGy-{n_projections}proj.png",
        dpi=300,
        bbox_inches="tight",
    )

    offset, k, x0, sigma = popt
    print(f"{offset=:.3f} {k=:.3f} {x0=:.3f} {sigma=:.3f}")
    print(f"{k10=:.3f} {sigma_k10=:.3f}")


if __name__ == "__main__":
    typer.run(main)
