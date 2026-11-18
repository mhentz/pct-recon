"""Compare the TV and dTV reconstructions of the Catphan phantom, with closeups of the line pairs."""

from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import Size, make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.style.use(f"{REPO_ROOT}/analysis/style.txt")

ROI_2 = (slice(540, 640), slice(500, 600))  # 0.3 lp/mm
ROI_3 = (slice(480, 580), slice(550, 650))  # 0.3 lp/mm
ROI_4 = (slice(410, 510), slice(570, 670))  # 0.4 lp/mm
ROI_5 = (slice(355, 455), slice(570, 670))  # 0.5 lp/mm
ROI_6 = (slice(295, 395), slice(550, 650))  # 1 lp/mm
ROI_6 = (slice(310, 370), slice(570, 630))  # 1 lp/mm

ROI_10 = (slice(200, 300), slice(360, 460))  # 1 lp/mm

ROI_9_10 = (slice(200, 300), slice(380, 480))  # 1 lp/mm


def plot_with_closeup(  # noqa: C901, PLR0915
    ax: plt.Axes,
    im: np.ndarray,
    *,
    with_colorbar: bool = False,
    plot_line_pair_labels: bool = True,
) -> tuple[AxesImage, AxesImage]:
    """Show an image with three closeups of its line-pair inserts below it.

    Args:
        ax: Axes to draw on.
        im: Image to display.
        with_colorbar: Add a colour bar to the image.
        plot_line_pair_labels: Label the closeups with their line-pair frequencies.

    Returns:
        Image handles of the full image and of the last closeup.
    """
    # close-up inserts
    pad = 0.0
    axes_class = None

    divider = make_axes_locatable(ax)
    pad_size = pad * Size.AxesX(ax)

    xsize = ((1 - 2 * pad) / 3) * Size.AxesX(ax)
    ysize = ((1 - 2 * pad) / 3) * Size.AxesY(ax)

    divider.set_horizontal([xsize, pad_size, xsize, pad_size, xsize])
    divider.set_vertical([ysize, pad_size, Size.AxesY(ax)])

    ax.set_axes_locator(divider.new_locator(0, 2, nx1=-1))

    ax_inserts = []
    if axes_class is None:
        try:
            axes_class = ax._axes_class  # noqa: SLF001  # ty: ignore[missing-attribute]
        except AttributeError:
            axes_class = type(ax)

    for nx in [0, 2, 4]:
        ax1 = axes_class(ax.figure, ax.get_position(original=True))
        locator = divider.new_locator(nx=nx, ny=0)
        ax1.set_axes_locator(locator)
        for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            t.set_visible(False)
        try:
            for axis in ax1.axis.values():  # ty: ignore[missing-union-attribute]
                axis.major_ticklabels.set_visible(False)
        except AttributeError:
            pass

        ax_inserts.append(ax1)

    fig = ax.figure
    for ax1 in ax_inserts:
        fig.add_axes(ax1)

    # inserts_vmin = 0
    inserts_vmin = 1
    spl = ax_inserts[0].imshow(im[ROI_3], cmap="gray", vmin=inserts_vmin, vmax=2.2)
    ax_inserts[1].imshow(im[ROI_4], cmap="gray", vmin=inserts_vmin, vmax=2.2)
    ax_inserts[2].imshow(im[ROI_5], cmap="gray", vmin=inserts_vmin, vmax=2.2)

    # main plot
    main_plot = ax.imshow(im, cmap="gray", vmin=0, vmax=2.2)
    ax.text(0.03, 0.02, "880 x 880", fontsize=10, color="w", weight="bold", transform=ax.transAxes)

    if with_colorbar:
        cbbox = inset_axes(ax, "20%", "90%", loc=7, borderpad=0.2)
        for k in cbbox.spines:
            cbbox.spines[k].set_visible(False)
        cbbox.tick_params(
            axis="both",
            left="off",
            top="off",
            right="off",
            bottom="off",
            labelleft="off",
            labeltop="off",
            labelright="off",
            labelbottom="off",
            labelsize=8,
        )
        cbbox.set_xticks([])
        cbbox.set_yticks([])
        cbbox.set_facecolor([1, 1, 1, 0.7])

        cbaxes = inset_axes(cbbox, "20%", "95%", loc=6)

        cb = fig.colorbar(main_plot, cax=cbaxes, format="%.1f", ticks=np.arange(0, 2.3, 0.2))
        cb.ax.tick_params(size=2, width=0.5, labelsize=5, pad=2)

    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.spines["right"].set_color("w")
    ax.spines["left"].set_color("w")
    ax.spines["top"].set_color("w")
    ax.spines["bottom"].set_color("w")

    for ax_insert in ax_inserts:
        ax_insert.spines["right"].set_visible(True)
        ax_insert.spines["top"].set_visible(True)
        ax_insert.set_xticks([])
        ax_insert.set_yticks([])
        ax_insert.set_xticklabels([])
        ax_insert.set_yticklabels([])

        ax_insert.spines["right"].set_color("w")
        ax_insert.spines["left"].set_color("w")
        ax_insert.spines["top"].set_color("w")
        ax_insert.spines["bottom"].set_color("w")

    if plot_line_pair_labels:
        ax.text(
            0.74,
            0.37,
            "0.3 lp/mm",
            fontsize=6,
            color="black",
            weight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": [1, 1, 1, 0.5], "edgecolor": "none", "pad": 1.0},
        )
        ax.text(
            0.758,
            0.46,
            "0.4 lp/mm",
            fontsize=6,
            color="black",
            weight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": [1, 1, 1, 0.5], "edgecolor": "none", "pad": 1.0},
        )
        ax.text(
            0.748,
            0.54,
            "0.5 lp/mm",
            fontsize=6,
            color="black",
            weight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": [1, 1, 1, 0.5], "edgecolor": "none", "pad": 1.0},
        )

    ax_inserts[0].text(
        0.05, 0.85, "0.3 lp/mm", fontsize=8, color="w", weight="bold", transform=ax_inserts[0].transAxes
    )
    ax_inserts[1].text(
        0.05, 0.85, "0.4 lp/mm", fontsize=8, color="w", weight="bold", transform=ax_inserts[1].transAxes
    )
    ax_inserts[2].text(
        0.05, 0.85, "0.5 lp/mm", fontsize=8, color="w", weight="bold", transform=ax_inserts[2].transAxes
    )

    return main_plot, spl


def main(
    setup: Annotated[Literal["ideal", "realistic"], typer.Argument()],
    alpha_type: Annotated[Literal["ideal", "better"], typer.Argument()],
) -> None:
    """Plot the TV and dTV reconstructions at 5 mGy and 720 projections side by side.

    Args:
        setup: Dataset, ``"ideal"`` or ``"realistic"``.
        alpha_type: ``"ideal"`` takes the regularisation weight from the L-curve analysis,
            ``"better"`` a smaller weight chosen by eye.
    """
    dose = 5.0
    n_proj = 720

    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    fig, ax = plt.subplots(1, 2, figsize=(5.5, 5.5 / 1.5), dpi=300)

    if alpha_type == "ideal":
        alpha = get_ideal_alpha(f"ctp528-{setup}", dose, n_proj)
    else:
        alpha = "7.1e-4" if setup == "ideal" else "1e-3"

    base_path = Path(f"{RECON_DATA_DIR}/ctp528-{setup}/{dose_str}mGy/{n_proj}proj/recon")

    img_dir_tv: Path = max(
        base_path.glob(f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_TV_iso_alpha_{alpha}_iter*")
    )
    xi = np.load(img_dir_tv / "processed" / "yk_min_slice.npy")

    plot_with_closeup(ax[0], xi, with_colorbar=False, plot_line_pair_labels=True)

    img_dir_dtv: Path = max(
        base_path.glob(
            f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_dTV_iso_alpha_{alpha}_eta_2e-1_iter*"
        )
    )
    xi = np.load(img_dir_dtv / "processed" / "yk_min_slice.npy")

    dtv_plot, _ = plot_with_closeup(ax[1], xi, with_colorbar=False, plot_line_pair_labels=False)

    plt.tight_layout()
    ax[0].set_title("TV")
    ax[1].set_title("dTV")

    # Colorbar box for dTV
    cbbox = inset_axes(ax[1], "17%", "90%", loc=7, borderpad=0.3)
    for k in cbbox.spines:
        cbbox.spines[k].set_visible(False)

    cbbox.set_xticks([])
    cbbox.set_yticks([])
    cbbox.set_facecolor([1, 1, 1, 0.7])

    cbaxes = inset_axes(parent_axes=cbbox, width="20%", height="90%", loc=6)
    cb = fig.colorbar(dtv_plot, cax=cbaxes, format="%.1f", ticks=np.arange(0, 2.3, 0.5))
    cb.ax.set_frame_on(False)
    cb.ax.tick_params(size=2, width=1, labelsize=8, pad=2)

    # Formatting and saving figure
    plt.rcParams["figure.dpi"] = 300
    plt.subplots_adjust(top=0.90)

    # plt.show()

    alpha_str = "-better-alpha" if alpha_type == "better" else ""
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_to_path = figures_dir / f"catphan-comparison-{setup}-5.0mGy-720proj{alpha_str}.png"

    logger.info(str(save_to_path))
    plt.savefig(save_to_path, dpi=300)
    print(f"alpha: {alpha}")


if __name__ == "__main__":
    typer.run(main)
