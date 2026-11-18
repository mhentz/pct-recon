"""Tile the line-pair inserts of the Catphan phantom for TV and dTV at every dose and projection count."""

from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import Size, make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]

ROI_2 = (slice(540, 640), slice(500, 600))  # 0.3 lp/mm
ROI_3 = (slice(480, 580), slice(550, 650))  # 0.3 lp/mm
ROI_4 = (slice(410, 510), slice(570, 670))  # 0.4 lp/mm
ROI_5 = (slice(355, 455), slice(570, 670))  # 0.5 lp/mm
ROI_6 = (slice(310, 370), slice(570, 630))  # 0.6 lp/mm
ROI_7 = (slice(265, 325), slice(530, 590))  # 0.7 lp/mm
ROI_8 = (slice(235, 295), slice(483, 543))  # 0.8 lp/mm
ROI_10 = (slice(200, 300), slice(360, 460))  # 1 lp/mm
ROI_9_10 = (slice(200, 300), slice(380, 480))  # 1 lp/mm


def plot_with_closeup(  # noqa: C901, PLR0912, PLR0915
    ax: plt.Axes,
    ims: list[np.ndarray],
    with_colorbar: bool = False,  # noqa: FBT001, FBT002
    plot_line_pair_labels: bool = True,  # noqa: FBT001, FBT002
    left: bool = False,  # noqa: FBT001, FBT002
    title: str | None = None,
    *,
    label_x_axis: bool,
) -> tuple[AxesImage, AxesImage]:
    """Show a grid of line-pair closeups, one row per image.

    Args:
        ax: Axes to draw on.
        ims: Images to take the closeups from, one per dose.
        with_colorbar: Add a colour bar.
        plot_line_pair_labels: Label the closeups with their line-pair frequencies.
        left: Label the rows with their doses. Used for the left-most column.
        title: Title of the grid, if any.
        label_x_axis: Label the columns with their line-pair frequencies.

    Returns:
        Image handles of the full image and of the last closeup.
    """
    pad = 0.0
    axes_class = None

    divider = make_axes_locatable(ax)

    pad_size_x = pad * Size.AxesX(ax)
    pad_size_y = pad * Size.AxesY(ax)

    xsize = ((1 - 2 * pad) / 3) * Size.AxesX(ax)
    ysize = ((1 - 2 * pad) / 3) * Size.AxesY(ax)

    divider.set_horizontal([xsize, pad_size_x, xsize, pad_size_x, xsize])
    divider.set_vertical([ysize, pad_size_y, ysize, pad_size_y, ysize])

    ax.set_axes_locator(divider.new_locator(0, 2, nx1=-1))

    ax_rgb = []
    if axes_class is None:
        try:
            axes_class = ax._axes_class  # noqa: SLF001  # ty: ignore[missing-attribute]
        except AttributeError:
            axes_class = type(ax)

    for ny in [0, 2, 4]:
        for nx in [0, 2, 4]:
            ax1 = axes_class(ax.figure, ax.get_position(original=True))
            locator = divider.new_locator(nx=nx, ny=ny)
            ax1.set_axes_locator(locator)
            for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
                t.set_visible(False)
            try:
                for axis in ax1.axis.values():  # ty: ignore[missing-union-attribute]
                    axis.major_ticklabels.set_visible(False)
            except AttributeError:
                pass

            ax_rgb.append(ax1)

    fig = ax.figure
    for ax1 in ax_rgb:
        fig.add_axes(ax1)

    pl = ax.imshow(ims[0], cmap="gray", vmin=0, vmax=2.2)

    spl = ax_rgb[0].imshow(ims[2][ROI_3], cmap="gray", vmin=1, vmax=2.2)
    ax_rgb[1].imshow(ims[2][ROI_4], cmap="gray", vmin=1, vmax=2.2)
    ax_rgb[2].imshow(ims[2][ROI_5], cmap="gray", vmin=1, vmax=2.2)

    spl = ax_rgb[3].imshow(ims[1][ROI_3], cmap="gray", vmin=1, vmax=2.2)
    ax_rgb[4].imshow(ims[1][ROI_4], cmap="gray", vmin=1, vmax=2.2)
    ax_rgb[5].imshow(ims[1][ROI_5], cmap="gray", vmin=1, vmax=2.2)

    spl = ax_rgb[6].imshow(ims[0][ROI_3], cmap="gray", vmin=1, vmax=2.2)
    ax_rgb[7].imshow(ims[0][ROI_4], cmap="gray", vmin=1, vmax=2.2)
    ax_rgb[8].imshow(ims[0][ROI_5], cmap="gray", vmin=1, vmax=2.2)

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
            labelsize=6,
        )
        cbbox.set_xticks([])
        cbbox.set_yticks([])
        cbbox.set_facecolor([1, 1, 1, 0.7])

        cbaxes = inset_axes(cbbox, "20%", "95%", loc=6)

        cb = fig.colorbar(pl, cax=cbaxes, format="%.1f", ticks=np.arange(0, 2.3, 0.2))
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

    for axx in ax_rgb:
        axx.spines["right"].set_visible(True)
        axx.spines["top"].set_visible(True)
        axx.set_xticks([])
        axx.set_yticks([])
        axx.set_xticklabels([])
        axx.set_yticklabels([])

        axx.spines["right"].set_color("w")
        axx.spines["left"].set_color("w")
        axx.spines["top"].set_color("w")
        axx.spines["bottom"].set_color("w")

    if plot_line_pair_labels:
        ax.text(
            0.74,
            0.37,
            "0.3 lp/mm",
            fontsize=5,
            color="black",
            weight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": [1, 1, 1, 0.5], "edgecolor": "none", "pad": 1.0},
        )
        ax.text(
            0.758,
            0.46,
            "0.4 lp/mm",
            fontsize=5,
            color="black",
            weight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": [1, 1, 1, 0.5], "edgecolor": "none", "pad": 1.0},
        )
        ax.text(
            0.748,
            0.54,
            "0.5 lp/mm",
            fontsize=5,
            color="black",
            weight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": [1, 1, 1, 0.5], "edgecolor": "none", "pad": 1.0},
        )

        # ax.text(0.02, 0.92, "[0, 2.2]", color='w', weight='bold', transform=ax.transAxes,)
        # ax_rgb[2].text(
        #     0.5, 0.05, "[1, 2.2]", fontsize=5, color="w", weight="bold", transform=ax_rgb[2].transAxes
        # )

    if label_x_axis:
        ax_rgb[0].set_xlabel("0.3")
        ax_rgb[1].set_xlabel("0.4\nSpatial frequency [lp/mm]")
        ax_rgb[2].set_xlabel("0.5")

    if left:
        ax_rgb[6].set_ylabel("5.0")
        ax_rgb[3].set_ylabel("Dose [mGy]\n1.0")
        ax_rgb[0].set_ylabel("0.2")

    if title:
        ax_rgb[6].set_title(title)

    return pl, spl


def load_im(
    setup: Literal["ideal", "realistic"],
    reg: Literal["tv", "dtv"],
    dose: float,
    n_proj: int,
) -> np.ndarray:
    """Load the central slice of the minimum-error reconstruction of the Catphan phantom.

    Args:
        setup: Dataset, ``"ideal"`` or ``"realistic"``.
        reg: Regulariser, ``"tv"`` or ``"dtv"``.
        dose: Dose in mGy.
        n_proj: Number of projections.

    Returns:
        The image.

    Raises:
        ValueError: If the regulariser is unknown.
    """
    alpha = get_ideal_alpha(f"ctp528-{setup}", dose, n_proj)
    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    if reg == "tv":
        reg_str = f"TV_iso_alpha_{alpha}"
    elif reg == "dtv":
        reg_str = f"dTV_iso_alpha_{alpha}_eta_2e-1"
    else:
        msg = f"Unknown regularisation type: {reg}. Only 'tv' and 'dtv' are supported."
        raise ValueError(msg)

    path = max(
        Path(f"{RECON_DATA_DIR}/ctp528-{setup}/{dose_str}mGy/{n_proj}proj/recon").glob(
            f"fista_exact_init_zero_880pix_{dose:.1f}mGy_{n_proj}proj_{reg_str}_iter*"
        )
    )

    return np.load(path / "processed" / "yk_min_slice.npy")


def main(
    setup: Annotated[
        Literal["ideal", "realistic"], typer.Argument(help="Setup type: 'ideal' or 'realistic'.")
    ],
) -> None:
    """Plot the insert tiles for TV and dTV with 720 and 45 projections."""
    plt.style.use(f"{REPO_ROOT}/analysis/plotting/style.txt")

    plt.rcParams["figure.figsize"] = [5.5, 5.5]
    plt.rcParams["figure.dpi"] = 300

    _, ax = plt.subplots(2, 2)

    n_projections = [720, 45]
    for i, n_proj in enumerate(n_projections):
        ax[i, 0].set_title("TV")
        ims = [
            load_im(setup, "tv", 5.0, n_proj),
            load_im(setup, "tv", 1.0, n_proj),
            load_im(setup, "tv", 0.2, n_proj),
        ]
        plot_with_closeup(
            ax[i, 0],
            ims,
            plot_line_pair_labels=False,
            left=True,
            title=f"TV, {n_proj} projections",
            label_x_axis=i == 1,
        )

        ax[i, 1].set_title("dTV")
        ims = [
            load_im(setup, "dtv", 5.0, n_proj),
            load_im(setup, "dtv", 1.0, n_proj),
            load_im(setup, "dtv", 0.2, n_proj),
        ]
        plot_with_closeup(
            ax[i, 1],
            ims,
            with_colorbar=False,
            plot_line_pair_labels=False,
            title=f"dTV, {n_proj} projections",
            label_x_axis=i == 1,
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, right=0.98, bottom=0.05)
    # plt.show()

    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(
        figures_dir / f"catphan-inserts-tiles-{setup}.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    typer.run(main)
