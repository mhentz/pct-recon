"""Flat-region analysis comparing TV and dTV reconstructions.

Two subcommands:
    overlay: GT images with flat-ROI rectangles
    plot: tiles, profiles, and error maps
"""

from pathlib import Path
from typing import Annotated, Literal

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line

from analysis.flat_regions.constants import PHANTOMS
from analysis.ideal_alpha import get_ideal_alpha
from analysis.plotting.utils import divnorm, get_path, load_image

plt.style.use(str(Path(__file__).resolve().parent.parent / "style.txt"))

# RdBu_r midpoint is ~(0.97, 0.97, 0.96), not pure white.
# Force the centre to (1, 1, 1) so that values mapping to 0.5 render as white.
_N = 256
_base_vals = plt.get_cmap("RdBu_r")(_lut := np.linspace(0, 1, _N))
_mid = _N // 2
_base_vals[_mid - 1 : _mid + 2] = [1.0, 1.0, 1.0, 1.0]
RDBU_WHITE = mcolors.LinearSegmentedColormap.from_list("RdBu_r_white", _base_vals, N=_N)

FIGURES_DIR = Path(__file__).parent / "figures"

TITLE_FONTSIZE = 20

app = typer.Typer(pretty_exceptions_show_locals=False)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save fig to the figures directory next to this script."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    typer.echo(f"Saved {out}")


def load_tv_dtv(
    setup_prefix: str,
    setup: str,
    dose: float,
    n_proj: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load TV and dTV reconstructions for a given phantom configuration."""
    alpha = get_ideal_alpha(f"{setup_prefix}-{setup}", dose, n_proj)
    full_setup = f"{setup_prefix}-{setup}"

    x_tv, _ = load_image(
        get_path(setup=full_setup, dose=dose, n_projections=n_proj, alpha=alpha, regulariser="TV")
    )
    x_dtv, _ = load_image(
        get_path(setup=full_setup, dose=dose, n_projections=n_proj, alpha=alpha, regulariser="dTV")
    )
    return x_tv, x_dtv


@app.command()
def overlay() -> None:
    """Plot each ground truth image with the flat ROI highlighted."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    for ax, ph in zip(axes, PHANTOMS, strict=False):
        gt = np.load(ph["gt_path"])
        roi = ph["roi"]

        ax.imshow(gt, cmap="gray", vmin=0.0, vmax=1.5)
        ax.set_title(ph["name"], fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

        y0, y1 = roi[0].start, roi[0].stop
        x0, x1 = roi[1].start, roi[1].stop
        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            boxstyle="round,pad=0",
        )
        ax.add_patch(rect)

    save_figure(fig, "gt-roi-overlay.png")
    plt.close(fig)


def _plot_tiles(setup: str, dose: float, n_proj: int) -> None:
    """Compare zoomed side-by-side plots of flat regions: ground truth / TV / dTV."""
    fig, axes = plt.subplots(
        3,
        # 3,
        2,
        # figsize=(8, 7),
        figsize=(6, 7),
        constrained_layout=True,
    )

    # col_titles = [r"$x_{\mathrm{ROI,GT}}$", r"$x_{\mathrm{ROI,TV}}$", r"$x_{\mathrm{ROI,dTV}}$"]
    col_titles = [r"$x_{\mathrm{ROI,TV}}$", r"$x_{\mathrm{ROI,dTV}}$"]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=TITLE_FONTSIZE)

    for row, ph in enumerate(PHANTOMS):
        # gt = np.load(ph["gt_path"])
        x_tv, x_dtv = load_tv_dtv(ph["setup_prefix"], setup, dose, n_proj)

        roi = ph["roi"]
        # crops = [gt[roi], x_tv[roi], x_dtv[roi]]
        crops = [x_tv[roi], x_dtv[roi]]

        # Symmetric window centred on 1.0 so GT is always mid-tone and
        # deviations above / below are equally visible.
        # recon_vals = np.concatenate([crops[1].ravel(), crops[2].ravel()])
        recon_vals = np.concatenate([crops[0].ravel(), crops[1].ravel()])
        half_w = float(np.max(np.abs(recon_vals - 1.0)))
        half_w = max(half_w * 1.3, 1e-4)  # 30 % padding, minimum guard
        vmin, vmax = 1.0 - half_w, 1.0 + half_w
        norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)

        axes[row, 0].set_ylabel(ph["name"], fontsize=TITLE_FONTSIZE)
        axes[row, 0].yaxis.set_label_coords(-0.22, 0.5)

        for col, img in enumerate(crops):
            im = axes[row, col].imshow(
                img,
                cmap=RDBU_WHITE,
                norm=norm,
                extent=[0, img.shape[1], img.shape[0], 0],
            )
            ax = axes[row, col]
            h, w = img.shape

            ax.set_xticks([0, w // 2, w])
            ax.set_xticklabels([1, w // 2, w])
            ax.set_xticks([w // 4, 3 * w // 4], minor=True)

            ax.set_yticks([0, h // 2, h])
            ax.set_yticklabels([1, h // 2, h] if col == 0 else [""] * 3)
            ax.set_yticks([h // 4, 3 * h // 4], minor=True)

            ax.tick_params(which="both", direction="in", right=True)
            ax.tick_params(which="major", direction="in", top=True)

            # Annotate mean ± std (skip GT since it is exactly 1.0)
            # if col > 0:
            if col >= 0:
                mu = float(np.mean(img))
                sigma = float(np.std(img))
                axes[row, col].text(
                    0.03,
                    0.97,
                    f"$\\mu$={mu:.5f}\n$\\sigma$={sigma:.1e}",
                    fontsize=11,
                    transform=axes[row, col].transAxes,
                    verticalalignment="top",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2.0},
                )

        # Single colorbar at the end of each row
        cb = fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.02, format="%.4f")
        cb.outline.set_visible(False)
        if row == 0:
            title_pad = plt.rcParams.get("axes.titlepad", 6.0)
            offset = ScaledTranslation(0, title_pad / 72, fig.dpi_scale_trans)
            trans = blended_transform_factory(cb.ax.transAxes, axes[0, 0].transAxes) + offset
            fig.text(
                0.0,
                1.0,
                "RSP",
                transform=trans,
                fontsize=TITLE_FONTSIZE,
                ha="left",
                va="baseline",
            )

    save_figure(fig, f"flat-region-tiles-{setup}-{dose:.1f}mGy-{n_proj}proj.png")
    plt.close(fig)


def _plot_profiles(setup: str, dose: float, n_proj: int) -> None:
    """Plot line profiles through flat regions: ground truth / TV / dTV."""
    fig, ax = plt.subplots(3, 1, figsize=(5.5, 7), sharex=False, constrained_layout=True)

    for row, ph in enumerate(PHANTOMS):
        gt = np.load(ph["gt_path"])
        x_tv, x_dtv = load_tv_dtv(ph["setup_prefix"], setup, dose, n_proj)

        roi = ph["roi"]
        start = ph["profile_start"]
        end = ph["profile_end"]

        pr_gt = profile_line(gt[roi], start, end, order=1, linewidth=1)
        pr_tv = profile_line(x_tv[roi], start, end, order=1, linewidth=1)
        pr_dtv = profile_line(x_dtv[roi], start, end, order=1, linewidth=1)

        ax[row].plot(pr_gt, lw=1.5, ls="dashed", label="GT", color="black")
        ax[row].plot(pr_tv, lw=2, ls="dotted", label="TV", color="black")
        ax[row].plot(pr_dtv, lw=2, label="dTV", color="black")

        ax[row].set_ylabel("RSP")
        ax[row].spines["top"].set_visible(False)
        ax[row].spines["right"].set_visible(False)

        ax[row].text(
            0.03,
            0.93,
            ph["name"],
            color="black",
            weight="bold",
            transform=ax[row].transAxes,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.0},
        )

    ax[0].legend(loc="upper right")
    ax[2].set_xlabel("Distance [pixels]")

    save_figure(fig, f"flat-profiles-{setup}-{dose:.1f}mGy-{n_proj}proj.png")
    plt.close(fig)


def _plot_errors(setup: str, dose: float, n_proj: int) -> None:
    """Error maps (image - ground truth) in flat regions."""
    fig, ax = plt.subplots(3, 2, figsize=(6, 7))

    ax[0, 0].set_title("TV $-$ GT", fontsize=20)
    ax[0, 1].set_title("dTV $-$ GT", fontsize=20)

    for row, ph in enumerate(PHANTOMS):
        gt = np.load(ph["gt_path"])
        x_tv, x_dtv = load_tv_dtv(ph["setup_prefix"], setup, dose, n_proj)

        roi = ph["roi"]
        err_tv = x_tv[roi] - gt[roi]
        err_dtv = x_dtv[roi] - gt[roi]

        for col, (err, _) in enumerate([(err_tv, "TV"), (err_dtv, "dTV")]):
            max_val = max(np.max(np.abs(err_tv)), np.max(np.abs(err_dtv)))
            im = ax[row, col].imshow(err, cmap="seismic", norm=divnorm(max_val))
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

            # Colorbar
            divider = make_axes_locatable(ax[row, col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical", format="%.3f")

            # Annotate mean and std
            mu = np.mean(err)
            sigma = np.std(err)
            ax[row, col].text(
                0.03,
                0.93,
                f"$\\mu$={mu:.4f}\n$\\sigma$={sigma:.4f}",
                color="black",
                fontsize=7,
                transform=ax[row, col].transAxes,
                verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.0},
            )

        ax[row, 0].set_ylabel(ph["name"], fontsize=20)

    plt.tight_layout()
    save_figure(fig, f"flat-errors-{setup}-{dose:.1f}mGy-{n_proj}proj.png")
    plt.close(fig)


@app.command()
def plot(
    setup: Annotated[
        Literal["ideal", "realistic"],
        typer.Argument(help="Phantom setup (ideal or realistic)."),
    ],
    dose: Annotated[float, typer.Argument(help="Dose level in mGy.")],
    n_proj: Annotated[Literal[720, 45], typer.Argument(help="Number of projections.")],
) -> None:
    """Generate flat-region tile figures."""
    _plot_tiles(setup, dose, n_proj)


@app.command()
def extras(
    setup: Annotated[
        Literal["ideal", "realistic"],
        typer.Argument(help="Phantom setup (ideal or realistic)."),
    ],
    dose: Annotated[float, typer.Argument(help="Dose level in mGy.")],
    n_proj: Annotated[Literal[720, 45], typer.Argument(help="Number of projections.")],
) -> None:
    """Generate flat-region profiles and error maps."""
    _plot_profiles(setup, dose, n_proj)
    _plot_errors(setup, dose, n_proj)


if __name__ == "__main__":
    app()
