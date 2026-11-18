"""Select the minimum-error TV iterate for each regularisation weight and save the results."""

from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from blosc2 import load_array
from click import Choice
from loguru import logger
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.paths import RECON_DATA_DIR


def divnorm(e):  # noqa: ANN001, ANN201
    """Return a diverging colour normalisation centred on zero for an error image.

    Args:
        e: Error image.

    Returns:
        Two-slope colour normalisation spanning the range of ``e``.
    """
    return colors.TwoSlopeNorm(vmin=np.min(e) - 1e-6, vcenter=0, vmax=np.max(e) + 1e-6)


class Setup(StrEnum):
    """Phantom and setup combinations accepted by the script."""

    CTP528_IDEAL = "ctp528-ideal"
    CTP528_REALISTIC = "ctp528-realistic"
    GAMMEX_IDEAL = "gammex-ideal"
    GAMMEX_REALISTIC = "gammex-realistic"
    SPIRAL_IDEAL = "spiral-ideal"
    SPIRAL_REALISTIC = "spiral-realistic"


class InitType(StrEnum):
    """Initial estimates of the iterative reconstruction."""

    ZERO = "zero"
    FBP = "fbp"


class NoOfProjections(int, Enum):
    """Numbers of projections for which reconstructions exist."""

    N15 = 15
    N45 = 45
    N720 = 720


class Dose(StrEnum):
    """Doses in mGy for which reconstructions exist."""

    D5 = "5"
    D1 = "1"
    D0_2 = "0.2"


def get_regularisation_values(metrics: np.ndarray) -> np.ndarray:
    """Return the regularisation term from each row of a metrics array.

    Args:
        metrics: Metrics array with one row per iteration.

    Returns:
        Last column of the metrics array.
    """
    return metrics[:, -1]


def get_squared_residuals(metrics: np.ndarray) -> np.ndarray:
    """Return the squared residual norm from each row of a metrics array.

    Args:
        metrics: Metrics array with one row per iteration.

    Returns:
        Second column of the metrics array.
    """
    return metrics[:, 1]


def main(  # noqa: PLR0915
    setup: Annotated[
        str,
        typer.Argument(
            ...,
            click_type=Choice([e.value for e in Setup], case_sensitive=True),
            help="Which setup to pick",
        ),
    ],
    x_init: Annotated[
        str,
        typer.Argument(
            ...,
            click_type=Choice([e.value for e in InitType], case_sensitive=True),
            help="Which init type to pick",
        ),
    ],
    n_projections: Annotated[
        int,
        typer.Argument(
            ...,
            click_type=Choice([e.value for e in NoOfProjections], case_sensitive=False),
            help="Number of projections to use",
        ),
    ],
    dose: Annotated[
        str,
        typer.Argument(
            ...,
            click_type=Choice([e.value for e in Dose], case_sensitive=False),
            help="Dose in mGy to use",
        ),
    ],
) -> None:
    """Find the minimum-error iterate of each regularisation weight and save the results.

    Args:
        setup: Phantom and setup name, e.g. ``"ctp528-ideal"``.
        x_init: Initial estimate, ``"zero"`` or ``"fbp"``.
        n_projections: Number of projections.
        dose: Dose in mGy.
    """
    logger.info(f"Processing {setup} at {dose} mGy with x_init {x_init} and {n_projections} n_projections")

    image_w = 880 if setup.split("-")[0] != "gammex" else 700

    dose = float(dose)  # ty: ignore[invalid-assignment]
    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    grid_size = (3, image_w, image_w)
    im_gt_2d = np.load(
        Path(__file__).parent.parent / f"phantoms/ground_truths/{setup.split('-')[0]}_gt_{image_w}pix.npy",
    ).astype(
        np.float32,  # to match reconstructed images
    )
    im_gt = np.stack([im_gt_2d] * grid_size[0])

    alphas_strings = [
        "1e-5",
        "2.5e-5",
        "5e-5",
        "1e-4",
        "2.5e-4",
        "5e-4",
        "7.1e-4",
        "1e-3",
        "1.58e-3",
        "2.5e-3",
        "3.54e-3",
        "5e-3",
        "1e-2",
        "0.0224",
        "2.24e-2",
        "2.5e-2",
        "5e-2",
        "1e-1",
        "5e-1",
    ]

    alphas = []

    # error_images = []
    error_norm_values = []
    min_error_norm_values = []
    min_error_index_values = []
    min_error_images = []
    yk_min_list = []
    yk_min_slice_list = []

    residuals = []
    regularisation_values = []

    for alpha in alphas_strings:
        logger.info(f"Processing alpha: {alpha}")

        filename = (
            f"{RECON_DATA_DIR}/{setup}/{dose_str}mGy/{n_projections}proj/recon/"
            f"fista_exact_init_{x_init}_{image_w}pix_{dose:.1f}mGy_{n_projections}proj_TV_iso_alpha_{alpha}_iter200/yk_final.bl2"
        )

        if not Path(filename).exists():
            logger.warning(f"File for {alpha} does not exist, skipping.")
            continue

        alphas.append(alpha)

        # if (Path(filename).parent / "yk_min.npy").exists():
        #     logger.warning(f"Already processed {alpha}, skipping.")
        #     continue

        yk = load_array(filename)

        error = (yk[:, 1, ...] - im_gt[0]).astype(np.float32)  # error in central slice
        error_norm: np.ndarray = np.linalg.norm(error, axis=(1, 2))

        if x_init == "zero":
            min_error_index = error_norm.argmin()

        elif x_init == "fbp":
            min_error_index = error_norm[1:].argmin() + 1  # account for the initial FBP image

        else:
            msg = f"Unsupported value for x_init: {x_init}"
            raise ValueError(msg)

        logger.info(f"Minimum error index: {min_error_index}")

        yk_min = yk[min_error_index]
        yk_min_slice = yk_min[1]  # take central slice

        np.save(Path(filename).parent / "yk_min.npy", yk_min)
        np.save(Path(filename).parent / "yk_min_slice.npy", yk_min_slice)

        # error_images.append(error)  # very large
        error_norm_values.append(error_norm)
        min_error_norm_values.append(error_norm[min_error_index])
        min_error_index_values.append(min_error_index)
        min_error_images.append(error[min_error_index])
        yk_min_list.append(yk_min)
        yk_min_slice_list.append(yk_min_slice)

        metrics = load_array(
            f"{RECON_DATA_DIR}/{setup}/{dose_str}mGy/{n_projections}proj/recon/"
            f"fista_exact_init_{x_init}_{image_w}pix_{dose:.1f}mGy_{n_projections}proj_TV_iso_alpha_{alpha}_iter200/metrics_final.bl2",
        )

        res_norm_sq = get_squared_residuals(metrics)[min_error_index - 1]
        reg_val = get_regularisation_values(metrics)[min_error_index - 1] / float(alpha)

        residuals.append(res_norm_sq)
        regularisation_values.append(reg_val)

        # imshow image
        cm = 1 / 2.54  # centimeters in inches
        fig, ax = plt.subplots(figsize=(10 * cm, 8 * cm), dpi=300)
        ax.set_title(f"$x_{{{min_error_index}}}$")
        im = ax.imshow(yk_min_slice, cmap="gray")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

        plt.tight_layout(h_pad=0.1)
        plt.savefig(Path(filename).parent / "yk_min.png")

        plt.close(fig)

        # imshow error image
        fig, ax = plt.subplots(figsize=(10 * cm, 8 * cm), dpi=300)

        im = ax.imshow(error[min_error_index], cmap="seismic", norm=divnorm(error[min_error_index]))
        ax.set_title(f"Error = $x_{{{min_error_index}}} - x_{{\\rm GT}}$")
        # ax.set_xlim(0, 700)
        # ax.set_xlim(0, 700)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

        plt.tight_layout(h_pad=0.1)
        plt.savefig(Path(filename).parent / "yk_min_err.png")

        plt.close(fig)

    output_dir = Path(f"{RECON_DATA_DIR}/{setup}/{dose_str}mGy/{n_projections}proj/recon")

    np.save(output_dir / "tv_alphas_str.npy", alphas)
    np.save(output_dir / "tv_residuals.npy", residuals)
    np.save(output_dir / "tv_regularisation_values.npy", regularisation_values)

    # np.save(output_dir / "tv_error_images.npy", np.concatenate(np.array(error_images)[np.newaxis], axis=0))
    np.save(
        output_dir / "tv_error_norm_values.npy",
        np.concatenate(np.array(error_norm_values)[np.newaxis], axis=0),
    )
    np.save(output_dir / "tv_min_error_norm_values.npy", min_error_norm_values)
    np.save(output_dir / "tv_min_error_index_values.npy", min_error_index_values)
    np.save(
        output_dir / "tv_min_error_images.npy",
        np.concatenate(np.array(min_error_images)[np.newaxis], axis=0),
    )
    np.save(output_dir / "tv_yk_min.npy", np.concatenate(np.array(yk_min_list)[np.newaxis], axis=0))
    np.save(
        output_dir / "tv_yk_min_slice.npy",
        np.concatenate(np.array(yk_min_slice_list)[np.newaxis], axis=0),
    )


if __name__ == "__main__":
    # setup = "ctp528-ideal"
    # setup = "gammex-ideal"
    # setup = "spiral-ideal"

    # setup = "ctp528-realistic"
    setup = "gammex-realistic"
    # setup = "spiral-realistic"

    n_projections = 720
    dose = 5

    # main(setup=setup, x_init="zero", n_projections=n_projections, dose=dose)

    typer.run(main)
