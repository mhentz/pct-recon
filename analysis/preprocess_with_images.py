"""Save the minimum-error iterate of a reconstruction, with its error images and summary."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer
from blosc2 import load_array, save_array
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def divnorm(e: np.ndarray, *, symmetric: bool = False) -> mpl.colors.TwoSlopeNorm:
    """Return a diverging colour normalisation centred on zero for an error image.

    Args:
        e: Error image.
        symmetric: Use the largest absolute value to set both colour limits.

    Returns:
        Two-slope colour normalisation.
    """
    if symmetric:
        return mpl.colors.TwoSlopeNorm(vmin=-np.max(abs(e)) - 1e-6, vcenter=0, vmax=np.max(abs(e)) + 1e-6)

    return mpl.colors.TwoSlopeNorm(vmin=np.min(e) - 1e-6, vcenter=0, vmax=np.max(e) + 1e-6)


def main(  # noqa: PLR0915
    setup: str,
    filepath: Path,
    ground_truth_dir: Path = Path(f"{REPO_ROOT}/phantoms/ground_truths"),
    *,
    errors_symmetric_colour_scale: bool = True,
) -> None:
    """Find the minimum-error iterate of a reconstruction and save it with its metrics.

    The minimum-error iterate is the one that minimises ||x_k - x_gt|| against the
    ground truth ``<phantom>_gt_<width>pix.npy`` in ``ground_truth_dir``. The phantom,
    initialisation, width and alpha are read from the name of the run directory written
    by ``recon/algo/fista.py``:

        `..._{phantom}_..._init_{x_init}_{image_width}pix_..._alpha_{alpha}_..._eta_{eta}_...`

    The directory of ``filepath`` must hold the two Blosc2 arrays written by FISTA:
    ``yk_final.bl2``, the iterates, of shape (n_iterations, 3, height, width), with the
    initial image at index 0; and ``metrics_final.bl2``, of shape (n_iterations, 3),
    whose columns are the cost, ||A x_k - b||^2 and alpha * regulariser(x_k).

    Written to ``processed/`` next to ``filepath``: ``error_norm.npy`` and ``.png``,
    ``min_error_index.txt``, ``min_error_norm.txt``, ``res_norm_sq.npy``, ``reg_val.npy``,
    ``yk_min_3d.npy``, ``yk_min_slice.npy`` and ``.png``, ``yk_min_slice_err.png`` (error
    of the central slice) and ``yk_slices.bl2`` (the central slice of every iterate).

    Args:
        setup: Phantom and acquisition setup, e.g. "spiral-ideal".
        filepath: Path to ``yk_final.bl2`` or ``metrics_final.bl2``.
        ground_truth_dir: Directory of the ground truth images.
        errors_symmetric_colour_scale: Whether to use a symmetric colour scale for error images.
    """
    phantom = setup.split("-", maxsplit=1)[0]

    filedir = filepath.parent
    x_init = filepath.parent.name.split("init_")[1].split("_")[0]

    outdir = filedir / "processed"
    outdir.mkdir(parents=True, exist_ok=True)

    if x_init not in ["zero", "fbp"]:
        msg = f"Unsupported value for x_init: {x_init}"
        raise ValueError(msg)

    logger.info(f"Processing file '{filepath}'")

    yk = load_array(str(filepath))

    image_w = int(filepath.parent.name.split("pix_")[0].split("_")[-1])

    if image_w not in [700, 880, 440]:
        msg = f"Unsupported image width: {image_w}"
        raise ValueError(msg)

    # grid_size = (3, image_w, image_w)
    im_gt_2d = np.load(ground_truth_dir / f"{phantom}_gt_{image_w}pix.npy").astype(np.float32)
    # im_gt = np.stack([im_gt_2d] * grid_size[0])

    error = (yk[:, 1, ...] - im_gt_2d).astype(np.float32)  # error in central slice
    error_norm = np.linalg.norm(error, axis=(1, 2))

    min_error_index = error_norm.argmin()

    if x_init == "zero":
        min_error_index = error_norm.argmin()

    elif x_init == "fbp":
        min_error_index = error_norm[1:].argmin() + 1  # account for the initial FBP image

    else:
        msg = f"Unsupported value for x_init: {x_init}"
        raise ValueError(msg)

    logger.info(f"Minimum error index: {min_error_index}")

    with (outdir / "min_error_index.txt").open("w") as f:
        f.write(f"{min_error_index}\n")

    with (outdir / "min_error_norm.txt").open("w") as f:
        f.write(f"{error_norm[min_error_index]:.3f}\n")

    yk_min = yk[min_error_index]
    yk_min_slice = yk_min[1]  # take central slice

    np.save(outdir / "yk_min_3d.npy", yk_min)
    np.save(outdir / "yk_min_slice.npy", yk_min_slice)

    metrics = load_array(str(filepath.parent / "metrics_final.bl2"))

    alpha: str = filepath.parent.name.split("alpha_")[1].split("_")[0]

    res_norm_sq = get_squared_residuals(metrics)[min_error_index - 1]
    reg_val = get_regularisation_values(metrics)[min_error_index - 1] / float(alpha)

    np.save(outdir / "res_norm_sq.npy", res_norm_sq)
    np.save(outdir / "reg_val.npy", reg_val)

    # imshow image
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(10 * cm, 8 * cm), dpi=300)
    ax.set_title(f"$x_{{{min_error_index}}}$")
    im = ax.imshow(yk_min_slice, cmap="gray")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

    plt.tight_layout(h_pad=0.1)
    plt.savefig(outdir / "yk_min_slice.png")

    # imshow error image
    fig, ax = plt.subplots(figsize=(10 * cm, 8 * cm), dpi=300)

    im = ax.imshow(
        error[min_error_index],
        cmap="seismic",
        norm=divnorm(error[min_error_index], symmetric=errors_symmetric_colour_scale),
    )
    ax.set_title(f"Error = $x_{{{min_error_index}}} - x_{{\\rm GT}}$")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

    plt.tight_layout(h_pad=0.1)
    plt.savefig(outdir / "yk_min_slice_err.png")

    # log-error
    fig, ax = plt.subplots(figsize=(10 * cm, 7 * cm), dpi=200)
    ax.semilogy(error_norm, lw=2, color="black")
    ax.set_xlabel("Iteration number")
    ax.set_ylabel(r"Error norm, $\log(\|x - x_{GT}\|)$")

    plt.tight_layout(h_pad=0.2, w_pad=0.2)
    plt.savefig(outdir / "error_norm.png")

    # np.save(outdir / "yk_min.npy", yk[error_norm.argmin()])

    np.save(outdir / "error_norm.npy", error_norm)

    logger.info(f"Save compressed array to '{outdir / 'yk_slices.npy'!s}'")
    # np.savez_compressed(filedir / "yk_slices.npy", yk=yk[:, 1, ...])

    save_array(
        np.ascontiguousarray(yk[:, 1, ...]),
        str(outdir / "yk_slices.bl2"),
        mode="w",  # ty: ignore[invalid-argument-type]
        cparams={"clevel": 5, "nthreads": 4},
    )


if __name__ == "__main__":
    typer.run(main)

    # main(
    #     setup="spiral-ideal",
    #     filepath=Path(
    #         f"{RECON_DATA_DIR}/spiral-ideal/1mGy/720proj/recon/"
    #         "fista_exact_init_zero_880pix_1.0mGy_720proj_dTV_iso_alpha_1e-3_eta_2e-1_iter200/yk_final.bl2"
    #     ),
    # )
