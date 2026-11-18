"""Plot L-curves for the three phantoms and locate their corner points."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.backend_bases import MouseEvent
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter

from analysis.L_curve.plotting import plot_curve_on_ax
from analysis.paths import RECON_DATA_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)


def exclude_alpha_from_slice(
    alphas: list[str],
    alphas_slice: slice,
    exclude_alphas: list[str],
) -> tuple[list[str], np.ndarray, slice]:
    """Exclude specific alphas from the list and adjust the slice accordingly.

    Args:
        alphas: The original list of alphas.
        alphas_slice: The original slice object.
        exclude_alphas: The list of alphas to exclude.

    Returns:
        A tuple containing the filtered list of alphas, a boolean mask indicating which alphas were kept,
        and the adjusted slice object.

    """
    # Filter out excluded alphas
    keep_mask = np.array([alpha not in exclude_alphas for alpha in alphas])
    filtered_alphas = [alpha for alpha, keep in zip(alphas, keep_mask, strict=False) if keep]

    # Adjust slice to account for removed elements
    original_indices = np.arange(len(keep_mask))
    filtered_indices = original_indices[keep_mask]

    start = alphas_slice.start if alphas_slice.start is not None else 0
    stop = alphas_slice.stop if alphas_slice.stop is not None else len(filtered_indices)

    # Handle negative indices
    if start < 0:
        start = len(filtered_indices) + start
    if stop < 0:
        stop = len(filtered_indices) + stop

    new_alphas_slice = slice(start, stop)

    return filtered_alphas, keep_mask, new_alphas_slice


def get_subset(phantom: str, type_: str, dose: float, n_proj: int) -> slice:  # noqa: C901, PLR0911, PLR0912
    """Return the slice of regularisation weights to keep for one configuration.

    Args:
        phantom: Phantom name, one of ``"ctp528"``, ``"gammex"`` or ``"spiral"``.
        type_: Setup type, ``"ideal"`` or ``"realistic"``.
        dose: Dose in mGy.
        n_proj: Number of projections.

    Returns:
        Slice selecting the alphas to keep.

    Raises:
        ValueError: If no subset is defined for the configuration.
    """
    match (phantom, type_, dose, n_proj):
        # ctp528
        case "ctp528", "ideal", 5.0, 720:
            return slice(4, -4)
        case "ctp528", "ideal", 5.0, 45:
            return slice(2, -2)
        case "ctp528", "ideal", 1.0, 720:
            return slice(3, -2)
        case "ctp528", "ideal", 1.0, 45:
            return slice(3, -2)
        case "ctp528", "ideal", 0.2, 720:
            return slice(4, -2)
        case "ctp528", "ideal", 0.2, 45:
            return slice(4, -2)

        case "ctp528", "realistic", 5.0, 720:
            return slice(2, -3)
        case "ctp528", "realistic", 5.0, 45:
            return slice(2, -3)
        case "ctp528", "realistic", 1.0, 720:
            return slice(4, -2)
        case "ctp528", "realistic", 1.0, 45:
            return slice(4, -2)
        case "ctp528", "realistic", 0.2, 720:
            return slice(5, -2)
        case "ctp528", "realistic", 0.2, 45:
            return slice(5, -2)

        # gammex
        case "gammex", "ideal", 5.0, 720:
            return slice(2, -3)
        case "gammex", "ideal", 5.0, 45:
            return slice(2, -3)
        case "gammex", "ideal", 1.0, 720:
            return slice(2, -2)
        case "gammex", "ideal", 1.0, 45:
            return slice(2, -2)
        case "gammex", "ideal", 0.2, 720:
            return slice(4, -2)
        case "gammex", "ideal", 0.2, 45:
            return slice(4, -2)

        case "gammex", "realistic", 5.0, 720:
            return slice(2, -2)
        case "gammex", "realistic", 5.0, 45:
            return slice(2, -2)
        case "gammex", "realistic", 1.0, 720:
            return slice(4, -2)
        case "gammex", "realistic", 1.0, 45:
            return slice(4, -2)
        case "gammex", "realistic", 0.2, 720:
            return slice(4, -2)
        case "gammex", "realistic", 0.2, 45:
            return slice(4, -2)

        # spiral
        case "spiral", "ideal", 5.0, 720:
            return slice(2, -2)
        case "spiral", "ideal", 5.0, 45:
            return slice(2, -2)
        case "spiral", "ideal", 1.0, 720:
            return slice(3, -2)
        case "spiral", "ideal", 1.0, 45:
            return slice(3, -2)
        case "spiral", "ideal", 0.2, 720:
            return slice(5, -2)
        case "spiral", "ideal", 0.2, 45:
            return slice(5, -2)

        case "spiral", "realistic", 5.0, 720:
            return slice(2, -2)
        case "spiral", "realistic", 5.0, 45:
            return slice(2, -2)
        case "spiral", "realistic", 1.0, 720:
            return slice(4, -2)
        case "spiral", "realistic", 1.0, 45:
            return slice(4, -2)
        case "spiral", "realistic", 0.2, 720:
            return slice(6, -2)
        case "spiral", "realistic", 0.2, 45:
            return slice(6, -2)
        case _:
            msg = f"No subset defined for {phantom}-{type_}, {dose}mGy, {n_proj}proj"
            raise ValueError(msg)


def read_data_tv(
    setup: str,
    n_proj: int,
    dose: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the squared residuals and regularisation values from the data files."""
    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"
    input_dir = Path(f"{RECON_DATA_DIR}/{setup}/{dose_str}mGy/{n_proj}proj/recon/")

    alphas = np.load(input_dir / "tv_alphas_str.npy")
    res_values = np.load(input_dir / "tv_residuals.npy")
    reg_values = np.load(input_dir / "tv_regularisation_values.npy")

    yk_min_arr = np.load(input_dir / "tv_yk_min.npy")
    yk_min_slice_arr = np.load(input_dir / "tv_yk_min_slice.npy")

    err_values = np.load(input_dir / "tv_min_error_norm_values.npy")

    return alphas, res_values, reg_values, yk_min_arr, yk_min_slice_arr, err_values


def apply_smoothing(
    index_slice: slice,
    alphas: list[str],
    res_values: np.ndarray,
    reg_values: np.ndarray,
    threshold: float = 0.4,
) -> tuple:
    """Interpolate and smooth an L-curve, then locate its corner by curvature.

    Args:
        index_slice: Slice of the alphas used for the fit.
        alphas: Regularisation weights as strings.
        res_values: Squared residual norms for each alpha.
        reg_values: Regularisation values for each alpha.
        threshold: Fraction of the maximum curvature used to detect the corner peak.

    Returns:
        Log residuals and regularisation values on the fitted subset, the smoothed
        curves, the log-alpha grid, the corner coordinates, the curvature, and the
        corner index on the grid.
    """
    xx = np.log10(res_values).astype(np.float64)
    yy = np.log10(reg_values).astype(np.float64)

    log_alpha = np.log10(np.asarray(alphas).astype(np.float64))[index_slice]

    x = xx[index_slice]
    y = yy[index_slice]

    x_pchip = PchipInterpolator(log_alpha, x)
    y_pchip = PchipInterpolator(log_alpha, y)

    n_grid = 15 * len(log_alpha)
    log_alpha_grid = np.linspace(log_alpha.min(), log_alpha.max(), num=n_grid)

    x_interp = x_pchip(log_alpha_grid)
    y_interp = y_pchip(log_alpha_grid)

    window = 15
    # window = 21
    # window = 37
    # window = 31
    poly = 3
    x_smooth = savgol_filter(x_interp, window, poly, mode="interp")
    y_smooth = savgol_filter(y_interp, window, poly, mode="interp")

    dx = np.gradient(x_smooth, log_alpha_grid)
    dy = np.gradient(y_smooth, log_alpha_grid)
    d2x = np.gradient(dx, log_alpha_grid)
    d2y = np.gradient(dy, log_alpha_grid)

    kappa = np.abs(dx * d2y - dy * d2x) / (dx * dx + dy * dy) ** 1.5

    # kappa = medfilt(kappa, kernel_size=31)

    # threshold = 0.3
    # threshold = 0.4
    idx = find_peaks(kappa, height=threshold * np.nanmax(kappa))[0][0]

    x_corner = x_smooth[idx]
    y_corner = y_smooth[idx]

    logger.info(
        f"Corner point at alpha = {10 ** log_alpha_grid[idx]:.2e}, "
        f"x(alpha) = {x_corner:.6f}, y(alpha) = {y_corner:.6f}",
    )

    # print(find_peaks(kappa)[0])
    # print([f"{log_alpha_grid[i]:.3f}" for i in find_peaks(kappa)[0]])
    # print([f"{10**log_alpha_grid[i]:.2e}" for i in find_peaks(kappa)[0]])

    n_grid, len(alphas), len(log_alpha)

    return x, y, x_smooth, y_smooth, log_alpha_grid, x_corner, y_corner, kappa, idx


reg = "TV"

# alphas = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
alphas = ["1e-4", "1e-3", "1e-2", "1e-1"]
# alphas = ["1e-3", "1e-2", "1e-1"]

phantoms = ["ctp528", "gammex", "spiral"]


@app.command()
def plot_curves(
    type_: str,
    dose: float,
    n_proj: int,
    *,
    show: Annotated[bool, typer.Option()] = False,
) -> None:
    """Plot the L-curves of the three phantoms for one dose and projection count.

    Args:
        type_: Setup type, ``"ideal"`` or ``"realistic"``.
        dose: Dose in mGy.
        n_proj: Number of projections.
        show: Show the figure interactively instead of saving it.
    """
    # setups = [f"{phantom}-{type_}" for phantom in phantoms]

    logger.info("Read CTP528 data")
    alphas_ctp, res_ctp, reg_ctp, _yk_min_ctp, _yk_min_slice_ctp, _err_ctp = read_data_tv(
        f"ctp528-{type_}",
        n_proj,
        dose,
    )
    (
        x_ctp,
        y_ctp,
        x_smooth_ctp,
        y_smooth_ctp,
        log_alpha_grid_ctp,
        x_corner_ctp,
        y_corner_ctp,
        _kappa_ctp,
        idx_ctp,
    ) = apply_smoothing(
        get_subset("ctp528", type_, dose, n_proj),
        alphas_ctp,
        res_ctp,
        reg_ctp,
        threshold=0.4,
    )

    logger.info("Read Gammex data")
    alphas_gammex, res_gammex, reg_gammex, _yk_min_gammex, _yk_min_slice_gammex, _err_gammex = read_data_tv(
        f"gammex-{type_}",
        n_proj,
        dose,
    )
    alphas_slice = get_subset("gammex", type_, dose, n_proj)
    (
        x_gammex,
        y_gammex,
        x_smooth_gammex,
        y_smooth_gammex,
        log_alpha_grid_gammex,
        x_corner_gammex,
        y_corner_gammex,
        _kappa_gammex,
        idx_gammex,
    ) = apply_smoothing(alphas_slice, alphas_gammex, res_gammex, reg_gammex, threshold=0.4)

    logger.info("Read Spiral data")
    alphas_spiral, res_spiral, reg_spiral, _yk_min_spiral, _yk_min_slice_spiral, _err_spiral = read_data_tv(
        f"spiral-{type_}",
        n_proj,
        dose,
    )
    alphas_slice = get_subset("spiral", type_, dose, n_proj)

    if type_ == "realistic" and dose == 5.0 and n_proj == 720:  # noqa: PLR2004
        alphas_spiral, keep_mask, alphas_slice = exclude_alpha_from_slice(
            alphas=alphas_spiral,
            alphas_slice=get_subset("spiral", type_, dose, n_proj),
            exclude_alphas=["2.5e-3"],
            # exclude_alphas=[],
        )
        res_spiral = res_spiral[keep_mask]
        reg_spiral = reg_spiral[keep_mask]

    (
        x_spiral,
        y_spiral,
        x_smooth_spiral,
        y_smooth_spiral,
        log_alpha_grid_spiral,
        x_corner_spiral,
        y_corner_spiral,
        _kappa_spiral,
        idx_spiral,
    ) = apply_smoothing(alphas_slice, alphas_spiral, res_spiral, reg_spiral, threshold=0.4)

    # res_dict, reg_dict, labels = read_all_data(setups, alphas, n_proj, dose, reg)

    # plt.style.use(f"{os.path.dirname(os.path.realpath(__file__))}/style.txt")
    # plt.rcParams["figure.figsize"] = [5.5, 5.5 * np.sqrt(2)]
    plt.rcParams["figure.figsize"] = [6.5, 9.0]

    fig, ax = plt.subplots(3, 2)
    # axins_list = []

    # fig.suptitle(
    #     f"L-curves for {type_.capitalize()} TV Reconstructions\n{n_proj} projections at {dose:.1f} mGy",
    #     y=0.94,
    #     fontsize=12,
    # )

    ax[0, 0].set_ylabel("$\\log\\rm(TV)$, Catphan")
    plot_curve_on_ax(
        ax[0],
        setup=f"ctp528-{type_}",
        dose=dose,
        n_projections=n_proj,
        residuals=res_ctp,
        reg_val=reg_ctp,
        x=x_ctp,
        y=y_ctp,
        labels=alphas_ctp,
        x_smooth=x_smooth_ctp,
        y_smooth=y_smooth_ctp,
        log_alpha_grid=log_alpha_grid_ctp,
        subset_slice=get_subset("ctp528", type_, dose, n_proj),
        x_corner=x_corner_ctp,
        y_corner=y_corner_ctp,
        idx=idx_ctp,
    )

    ax[1, 0].set_ylabel("$\\log\\rm(TV)$, Gammex")
    plot_curve_on_ax(
        ax[1],
        setup=f"gammex-{type_}",
        dose=dose,
        n_projections=n_proj,
        residuals=res_gammex,
        reg_val=reg_gammex,
        x=x_gammex,
        y=y_gammex,
        labels=alphas_gammex,
        x_smooth=x_smooth_gammex,
        y_smooth=y_smooth_gammex,
        log_alpha_grid=log_alpha_grid_gammex,
        subset_slice=get_subset("gammex", type_, dose, n_proj),
        x_corner=x_corner_gammex,
        y_corner=y_corner_gammex,
        idx=idx_gammex,
    )

    ax[2, 0].set_ylabel("$\\log\\rm(TV)$, Spiral")
    plot_curve_on_ax(
        ax[2],
        setup=f"spiral-{type_}",
        dose=dose,
        n_projections=n_proj,
        residuals=res_spiral,
        reg_val=reg_spiral,
        x=x_spiral,
        y=y_spiral,
        labels=alphas_spiral,
        x_smooth=x_smooth_spiral,
        y_smooth=y_smooth_spiral,
        log_alpha_grid=log_alpha_grid_spiral,
        subset_slice=get_subset("spiral", type_, dose, n_proj),
        x_corner=x_corner_spiral,
        y_corner=y_corner_spiral,
        idx=idx_spiral,
    )

    ax[2, 0].set_xlabel("$\\log\\rm(Residuals)$")
    ax[2, 1].set_xlabel("$\\log\\rm(Residuals)$")

    # ax[1].set_ylabel("Gammex, TV(x)")
    # plot_curve_on_ax(
    #     ax[1],
    #     res_gammex,
    #     reg_gammex,
    #     labels=alphas_gammex,
    #     # label_pos=label_pos[f"gammex-{type_}"][f"720-{dose:.1f}"],
    #     highlight_index=2,
    # )

    # ax[2].set_xlabel(f"Squared residuals, $||Ax - b||^2$")
    # ax[2].set_ylabel("Spiral, TV(x)")
    # plot_curve_on_ax(
    #     ax[2],
    #     res_spiral,
    #     reg_spiral,
    #     labels=alphas_spiral,
    #     # label_pos=label_pos[f"spiral-{type_}"][f"720-{dose:.1f}"],
    #     highlight_index=2,
    # )

    # plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    for axes in ax.flatten():
        # axes.xaxis.major.formatter._useMathText = True

        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

    xy = []

    def onclick(event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            msg = "Click was outside the axes"
            raise ValueError(msg)
        xy.append((int(event.xdata), int(event.ydata)))
        print(xy)

    fig.canvas.mpl_connect("button_press_event", onclick)

    if show:
        plt.show()
    else:
        # plt.subplots_adjust(hspace=0.3)
        figures_dir = Path(__file__).parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        plt.savefig(
            figures_dir / f"L-curves-{type_}-{dose}mGy-{n_proj}proj.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    # run(plot_curves, pretty_exceptions_show_locals=False)
    app()
