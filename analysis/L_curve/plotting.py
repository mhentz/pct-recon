"""Plotting helpers for L-curves, including axis limits and annotated alpha labels."""

# ruff: noqa: PLR2004
from collections.abc import Sequence

import numpy as np
from loguru import logger


def get_lims(  # noqa: C901, PLR0912, PLR0915
    setup: str,
    dose: float,
    n_projections: int,
) -> tuple[float, float, float, float]:
    """Return the axis limits of the L-curve panel for one setup and dose.

    Args:
        setup: Setup name, e.g. ``"ctp528-ideal"``.
        dose: Dose in mGy.
        n_projections: Number of projections.

    Returns:
        Lower and upper x-limits, followed by lower and upper y-limits.
    """
    match setup, dose, n_projections:
        case "ctp528-ideal", 5, 720:
            xmin = 8.056
            xmax = 8.085
            ymin = 4.025
            ymax = 4.2
        case "ctp528-ideal", 5, 45:
            xmin = 8.08
            xmax = 8.097
            ymin = 4.025
            ymax = 4.2
        case "ctp528-ideal", 1, 720:
            xmin = 7.355
            xmax = 7.38
            ymin = 4.025
            ymax = 4.2
        case "ctp528-ideal", 1, 45:
            xmin = 7.382
            xmax = 7.405
            ymin = 4.02
            ymax = 4.16
        case "ctp528-ideal", 0.2, 720:
            xmin = 6.65
            xmax = 6.69
            ymin = 4.02
            ymax = 4.18
        case "ctp528-ideal", 0.2, 45:
            xmin = 6.684
            xmax = 6.71
            ymin = 4.02
            ymax = 4.18
        case "gammex-ideal", 5, 720:
            xmin = 8.4755
            xmax = 8.478
            ymin = 3.92
            ymax = 3.945
        case "gammex-ideal", 5, 45:
            xmin = 8.4788
            xmax = 8.481
            ymin = 3.9225
            ymax = 3.940
        case "gammex-ideal", 1, 720:
            xmin = 7.775
            xmax = 7.783
            ymin = 3.917
            ymax = 3.97
        case "gammex-ideal", 1, 45:
            xmin = 7.777
            xmax = 7.785
            ymin = 3.915
            ymax = 3.97
        case "gammex-ideal", 0.2, 720:
            xmin = 7.072
            xmax = 7.082
            ymin = 3.915
            ymax = 3.96
        case "gammex-ideal", 0.2, 45:
            xmin = 7.076
            xmax = 7.086
            ymin = 3.915
            ymax = 3.96
        case "spiral-ideal", 5, 720:
            xmin = 8.095
            xmax = 8.103
            ymin = 4.25
            ymax = 4.3
        case "spiral-ideal", 5, 45:
            xmin = 8.113
            xmax = 8.121
            ymin = 4.25
            ymax = 4.3
        case "spiral-ideal", 1, 720:
            xmin = 7.391
            xmax = 7.401
            ymin = 4.25
            ymax = 4.31
        case "spiral-ideal", 1, 45:
            xmin = 7.41
            xmax = 7.42
            ymin = 4.25
            ymax = 4.31
        case "spiral-ideal", 0.2, 720:
            xmin = 6.675
            xmax = 6.70
            ymin = 4.24
            ymax = 4.35
        case "spiral-ideal", 0.2, 45:
            xmin = 6.695
            xmax = 6.73
            ymin = 4.24
            ymax = 4.35
        case "ctp528-realistic", 5, 720:
            xmin = 8.2435
            xmax = 8.2495
            ymin = 4.04
            ymax = 4.11
        case "ctp528-realistic", 5, 45:
            xmin = 8.246
            xmax = 8.2515
            ymin = 4.04
            ymax = 4.11
        case "ctp528-realistic", 1, 720:
            xmin = 7.543
            xmax = 7.557
            ymin = 4.015
            ymax = 4.125
        case "ctp528-realistic", 1, 45:
            xmin = 7.545
            xmax = 7.56
            ymin = 4.015
            ymax = 4.125
        case "ctp528-realistic", 0.2, 720:
            xmin = 6.835
            xmax = 6.853
            ymin = 4.015
            ymax = 4.125
        case "ctp528-realistic", 0.2, 45:
            xmin = 6.84
            xmax = 6.859
            ymin = 4.015
            ymax = 4.125
        case "gammex-realistic", 5, 720:
            xmin = 8.62
            xmax = 8.624
            ymin = 3.915
            ymax = 3.945
        case "gammex-realistic", 5, 45:
            xmin = 8.62
            xmax = 8.626
            ymin = 3.915
            ymax = 3.95
        case "gammex-realistic", 5, 15:
            xmin = 8.61
            xmax = 8.626
            ymin = 3.915
            ymax = 3.95
        case "gammex-realistic", 1, 720:
            xmin = 7.919
            xmax = 7.926
            ymin = 3.915
            ymax = 3.945
        case "gammex-realistic", 1, 45:
            xmin = 7.921
            xmax = 7.9265
            ymin = 3.915
            ymax = 3.945
        case "gammex-realistic", 0.2, 720:
            xmin = 7.216
            xmax = 7.226
            ymin = 3.915
            ymax = 3.97
        case "gammex-realistic", 0.2, 45:
            xmin = 7.217
            xmax = 7.227
            ymin = 3.915
            ymax = 3.97
        case "spiral-realistic", 5, 720:
            xmin = 8.339
            xmax = 8.3414
            ymin = 4.25
            ymax = 4.285
        case "spiral-realistic", 5, 45:
            xmin = 8.339
            xmax = 8.3414
            ymin = 4.255
            ymax = 4.29
        case "spiral-realistic", 1, 720:
            xmin = 7.636
            xmax = 7.644
            ymin = 4.25
            ymax = 4.315
        case "spiral-realistic", 1, 45:
            xmin = 7.637
            xmax = 7.644
            ymin = 4.25
            ymax = 4.30
        case "spiral-realistic", 0.2, 720:
            xmin = 6.93
            xmax = 6.95
            ymin = 4.25
            ymax = 4.30
        case "spiral-realistic", 0.2, 45:
            xmin = 6.93
            xmax = 6.95
            ymin = 4.25
            ymax = 4.30

        case _:
            msg = f"Unsupported setup, dose, or number of projections: {setup}, {dose}, {n_projections}"
            raise ValueError(
                msg,
            )
    return xmin, xmax, ymin, ymax


def plot_curve_on_ax(  # noqa: C901, PLR0912, PLR0915
    ax: np.ndarray,
    setup: str,
    dose: float,
    n_projections: int,
    residuals: np.ndarray,
    reg_val: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    x_smooth: np.ndarray,
    y_smooth: np.ndarray,
    log_alpha_grid: np.ndarray,
    subset_slice: slice,
    x_corner: float,
    y_corner: float,
    idx: int,
) -> None:
    """Plot the raw and smoothed L-curves of one setup on a pair of axes.

    Args:
        ax: Two axes; the first shows the raw L-curve, the second the smoothed fit.
        setup: Setup name, e.g. ``"ctp528-ideal"``.
        dose: Dose in mGy.
        n_projections: Number of projections.
        residuals: Squared residual norms for each alpha.
        reg_val: Regularisation values for each alpha.
        x: Log residuals on the fitted subset.
        y: Log regularisation values on the fitted subset.
        labels: Regularisation weight labels to annotate.
        x_smooth: Smoothed log residuals on the alpha grid.
        y_smooth: Smoothed log regularisation values on the alpha grid.
        log_alpha_grid: Log10 of the alpha grid.
        subset_slice: Slice of the alphas that were fitted.
        x_corner: Smoothed log residual at the corner.
        y_corner: Smoothed log regularisation value at the corner.
        idx: Index of the corner on the alpha grid.
    """
    # ax.semilogy(residuals, reg_val, color="0.7", lw=2)/
    # ax.semilogy(residuals, reg_val, ".", markeredgecolor="0.5", markerfacecolor="0.5", markersize=10)

    # if highlight_index:
    #     ax.semilogy(
    #         residuals[highlight_index], reg_val[highlight_index], ".", color="black", alpha=1, markersize=10
    #     )

    # if index_slice is not None:
    #     px = residuals[index_slice]
    #     py = reg_val[index_slice]
    # else:
    #     px = residuals
    #     py = reg_val

    x0 = np.log10(residuals)
    y0 = np.log10(reg_val)

    ax[0].scatter(x0, y0, color="black", s=10, label="Sample")
    # # ax[0].scatter(px, py, color="black", s=10, label="Sample")
    # ax[0].plot(x_smooth, y_smooth, "-r", lw=0.5, label="Smoothed")
    ax[0].plot(x0, y0, "-", color="black", lw=1, label="L-curve")
    ax[0].legend(fontsize=7, loc="upper center")

    # x = residuals[subset_slice]
    # y = reg_val[subset_slice]

    # print(x[subset_slice])
    ax[1].scatter(x, y, color="black", s=15, label="Sample")
    ax[1].plot(x_smooth, y_smooth, "-", color="black", lw=1, label="Smoothed L-curve")
    ax[1].scatter(
        x_corner,
        y_corner,
        s=25,
        # s=50,
        facecolors="none",
        edgecolors="black",
        label=f"Corner point: $\\alpha=${10 ** log_alpha_grid[idx]:.2e}",
    )

    ax[1].legend(fontsize=7, loc="upper right")

    xmin, xmax, ymin, ymax = get_lims(setup=setup, dose=dose, n_projections=n_projections)

    for i, label in enumerate(labels[subset_slice]):
        # if x[i] >= xmin and x[i] <= xmax and y[i] >= ymin and y[i] <= ymax:

        if setup == "ctp528-ideal":
            if dose == 0.2 and n_projections in [720, 45]:
                ax[1].text(x[i] + 0.0007, y[i] + 0.0015, label, fontsize=6)
            else:
                ax[1].text(x[i] + 0.0005, y[i] + 0.001, label, fontsize=7)

        elif setup == "ctp528-realistic":
            if dose == 5 and n_projections == 720:
                ax[1].text(x[i] + 0.0001, y[i] + 0.0001, label, fontsize=7)

            elif (dose == 1 and n_projections == 720) or (dose == 1 and n_projections == 45):
                ax[1].text(x[i] + 0.0003, y[i] + 0.0001, label, fontsize=7)

            elif dose == 0.2 and n_projections == 720:
                ax[1].text(x[i] + 0.001, y[i] + 0.0004, label, fontsize=7)

            elif dose == 0.2 and n_projections == 45:
                ax[1].text(x[i] + 0.001, y[i] + 0.0008, label, fontsize=7)

            else:
                ax[1].text(x[i] + 0.0001, y[i] + 0.0001, label, fontsize=7)

        elif setup == "gammex-ideal":
            if dose == 5 and n_projections == 720:
                # ax[1].text(x[i] + 0.00008, y[i] + 0.0001, label, fontsize=7)
                ax[1].text(x[i] + 0.00005, y[i] + 0.0001, label, fontsize=7)

            elif dose == 5 and n_projections == 45:
                ax[1].text(x[i] + 0.00005, y[i] + 0.0001, label, fontsize=7)

            elif dose == 1 and n_projections == 720:
                ax[1].text(x[i] + 0.0003, y[i] + 0.0004, label, fontsize=6)

            elif dose == 1 and n_projections == 45:
                ax[1].text(x[i] + 0.0003, y[i] + 0.0001, label, fontsize=6)

            elif dose == 0.2 and n_projections == 720:
                ax[1].text(x[i] + 0.0004, y[i] + 0.0001, label, fontsize=7)

            elif dose == 0.2 and n_projections == 45:
                ax[1].text(x[i] + 0.0003, y[i] + 0.0001, label, fontsize=7)

        elif setup == "gammex-realistic":
            if (dose == 5 and n_projections == 720) or (dose == 5 and n_projections == 45):
                ax[1].text(x[i] + 0.00007, y[i] + 0.0005, label, fontsize=6)

            elif (dose == 1 and n_projections == 720) or (dose == 1 and n_projections == 45):
                ax[1].text(x[i] + 0.0001, y[i] + 0.0005, label, fontsize=7)

            elif dose == 0.2 and n_projections == 720:
                ax[1].text(x[i] + 0.0002, y[i] + 0.0005, label, fontsize=7)

            elif dose == 0.2 and n_projections == 45:
                ax[1].text(x[i] + 0.0002, y[i] + 0.0003, label, fontsize=7)

            else:
                ax[1].text(x[i] + 0.0005, y[i] + 0.0005, label, fontsize=7)

        elif setup == "spiral-ideal":
            if dose == 5 and n_projections == 720:
                ax[1].text(x[i] + 0.0002, y[i] + 0.0005, label, fontsize=6)

            elif dose == 5 and n_projections == 45:
                ax[1].text(x[i] + 0.0001, y[i] + 0.0005, label, fontsize=6)

            elif dose == 1 and n_projections == 720:
                ax[1].text(x[i] + 0.0002, y[i] + 0.0005, label, fontsize=7)

            elif dose == 1 and n_projections == 45:
                ax[1].text(x[i] + 0.0002, y[i] + 0.0006, label, fontsize=7)

            elif dose == 0.2 and n_projections == 720:
                # ax[1].text(x[i] + 0.003, y[i] + 0.0005, label, fontsize=6)
                ax[1].text(x[i] + 0.001, y[i] + 0.0005, label, fontsize=7)

            elif dose == 0.2 and n_projections == 45:
                ax[1].text(x[i] + 0.001, y[i] + 0.0005, label, fontsize=7)

        elif setup == "spiral-realistic":
            if dose == 5 and n_projections == 720:
                ax[1].text(x[i] + 0.0001, y[i] + 0.0001, label, fontsize=6)

            elif dose == 5 and n_projections == 45:
                ax[1].text(x[i] + 0.00015, y[i] + 0.0002, label, fontsize=5.5)

            elif dose == 1 and n_projections == 720:
                ax[1].text(x[i] + 0.0002, y[i] + 0.0005, label, fontsize=7)

            elif dose == 1 and n_projections == 45:
                ax[1].text(x[i] + 0.0001, y[i] + 0.0005, label, fontsize=7)

            elif dose == 0.2 and n_projections == 720:
                # ax[1].text(x[i] + 0.001, y[i] + 0.003, label, fontsize=6)
                ax[1].text(x[i] + 0.0005, y[i] + 0.0005, label, fontsize=7)

            elif dose == 0.2 and n_projections == 45:
                ax[1].text(x[i] + 0.0005, y[i] + 0.0005, label, fontsize=7)

        else:
            ax[1].text(x[i] + 0.0005, y[i] + 0.0005, label, fontsize=7)

        # else:
        #     raise ValueError(f"Unsupported setup: {setup}")

        # ax[1].text(x[i], y[i], label, fontsize=7)

    # if setup == "spiral-ideal":
    #     print(subset_slice)
    #     print(labels)
    #     print(labels[subset_slice])

    # print(setup)
    # print(subset_slice)
    # print(labels)
    logger.debug(setup, labels[subset_slice])

    lines_xmin = 0.999 * xmin
    lines_xmax = 1.001 * xmax
    lines_ymin = 0.99 * ymin
    lines_ymax = 1.005 * ymax

    ax[0].hlines(y=lines_ymin, xmin=lines_xmin, xmax=lines_xmax, color="black", linestyle="-", lw=1)
    ax[0].hlines(y=lines_ymax, xmin=lines_xmin, xmax=lines_xmax, color="black", linestyle="-", lw=1)
    ax[0].vlines(x=lines_xmin, ymin=lines_ymin, ymax=lines_ymax, color="black", linestyle="-", lw=1)
    ax[0].vlines(x=lines_xmax, ymin=lines_ymin, ymax=lines_ymax, color="black", linestyle="-", lw=1)

    ax[0].get_xaxis().get_offset_text().set_visible(False)
    ax[1].get_xaxis().get_offset_text().set_visible(False)

    # ax[0].text(
    #     0.97,
    #     -0.2,
    #     rf"x10$^{{{floor(np.log10(max(residuals)))}}}$",
    #     fontsize=6,
    #     color="black",
    #     transform=ax[0].transAxes,
    # )
    # ax[1].text(
    #     0.97,
    #     -0.2,
    #     rf"x10$^{{{floor(np.log10(max(residuals)))}}}$",
    #     fontsize=6,
    #     color="black",
    #     transform=ax[1].transAxes,
    # )

    # if label_pos is not None:
    #     for i, label in enumerate(labels):
    #         ax.text(
    #             *label_pos[i],
    #             rf"$\alpha = 10^{{{label[-2:]}}}$",
    #             fontsize=8,
    #             ha="center",
    #             va="center",
    #             color="black",
    #         )
