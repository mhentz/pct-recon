"""Plot profiles through the line pairs of the Catphan phantom for TV and dTV reconstructions."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from skimage.measure import profile_line

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.style.use(f"{REPO_ROOT}/analysis/plotting/style.txt")

plt.rcParams["figure.figsize"] = [5.5, 5.5]
plt.rcParams["figure.dpi"] = 300

ROI_2 = (slice(540, 640), slice(500, 600))  # 0.3 lp/mm
ROI_3 = (slice(480, 580), slice(550, 650))  # 0.3 lp/mm
ROI_4 = (slice(410, 510), slice(570, 670))  # 0.4 lp/mm
ROI_5 = (slice(355, 455), slice(570, 670))  # 0.5 lp/mm
ROI_6 = (slice(310, 370), slice(570, 630))  # 0.6 lp/mm
ROI_7 = (slice(265, 325), slice(530, 590))  # 0.7 lp/mm
ROI_8 = (slice(235, 295), slice(483, 543))  # 0.8 lp/mm
ROI_10 = (slice(200, 300), slice(360, 460))  # 1 lp/mm
ROI_9_10 = (slice(200, 300), slice(380, 480))  # 1 lp/mm


class DoseChoice(float, Enum):
    """Dose levels in mGy."""

    D5 = 5.0
    D1 = 1.0
    D02 = 0.2


def main(
    setup: Annotated[
        Literal["ideal", "realistic"],
        typer.Argument(help="Choose whether to plot ideal or realistic phantom reconstructions."),
    ],
    dose: Annotated[DoseChoice, typer.Argument(help="Dose level in mGy.")],
    n_proj: Annotated[Literal[720, 45], typer.Argument(help="Number of projections.")],
) -> None:
    """Plot the ground truth, TV and dTV profiles across six groups of line pairs."""
    x_gt = np.load(f"{REPO_ROOT}/phantoms/ground_truths/ctp528_gt_880pix.npy")

    alpha: str = get_ideal_alpha(f"ctp528-{setup}", dose, n_proj)
    dose = float(dose)  # ty: ignore[invalid-assignment]
    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    base_path = Path(f"{RECON_DATA_DIR}/ctp528-{setup}/{dose_str}mGy/{n_proj}proj/recon/")

    if not base_path.exists():
        msg = f"This directory does not exist: {base_path}"
        raise FileNotFoundError(msg)

    x_tv = np.load(
        max(
            base_path.glob(
                f"fista_exact_init_zero_880pix_{dose:.1f}mGy_{n_proj}proj_TV_iso_alpha_{alpha}_iter*"
            )
        )
        / "processed"
        / "yk_min_slice.npy"
    )
    x_dtv = np.load(
        max(
            base_path.glob(
                f"fista_exact_init_zero_880pix_{dose:.1f}mGy_{n_proj}proj_dTV_iso_alpha_{alpha}_eta_2e-1_iter*"
            )
        )
        / "processed"
        / "yk_min_slice.npy"
    )

    # profile_endpoints = {
    #     "ROI_3": [(25, 70), (80, 40)],
    #     "ROI_4": [(30, 60), (76, 54)],
    #     "ROI_5": [(20, 48), (60, 60)],
    #     "ROI_6": [(8, 17), (46, 42)],
    #     "ROI_7": [(15, 15), (40, 47)],
    #     "ROI_8": [(21, 12), (38, 47)],
    #     "ROI_9": [(50, 65), (53, 95)],
    #     "ROI_10": [(51, 15), (47, 40)],
    # }

    start = [(25, 70), (30, 60), (20, 48), (8, 17), (15, 15), (21, 12), (50, 65)]
    end = [(80, 40), (76, 54), (60, 60), (46, 42), (40, 47), (38, 47), (53, 95)]

    rois = [ROI_3, ROI_4, ROI_5, ROI_6, ROI_7, ROI_8]
    lab = ["0.3 lp/mm", "0.4 lp/mm", "0.5 lp/mm", "0.6 lp/mm", "0.7 lp/mm", "0.8 lp/mm"]

    # rois = [ROI_2, ROI_3, ROI_4, ROI_5, ROI_6, ROI_7]
    # lab = ["0.3 lp/mm", "0.3 lp/mm", "0.4 lp/mm", "0.5 lp/mm", "0.6 lp/mm", "0.7 lp/mm"]

    _, ax = plt.subplots(3, 2, sharey=True)

    for i in range(len(rois)):
        pr_gt = profile_line(x_gt[rois[i]], start[i], end[i], order=1, linewidth=1)
        pr_tv = profile_line(x_tv[rois[i]], start[i], end[i], order=1)
        pr_dtv = profile_line(x_dtv[rois[i]], start[i], end[i], order=1)

        ii = i % 3
        jj = i // 3
        ax[ii, jj].plot(pr_gt, lw=0.75, ls="dashed", label="GT", color="black", alpha=1)
        ax[ii, jj].plot(pr_tv, lw=3, label="TV", color="gray", alpha=0.7)
        ax[ii, jj].plot(pr_dtv, lw=3, label="dTV", color="black")

        ax[ii, jj].spines["top"].set_visible(False)
        ax[ii, jj].spines["right"].set_visible(False)

        ax[ii, 0].set_ylabel("RSP")

        ax[ii, jj].text(
            0.03,
            0.93,
            lab[i],
            color="black",
            weight="bold",
            transform=ax[ii, jj].transAxes,
            bbox={"facecolor": "gray", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        )

    ax[0, 1].legend(loc=(0.88, 0.6))

    ax[2, 0].set_xlabel("Distance [pixels]")
    ax[2, 1].set_xlabel("Distance [pixels]")
    plt.tight_layout()

    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(
        figures_dir / f"catphan-profiles-{setup}-{dose:.1f}mGy-{n_proj}proj.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    typer.run(main)
