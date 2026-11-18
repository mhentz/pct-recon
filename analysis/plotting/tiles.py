"""Plot reconstruction and error tiles for the Catphan and Gammex phantoms at one dose."""

import os
import sys
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.ideal_alpha import get_ideal_alpha
from analysis.plotting.utils import divnorm, get_path, load_image, round_to_nearest_tick
from recon.algo.utils import compute_zero_mask

REPO_ROOT = Path(__file__).resolve().parents[2]

if os.getenv("VS_CODE_DEBUG", None) is None:
    try:
        dose = float(sys.argv[1])
        n_projections = int(sys.argv[2])
        type_ = sys.argv[3]
        regulariser = cast("Literal['TV', 'dTV']", sys.argv[4])
    except IndexError:
        msg = (
            "Error: insufficient command line arguments provided.\n"
            "Usage: script.py <dose> <n_projections> <setup=ideal | realistic> <regulariser=TV | dTV>"
        )
        print(msg)
        sys.exit(1)
    except ValueError:
        msg = (
            "Error: Dose should be a float, n_projections an integer, and setup a string.\n"
            "Usage: script.py <dose> <n_projections> <setup=ideal | realistic> <regulariser=TV | dTV>"
        )
        print(msg)
        sys.exit(1)
else:
    dose = 5.0
    n_projections = 720
    type_ = "ideal"

logger.info("Plotting {} {} recons for {} mGy and {} projections", type_, regulariser, dose, n_projections)

# Shared symmetric error-colourbar limits per (phantom, setup): max abs error over all
# dose/projection/regulariser combinations, so TV and dTV error maps (and the appendix
# figures) of the same phantom+setup are directly comparable.
ERR_LIMITS = {
    ("ctp528", "ideal"): 1.005,
    ("ctp528", "realistic"): 1.005,
    ("gammex", "ideal"): 0.61,
    ("gammex", "realistic"): 0.61,
    ("spiral", "ideal"): 0.84,
    ("spiral", "realistic"): 0.84,
}


plt.style.use(f"{REPO_ROOT}/analysis/plotting/style.txt")
plt.rcParams["figure.figsize"] = [5.5, 5.5 * 1.3]
plt.rcParams["figure.dpi"] = 300

fig, ax = plt.subplots(3, 2, sharey="row")
d_max = []


# ------------------------------------------------------------------------------------------
#  Catphan
# ------------------------------------------------------------------------------------------

gt = np.load(f"{REPO_ROOT}/phantoms/ground_truths/ctp528_gt_880pix.npy")

alpha: str = get_ideal_alpha(f"ctp528-{type_}", dose, n_projections)
xi, i = load_image(
    path=get_path(
        setup=f"ctp528-{type_}",
        dose=dose,
        n_projections=n_projections,
        alpha=alpha,
        regulariser=regulariser,
    ),
)

zero_mask = compute_zero_mask((1, 880, 880), 400, 3)
# xi[np.newaxis, ...][zero_mask] = 0

# image
ax[0, 0].text(0.02, 0.93, f"$i$ = {i}", color="white", transform=ax[0, 0].transAxes, weight="bold")
ax[0, 0].set_title(f"$\\alpha =${alpha}")
im = ax[0, 0].imshow(xi, cmap="gray", vmin=0, vmax=2.2)

ax[0, 0].set_xlim(0, 880)


divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes("right", size="3%", pad=0.05, frameon=False)
fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

# error
d = xi - gt
d_max.append(np.max(np.abs(d)))

err_limit = ERR_LIMITS[("ctp528", type_)]
im = ax[0, 1].imshow(d, cmap="seismic", norm=divnorm(err_limit))
ax[0, 1].set_title(f"$x_{{{i}}} - x_{{\\rm GT}}$")
ax[0, 1].set_xlim(0, 880)

divider = make_axes_locatable(ax[0, 1])
cax = divider.append_axes("right", size="3%", pad=0.05, frameon=False)
cbar = fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

max_tick = round_to_nearest_tick(err_limit)
cbar.set_ticks(np.arange(-max_tick, max_tick + 0.01, 0.2).tolist())


# ------------------------------------------------------------------------------------------
#  Gammex
# ------------------------------------------------------------------------------------------

gt = np.load(f"{REPO_ROOT}/phantoms/ground_truths/gammex_gt_700pix.npy")

alpha: str = get_ideal_alpha(f"gammex-{type_}", dose, n_projections)
xi, i = load_image(
    path=get_path(
        setup=f"gammex-{type_}",
        dose=dose,
        n_projections=n_projections,
        alpha=alpha,
        regulariser=regulariser,
    ),
)

zero_mask = compute_zero_mask((1, 700, 700), 330, 3)
# xi[np.newaxis, ...][zero_mask] = 0

# image
ax[1, 0].text(0.02, 0.93, f"$i$ = {i}", color="white", transform=ax[1, 0].transAxes, weight="bold")
ax[1, 0].set_title(f"$\\alpha =${alpha}")
im = ax[1, 0].imshow(xi, cmap="gray", vmin=0, vmax=1.7)

divider = make_axes_locatable(ax[1, 0])
cax = divider.append_axes("right", size="3%", pad=0.05, frameon=False)
fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

# error
d = xi - gt
d_max.append(np.max(np.abs(d)))

err_limit = ERR_LIMITS[("gammex", type_)]
im = ax[1, 1].imshow(d, cmap="seismic", norm=divnorm(err_limit))
ax[1, 1].set_title(f"$x_{{{i}}} - x_{{\\rm GT}}$")

divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes("right", size="3%", pad=0.05, frameon=False)
cbar = fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

max_tick = round_to_nearest_tick(err_limit)
cbar.set_ticks(np.arange(-max_tick, max_tick + 0.01, 0.2).tolist())


# ------------------------------------------------------------------------------------------
#  Spiral
# ------------------------------------------------------------------------------------------

gt = np.load(f"{REPO_ROOT}/phantoms/ground_truths/spiral_gt_880pix.npy")

alpha: str = get_ideal_alpha(f"spiral-{type_}", dose, n_projections)
xi, i = load_image(
    path=get_path(
        setup=f"spiral-{type_}",
        dose=dose,
        n_projections=n_projections,
        alpha=alpha,
        regulariser=regulariser,
    ),
)

zero_mask = compute_zero_mask((1, 880, 880), 400, 3)
# xi[np.newaxis, ...][zero_mask] = 0

# image
ax[2, 0].text(0.02, 0.93, f"$i$ = {i}", color="white", transform=ax[2, 0].transAxes, weight="bold")
ax[2, 0].set_title(f"$\\alpha =${alpha}")
im = ax[2, 0].imshow(xi, cmap="gray", vmin=0, vmax=2.2)
ax[2, 0].set_xlim(0, 880)

divider = make_axes_locatable(ax[2, 0])
cax = divider.append_axes("right", size="3%", pad=0.05, frameon=False)
fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

ax[2, 0].spines["right"].set_visible(True)
ax[2, 0].spines["top"].set_visible(True)

# error
d = xi - gt
d_max.append(np.max(np.abs(d)))

err_limit = ERR_LIMITS[("spiral", type_)]
im = ax[2, 1].imshow(d, cmap="seismic", norm=divnorm(err_limit))
ax[2, 1].set_title(f"$x_{{{i}}} - x_{{\\rm GT}}$")
ax[2, 1].set_xlim(0, 880)

divider = make_axes_locatable(ax[2, 1])
cax = divider.append_axes("right", size="3%", pad=0.05, frameon=False)
cbar = fig.colorbar(im, cax=cax, orientation="vertical", format="%.1f")

max_tick = round_to_nearest_tick(err_limit)
cbar.set_ticks(np.arange(-max_tick, max_tick + 0.01, 0.2).tolist())


# ------------------------------------------------------------------------------------------
#  Plot settings
# ------------------------------------------------------------------------------------------

ax[2, 1].spines["right"].set_visible(True)
ax[2, 1].spines["top"].set_visible(True)

ax[0, 0].spines["right"].set_visible(True)
ax[0, 0].spines["top"].set_visible(True)
ax[0, 1].spines["right"].set_visible(True)
ax[0, 1].spines["top"].set_visible(True)

ax[1, 0].spines["right"].set_visible(True)
ax[1, 0].spines["top"].set_visible(True)
ax[1, 1].spines["right"].set_visible(True)
ax[1, 1].spines["top"].set_visible(True)


# ax[0, 0].text(0.66, 0.93, "[0, 2.1]", color="white", transform=ax[0, 0].transAxes, weight="bold")
# ax[1, 0].text(0.66, 0.93, "[0, 1.7]", color="white", transform=ax[1, 0].transAxes, weight="bold")
# ax[2, 0].text(0.66, 0.93, "[0, 2.2]", color="white", transform=ax[2, 0].transAxes, weight="bold")

# ax[0, 0].text(0.02, 0.02, "880 x 880", color='white', transform=ax[0, 0].transAxes, weight='bold')
# ax[1, 0].text(0.02, 0.02, "700 x 700", color='white', transform=ax[1, 0].transAxes, weight='bold')
# ax[2, 0].text(0.02, 0.02, "880 x 880", color='white', transform=ax[2, 0].transAxes, weight='bold')


for i in [0, 2]:
    for j in [0, 1]:
        ax[i, j].set_xticks([0, 439, 879])
        ax[i, j].set_yticks([0, 439, 879])
        ax[i, j].set_xticks([220, 660], minor=True)
        ax[i, j].set_yticks([220, 660], minor=True)

for j in [0, 1]:
    ax[1, j].set_xticks([0, 349, 699])
    ax[1, j].set_yticks([0, 349, 699])
    ax[1, j].set_xticks([175, 525], minor=True)
    ax[1, j].set_yticks([175, 525], minor=True)

# ax[0, 0].tick_params(axis='both', which='both', direction='in', color='white', right=True)
# ax[0, 0].tick_params(axis='both', which='major', direction='in', color='white', top=True)

# ax[0, 0].set_ylabel("Catphan")
# ax[1, 0].set_ylabel("Gammex")
# ax[2, 0].set_ylabel("Spiral")

plt.subplots_adjust(hspace=0.25)
plt.tight_layout(pad=1.05)


ax[0, 0].set_xticks([1, 440, 880])
ax[0, 0].set_yticks([1, 440, 880])
ax[0, 0].set_xticks([220, 660], minor=True)
ax[0, 0].set_yticks([220, 660], minor=True)

ax[0, 0].tick_params(axis="both", which="both", direction="in", color="white", right=True)
ax[0, 0].tick_params(axis="both", which="major", direction="in", color="white", top=True)


ax[1, 0].set_xticks([1, 350, 700])
ax[1, 0].set_yticks([1, 350, 700])
ax[1, 0].set_xticks([175, 525], minor=True)
ax[1, 0].set_yticks([175, 525], minor=True)

ax[1, 0].tick_params(axis="both", which="both", direction="in", color="white", right=True)
ax[1, 0].tick_params(axis="both", which="major", direction="in", color="white", top=True)


ax[2, 0].set_xticks([1, 440, 880])
ax[2, 0].set_yticks([1, 440, 880])
ax[2, 0].set_xticks([220, 660], minor=True)
ax[2, 0].set_yticks([220, 660], minor=True)

ax[2, 0].tick_params(axis="both", which="both", direction="in", color="white", right=True)
ax[2, 0].tick_params(axis="both", which="major", direction="in", color="white", top=True)


ax[0, 1].set_xticks([1, 440, 880])
ax[0, 1].set_yticks([1, 440, 880])
ax[0, 1].set_xticks([220, 660], minor=True)
ax[0, 1].set_yticks([220, 660], minor=True)
# ax[0, 1].tick_params(axis="both", which="both", direction="in")
ax[0, 1].tick_params(axis="both", which="both", direction="in", right=True)
ax[0, 1].tick_params(axis="both", which="major", direction="in", top=True)


ax[1, 1].set_xticks([1, 350, 700])
ax[1, 1].set_yticks([1, 350, 700])
ax[1, 1].set_xticks([175, 525], minor=True)
ax[1, 1].set_yticks([175, 525], minor=True)
# ax[1, 1].tick_params(axis="both", which="both", direction="in")
ax[1, 1].tick_params(axis="both", which="both", direction="in", right=True)
ax[1, 1].tick_params(axis="both", which="major", direction="in", top=True)

ax[2, 1].set_xticks([1, 440, 880])
ax[2, 1].set_yticks([1, 440, 880])
ax[2, 1].set_xticks([220, 660], minor=True)
ax[2, 1].set_yticks([220, 660], minor=True)
# ax[2, 1].tick_params(axis="both", which="both", direction="in")
ax[2, 1].tick_params(axis="both", which="both", direction="in", right=True)
ax[2, 1].tick_params(axis="both", which="major", direction="in", top=True)

plt.subplots_adjust(hspace=0.25)
plt.tight_layout(pad=1.05)
# plt.show()

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.3)

figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)
plt.savefig(
    figures_dir / f"tiles-{regulariser.lower()}-{type_}-{dose:.1f}mGy-{n_projections}proj.png",
    dpi=300,
    bbox_inches="tight",
)

logger.info("Max abs diff values Catphan: {:.3f}", d_max[0])
logger.info("Max abs diff values Gammex:  {:.3f}", d_max[1])
logger.info("Max abs diff values Spiral:  {:.3f}", d_max[2])
