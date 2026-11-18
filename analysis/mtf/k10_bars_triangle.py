"""Plot k10 bar charts for TV and dTV reconstructions at 1 and 5 mGy."""

import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR

N_MIN_ARGS = 2

if len(sys.argv) < N_MIN_ARGS:
    msg = "Please provide setup name as argument (ideal or realistic)"
    raise ValueError(msg)

setup = sys.argv[1]

if setup not in ["ideal", "realistic"]:
    msg = "Setup name must be 'ideal' or 'realistic'"
    raise ValueError(msg)

plt.style.use(Path(__file__).parent / "style.txt")

reg = ["tv", "dtv"]
doses = [5.0, 1.0, 0.2]
n_proj_list = [720, 45]

bars = {
    "tv": defaultdict(dict),
    "dtv": defaultdict(dict),
}
bars_sig = {
    "tv": defaultdict(dict),
    "dtv": defaultdict(dict),
}

total_bars = {
    "tv": defaultdict(list),
    "dtv": defaultdict(list),
}
total_bars_sig = {
    "tv": defaultdict(list),
    "dtv": defaultdict(list),
}

for dose in doses:
    dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"

    for n_proj in n_proj_list:
        alpha = get_ideal_alpha(f"spiral-{setup}", dose, n_proj)

        base_path = Path(f"{RECON_DATA_DIR}/spiral-{setup}/{dose_str}mGy/{n_proj}proj/recon")
        tv_path: Path = max(
            base_path.glob(
                f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_TV_iso_alpha_{alpha}_iter*/processed/mtf",
            ),
        )
        dtv_path: Path = max(
            base_path.glob(
                f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_dTV_iso_alpha_{alpha}_eta_2e-1_iter*/processed/mtf",
            ),
        )

        bars["tv"][dose][n_proj] = pd.read_csv(tv_path / "k10_per_radius.csv")["k10"].to_numpy()
        bars_sig["tv"][dose][n_proj] = pd.read_csv(tv_path / "k10_per_radius.csv")["sigma_k10"].to_numpy()

        bars["dtv"][dose][n_proj] = pd.read_csv(dtv_path / "k10_per_radius.csv")["k10"].to_numpy()
        bars_sig["dtv"][dose][n_proj] = pd.read_csv(dtv_path / "k10_per_radius.csv")["sigma_k10"].to_numpy()

        mean_k10_df = pd.read_csv(tv_path / "k10_mean.csv")
        total_bars["tv"][n_proj].append(mean_k10_df["k10"].iloc[0])
        total_bars_sig["tv"][n_proj].append(mean_k10_df["sigma_k10"].iloc[0])

        mean_k10_df = pd.read_csv(dtv_path / "k10_mean.csv")
        total_bars["dtv"][n_proj].append(mean_k10_df["k10"].iloc[0])
        total_bars_sig["dtv"][n_proj].append(mean_k10_df["sigma_k10"].iloc[0])


def plot_bar_tile(
    ax: plt.Axes,
    x: np.ndarray,
    bars: Mapping[int, np.ndarray],
    width: float,
    alpha: float = 0.5,
    offset: float = 0,
    *,
    bars_sig: Mapping[int, np.ndarray],
) -> None:
    """Draw the k10 bars of the 45- and 720-projection series for one tile.

    Args:
        ax: Axes to draw on.
        x: Horizontal positions of the bar groups.
        bars: k10 values keyed by projection count (45 or 720).
        width: Width of each bar.
        alpha: Opacity of the bars.
        offset: Horizontal offset applied to both bars of a group.
        bars_sig: Uncertainties of the k10 values, keyed by projection count.
    """
    error_kwargs = {"elinewidth": 1.5, "capsize": 0, "ecolor": "k"}

    ax.bar(
        x - width / 2 + offset,
        bars[45],
        width,
        color="C1",
        alpha=alpha,
        ec="w",
        # yerr=bars_sig[45] * 1.96,  # 95% confidence interval for large sample size
        yerr=bars_sig[45],
        error_kw=error_kwargs,
    )
    ax.bar(
        x + width / 2 + offset,
        bars[720],
        width,
        color="C0",
        alpha=alpha,
        ec="w",
        # yerr=bars_sig[720] * 2.776,  # 95% confidence interval for small sample size
        yerr=bars_sig[720],
        error_kw=error_kwargs,
    )


# Plotting
plt.rcParams["figure.figsize"] = [6.2, 6 / 2.2]
fig = plt.figure(figsize=(5.3, 5.3))

x = np.array([10, 25, 40, 55, 70])  # x locations for the groups of bars
width = 3  # the width of the bars

ax = []

ax.append(plt.subplot2grid((4, 4), (0, 0), 2, 2, fig=fig))
plot_bar_tile(ax[0], x, bars["dtv"][5.0], width, alpha=1, offset=width, bars_sig=bars_sig["dtv"][5.0])
plot_bar_tile(ax[0], x, bars["tv"][5.0], width, alpha=0.4, offset=-width, bars_sig=bars_sig["tv"][5.0])

ax.append(plt.subplot2grid((4, 4), (0, 2), 2, 2, fig=fig))
plot_bar_tile(ax[1], x, bars["dtv"][1.0], width, alpha=1, offset=width, bars_sig=bars_sig["dtv"][1.0])
plot_bar_tile(ax[1], x, bars["tv"][1.0], width, alpha=0.4, offset=-width, bars_sig=bars_sig["tv"][1.0])

ax.append(plt.subplot2grid((4, 4), (2, 1), 2, 2, fig=fig))
plot_bar_tile(ax[2], x, bars["dtv"][0.2], width, alpha=1, offset=width, bars_sig=bars_sig["dtv"][0.2])

# print(bars["tv"][0.2])
plot_bar_tile(ax[2], x, bars["tv"][0.2], width, alpha=0.4, offset=-width, bars_sig=bars_sig["tv"][0.2])


# Legends
dtv = ax[0].bar([0], [0], 0, color="black", ec="none")
tv = ax[0].bar([0], [0], 0, color=(0.5, 0.5, 0.5), alpha=0.5, ec="none")
ax[0].legend([dtv, tv], ["dTV", "TV"], loc=(0.03, 0.8))

proj_45 = ax[0].bar([0], [0], 0, color="C1", ec="none")
proj_720 = ax[0].bar([0], [0], 0, color="C0", ec="none")
ax[1].legend([proj_720, proj_45], ["720", "45"], loc=(0.03, 0.8))


# Labels
ax[0].set_ylabel("$k_{10\\%}$ [lp/mm]")
ax[2].set_ylabel("$k_{10\\%}$ [lp/mm]")

for j in range(3):
    ax[j].set_xticks(x)
    ax[j].set_xticklabels(["0", "22.5", "45", "67.5", "90"])

    if setup == "ideal":
        ax[j].set_ylim(0, 3.1)
    else:
        ax[j].set_ylim(0, 2.1)

    ax[j].spines["right"].set_visible(False)
    ax[j].spines["top"].set_visible(False)

ax[0].set_title("5.0 mGy")
ax[1].set_title("1.0 mGy")
ax[2].set_title("0.2 mGy")

ax[2].set_xlabel("Distance from centre [mm]")

plt.subplots_adjust(wspace=0.5, hspace=1)

figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)
plt.savefig(figures_dir / f"k10-bars-{setup}.png", dpi=300, bbox_inches="tight")
