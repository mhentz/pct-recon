"""Plot the SSIM of the dTV reconstructions against eta, the three phantoms in one column.

The script searches ``--root`` (default ``RECON_DATA_DIR``) for dTV runs, which are
directories named

    <phantom>-<setup>/<dose>mGy/<n_proj>proj/recon/
        fista_exact_init_zero_<width>pix_<dose>mGy_<n_proj>proj_dTV_iso_alpha_<alpha>_eta_<eta>_iter<n>

and keeps the runs whose alpha is the one chosen from the L-curves for that phantom,
dose and projection count (``analysis/ideal_alpha.py``). For each of them it reads
``processed/yk_min_slice.npy``, the central slice of the minimum-error iterate, and
computes its SSIM against the ground truth in ``phantoms/ground_truths``, with the
range of the ground truth as the data range.

The figure has one panel per phantom (CTP528, Gammex, spiral). In each panel the bars
are grouped by eta; the grey level gives the dose and the hatching the number of
projections, and the legend lists the alpha of each combination. The y-axis is zoomed
to the range of the SSIM values, which differ only in the third or fourth decimal.

The figure is saved as ``ssim_vs_eta_bars_column_<setup-type>.png`` in ``--save-dir``
(default: ``figures/`` next to this script).

Usage, from the repository root:

    PYTHONPATH=. python analysis/ssim-dtv-eta/ssim_dtv_bars_column.py --setup-type ideal
    PYTHONPATH=. python analysis/ssim-dtv-eta/ssim_dtv_bars_column.py --setup-type realistic
"""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from skimage.metrics import structural_similarity as ssim

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable

# Example dir name: fista_exact_init_zero_700pix_5.0mGy_720proj_dTV_iso_alpha_1.58e-3_eta_2e-1_iter200
DIRECTORY_REGEX = re.compile(
    r"fista_exact_init_zero_(700|880)pix_"
    r"(?P<dose>[0-9]+(?:\.[0-9])?)mGy_"
    r"(?P<nproj>[0-9]+)proj_"
    r"dTV_iso_"
    r"alpha_(?P<alpha>[0-9]+\.?[0-9]*e-[0-9]+)_"
    r"eta_(?P<eta>[0-9]+\.?[0-9]*e-[0-9]+)_"
    r"iter(?P<iter>[0-9]+)"
)

CANDIDATE_FILENAMES = [
    Path("processed/yk_min_slice.npy"),
    # Path("yk_min_slice.npy"),
]

GT_FILENAMES = {
    "ctp528": "ctp528_gt_880pix.npy",
    "gammex": "gammex_gt_700pix.npy",
    "spiral": "spiral_gt_880pix.npy",
}


@dataclass
class ReconSample:
    """Parameters and location of one dTV reconstruction directory."""

    path: Path
    phantom_setup: str  # e.g. gammex-ideal
    dose: float
    n_proj: int
    alpha: str
    eta: str
    iteration: int
    image_path: Path


def _extract_phantom_setup(dirpath: Path) -> str:
    """Return the phantom setup of a reconstruction directory, e.g. ``gammex-ideal``.

    The setup is the directory three levels above the last ``recon`` in the path
    (``<phantom-setup>/<dose>mGy/<nproj>proj/recon/<recon_dir>``), so that a ``recon``
    higher up in the root path is not picked up.
    """
    parts = dirpath.parts
    recon_indices = [i for i, p in enumerate(parts) if p == "recon"]
    if not recon_indices:
        return "unknown"
    recon_idx = recon_indices[-1]
    if recon_idx < 3:  # noqa: PLR2004
        return "unknown"
    return parts[recon_idx - 3]


def scan_recons(root: Path) -> list[ReconSample]:
    """Find dTV reconstruction directories that use the ideal alpha.

    Args:
        root: Directory to search recursively.

    Returns:
        One sample per matching reconstruction directory that contains an image file.
    """
    logger.info(f"Scanning for dTV recon directories under {root}")
    samples: list[ReconSample] = []

    for dirpath in root.rglob("fista_exact_init_zero_*pix_*_dTV_iso_alpha_*"):
        if not dirpath.is_dir():
            continue
        regex_match = DIRECTORY_REGEX.match(dirpath.name)
        if not regex_match:
            continue

        dose = float(regex_match.group("dose"))
        n_proj = int(regex_match.group("nproj"))
        alpha = regex_match.group("alpha")
        eta = regex_match.group("eta")
        iteration = int(regex_match.group("iter"))

        phantom_setup = _extract_phantom_setup(dirpath)

        if alpha != get_ideal_alpha(phantom_setup, dose, n_proj):
            continue

        image_path = None
        for fname in CANDIDATE_FILENAMES:
            p = dirpath / fname
            if p.exists():
                image_path = p
                break
        if image_path is None:
            logger.warning(f"No image file found in {dirpath}")
            continue

        samples.append(
            ReconSample(
                path=dirpath,
                phantom_setup=phantom_setup,
                dose=dose,
                n_proj=n_proj,
                alpha=alpha,
                eta=eta,
                iteration=iteration,
                image_path=image_path,
            )
        )
    logger.info(f"Found {len(samples)} dTV samples")
    return samples


def compute_single_ssim(image: np.ndarray, reference: np.ndarray) -> float:
    """Compute the SSIM of an image against a reference.

    Args:
        image: Reconstructed image.
        reference: Reference image, used to set the data range.

    Returns:
        SSIM value of the image relative to the reference.
    """
    dr = float(reference.max() - reference.min()) or 1.0
    return float(ssim(reference, image, data_range=dr))


def load_ground_truth(phantom_setup: str) -> np.ndarray | None:
    """Load the ground truth of a phantom setup from ``phantoms/ground_truths``.

    Returns None if the phantom is not one of ``GT_FILENAMES`` or the file is missing.
    """
    fname = GT_FILENAMES.get(phantom_setup.split("-", maxsplit=1)[0])
    if not fname:
        return None

    gt_path = (Path(__file__).resolve().parent.parent.parent / "phantoms" / "ground_truths" / fname).resolve()
    if not gt_path.exists():
        logger.warning(f"Ground truth file not found: {gt_path}")
        return None
    return np.load(gt_path)


class PhantomGroup:
    """All (dose, n_proj) combinations of one phantom setup.

    ``data`` maps (dose, n_proj) to eta to ReconSample, keeping the first sample if
    there are duplicates. All samples use the ideal alpha of their combination.
    """

    def __init__(self, phantom_setup: str) -> None:
        """Create an empty group for one phantom setup, e.g. ``"gammex-ideal"``."""
        self.phantom_setup = phantom_setup
        self.data: dict[tuple[float, int], dict[str, ReconSample]] = {}

    def add(self, sample: ReconSample) -> None:
        """Add a sample, unless its (dose, n_proj, eta) is already there."""
        key = (sample.dose, sample.n_proj)
        self.data.setdefault(key, {})
        self.data[key].setdefault(sample.eta, sample)

    def __repr__(self) -> str:
        """Return the representation of the grouped data dictionary."""
        return repr(self.data)


def group_samples_by_phantom(samples: Iterable[ReconSample]) -> dict[str, PhantomGroup]:
    """Group samples by phantom_setup only."""
    groups: dict[str, PhantomGroup] = {}
    for sample in samples:
        key = sample.phantom_setup
        if key not in groups:
            groups[key] = PhantomGroup(sample.phantom_setup)
        groups[key].add(sample)
    return groups


def _plot_phantom_group_bars(  # noqa: C901, PLR0912, PLR0915
    group: PhantomGroup,
    ax: plt.Axes,
    eta_order: list[str] | None = None,
    *,
    show_xlabel: bool = True,
) -> bool:
    """Plot the SSIM against eta of one phantom, all (dose, n_proj) combinations.

    Each eta has one bar per combination, all at the ideal alpha of that combination.

    Args:
        group: Samples of the phantom.
        ax: Axes to plot on.
        eta_order: Eta values of the x-axis. If None, the eta values of the group.
        show_xlabel: Whether to label the x-axis (False above the bottom panel).

    Returns:
        False if the ground truth is missing and nothing was plotted, True otherwise.
    """
    gt = load_ground_truth(group.phantom_setup)
    if gt is None:
        logger.warning(f"Skipping phantom {group.phantom_setup} (no ground truth)")
        return False

    dose_nproj_pairs = list(group.data.keys())
    dose_nproj_pairs.sort(key=lambda t: (t[1], t[0]))  # sort by n_proj first, then dose

    doses = sorted({dose for dose, _ in dose_nproj_pairs})
    nprojs = sorted({nproj for _, nproj in dose_nproj_pairs})

    # Darkest grey for the highest dose
    grey_levels = [
        "#303030",  # 5 mGy
        "#808080",  # 1 mGy
        "#C0C0C0",  # 0.2 mGy
    ]

    dose_color_map = {dose: grey_levels[i % len(grey_levels)] for i, dose in enumerate(reversed(doses))}

    hatch_patterns = ["", "//", "\\\\", "..", "xx", "++", "||", "--"]
    nproj_hatch_map = {nproj: hatch_patterns[i % len(hatch_patterns)] for i, nproj in enumerate(nprojs)}

    if eta_order is None:
        all_eta_strs = set()
        for eta_map in group.data.values():
            all_eta_strs.update(eta_map.keys())
        eta_order = sorted(all_eta_strs, key=float)

    # eta_index -> (dose, n_proj) -> SSIM
    eta_data = {}
    for eta_idx, eta_str in enumerate(eta_order):
        eta_data[eta_idx] = {}
        for dose, n_proj in dose_nproj_pairs:
            sample = group.data[(dose, n_proj)].get(eta_str)
            if sample is not None:
                eta_data[eta_idx][(dose, n_proj)] = compute_single_ssim(np.load(sample.image_path), gt)

    n_eta = len(eta_order)
    n_combinations = len(dose_nproj_pairs)
    bar_width = 0.8 / n_combinations if n_combinations > 0 else 0.8

    legend_elements = []
    legend_labels = []

    for combo_idx, (dose, n_proj) in enumerate(dose_nproj_pairs):
        eta_indices = []
        ssim_values = []

        for eta_idx in range(n_eta):
            if (dose, n_proj) in eta_data[eta_idx]:
                eta_indices.append(eta_idx)
                ssim_values.append(eta_data[eta_idx][(dose, n_proj)])

        if not ssim_values:
            continue

        x_positions = [x + (combo_idx - n_combinations / 2 + 0.5) * bar_width for x in eta_indices]

        ideal_alpha = get_ideal_alpha(group.phantom_setup, dose, n_proj)

        color = dose_color_map[dose]
        hatch = nproj_hatch_map[n_proj]
        label = f"dose={dose:g}mGy, n_proj={n_proj}, alpha={ideal_alpha}"

        bars = ax.bar(
            x_positions,
            ssim_values,
            width=bar_width,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.3,
            alpha=1.0,
            label=label,
            zorder=3,
        )

        if bars:
            # Denser hatching in the legend, where the patches are small
            if hatch:
                dense_hatch = hatch * 3
                legend_patch = plt.Rectangle(
                    (0, 0), 1, 1, facecolor=color, hatch=dense_hatch, edgecolor="white", linewidth=0.3
                )
            else:
                legend_patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", linewidth=0.3)
            legend_elements.append(legend_patch)
            legend_labels.append(label)

    phantom_name = group.phantom_setup.split("-")[0]
    phantom_name = phantom_name.upper() if phantom_name.lower() == "ctp528" else phantom_name.capitalize()

    text = ax.text(
        0.5,
        0.98,
        phantom_name,
        transform=ax.transAxes,
        fontsize=14,
        # fontweight="bold",
        va="top",
        ha="center",
        zorder=10,
    )

    # White box behind the title, padded more horizontally, so that the grid lines
    # stop short of it
    fig = ax.figure
    fig.canvas.draw()
    bbox_text = text.get_window_extent(renderer=fig.canvas.get_renderer())  # ty: ignore[missing-attribute]
    bbox_axes = bbox_text.transformed(ax.transAxes.inverted())

    h_pad = 0.02  # axes fraction
    v_pad = 0.01

    rect = plt.Rectangle(
        (bbox_axes.x0 - h_pad, bbox_axes.y0 - v_pad),
        bbox_axes.width + 2 * h_pad,
        bbox_axes.height + 2 * v_pad,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        zorder=9,
    )
    ax.add_patch(rect)

    if show_xlabel:
        ax.set_xlabel("$\\eta$", fontsize=14)
    ax.set_ylabel("SSIM", fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_zorder(5)
    ax.spines["bottom"].set_zorder(5)

    # ax.grid(visible=True, alpha=0.3, axis="y", zorder=1, linewidth=0.6)
    ax.grid(visible=True, alpha=0.6, axis="y", zorder=1, linewidth=0.6)

    ax.tick_params(axis="both", which="major", labelsize=12, width=0.5)
    ax.tick_params(axis="both", which="minor", width=0.3)

    ax.set_xticks(range(len(eta_order)))
    if show_xlabel:
        ax.set_xticklabels(eta_order, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])

    if legend_elements:
        # Legend as a table: dose | n_proj | alpha, aligned under the title
        formatted_labels = []
        for dose, n_proj in dose_nproj_pairs:
            ideal_alpha = get_ideal_alpha(group.phantom_setup, dose, n_proj)
            matching = [label for label in legend_labels if f"dose={dose:g}mGy, n_proj={n_proj}" in label]
            if matching:
                label = f"{dose:>7.1f} | {n_proj:>6d} | {ideal_alpha}"
                formatted_labels.append(label)

        original_hatch_linewidth = mpl.rcParams["hatch.linewidth"]
        mpl.rcParams["hatch.linewidth"] = 1.5

        legend = ax.legend(
            legend_elements[: len(formatted_labels)],
            formatted_labels,
            alignment="left",
            fontsize=11,
            loc="upper left",
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor="gray",
            framealpha=0.85,
            labelspacing=0.3,
            handletextpad=0.5,
            title="  Dose (mGy)| # proj | $\\alpha$",
        )
        legend.get_frame().set_linewidth(0.6)
        legend.get_title().set_fontfamily("monospace")
        legend.get_title().set_fontsize(11)
        legend.get_title().set_horizontalalignment("left")

        for text in legend.get_texts():
            text.set_fontfamily("monospace")

        mpl.rcParams["hatch.linewidth"] = original_hatch_linewidth

    all_ssim_values = []
    for eta_idx in range(n_eta):
        all_ssim_values.extend(eta_data[eta_idx].values())

    if all_ssim_values:
        lo = min(all_ssim_values)
        hi = max(all_ssim_values)
        margin = max(1e-4, 0.05 * (hi - lo) if hi > lo else 0.002)
        y_max = max(hi + margin, hi)
        ax.set_ylim(lo - margin, y_max)

    return True


def plot_phantom_column(
    groups: dict[str, PhantomGroup],
    setup_type: str,
    save_dir: Path,
    *,
    show_bars: bool = False,
) -> bool:
    """Plot the three phantoms of one setup type in a column and save the figure.

    Args:
        groups: Samples per phantom setup.
        setup_type: ``ideal`` or ``realistic``.
        save_dir: Directory to save the figure in.
        show_bars: Whether to also show the figure.

    Returns:
        True if at least one phantom was plotted and the figure saved.
    """
    # Thin hatching in the bars; the legend sets its own width
    original_hatch_linewidth = mpl.rcParams.get("hatch.linewidth", 1.0)
    mpl.rcParams["hatch.linewidth"] = 0.4

    phantom_names = ["ctp528", "gammex", "spiral"]
    phantom_setups = [f"{name}-{setup_type}" for name in phantom_names]

    available_phantoms = [setup for setup in phantom_setups if setup in groups]

    if not available_phantoms:
        logger.warning(f"No phantoms found for setup type '{setup_type}'")
        mpl.rcParams["hatch.linewidth"] = original_hatch_linewidth
        return False

    # Same eta values on every panel, so that the shared x-axis lines up
    all_eta_strs = set()
    for phantom_setup in available_phantoms:
        group = groups[phantom_setup]
        for eta_map in group.data.values():
            all_eta_strs.update(eta_map.keys())
    eta_order = sorted(all_eta_strs, key=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), dpi=100, sharex=True, gridspec_kw={"hspace": 0.1})

    success_count = 0
    for idx, (phantom_setup, ax) in enumerate(zip(phantom_setups, axes, strict=True)):
        if phantom_setup in groups:
            group = groups[phantom_setup]
            show_xlabel = idx == len(phantom_setups) - 1
            success = _plot_phantom_group_bars(group, ax, eta_order=eta_order, show_xlabel=show_xlabel)
            if success:
                success_count += 1
        else:
            ax.set_visible(False)
            logger.warning(f"Phantom setup '{phantom_setup}' not found in data")

    if success_count > 0:
        out_path = save_dir / f"ssim_vs_eta_bars_column_{setup_type}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        logger.info(f"Saved {out_path}")

        if show_bars:
            with suppress(Exception):
                fig.show()
                input("Press Enter to continue...")
        else:
            plt.close(fig)
    else:
        plt.close(fig)
        logger.warning(f"No successful plots for setup type '{setup_type}'")

    mpl.rcParams["hatch.linewidth"] = original_hatch_linewidth

    return success_count > 0


def main(
    root: Annotated[
        Path,
        typer.Option(help="Root directory containing phantom-setup subfolders"),
    ] = RECON_DATA_DIR,
    save_dir: Annotated[
        Path | None,
        typer.Option(help="Directory to save plots (defaults to script_dir/figures)"),
    ] = None,
    setup_type: Annotated[
        str,
        typer.Option(help="Setup type to process: 'ideal' or 'realistic'"),
    ] = "ideal",
    *,
    show_bars: Annotated[
        bool,
        typer.Option("--show-bars", help="Show bar plots with plt.show()"),
    ] = False,
) -> None:
    """Generate SSIM vs eta bar plots in a column layout (3 phantoms per figure)."""
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_recons(root)
    if not samples:
        logger.warning("No samples found. Check that the root path is correct and mounted.")
        return

    samples = [s for s in samples if s.phantom_setup.endswith(f"-{setup_type}")]
    if not samples:
        logger.warning(f"No samples found for setup type '{setup_type}'. Available setup types:")
        all_setups = {s.phantom_setup.split("-")[-1] for s in scan_recons(root)}
        for s in sorted(all_setups):
            logger.info(f"  - {s}")
        return

    groups = group_samples_by_phantom(samples)
    plot_phantom_column(groups, setup_type, save_dir, show_bars=show_bars)


if __name__ == "__main__":
    typer.run(main)
