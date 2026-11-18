"""Tests of the mid-plane positions and the statistical data cuts in ``recon/preprocess/cuts.py``."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from recon.preprocess.cuts import apply_data_cuts, compute_positions_at_midplane, compute_proton_stats

C_Z = 100.0
HALFSPACE = 2.0  # with the default 2 mm bins: edges -2, 0, 2 in x and y, so 2 x 2 bins
BIN_CENTRES = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
N_PER_BIN = 200


def gaussian_quantiles(n: int) -> np.ndarray:
    """Return n standard normal values spaced evenly in probability, all within 3 sigma."""
    return norm.ppf((np.arange(n) + 0.5) / n)


def make_binned_protons() -> pd.DataFrame:
    """Protons on straight paths through the bin centres, Gaussian in WEPL and both angles.

    Each bin has its own WEPL mean. The relative angles are spread by permuting the same
    quantiles, so the three quantities are not identical within a bin.
    """
    q = gaussian_quantiles(N_PER_BIN)
    rng = np.random.default_rng(0)
    frames = []
    for i, (x, y) in enumerate(BIN_CENTRES):
        frames.append(
            pd.DataFrame(
                {
                    "x_hull_in": x,
                    "y_hull_in": y,
                    "z_hull_in": C_Z - 50,
                    "x_hull_out": x,
                    "y_hull_out": y,
                    "z_hull_out": C_Z + 50,
                    "wepl": 100.0 + 10 * i + q,
                    "ang_xz_in": 0.0,
                    "ang_yz_in": 0.0,
                    "ang_xz_out": 0.001 * rng.permutation(q),
                    "ang_yz_out": 0.001 * rng.permutation(q),
                },
            ),
        )
    return pd.concat(frames, ignore_index=True)


def test_compute_positions_at_midplane_on_a_straight_line():
    protons = pd.DataFrame(
        {
            "x_hull_in": [-5.0],
            "y_hull_in": [1.0],
            "z_hull_in": [50.0],
            "x_hull_out": [5.0],
            "y_hull_out": [3.0],
            "z_hull_out": [150.0],
        },
    )

    result = compute_positions_at_midplane(protons, C_Z)

    assert result["x_mid"].tolist() == pytest.approx([0.0])
    assert result["y_mid"].tolist() == pytest.approx([2.0])


def test_compute_proton_stats_bins_and_drops_protons_outside_the_image():
    protons = make_binned_protons()
    outside = protons.iloc[[0]].assign(x_hull_in=HALFSPACE + 1, x_hull_out=HALFSPACE + 1)
    protons = pd.concat([protons, outside], ignore_index=True)

    kept, stats, bin_edges = compute_proton_stats(protons, C_Z, HALFSPACE, HALFSPACE)

    assert len(kept) == 4 * N_PER_BIN
    np.testing.assert_allclose(bin_edges[0], [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(bin_edges[1], [-2.0, 0.0, 2.0])
    assert stats["mean_wepl"].shape == (2, 2)
    # Bins are indexed [x, y]; BIN_CENTRES lists them in that order, with WEPL means 100 + 10 i.
    np.testing.assert_allclose(stats["mean_wepl"], [[100.0, 110.0], [120.0, 130.0]])
    np.testing.assert_allclose(kept["rel_ang_xz"], kept["ang_xz_out"])


def test_apply_data_cuts_removes_exactly_the_outliers():
    protons = make_binned_protons()
    outliers = [5, 250, 777]  # one per quantity, in three different bins
    protons.loc[5, "wepl"] += 20.0
    protons.loc[250, "ang_xz_out"] = 0.05
    protons.loc[777, "ang_yz_out"] = -0.05
    protons, stats, bin_edges = compute_proton_stats(protons, C_Z, HALFSPACE, HALFSPACE)

    kept = apply_data_cuts(protons, 3.0, stats, bin_edges)

    removed = sorted(set(protons.index) - set(kept.index))
    assert removed == outliers


def test_compute_proton_stats_drops_protons_on_the_upper_edge():
    protons = make_binned_protons()
    typical = protons.iloc[[N_PER_BIN // 2]]  # WEPL and angles at the median of the first bin
    on_upper = typical.assign(x_hull_in=HALFSPACE, x_hull_out=HALFSPACE)
    on_lower = typical.assign(x_hull_in=-HALFSPACE, x_hull_out=-HALFSPACE)
    protons = pd.concat([protons, on_upper, on_lower], ignore_index=True)
    i_upper, i_lower = len(protons) - 2, len(protons) - 1

    kept, stats, bin_edges = compute_proton_stats(protons, C_Z, HALFSPACE, HALFSPACE)
    after_cuts = apply_data_cuts(kept, 3.0, stats, bin_edges)

    assert i_upper not in kept.index
    assert i_lower in kept.index
    assert i_lower in after_cuts.index


def test_apply_data_cuts_removes_a_single_proton_bin():
    protons = make_binned_protons()
    i_lone = 3 * N_PER_BIN  # the first proton of the last bin, which keeps no other
    protons = protons.iloc[: i_lone + 1]
    protons, stats, bin_edges = compute_proton_stats(protons, C_Z, HALFSPACE, HALFSPACE)

    kept = apply_data_cuts(protons, 3.0, stats, bin_edges)

    removed = sorted(set(protons.index) - set(kept.index))
    assert removed == [i_lone]
