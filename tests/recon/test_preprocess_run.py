"""Tests of the preprocessing helpers in ``recon/preprocess/run.py`` that need no ROOT files."""

import os

import click
import numpy as np
import pandas as pd
import pytest
import typer

from recon.preprocess.run import convert_energy_to_wepl, degrade_measurements, preprocess_protons

COLUMNS = ["x_in", "y_in", "x_out", "y_out", "ang_xz_in", "ang_yz_in", "ang_xz_out", "ang_yz_out"]

# The positional arguments as scripts/preprocess.job passes them.
JOB_ARGS = ["gammex-realistic", "360", "10.0", "1.0", "0", "90", "4", "in/", "out/"]


def make_protons(n: int) -> pd.DataFrame:
    return pd.DataFrame({**{column: np.zeros(n) for column in COLUMNS}, "wepl": np.full(n, 100.0)})


def make_command() -> click.Command:
    """Build the Click command that ``typer.run(preprocess_protons)`` runs."""
    app = typer.Typer()
    app.command()(preprocess_protons)
    return typer.main.get_command(app)


def test_convert_energy_to_wepl_with_linear_lut():
    # LUT entry i is 2 i and the default bin size is 1e-4; energies sit mid-bin, so they map to
    # bins ceil(7.5) = 8 and ceil(2.5) = 3, giving WEPL 2 * 8 - 2 * 3 = 10.
    lut = 2.0 * np.arange(10)
    protons = pd.DataFrame({"energy_in": [0.00075, 0.00085], "energy_out": [0.00025, 0.00085]})

    result = convert_energy_to_wepl(protons, lut)

    np.testing.assert_allclose(result["wepl"], [10.0, 0.0])


def test_degrade_measurements_is_reproducible_for_a_seed():
    first = make_protons(50)
    second = make_protons(50)
    other = make_protons(50)

    degrade_measurements(first, "ctp528", random_seed=3)
    degrade_measurements(second, "ctp528", random_seed=3)
    degrade_measurements(other, "ctp528", random_seed=4)

    pd.testing.assert_frame_equal(first, second)
    assert not np.allclose(first["x_in"], other["x_in"])


@pytest.mark.parametrize(
    ("phantom", "sigma_in", "sigma_out"),
    [("gammex", 0.0022, 0.0035), ("ctp528", 0.0023, 0.005)],
)
def test_degrade_measurements_spreads(phantom, sigma_in, sigma_out):
    protons = make_protons(20_000)

    degraded = degrade_measurements(protons, phantom, random_seed=0)

    std = degraded.std()
    for column in ["x_in", "y_in", "x_out", "y_out"]:
        assert std[column] == pytest.approx(0.07, rel=0.05)
    for column in ["ang_xz_in", "ang_yz_in"]:
        assert std[column] == pytest.approx(sigma_in, rel=0.05)
    for column in ["ang_xz_out", "ang_yz_out"]:
        assert std[column] == pytest.approx(sigma_out, rel=0.05)

    assert std["wepl"] == pytest.approx(100 * 0.0013, rel=0.05)


def test_cli_parses_the_preprocess_job_command_line(mocker):
    mocker.patch.dict(os.environ, {"WEPL_LUT_PATH": "lut.npy"})
    command = make_command()

    with command.make_context("run.py", [*JOB_ARGS, "--n-threads", "4", "--degrade"]) as ctx:
        params = ctx.params

    assert params == {
        "setup": "gammex-realistic",
        "n_projections": 360,
        "actual_dose": 10.0,
        "target_dose": 1.0,
        "start_index": 0.0,
        "end_index": 90.0,
        "n_chunks": 4,
        "input_dir": "in/",
        "output_dir": "out/",
        "wepl_lut_path": "lut.npy",
        "n_slices": 3,
        "slice_thickness": 1.0,
        "n_sigma": 3.0,
        "n_threads": 4,
        "degrade": True,
    }


def test_cli_requires_the_wepl_lut_path(mocker):
    mocker.patch.dict(os.environ)
    os.environ.pop("WEPL_LUT_PATH", None)
    command = make_command()

    with pytest.raises(click.exceptions.MissingParameter, match="wepl_lut_path"):
        command.make_context("run.py", JOB_ARGS)
