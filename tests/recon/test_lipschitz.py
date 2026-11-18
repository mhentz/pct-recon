"""Tests of the step-size conversion in ``recon/lipschitz/convert.py``."""

from pathlib import Path

import pytest
import yaml

from recon.lipschitz import convert
from recon.lipschitz.convert import format_step_size

STEP_SIZES_PATH = Path(convert.__file__).parent / "step_sizes.yml"


@pytest.mark.parametrize(
    ("lipschitz_c", "expected"),
    [
        (348124.1875, "2.872e-6"),  # 1 / L = 2.87254e-6, which rounding would turn into 2.873e-6
        (1e5, "1.000e-5"),
        (8.0, "1.250e-1"),
        (3.0, "3.333e-1"),
    ],
)
def test_format_step_size_rounds_down(lipschitz_c, expected):
    assert format_step_size(lipschitz_c) == expected


def test_format_step_size_reproduces_recorded_step_sizes():
    with convert.LIPSCHITZ_CONSTANTS_PATH.open() as f:
        constants = yaml.safe_load(f)
    with STEP_SIZES_PATH.open() as f:
        step_sizes = yaml.safe_load(f)

    mismatches = [
        (setup, dose, n_projections, image_w)
        for setup in constants
        for dose in constants[setup]
        for n_projections in constants[setup][dose]
        for image_w, lipschitz_c in constants[setup][dose][n_projections].items()
        if float(format_step_size(lipschitz_c)) != step_sizes[setup][dose][n_projections][image_w]
    ]

    # The one recorded value that was rounded up (3.238e-6 for 1 / L = 3.2379e-6).
    assert mismatches == [("gammex-realistic", 1.0, 45, 700)]


def test_main_prints_constant_and_step_size(tmp_path, capsys):
    path = tmp_path / "constants.yml"
    path.write_text("ctp528-ideal:\n  5.0:\n    720:\n      880: 348124.1875\n")

    convert.main(path)

    out = capsys.readouterr().out
    assert "ctp528-ideal 5.0 720 880" in out
    assert "lips: 348124.1875" in out
    assert "step: 2.872e-6" in out
