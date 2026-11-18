"""Tests of the configuration loader and the geometry derived from it."""

from pathlib import Path

import pytest

from recon.algo.utils import RegulariserChoice
from recon.config import ConfigError, ReconConfig, load_config

MINIMAL = {"SETUP": "ctp528-ideal", "N_PROJECTIONS": "720"}


def load(**environ: str) -> ReconConfig:
    return load_config(env_file="", environ={**MINIMAL, **environ})


def test_minimal_environment_uses_defaults():
    config = load()
    assert config.setup == "ctp528-ideal"
    assert config.n_projections == 720
    assert config.target_dose == 5.0
    assert config.pixel_w == 0.5
    assert config.regulariser is RegulariserChoice.NONE
    assert config.alpha is None
    assert config.matrix_dir is None


def test_values_are_parsed_to_field_types():
    config = load(
        PIXEL_W="0.25",
        N_SLICES="5",
        REGULARISER="dTV",
        ALPHA="1e-3",
        ETA="0.1",
        MATRIX_DIR="/data/matrix",
        PROX_MAX_LOOPS="2000",
    )
    assert config.pixel_w == 0.25
    assert config.n_slices == 5
    assert config.regulariser is RegulariserChoice.DIRECTIONAL_TV
    assert config.alpha == pytest.approx(1e-3)
    assert config.eta == pytest.approx(0.1)
    assert config.matrix_dir == Path("/data/matrix")
    assert config.prox_max_loops == 2000


def test_empty_values_are_treated_as_unset():
    config = load(ALPHA="", ETA="")
    assert config.alpha is None
    assert config.eta is None


def test_missing_required_variable():
    with pytest.raises(ConfigError, match="N_PROJECTIONS"):
        load_config(env_file="", environ={"SETUP": "ctp528-ideal"})


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"N_PROJECTIONS": "many"}, "N_PROJECTIONS='many' is not a valid int"),
        ({"REGULARISER": "tv"}, "REGULARISER='tv' is not a valid RegulariserChoice"),
        ({"SETUP": "ctp528"}, "SETUP must be of the form"),
        ({"SETUP": "unknown-ideal"}, "Unknown phantom"),
        ({"SETUP": "ctp528-blurry"}, "Unknown phantom or dataset"),
        ({"X_INIT": "random"}, "X_INIT must be 'zero' or 'fbp'"),
    ],
)
def test_invalid_values(environ: dict[str, str], message: str):
    with pytest.raises(ConfigError, match=message):
        load(**environ)


def test_environment_overrides_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("SETUP=gammex-realistic\nN_PROJECTIONS=45\nPIXEL_W=1.0\n")
    config = load_config(env_file=env_file, environ={"PIXEL_W": "0.5"})
    assert config.setup == "gammex-realistic"
    assert config.n_projections == 45
    assert config.pixel_w == 0.5


def test_overrides_take_precedence_and_must_be_known():
    config = load_config(env_file="", environ=MINIMAL, n_projections=90, alpha=2e-3)
    assert config.n_projections == 90
    assert config.alpha == 2e-3
    with pytest.raises(ConfigError, match="Unknown configuration fields: gamma"):
        load_config(env_file="", environ=MINIMAL, gamma=1.0)


def test_config_is_frozen():
    config = load()
    with pytest.raises(AttributeError):
        config.pixel_w = 1.0  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("setup", "pixel_w", "radius", "detector_distance", "image_w"),
    [
        ("ctp528-ideal", 0.25, 100, 10, 880),
        ("ctp528-realistic", 0.5, 100, 150, 440),
        ("spiral-ideal", 0.5, 100, 10, 440),
        ("gammex-ideal", 0.5, 165, 10, 700),
        ("gammex-realistic", 1.0, 165, 150, 350),
    ],
)
def test_derived_geometry(setup: str, pixel_w: float, radius: int, detector_distance: int, image_w: int):
    config = load(SETUP=setup, PIXEL_W=str(pixel_w))
    assert (config.phantom, config.dataset) == tuple(setup.split("-"))
    assert config.radius == radius
    assert config.detector_distance == detector_distance
    assert config.centre_of_rotation == radius + detector_distance
    assert config.image_halfspace_xz == radius + 10
    assert config.image_halfspace_yz == 1.5
    assert config.image_w == image_w
    assert config.image_shape == (3, image_w, image_w)


def test_as_dict_is_plain_and_includes_geometry():
    values = load(MATRIX_DIR="/data/matrix", REGULARISER="TV").as_dict()
    assert values["matrix_dir"] == "/data/matrix"
    assert values["regulariser"] == "TV"
    assert values["image_w"] == 440
    assert values["phantom"] == "ctp528"
    assert all(isinstance(v, str | int | float | type(None)) for v in values.values())
