"""Tests of the FISTA helpers that do not need pct-tools data or the CCPi toolkit."""

from pathlib import Path

import numpy as np
import pytest
from blosc2 import load_array
from loguru import logger

from phantoms import PHANTOMS_DIR
from recon.algo.fista import (
    build_regulariser,
    collect_chunks,
    dump_iterates,
    format_alpha_string,
    load_initial_image,
    load_step_size,
    load_xray_ref,
    log_settings,
    output_dir_name,
)
from recon.algo.regularisation import DirectionalTVRegulariser, NoRegulariser, TVRegulariser
from recon.algo.utils import XrayReference, compute_zero_mask
from recon.config import ConfigError, ReconConfig, load_config

# ctp528 has a 110 mm xz halfspace, so 27.5 mm pixels give an 8-pixel image.
SMALL = {"SETUP": "ctp528-ideal", "N_PROJECTIONS": "4", "PIXEL_W": "27.5"}


def make_config(**environ: str) -> ReconConfig:
    return load_config(env_file="", environ={**SMALL, **environ})


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (1e-3, "1e-3"),
        (1.58e-3, "1.58e-3"),
        (2.5e-3, "2.5e-3"),
        (0.1, "1e-1"),
        (2e-4, "2e-4"),
        (1.0, "1e0"),
        ("1e-3", "1e-3"),
    ],
)
def test_format_alpha_string(alpha, expected):
    assert format_alpha_string(alpha) == expected


def test_format_alpha_string_rejects_bad_input():
    with pytest.raises(TypeError):
        format_alpha_string(1)
    with pytest.raises(ValueError, match="positive"):
        format_alpha_string(-1e-3)


def test_compute_zero_mask_counts_and_symmetry():
    # 8x8 grid, pixel centres at offsets +-0.5, +-1.5, ... from the middle; with radius 2
    # and no margin the inside is the 12 pixels with (dx, dy) in {(+-0.5, +-0.5), (+-0.5, +-1.5),
    # (+-1.5, +-0.5)}, so 52 of 64 pixels are masked, in every slice.
    mask = compute_zero_mask((3, 8, 8), radius=2, margin=0)

    k, i, j = (np.array(m) for m in mask)
    assert len(k) == 3 * 52
    assert set(k) == {0, 1, 2}

    masked = {(a, b) for a, b in zip(i[k == 0], j[k == 0], strict=True)}
    assert len(masked) == 52
    assert masked == {(7 - a, b) for a, b in masked}
    assert masked == {(a, 7 - b) for a, b in masked}
    assert masked == {(b, a) for a, b in masked}


def test_compute_zero_mask_margin_adds_to_radius():
    assert len(compute_zero_mask((1, 8, 8), radius=1, margin=1)[0]) == len(
        compute_zero_mask((1, 8, 8), radius=2, margin=0)[0],
    )


def test_load_step_size_reads_the_thesis_table():
    config = load_config(
        env_file="", environ={"SETUP": "ctp528-ideal", "N_PROJECTIONS": "720", "PIXEL_W": "0.25"}
    )
    assert config.image_w == 880
    assert load_step_size(config) == pytest.approx(2.872e-6)


def test_load_initial_image_zero():
    x0 = load_initial_image(make_config())
    assert x0.shape == (3, 8, 8)
    assert not x0.any()


def test_load_initial_image_fbp_requires_fbp_dir():
    with pytest.raises(ConfigError, match="FBP_DIR"):
        load_initial_image(make_config(X_INIT="fbp"))


def test_load_initial_image_fbp_stacks_a_2d_image(tmp_path: Path):
    fbp = np.arange(64, dtype=np.float32).reshape(8, 8)
    np.save(tmp_path / "fbp_5mGy_8pix.npy", fbp)

    x0 = load_initial_image(make_config(X_INIT="fbp", FBP_DIR=str(tmp_path), TARGET_DOSE="5.0"))

    assert x0.shape == (3, 8, 8)
    np.testing.assert_array_equal(x0[1], fbp)


def test_collect_chunks_orders_by_angle_and_pairs_vectors(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    for stem in ["270.0", "0.0_1", "90.0", "0.0_0", "180.0"]:
        (matrix_dir / f"{stem}.mat").touch()
    matrices, vectors = collect_chunks(matrix_dir, tmp_path / "b", n_projections=4)
    stems = [Path(m).stem for m in matrices]

    assert stems == ["0.0_0", "0.0_1", "90.0", "180.0", "270.0"]
    assert vectors == [str(tmp_path / "b" / f"b_{stem}.vec") for stem in stems]


def test_dump_iterates_round_trip_and_overwrite(tmp_path: Path):
    yk_list = [np.zeros((1, 2, 2)), np.ones((1, 2, 2))]
    metrics_list = [[3.0, 2.0, 1.0]]
    dump_iterates(tmp_path, "final", [np.full((1, 2, 2), 7.0)], [[0.0, 0.0, 0.0]])

    dump_iterates(tmp_path, "final", yk_list, metrics_list)

    yk = load_array(str(tmp_path / "yk_final.bl2"))
    metrics = load_array(str(tmp_path / "metrics_final.bl2"))

    assert yk.dtype == np.float32
    np.testing.assert_array_equal(yk, np.array(yk_list))
    np.testing.assert_array_equal(metrics, np.array(metrics_list))


def test_output_dir_name_matches_the_thesis_layout():
    config = load_config(
        env_file="",
        environ={"SETUP": "ctp528-ideal", "N_PROJECTIONS": "720", "PIXEL_W": "0.25", "ITER_END": "200"},
    )
    assert (
        output_dir_name(config, "TV_iso_alpha_2e-3")
        == "fista_exact_init_zero_880pix_5.0mGy_720proj_TV_iso_alpha_2e-3_iter200"
    )


def test_log_settings_records_the_regularisation_settings():
    config = make_config(REGULARISER="dTV", ALPHA="2.5e-3", ETA="0.1", PROX_MAX_LOOPS="2000", ITER_END="200")
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")

    try:
        log_settings(config, step_size=1e-5)
    finally:
        logger.remove(sink_id)

    assert "Iterations: 0 to 200" in messages
    assert "Regulariser: dTV" in messages
    assert "Regularisation parameter alpha: 0.0025" in messages
    assert "dTV edge parameter eta: 0.1" in messages
    assert "Proximal step tolerance: 1e-07" in messages
    assert "Proximal step maximum iterations: 2000" in messages


def test_build_regulariser_none():
    regulariser, description = build_regulariser(make_config(), zero_mask=([0], [0], [0]))

    assert isinstance(regulariser, NoRegulariser)
    assert description == "no_reg"


def test_build_regulariser_tv():
    regulariser, description = build_regulariser(make_config(REGULARISER="TV", ALPHA="2e-3"), zero_mask=None)

    assert isinstance(regulariser, TVRegulariser)
    assert regulariser.alpha == pytest.approx(2e-3)
    assert description == "TV_iso_alpha_2e-3"


@pytest.mark.parametrize("environ", [{"REGULARISER": "TV"}, {"REGULARISER": "TV", "ALPHA": "0"}])
def test_build_regulariser_needs_a_positive_alpha(environ):
    with pytest.raises(ConfigError, match="ALPHA"):
        build_regulariser(make_config(**environ), zero_mask=None)


def test_build_regulariser_dtv(tmp_path: Path):
    ref = np.zeros((8, 8), dtype=np.float32)
    ref[:, 4:] = 1.0
    np.save(tmp_path / "ctp528_xray_ref_realistic_slice_filtered_8pix.npy", ref)
    config = make_config(REGULARISER="dTV", ALPHA="2e-3", ETA="0.1", REF_DIR=str(tmp_path))

    regulariser, description = build_regulariser(config, zero_mask=None)

    assert isinstance(regulariser, DirectionalTVRegulariser)
    assert regulariser.xray_ref.shape == (3, 8, 8)
    assert regulariser.xray_ref.dtype == np.float32
    assert regulariser.eta_effective == pytest.approx(0.1)  # eta times the step height of 1
    assert description == "dTV_iso_alpha_2e-3_eta_1e-1"


def test_load_xray_ref_keeps_the_stored_orientation(tmp_path: Path):
    ref = np.arange(64, dtype=np.float64).reshape(8, 8)
    np.save(tmp_path / "ctp528_xray_ref_realistic_slice_filtered_8pix.npy", ref)

    xray_ref = load_xray_ref(tmp_path, "ctp528", 8)

    for xray_ref_slice in xray_ref:
        np.testing.assert_array_equal(xray_ref_slice, ref)


@pytest.mark.parametrize(
    ("variant", "filename"),
    [
        (XrayReference.REALISTIC, "ctp528_xray_ref_realistic_8pix.npy"),
        (XrayReference.REALISTIC_SLICE, "ctp528_xray_ref_realistic_slice_8pix.npy"),
        (XrayReference.REALISTIC_SLICE_FILTERED, "ctp528_xray_ref_realistic_slice_filtered_8pix.npy"),
        (XrayReference.DETERIORATED, "ctp528_xray_ref_8pix_1.0sig_0.02n.npy"),
    ],
)
def test_load_xray_ref_loads_the_file_of_the_variant(tmp_path: Path, variant: XrayReference, filename: str):
    ref = np.ones((8, 8))
    np.save(tmp_path / filename, ref)

    xray_ref = load_xray_ref(tmp_path, "ctp528", 8, variant=variant)

    assert xray_ref.shape == (3, 8, 8)


@pytest.mark.parametrize(("phantom", "image_w"), [("ctp528", 880), ("gammex", 700), ("spiral", 880)])
def test_load_xray_ref_aligns_with_the_xray_ground_truth(phantom: str, image_w: int):
    ground_truth = np.load(PHANTOMS_DIR / "ground_truths" / f"{phantom}_xray_gt_{image_w}pix.npy")

    xray_ref = load_xray_ref(PHANTOMS_DIR / "xray_refs", phantom, image_w)[1]

    error = np.abs(xray_ref - ground_truth).mean()
    flipped_error = np.abs(np.flipud(xray_ref) - ground_truth).mean()
    assert error < flipped_error / 2


def test_build_regulariser_dtv_needs_eta_and_ref_dir():
    with pytest.raises(ConfigError, match="ETA and REF_DIR"):
        build_regulariser(make_config(REGULARISER="dTV", ALPHA="2e-3"), zero_mask=None)
