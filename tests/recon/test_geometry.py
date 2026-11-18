"""Tests of the per-phantom geometry in ``recon/geometry.py``."""

import pytest

from recon.geometry import (
    get_centre_of_rotation,
    get_detector_distance,
    get_image_halfspace_xz,
    get_image_halfspace_yz,
    get_image_w,
    get_radius,
)


@pytest.mark.parametrize(
    ("phantom", "dataset", "radius", "detector_distance", "halfspace_xz"),
    [
        ("ctp528", "ideal", 100, 10, 110),
        ("ctp528", "realistic", 100, 150, 110),
        ("spiral", "ideal", 100, 10, 110),
        ("spiral", "realistic", 100, 150, 110),
        ("gammex", "ideal", 165, 10, 175),
        ("gammex", "realistic", 165, 150, 175),
        ("toy_inserts", "ideal", 8, 2, 10),
        ("toy_water", "ideal", 8, 2, 10),
    ],
)
def test_geometry_table(phantom, dataset, radius, detector_distance, halfspace_xz):
    assert get_radius(phantom) == radius
    assert get_detector_distance(phantom, dataset) == detector_distance
    assert get_centre_of_rotation(phantom, dataset) == radius + detector_distance
    assert get_image_halfspace_xz(phantom) == halfspace_xz


@pytest.mark.parametrize(
    ("phantom", "pixel_w", "image_w"),
    [
        ("ctp528", 0.25, 880),
        ("ctp528", 0.5, 440),
        ("ctp528", 0.3, 733),  # 220 / 0.3 = 733.3, truncated
        ("gammex", 0.5, 700),
        ("gammex", 1.0, 350),
        ("toy_water", 0.5, 40),
    ],
)
def test_get_image_w(phantom, pixel_w, image_w):
    assert get_image_w(phantom, pixel_w) == image_w


@pytest.mark.parametrize(
    ("n_slices", "slice_thickness", "halfspace_yz"),
    [(3, 1.0, 1.5), (1, 0.5, 0.25), (8, 2.0, 8.0)],
)
def test_get_image_halfspace_yz(n_slices, slice_thickness, halfspace_yz):
    assert get_image_halfspace_yz(n_slices, slice_thickness) == halfspace_yz


def test_unknown_phantom_or_dataset_raises():
    with pytest.raises(ValueError, match="Unknown phantom: ctp404"):
        get_radius("ctp404")
    with pytest.raises(ValueError, match="Unknown phantom or dataset"):
        get_detector_distance("ctp528", "noisy")
