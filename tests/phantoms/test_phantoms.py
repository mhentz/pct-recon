"""Tests of the phantom drawing helpers and the simulated x-ray CT acquisition."""

from itertools import combinations, pairwise

import numpy as np
import pytest

from phantoms.ground_truth_spiral import calculate_insert_positions
from phantoms.utils import draw_circle_in_image, generate_realistic_ct_image, get_circle_image

# Spiral phantom: 100 mm radius, 6 mm inserts on five arms, as in ground_truth_spiral.main
SPIRAL_RADIUS = 100
INSERT_RADIUS = 6
ARM_ANGLES = [2.75 - i * 2 / 5 * np.pi for i in range(5)]


def test_get_circle_image_draws_a_centred_disk_of_the_expected_area():
    im = get_circle_image(radius=10, pixel_width=0.5, background_value=0.0, circle_value=1.0)

    n_inside = np.count_nonzero(im == 1.0)

    assert im.shape == (48, 48)  # 2 * (10 + 2) mm at 0.5 mm per pixel
    assert set(np.unique(im)) == {0.0, 1.0}
    assert abs(n_inside - np.pi * 20**2) < 2 * np.pi * 20  # within one ring of pixels
    np.testing.assert_array_equal(im, np.flipud(im))
    np.testing.assert_array_equal(im, np.fliplr(im))


def test_draw_circle_in_image_centres_the_disk_on_the_position_in_mm():
    im = np.zeros((40, 40))

    im = draw_circle_in_image(im, centre_in_mm=(5.0, 8.0), radius=2.0, pixel_width=0.5, circle_value=1.0)

    rows, cols = np.nonzero(im)
    assert rows.mean() == pytest.approx(5.0 / 0.5 - 0.5)
    assert cols.mean() == pytest.approx(8.0 / 0.5 - 0.5)
    assert np.count_nonzero(im) == pytest.approx(np.pi * 4**2, abs=2 * np.pi * 4)


@pytest.mark.parametrize("theta_init", ARM_ANGLES)
def test_calculate_insert_positions_places_four_inserts_inside_the_phantom(theta_init: float):
    positions = calculate_insert_positions(theta_init)

    radii = [np.hypot(x, y) for x, y in positions]
    assert len(positions) == 4
    assert all(r + INSERT_RADIUS < SPIRAL_RADIUS for r in radii)
    assert all(r_in < r_out for r_in, r_out in pairwise(radii))


def test_spiral_inserts_do_not_overlap():
    inserts = [(0, 0)] + [p for theta in ARM_ANGLES for p in calculate_insert_positions(theta)]

    distances = [np.hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in combinations(inserts, 2)]

    assert len(inserts) == 21
    assert min(distances) > 2 * INSERT_RADIUS


def make_ct_input() -> tuple[np.ndarray, np.ndarray]:
    im = get_circle_image(radius=10, pixel_width=0.5, background_value=0.0, circle_value=0.17)
    theta = np.linspace(0, 180, 40, endpoint=False)
    return im, theta


def test_generate_realistic_ct_image_reconstructs_the_attenuation():
    im, theta = make_ct_input()

    thick_slice, single_slice = generate_realistic_ct_image(
        im, theta, image_w=48, pixel_w=0.5, initial_intensity=7e9, random_seed=1
    )

    assert thick_slice.shape == single_slice.shape == (48, 48)
    assert np.isfinite(thick_slice).all()
    assert np.isfinite(single_slice).all()
    assert thick_slice[16:32, 16:32].mean() == pytest.approx(0.17, rel=0.05)


def test_generate_realistic_ct_image_averages_all_twelve_reconstructions(mocker):
    # The i-th reconstruction is uniformly i, in 1/mm; a pixel width of 10 mm makes the
    # conversion to 1/cm a factor of one.
    im, theta = make_ct_input()
    reconstructions = [np.full((4, 4), float(i)) for i in range(12)]
    mocker.patch("phantoms.utils.iradon", side_effect=reconstructions)

    thick_slice, single_slice = generate_realistic_ct_image(im, theta, image_w=4, pixel_w=10.0)

    assert thick_slice == pytest.approx(np.full((4, 4), 5.5))
    assert single_slice == pytest.approx(np.full((4, 4), 11.0))


def test_generate_realistic_ct_image_is_reproducible_with_a_seed():
    im, theta = make_ct_input()

    first = generate_realistic_ct_image(im, theta, image_w=48, pixel_w=0.5, random_seed=1)
    second = generate_realistic_ct_image(im, theta, image_w=48, pixel_w=0.5, random_seed=1)
    other_seed = generate_realistic_ct_image(im, theta, image_w=48, pixel_w=0.5, random_seed=2)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other_seed[0])
