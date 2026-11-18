"""Tests of the TV and dTV penalties and of the proximal steps (which need the CCPi toolkit)."""

import numpy as np
import pytest

from recon.algo.regularisation import (
    DirectionalTVRegulariser,
    Isotropy,
    NonNegativity,
    NoRegulariser,
    Regulariser,
    TVRegulariser,
    compute_grad,
    compute_max_edge_height,
)


def step_image(height: float = 1.0, *, vertical_edge: bool = True, width: int = 8) -> np.ndarray:
    """One square slice that jumps by ``height`` across its middle column (or row)."""
    image = np.zeros((1, width, width), dtype=np.float32)
    if vertical_edge:
        image[:, :, width // 2 :] = height
    else:
        image[:, width // 2 :, :] = height
    return image


def tv_regulariser(zero_mask=None) -> TVRegulariser:
    return TVRegulariser(
        alpha=0.1,
        prox_max_loops=100,
        prox_tolerance=1e-7,
        isotropy=Isotropy.ISOTROPIC,
        non_negativity=NonNegativity.ENABLED,
        zero_mask=zero_mask,
    )


def dtv_regulariser(
    xray_ref: np.ndarray, eta: float = 0.01, edge_height_method="absolute"
) -> DirectionalTVRegulariser:
    return DirectionalTVRegulariser(
        alpha=0.1,
        eta=eta,
        prox_max_loops=100,
        prox_tolerance=1e-7,
        isotropy=Isotropy.ISOTROPIC,
        non_negativity=NonNegativity.ENABLED,
        xray_ref=xray_ref,
        edge_height_method=edge_height_method,
    )


def test_regulariser_is_abstract():
    with pytest.raises(TypeError):
        Regulariser()  # ty: ignore[abstract-class-instantiation]


@pytest.mark.parametrize("shape", [(5, 6), (3, 5, 6)])
def test_compute_grad_is_a_forward_difference_with_zero_boundary(shape):
    rng = np.random.default_rng(0)
    x = rng.random(shape)

    grad = compute_grad(x)

    assert grad.shape == (len(shape), *shape)
    for component, axis in enumerate(reversed(range(len(shape)))):
        expected = np.zeros(shape)
        front = [slice(None)] * len(shape)
        front[axis] = slice(None, -1)
        expected[tuple(front)] = np.diff(x, axis=axis)
        np.testing.assert_array_equal(grad[component], expected)


def test_compute_grad_rejects_other_dimensions():
    with pytest.raises(ValueError, match="2 or 3"):
        compute_grad(np.zeros(8))


def test_compute_max_edge_height_absolute_and_robust():
    # 64x64 so that the 99th percentile falls inside the 64 unit edge pixels, not next to the spike
    ref = step_image(height=1.0, width=64)
    ref[0, 0, 0] = 50.0  # one spike in the corner

    absolute = compute_max_edge_height(ref, "absolute")
    robust = compute_max_edge_height(ref, "robust")

    assert absolute == pytest.approx(50.0 * np.sqrt(2))  # the spike has a 50 step in x and in y
    assert robust == pytest.approx(1.0)
    with pytest.raises(ValueError, match="edge height method"):
        compute_max_edge_height(ref, "median")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("ndim", [2, 3])
def test_tv_of_a_step_is_height_times_edge_length(ndim):
    image = step_image(height=2.5)
    image = image if ndim == 3 else image[0]

    tv = tv_regulariser()(image)

    assert tv == pytest.approx(2.5 * 8)


def test_dtv_with_aligned_reference_edges_is_almost_zero():
    image = step_image(height=2.5)
    reference = step_image(height=1.0)

    dtv = dtv_regulariser(reference, eta=0.01)(image)

    # the residual is TV * eta^2 / (1 + eta^2) for a unit reference step
    assert dtv == pytest.approx(2.5 * 8 * 1e-4 / (1 + 1e-4), rel=1e-6)
    assert dtv < 1e-3 * tv_regulariser()(image)


def test_dtv_with_orthogonal_reference_edges_equals_tv():
    image = step_image(height=2.5, vertical_edge=True)
    reference = step_image(height=1.0, vertical_edge=False)

    dtv = dtv_regulariser(reference)(image)

    assert dtv == pytest.approx(tv_regulariser()(image))


@pytest.mark.parametrize("edge_height_method", ["absolute", "robust"])
def test_dtv_matches_the_definition_of_ehrhardt_and_betcke(edge_height_method):
    rng = np.random.default_rng(1)
    image = rng.random((3, 6, 7)).astype(np.float32)
    reference = rng.random((3, 6, 7)).astype(np.float32)
    regulariser = dtv_regulariser(reference, eta=0.1, edge_height_method=edge_height_method)

    dtv = regulariser(image)

    # Definition 3.4 with Remark 3.5: xi = grad(v) / sqrt(|grad(v)|^2 + eta^2),
    # dTV = sum over voxels of |grad(u) - <grad(u), xi> xi|
    grad_v = compute_grad(reference)
    grad_norm = np.sqrt((grad_v**2).sum(axis=0))
    eta = 0.1 * (grad_norm.max() if edge_height_method == "absolute" else np.percentile(grad_norm, 99))
    xi = grad_v / np.sqrt(grad_norm**2 + eta**2)
    grad_u = compute_grad(image)
    expected = np.sqrt(((grad_u - (grad_u * xi).sum(axis=0) * xi) ** 2).sum(axis=0)).sum()
    assert regulariser.eta_effective == pytest.approx(eta)
    assert dtv == pytest.approx(expected, rel=1e-9)


def test_no_regulariser_prox_applies_only_the_constraints():
    image = np.array([[[-1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    regulariser = NoRegulariser(NonNegativity.ENABLED, zero_mask=([0], [1], [1]))

    result, info = regulariser.prox(image)

    assert info is None
    np.testing.assert_array_equal(result, [[[0.0, 2.0], [3.0, 0.0]]])
    assert regulariser(image) == 0.0


def test_no_regulariser_prox_without_non_negativity_leaves_the_image():
    image = np.array([[[-1.0, 2.0]]], dtype=np.float32)

    result, _ = NoRegulariser(NonNegativity.DISABLED, zero_mask=([0], [0], [0])).prox(image)

    np.testing.assert_array_equal(result, [[[-1.0, 2.0]]])


def test_tv_prox_reduces_the_penalty_and_applies_the_constraints():
    rng = np.random.default_rng(2)
    noisy = (step_image(height=1.0) + 0.1 * rng.standard_normal((1, 8, 8))).astype(np.float32)
    noisy[0, 0, 0] = -1.0
    regulariser = tv_regulariser(zero_mask=([0], [7], [7]))
    tv_before = regulariser(noisy)

    denoised, info = regulariser.prox(noisy.copy())

    assert denoised.shape == noisy.shape
    assert info.shape == (2,)
    assert regulariser(denoised) < tv_before
    assert denoised.min() >= 0.0
    assert denoised[0, 7, 7] == 0.0


def test_dtv_prox_reduces_the_penalty_and_keeps_the_aligned_edge():
    rng = np.random.default_rng(4)
    reference = step_image(height=1.0)
    noisy = (step_image(height=2.0) + 0.1 * rng.standard_normal((1, 8, 8))).astype(np.float32)
    regulariser = dtv_regulariser(reference, eta=0.01)
    dtv_before = regulariser(noisy)

    denoised, info = regulariser.prox(noisy.copy())

    assert denoised.shape == noisy.shape
    assert info.shape == (2,)
    assert regulariser(denoised) < dtv_before
    assert denoised[0, :, 4:].mean() - denoised[0, :, :4].mean() == pytest.approx(2.0, abs=0.1)
