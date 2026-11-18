"""Regularisers and proximal operators used in the FISTA reconstruction.

Each regulariser evaluates its penalty with ``__call__`` and applies its
proximal operator with ``prox``. The proximal steps of the total variation (TV)
and the directional total variation (dTV) are computed by the fast gradient
projection solvers of the CCPi regularisation toolkit; the penalties are
evaluated here with forward differences. dTV follows Ehrhardt & Betcke (2016):
the reference gradient is normalised as ``xi = grad(v) / sqrt(|grad(v)|^2 +
eta^2)`` and the penalty is the sum of ``|(I - xi xi^T) grad(u)|``.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from ccpi.filters.regularisers import FGP_TV, FGP_dTV
from loguru import logger

Image = TypeVar("Image", bound=np.ndarray[Any, np.dtype[np.float32]])
ProxInfo = TypeVar("ProxInfo", bound=np.ndarray[Any, np.dtype[np.float32]])
ZeroMask = tuple[list[int], list[int], list[int]]
EdgeHeightMethod = Literal["absolute", "robust"]


class Isotropy(int, Enum):
    """Choice between isotropic and anisotropic total variation."""

    ISOTROPIC = 0
    ANISOTROPIC = 1


class NonNegativity(int, Enum):
    """Choice of whether non-negativity is enforced."""

    DISABLED = 0
    ENABLED = 1


def compute_grad(x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Compute the forward-difference gradient of a 2D or 3D image.

    The difference at the last index along each axis is zero (zero boundary
    condition), as in the CCPi solvers.

    Args:
        x: Input image.

    Returns:
        The gradient stacked along the first axis, one component per image axis,
        in the order ``(x, y[, z])`` = axes ``(-1, -2[, -3])``.

    Raises:
        ValueError: If x is neither 2D nor 3D.
    """
    if x.ndim not in (2, 3):
        msg = f"The number of dimensions must be 2 or 3, got {x.ndim}."
        raise ValueError(msg)
    grad = np.zeros((x.ndim, *x.shape))
    for component, axis in enumerate(range(x.ndim - 1, -1, -1)):
        index = [slice(None)] * x.ndim
        index[axis] = slice(None, -1)
        grad[component][tuple(index)] = np.diff(x, axis=axis)
    return grad


def compute_max_edge_height(x: npt.NDArray[Any], method: EdgeHeightMethod = "absolute") -> float:
    """Return the largest gradient magnitude of an image.

    Args:
        x: Input image.
        method: ``absolute`` for the maximum of the gradient norm, ``robust`` for
            its 99th percentile, which ignores isolated spikes in the reference.

    Returns:
        The edge height.

    Raises:
        ValueError: If the method is unknown.
    """
    grad_norm = np.linalg.norm(compute_grad(x), axis=0)
    match method:
        case "absolute":
            return float(grad_norm.max())
        case "robust":
            return float(np.percentile(grad_norm, 99))
        case _:
            msg = f"Invalid edge height method: {method!r} (expected 'absolute' or 'robust')"
            raise ValueError(msg)


class Regulariser(ABC):
    """Base class for regularisers with a proximal step."""

    @abstractmethod
    def __call__(self, x: Image) -> float:
        """Evaluate the regularisation penalty for x."""

    @abstractmethod
    def prox(self, x: Image) -> tuple[Image, ProxInfo | None]:
        """Compute the proximal operator at x; return the image and the solver info, if any."""

    @staticmethod
    def apply_constraints(x: Image, non_negativity: NonNegativity, zero_mask: ZeroMask | None) -> Image:
        """Set negative voxels and the masked voxels to zero, in place.

        Args:
            x: Image to constrain.
            non_negativity: Whether to enforce non-negativity; the mask is only
                applied when it is enabled.
            zero_mask: Voxel indices to set to zero, or ``None``.

        Returns:
            The same array.
        """
        if non_negativity == NonNegativity.ENABLED:
            x[x < 0] = 0
            if zero_mask:
                x[zero_mask] = 0
        return x


class NoRegulariser(Regulariser):
    """Regulariser that applies only the non-negativity and zero-mask constraints."""

    def __init__(self, non_negativity: NonNegativity, zero_mask: ZeroMask | None = None) -> None:
        """Store the constraint settings."""
        self.non_negativity = non_negativity
        self.zero_mask = zero_mask

    def __call__(self, x: Image) -> float:  # noqa: ARG002
        """Return zero: there is no penalty."""
        return 0.0

    def prox(self, x: Image) -> tuple[Image, None]:
        """Apply the constraints only; there is no solver info."""
        return self.apply_constraints(x, self.non_negativity, self.zero_mask), None


class TVRegulariser(Regulariser):
    """Total variation regulariser solved with the FGP proximal method."""

    def __init__(
        self,
        alpha: float,
        prox_max_loops: int,
        prox_tolerance: float,
        isotropy: Isotropy,
        non_negativity: NonNegativity,
        zero_mask: ZeroMask | None = None,
    ) -> None:
        """Store the regularisation parameters.

        Args:
            alpha: Regularisation strength.
            prox_max_loops: Maximum number of iterations of the proximal solver.
            prox_tolerance: Tolerance of the proximal solver.
            isotropy: Isotropic or anisotropic total variation.
            non_negativity: Whether to enforce non-negativity.
            zero_mask: Mask of voxels to set to zero.
        """
        self.alpha = alpha
        self.prox_max_loops = prox_max_loops
        self.prox_tolerance = prox_tolerance
        self.isotropy = isotropy
        self.non_negativity = non_negativity
        self.zero_mask = zero_mask

    def __call__(self, x: Image) -> float:
        """Compute the isotropic total variation of a 2D or 3D image."""
        return float(np.linalg.norm(compute_grad(x), axis=0).sum())

    def prox(self, x: Image) -> tuple[Image, npt.NDArray[np.float32]]:
        """Apply the FGP total variation proximal step and the constraints."""
        info = np.zeros(2, dtype=np.float32)
        logger.debug(
            f"{x.shape=}, {self.alpha=}, {self.prox_max_loops=}, {self.prox_tolerance=}, "
            f"{self.isotropy.value=}, {NonNegativity.DISABLED.value=}, {info=}",
        )
        x = FGP_TV(
            x,
            self.alpha,
            self.prox_max_loops,
            self.prox_tolerance,
            self.isotropy.value,
            NonNegativity.DISABLED.value,
            infovector=info,
        )
        return self.apply_constraints(x, self.non_negativity, self.zero_mask), info


class DirectionalTVRegulariser(Regulariser):
    """Directional total variation regulariser solved with the FGP proximal method."""

    def __init__(
        self,
        alpha: float,
        eta: float,
        prox_max_loops: int,
        prox_tolerance: float,
        isotropy: Isotropy,
        non_negativity: NonNegativity,
        xray_ref: Image,
        zero_mask: ZeroMask | None = None,
        edge_height_method: EdgeHeightMethod = "absolute",
    ) -> None:
        """Store the regularisation parameters and the x-ray reference.

        The smoothing parameter of the reference gradient is ``eta`` times the
        largest edge height of the reference, measured with
        ``edge_height_method``; the same value is used by the penalty and by the
        proximal step.

        Args:
            alpha: Regularisation strength.
            eta: Smoothing factor, relative to the largest edge height of the reference.
            prox_max_loops: Maximum number of iterations of the proximal solver.
            prox_tolerance: Tolerance of the proximal solver.
            isotropy: Isotropic or anisotropic total variation.
            non_negativity: Whether to enforce non-negativity.
            xray_ref: Reference image that sets the edge directions.
            zero_mask: Mask of voxels to set to zero.
            edge_height_method: How the largest edge height is computed, see
                :func:`compute_max_edge_height`.
        """
        self.alpha = alpha
        self.prox_max_loops = prox_max_loops
        self.prox_tolerance = prox_tolerance
        self.isotropy = isotropy
        self.non_negativity = non_negativity
        self.zero_mask = zero_mask
        self.xray_ref = xray_ref
        self.eta = eta
        self.edge_height_method = edge_height_method
        self.eta_effective = eta * compute_max_edge_height(xray_ref, edge_height_method)

    def __call__(self, x: Image) -> float:
        """Compute the isotropic directional total variation of a 2D or 3D image.

        Args:
            x: Input image, of the same shape as the reference.

        Returns:
            The sum over voxels of the gradient of x projected orthogonally to the
            smoothed reference gradient direction.
        """
        grad_ref = compute_grad(self.xray_ref)
        xi = grad_ref / np.sqrt((grad_ref**2).sum(axis=0) + self.eta_effective**2)

        grad_x = compute_grad(x)
        projected = grad_x - (grad_x * xi).sum(axis=0) * xi

        return float(np.linalg.norm(projected, axis=0).sum())

    def prox(self, x: Image) -> tuple[Image, npt.NDArray[np.float32]]:
        """Apply the FGP directional total variation proximal step and the constraints."""
        info = np.zeros(2, dtype=np.float32)
        x = FGP_dTV(
            x,
            self.xray_ref,
            self.alpha,
            self.prox_max_loops,
            self.prox_tolerance,
            self.eta_effective,
            self.isotropy.value,
            NonNegativity.DISABLED.value,
            infovector=info,
        )
        return self.apply_constraints(x, self.non_negativity, self.zero_mask), info
