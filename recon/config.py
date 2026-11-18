"""Reconstruction settings, read from environment variables and a ``.env`` file.

Nothing is read at import time. Each entry point calls :func:`load_config` once
and passes the resulting :class:`ReconConfig` (or the values it needs) down as
arguments. The variable names are those used by the job scripts in
``scripts/`` and documented in ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import dotenv_values, find_dotenv

from recon import geometry
from recon.algo.utils import RegulariserChoice

if TYPE_CHECKING:
    from collections.abc import Mapping


class ConfigError(ValueError):
    """A required variable is missing or a variable cannot be parsed."""


@dataclass(frozen=True)
class ReconConfig:
    """Scalar settings of one reconstruction.

    Field names are the lower-case form of the environment variables. Geometry
    derived from the setup (phantom radius, detector distance, image size) is
    exposed as properties computed from :mod:`recon.geometry`.
    """

    # acquisition
    setup: str = field(metadata={"env": "SETUP"})
    n_projections: int = field(metadata={"env": "N_PROJECTIONS"})
    target_dose: float = field(default=5.0, metadata={"env": "TARGET_DOSE"})
    n_sigma: float = field(default=3.0, metadata={"env": "N_SIGMA"})

    # image grid (mm)
    pixel_w: float = field(default=0.5, metadata={"env": "PIXEL_W"})
    n_slices: int = field(default=3, metadata={"env": "N_SLICES"})
    slice_thickness: float = field(default=1.0, metadata={"env": "SLICE_THICKNESS"})

    # data locations
    matrix_dir: Path | None = field(default=None, metadata={"env": "MATRIX_DIR"})
    vector_dir: Path | None = field(default=None, metadata={"env": "VECTOR_DIR"})
    fbp_dir: Path | None = field(default=None, metadata={"env": "FBP_DIR"})
    output_dir: Path | None = field(default=None, metadata={"env": "OUTPUT_DIR"})
    ref_dir: Path | None = field(default=None, metadata={"env": "REF_DIR"})

    # reconstruction
    iter_start: int = field(default=0, metadata={"env": "ITER_START"})
    iter_end: int = field(default=100, metadata={"env": "ITER_END"})
    x_init: str = field(default="zero", metadata={"env": "X_INIT"})
    dump_modulo: int = field(default=100, metadata={"env": "DUMP_MODULO"})
    n_threads: int = field(default=4, metadata={"env": "N_THREADS"})

    # regularisation
    regulariser: RegulariserChoice = field(default=RegulariserChoice.NONE, metadata={"env": "REGULARISER"})
    alpha: float | None = field(default=None, metadata={"env": "ALPHA"})
    eta: float | None = field(default=None, metadata={"env": "ETA"})
    prox_tolerance: float = field(default=1e-7, metadata={"env": "PROX_TOLERANCE"})
    prox_max_loops: int = field(default=1000, metadata={"env": "PROX_MAX_LOOPS"})

    def __post_init__(self) -> None:
        """Validate the fields that the geometry depends on.

        Raises:
            ConfigError: If ``setup`` or ``x_init`` has an invalid value.
        """
        if "-" not in self.setup:
            msg = f"SETUP must be of the form <phantom>-<dataset>, got {self.setup!r}"
            raise ConfigError(msg)
        if self.x_init not in ("zero", "fbp"):
            msg = f"X_INIT must be 'zero' or 'fbp', got {self.x_init!r}"
            raise ConfigError(msg)
        try:
            geometry.get_radius(self.phantom)
            geometry.get_detector_distance(self.phantom, self.dataset)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    @property
    def phantom(self) -> str:
        """Phantom name: the part of ``setup`` before the first dash."""
        return self.setup.split("-", maxsplit=1)[0]

    @property
    def dataset(self) -> str:
        """Dataset name: the part of ``setup`` after the first dash."""
        return self.setup.split("-", maxsplit=1)[1]

    @property
    def radius(self) -> int:
        """Phantom radius in mm."""
        return geometry.get_radius(self.phantom)

    @property
    def detector_distance(self) -> int:
        """Distance between the phantom and the detector in mm."""
        return geometry.get_detector_distance(self.phantom, self.dataset)

    @property
    def centre_of_rotation(self) -> float:
        """Position of the centre of rotation in mm."""
        return geometry.get_centre_of_rotation(self.phantom, self.dataset)

    @property
    def image_halfspace_xz(self) -> float:
        """Half-width of the image in the xz plane in mm."""
        return geometry.get_image_halfspace_xz(self.phantom)

    @property
    def image_halfspace_yz(self) -> float:
        """Half-height of the image in the yz plane in mm."""
        return geometry.get_image_halfspace_yz(self.n_slices, self.slice_thickness)

    @property
    def image_w(self) -> int:
        """Image width in pixels."""
        return geometry.get_image_w(self.phantom, self.pixel_w)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Image shape as ``(n_slices, image_w, image_w)``."""
        return (self.n_slices, self.image_w, self.image_w)

    def as_dict(self) -> dict[str, Any]:
        """Return the fields and the derived geometry with plain values, for writing to YAML."""
        values: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            values[f.name] = str(value) if isinstance(value, Path | RegulariserChoice) else value
        values.update(
            phantom=self.phantom,
            dataset=self.dataset,
            radius=self.radius,
            detector_distance=self.detector_distance,
            image_halfspace_xz=self.image_halfspace_xz,
            image_halfspace_yz=self.image_halfspace_yz,
            image_w=self.image_w,
        )
        return values


def _parse(name: str, raw: str, annotation: str) -> Any:  # noqa: ANN401
    """Convert the string value of one environment variable to the field's type.

    Args:
        name: Environment variable name, for error messages.
        raw: Its string value.
        annotation: The dataclass field annotation, as a string.

    Returns:
        The parsed value.

    Raises:
        ConfigError: If the value cannot be parsed.
    """
    base = annotation.removesuffix(" | None")
    try:
        match base:
            case "int":
                return int(raw)
            case "float":
                return float(raw)
            case "Path":
                return Path(raw)
            case "RegulariserChoice":
                return RegulariserChoice(raw)
            case _:
                return raw
    except ValueError as exc:
        choices = f" (one of {', '.join(RegulariserChoice)})" if base == "RegulariserChoice" else ""
        msg = f"{name}={raw!r} is not a valid {base}{choices}"
        raise ConfigError(msg) from exc


def load_config(
    env_file: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    **overrides: Any,  # noqa: ANN401
) -> ReconConfig:
    """Build a :class:`ReconConfig` from a ``.env`` file, the environment and overrides.

    Precedence, lowest to highest: the ``.env`` file, ``environ``, ``overrides``.

    Args:
        env_file: Path of the ``.env`` file. By default the nearest ``.env`` found by
            ``python-dotenv`` is used; pass ``""`` to skip the file.
        environ: Environment to read from, by default ``os.environ``.
        **overrides: Already-typed values, keyed by field name.

    Returns:
        The configuration.

    Raises:
        ConfigError: If a required variable is missing, a value cannot be parsed or an
            override names an unknown field.
    """
    if env_file is None:
        env_file = find_dotenv(usecwd=True)
    values: dict[str, str] = {}
    if env_file:
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ if environ is None else environ)

    kwargs: dict[str, Any] = {}
    for f in fields(ReconConfig):
        if f.name in overrides:
            kwargs[f.name] = overrides.pop(f.name)
            continue
        env_name = f.metadata["env"]
        raw = values.get(env_name)
        if raw is None or raw == "":
            continue
        kwargs[f.name] = _parse(env_name, raw, str(f.type))

    if overrides:
        msg = f"Unknown configuration fields: {', '.join(sorted(overrides))}"
        raise ConfigError(msg)

    required = [f for f in fields(ReconConfig) if f.default is MISSING]
    missing = [f.metadata["env"] for f in required if f.name not in kwargs]
    if missing:
        msg = f"Missing required configuration variables: {', '.join(missing)}"
        raise ConfigError(msg)

    return ReconConfig(**kwargs)
