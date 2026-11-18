from enum import StrEnum  # noqa: D100


class DetectorChoice(StrEnum):
    """Enum for detector choice."""

    ENTRY = "in"
    EXIT = "out"


class Axis(StrEnum):
    """Enum for Cartesian axes."""

    X = "x"
    Y = "y"
    Z = "z"
