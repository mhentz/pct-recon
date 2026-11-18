"""Print the Lipschitz constants and step sizes stored in the YAML file."""

from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import yaml

LIPSCHITZ_CONSTANTS_PATH = Path(__file__).parent / "lipschitz_constants.yml"


def format_step_size(lipschitz_c: float, digits: int = 4) -> str:
    """Return the step size 1 / L, rounded down to a number of significant digits.

    Rounding down keeps the step size at or below 1 / L, as FISTA needs; ``step_sizes.yml``
    holds values rounded this way.

    Args:
        lipschitz_c: Lipschitz constant L.
        digits: Number of significant digits.

    Returns:
        The step size in the notation of ``step_sizes.yml``, e.g. ``2.872e-6``.
    """
    step_size = Decimal(1 / lipschitz_c)
    quantum = Decimal(1).scaleb(step_size.adjusted() - digits + 1)
    step_size = step_size.quantize(quantum, rounding=ROUND_FLOOR)
    return f"{step_size:.{digits - 1}e}".replace("e-0", "e-")


def main(path: Path = LIPSCHITZ_CONSTANTS_PATH) -> None:
    """Print the Lipschitz constant and step size of every setup in the YAML file.

    Args:
        path: Path to the YAML file of Lipschitz constants.
    """
    with path.open() as f:
        constants = yaml.safe_load(f)

    for setup in constants:
        for dose in constants[setup]:
            for n_projections in constants[setup][dose]:
                for image_w in constants[setup][dose][n_projections]:
                    lipschitz_c = constants[setup][dose][n_projections][image_w]

                    print(f"{setup} {dose} {n_projections} {image_w}: ")
                    print(f"         lips: {lipschitz_c}")
                    print(f"         step: {format_step_size(lipschitz_c)}")


if __name__ == "__main__":
    main()
