"""Look up the ideal regularisation weight alpha for a setup, dose and projection count."""


def get_ideal_alpha(setup: str, dose: float, n_projections: int) -> str:  # noqa: C901, PLR0912
    """Get ideal alpha value for given phantom setup, dose, and number of projections.

    The alpha values were determined from L-curve analysis of TV reconstructions.

    Args:
        setup: Phantom setup. One of "ctp528-ideal", "ctp528-realistic",
            "gammex-ideal", "gammex-realistic", "spiral-ideal", "spiral-realistic".
        dose: Dose in mGy. One of 0.2, 1.0, 5.0.
        n_projections: Number of projections. Either 45 or 720.
    """
    if n_projections not in {45, 720}:
        msg = "n_projections must be either 45 or 720"
        raise ValueError(msg)

    if dose not in {0.2, 1.0, 5.0}:
        msg = "dose must be either 0.2, 1.0, or 5.0 mGy"
        raise ValueError(msg)

    alpha = ""

    match setup, dose, n_projections:
        case "ctp528-ideal", 0.2, 45:
            alpha = "3.54e-3"
        case "ctp528-ideal", _, _:
            alpha = "2.5e-3"

        case "ctp528-realistic", 5.0, _:
            alpha = "1.58e-3"
        case "ctp528-realistic", 1.0, _:
            alpha = "2.5e-3"
        case "ctp528-realistic", 0.2, _:
            alpha = "3.54e-3"

        case "gammex-ideal", 0.2, 720:
            alpha = "3.54e-3"
        case "gammex-ideal", 0.2 | 1.0, 45:
            alpha = "2.5e-3"
        case "gammex-ideal", _, _:
            alpha = "1.58e-3"

        case "gammex-realistic", 5.0, 720:
            alpha = "1.58e-3"
        case "gammex-realistic", _, _:
            alpha = "2.5e-3"

        case "spiral-ideal", 0.2, _:
            alpha = "3.54e-3"
        case "spiral-ideal", 1.0, _:
            alpha = "2.5e-3"
        case "spiral-ideal", 5.0, _:
            alpha = "1.58e-3"

        case "spiral-realistic", 5.0, 720:
            alpha = "1.58e-3"
        case "spiral-realistic", 5.0, 45:
            alpha = "1e-3"
        case "spiral-realistic", 1.0, _:
            alpha = "2.5e-3"
        case "spiral-realistic", 0.2, _:
            alpha = "3.54e-3"

    if alpha == "":
        msg = f"No ideal alpha found for setup={setup}, dose={dose}, n_projections={n_projections}"
        raise ValueError(
            msg,
        )

    return alpha
