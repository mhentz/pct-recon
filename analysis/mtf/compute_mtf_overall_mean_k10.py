"""Print the mean k10 of the spiral phantom for every regulariser, dose and projection count."""

from itertools import product
from math import sqrt
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
from loguru import logger

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR


def main(setup: Annotated[Literal["ideal", "realistic"], typer.Argument()]) -> None:
    """Read the ``k10_mean.csv`` of each reconstruction and print the values as LaTeX table rows.

    Args:
        setup: Dataset, ``"ideal"`` or ``"realistic"``.
    """
    regularisations = ["TV", "dTV"]
    doses = [5.0, 1.0, 0.2]
    n_projections = [720, 45]

    k10_dict = {}

    for reg, dose, n_proj in product(regularisations, doses, n_projections):
        alpha = get_ideal_alpha(setup=f"spiral-{setup}", dose=dose, n_projections=n_proj)

        dose_str = f"{dose:.0f}" if float(dose).is_integer() else f"{dose:.1f}"
        base_path = Path(f"{RECON_DATA_DIR}/spiral-{setup}/{dose_str}mGy/{n_proj}proj/recon/")

        suffix = "_eta_2e-1" if reg == "dTV" else ""
        dir_path = max(
            base_path.glob(
                f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_{reg}_iso_alpha_{alpha}{suffix}_iter*"
            )
        )

        k10_df: pd.DataFrame = pd.read_csv(dir_path / "processed" / "mtf" / "k10_mean.csv")

        logger.info(
            f"reg={reg:<3}, dose={dose}, n_proj={n_proj:<3d}: "
            f"{k10_df['k10'].iloc[0]:.3f} ± {k10_df['sigma_k10'].iloc[0]:.3f} lp/mm"
        )

        k10_dict[(reg, dose, n_proj)] = (
            k10_df["k10"].iloc[0],
            k10_df["sigma_k10"].iloc[0],
            sqrt(k10_df["sample_var"].iloc[0]),
        )

    for item in k10_dict.items():
        reg, dose, n_proj = item[0]
        k10, sigma_k10, sample_stddev = item[1]
        print(f"{reg:<3} & {dose} & {n_proj:<3d} & {k10:.3f} & {sigma_k10:.3f} & {sample_stddev:.3f} \\\\")


if __name__ == "__main__":
    typer.run(main)
