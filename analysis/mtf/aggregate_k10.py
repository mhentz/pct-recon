"""Aggregate per-insert k10 values of the spiral phantom into per-ring and overall statistics.

Reads ``k10_values.csv`` (one row per insert, in ``ROI_LIST`` order: 1 central insert followed
by four rings of 5) from each selected ``processed/mtf`` directory and writes

* ``k10_per_radius.csv`` with columns radius_index, n_inserts, k10, sigma_k10, sample_var
* ``k10_mean.csv`` with columns k10, sigma_k10, sample_var

Formulas (N inserts overall, n inserts in a ring, sigma_i the propagated ESF fit uncertainty):

* mean k10 = mean over exactly the inserts of the ring / all N inserts
* sigma_k10 = sqrt(sum sigma_i^2) / n  (ring)  and  sqrt(sum sigma_i^2) / N  (overall)
* sample_var = sample variance of k10 over the inserts with ddof=1 (0 for a single insert)
"""

import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger

from analysis.ideal_alpha import get_ideal_alpha
from analysis.paths import RECON_DATA_DIR

DATA_ROOT = Path(f"{RECON_DATA_DIR}")
N_INSERTS = [1, 5, 5, 5, 5]  # Number of inserts at each radius, in ROI_LIST order

SETUPS = ["ideal", "realistic"]
REGULARISATIONS = ["TV", "dTV"]
DOSES = [5.0, 1.0, 0.2]
N_PROJECTIONS = [720, 45]


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-ring and overall k10 statistics from a per-insert table."""
    if len(df) != sum(N_INSERTS):
        msg = f"Expected {sum(N_INSERTS)} inserts, found {len(df)}"
        raise ValueError(msg)

    k10 = df["k10"].to_numpy()
    sigma = df["sigma_k10"].to_numpy()

    rows = []
    start = 0

    for radius_index, n in enumerate(N_INSERTS):
        ring_k10 = k10[start : start + n]
        ring_sigma = sigma[start : start + n]

        rows.append(
            {
                "radius_index": radius_index,
                "n_inserts": n,
                "k10": ring_k10.mean(),
                "sigma_k10": np.sqrt(np.sum(ring_sigma**2)) / n,
                "sample_var": ring_k10.var(ddof=1) if n > 1 else 0.0,
            },
        )

        start += n

    per_radius = pd.DataFrame(rows)
    overall = pd.DataFrame(
        {
            "k10": [k10.mean()],
            "sigma_k10": [np.sqrt(np.sum(sigma**2)) / len(k10)],
            "sample_var": [k10.var(ddof=1)],
        },
    )

    return per_radius, overall


def find_mtf_dir(setup: str, reg: str, dose: float, n_proj: int, alpha: str, eta: str) -> Path | None:
    """Locate the processed/mtf directory of a reconstruction (latest iteration count if several)."""
    dose_dir = f"{dose:.0f}" if dose.is_integer() else f"{dose:.1f}"
    eta_part = f"_eta_{eta}" if reg == "dTV" else ""
    pattern = (
        f"fista_exact_init_zero_880pix_{dose}mGy_{n_proj}proj_{reg}_iso_alpha_{alpha}{eta_part}"
        "_iter*/processed/mtf"
    )

    hits = sorted(
        (DATA_ROOT / f"spiral-{setup}" / f"{dose_dir}mGy" / f"{n_proj}proj" / "recon").glob(pattern),
    )

    return hits[-1] if hits else None


def backup(path: Path) -> None:
    """Copy ``path`` to ``path.bak`` unless a backup already exists."""
    bak = path.with_suffix(path.suffix + ".bak")

    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        logger.info(f"Backed up {path.name} -> {bak.name}")


def main(
    setup: Annotated[list[str] | None, typer.Option(help="ideal / realistic (default: both)")] = None,
    reg: Annotated[list[str] | None, typer.Option(help="TV / dTV (default: both)")] = None,
    dose: Annotated[list[float] | None, typer.Option(help="Dose in mGy (default: 5.0, 1.0, 0.2)")] = None,
    n_proj: Annotated[list[int] | None, typer.Option(help="Number of projections (default: 720, 45)")] = None,
    alpha: Annotated[
        str | None,
        typer.Option(help="Regularisation parameter (default: get_ideal_alpha for the setup)"),
    ] = None,
    eta: Annotated[str, typer.Option(help="dTV eta parameter")] = "2e-1",
    *,
    do_backup: Annotated[bool, typer.Option("--backup/--no-backup", help="Back up existing outputs")] = True,
) -> None:
    """Aggregate k10 per ring and overall for the selected spiral phantom reconstructions."""
    for s in setup or SETUPS:
        for d in dose or DOSES:
            for p in n_proj or N_PROJECTIONS:
                a = alpha or get_ideal_alpha(f"spiral-{s}", d, p)

                for r in reg or REGULARISATIONS:
                    mtf_dir = find_mtf_dir(s, r, d, p, a, eta)

                    if mtf_dir is None:
                        logger.warning(f"No reconstruction found: {s} {r} {d} mGy {p} proj alpha {a}")
                        continue

                    if not (mtf_dir / "k10_values.csv").exists():
                        logger.warning(f"Missing k10_values.csv in {mtf_dir}, skipping")
                        continue

                    per_radius, overall = aggregate(pd.read_csv(mtf_dir / "k10_values.csv"))

                    if do_backup:
                        backup(mtf_dir / "k10_per_radius.csv")
                        backup(mtf_dir / "k10_mean.csv")

                    per_radius.to_csv(mtf_dir / "k10_per_radius.csv", index=False)
                    overall.to_csv(mtf_dir / "k10_mean.csv", index=False)

                    logger.info(
                        f"{s} {r} {d} mGy {p} proj alpha {a}: k10 {overall['k10'][0]:.3f} "
                        f"sigma {overall['sigma_k10'][0]:.3f} s {np.sqrt(overall['sample_var'][0]):.3f}",
                    )


if __name__ == "__main__":
    typer.run(main)
