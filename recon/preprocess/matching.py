"""Match protons measured on the entry and exit detectors."""

import pandas as pd
from loguru import logger


def match_protons(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """Match proton measurements on entry and exit detectors.

    Args:
        df_in: Protons measured on the entry detector.
        df_out: Protons measured on the exit detector.

    Returns:
        Dataframe of protons with measurements of energy, position, and direction of travel measured on the
        entry ('_in' suffix) and exit ('_out' suffix) detectors.

    """
    logger.info("Matching protons on entry and exit detectors")
    return df_in.merge(
        df_out,
        how="inner",
        on=["run_id", "event_id"],
        suffixes=("_in", "_out"),
    ).reset_index(drop=True)
