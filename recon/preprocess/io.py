"""Read the entry and exit detector data from the ROOT and pickle files."""

from pathlib import Path

import pandas as pd
import uproot
from loguru import logger

from recon.preprocess.utils import clean_raw_df


def read_root_file(data_dir: str, proj_angle: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the entry and exit detector measurements of one projection.

    Args:
        data_dir: Directory containing the ROOT files.
        proj_angle: Projection angle as it appears in the file name.

    Returns:
        The entry and exit detector dataframes.
    """
    # proj_angle = int(proj_angle) if float(proj_angle).is_integer() else float(proj_angle)
    # proj_angle = int(proj_angle) if float(proj_angle).is_integer() else float(proj_angle)
    path_to_file = f"{data_dir}/PhaseSpace{proj_angle}.root"

    vars_to_save = ["RunID", "EventID", "Ekine", "X", "Y", "dX", "dY", "dZ"]
    rename_vars = {
        "RunID": "run_id",
        "EventID": "event_id",
        "Ekine": "energy",
        "X": "x",
        "Y": "y",
        "dX": "px",
        "dY": "py",
        "dZ": "pz",
    }

    logger.info(f"Opening file '{path_to_file}' with uproot")
    with uproot.open(path_to_file) as file:
        df_in_raw = file["In;1"].arrays(vars_to_save, library="pd")
        df_out_raw = file["Out;1"].arrays(vars_to_save, library="pd")
        df_in = clean_raw_df(df_in_raw, vars_to_save, rename_vars=rename_vars)
        df_out = clean_raw_df(df_out_raw, vars_to_save, rename_vars=rename_vars)

    return df_in, df_out


def read_pickled_df(input_dir: str | Path, proj_angle: float, n_chunks: int = 1) -> pd.DataFrame:
    """Read the processed proton dataframe of one projection, joining its chunks if it was split.

    Args:
        input_dir: Directory containing the pickle files written by ``preprocess/run.py``.
        proj_angle: Projection angle in degrees.
        n_chunks: Number of chunks the projection was split into.

    Returns:
        The protons of the projection, in chunk order.

    Raises:
        TypeError: If a pickle file does not hold a dataframe.
    """
    if n_chunks > 1:
        paths = [Path(input_dir) / f"df_{proj_angle:.1f}-{chunk}.pkl" for chunk in range(n_chunks)]
    else:
        paths = [Path(input_dir) / f"df_{proj_angle:.1f}.pkl"]

    dfs = []
    for path in paths:
        df = pd.read_pickle(path)  # noqa: S301
        if not isinstance(df, pd.DataFrame):
            msg = f"{path} does not hold a dataframe"
            raise TypeError(msg)
        dfs.append(df)

    return pd.concat(dfs) if len(dfs) > 1 else dfs[0]
