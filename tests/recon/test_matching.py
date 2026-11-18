"""Tests of the entry/exit matching in ``recon/preprocess/matching.py``."""

import pandas as pd

from recon.preprocess.matching import match_protons


def test_match_protons_keeps_only_shared_run_and_event_ids():
    df_in = pd.DataFrame(
        {
            "run_id": [0, 0, 0, 1],
            "event_id": [1, 2, 3, 1],
            "energy": [200.0, 201.0, 202.0, 203.0],
        },
        index=pd.Index([10, 11, 12, 13]),
    )
    # (0, 2) is missing on exit, (2, 1) only exists on exit, and (1, 3) shares an event id
    # with the entry data but not its run id.
    df_out = pd.DataFrame(
        {
            "run_id": [1, 0, 2, 0, 1],
            "event_id": [1, 3, 1, 1, 3],
            "energy": [150.0, 152.0, 160.0, 149.0, 170.0],
        },
    )

    protons = match_protons(df_in, df_out)

    assert protons.index.tolist() == [0, 1, 2]
    rows = protons[["run_id", "event_id", "energy_in", "energy_out"]].to_numpy().tolist()
    assert sorted(rows) == [[0, 1, 200.0, 149.0], [0, 3, 202.0, 152.0], [1, 1, 203.0, 150.0]]
