"""Tests of the pickle reader in ``recon/preprocess/io.py``."""

import pandas as pd
import pytest

from recon.preprocess.io import read_pickled_df


def test_read_pickled_df_single_file(tmp_path):
    protons = pd.DataFrame({"wepl": [1.0, 2.0]})
    protons.to_pickle(tmp_path / "df_90.0.pkl")

    result = read_pickled_df(tmp_path, 90)

    pd.testing.assert_frame_equal(result, protons)


def test_read_pickled_df_joins_chunks_in_order(tmp_path):
    chunks = [pd.DataFrame({"wepl": [float(i)]}) for i in range(3)]
    for i, chunk in enumerate(chunks):
        chunk.to_pickle(tmp_path / f"df_0.5-{i}.pkl")

    result = read_pickled_df(tmp_path, 0.5, n_chunks=3)

    assert result["wepl"].tolist() == [0.0, 1.0, 2.0]


def test_read_pickled_df_rejects_non_dataframes(tmp_path):
    pd.Series([1.0]).to_pickle(tmp_path / "df_0.0.pkl")

    with pytest.raises(TypeError, match="does not hold a dataframe"):
        read_pickled_df(tmp_path, 0.0)
