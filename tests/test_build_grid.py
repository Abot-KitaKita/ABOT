"""Pure helpers for the grid build. They do not read raw data."""
from __future__ import annotations

import pandas as pd
import pytest

from abot.build_grid import assert_full_raster, equal_count_bins, keep_spaced


def test_equal_count_bins_lowest_is_one():
    frame = pd.DataFrame({"psgc": ["b", "a", "c", "d"], "value": [3.0, 1.0, 4.0, 2.0]})
    bins = equal_count_bins(frame, "value", "psgc", 2)
    assert bins["a"] == 1 and bins["d"] == 1
    assert bins["b"] == 2 and bins["c"] == 2


def test_equal_count_bins_ties_break_by_id():
    frame = pd.DataFrame({"psgc": ["b", "a", "c", "d"], "value": [1.0, 1.0, 1.0, 1.0]})
    bins = equal_count_bins(frame, "value", "psgc", 2)
    assert list(bins.sort_index().to_numpy()) == [1, 1, 2, 2]


def test_keep_spaced_prefers_lower_priority_number():
    kept = keep_spaced(
        ids=["far", "near-low", "near-high"],
        xs=[0.0, 110.0, 100.0],
        ys=[0.0, 0.0, 0.0],
        priority=[0, 1, 0],
        metres=50,
    )
    assert kept == ["far", "near-high"]


def test_short_raster_is_refused(tmp_path):
    path = tmp_path / "phl_f_30_2020.tif"
    path.write_bytes(b"not a full raster")
    with pytest.raises(RuntimeError, match="still in progress"):
        assert_full_raster(path)
