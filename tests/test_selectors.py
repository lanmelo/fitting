"""Group selectors and per-group constants."""

# pylint: disable=redefined-outer-name

import numpy as np
import pandas as pd
import pytest

import fitting as ft
from fitting.selectors import resolve_curve_const, resolve_group_index


@pytest.fixture
def index() -> pd.MultiIndex:
    """A small two-level index with repeated replicate keys."""
    return pd.MultiIndex.from_tuples(
        [("R1", "a"), ("R1", "b"), ("R2", "a"), ("R3", "a"), ("R2", "b")],
        names=["replicate", "variant"],
    )


def test_by_level_factorises_groups(index: pd.MultiIndex) -> None:
    """By level factorises groups."""
    got = resolve_group_index(ft.by_level("replicate"), index, len(index), "p")
    assert np.array_equal(np.asarray(got), [0, 0, 1, 2, 1])


def test_by_level_multiple_levels(index: pd.MultiIndex) -> None:
    """By level multiple levels."""
    got = resolve_group_index(
        ft.by_level(["replicate", "variant"]), index, len(index), "p"
    )
    assert len(np.unique(np.asarray(got))) == 5


def test_none_means_one_global_group(index: pd.MultiIndex) -> None:
    """None means one global group."""
    assert resolve_group_index(None, index, len(index), "p") is None


def test_missing_level_raises(index: pd.MultiIndex) -> None:
    """Missing level raises."""
    with pytest.raises(ft.FittingError, match="no level"):
        resolve_group_index(ft.by_level("nope"), index, len(index), "p")


def test_by_level_const_gathers_without_dense_array(
    index: pd.MultiIndex,
) -> None:
    """One vector per replicate, gathered -- O(n_groups) memory."""
    values = {"R1": np.zeros(4), "R2": np.ones(4), "R3": np.full(4, 2.0)}
    got = np.asarray(
        resolve_curve_const(
            ft.by_level("replicate", values), index, len(index), "c"
        )
    )
    assert got.shape == (5, 4)
    assert np.array_equal(got[:, 0], [0.0, 0.0, 1.0, 2.0, 1.0])


def test_by_level_const_missing_group_raises(index: pd.MultiIndex) -> None:
    """By level const missing group raises."""
    with pytest.raises(ft.FittingError, match="no value supplied"):
        resolve_curve_const(
            ft.by_level("replicate", {"R1": np.zeros(4)}),
            index,
            len(index),
            "c",
        )


def test_by_level_const_accepts_dataframe(index: pd.MultiIndex) -> None:
    """By level const accepts dataframe."""
    frame = pd.DataFrame(
        np.arange(12.0).reshape(3, 4), index=["R1", "R2", "R3"]
    )
    got = np.asarray(
        resolve_curve_const(
            ft.by_level("replicate", frame), index, len(index), "c"
        )
    )
    assert got.shape == (5, 4)
