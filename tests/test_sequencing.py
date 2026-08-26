"""Spike-in handling and the sequencing wrapper."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from conftest import make_dissociation

import fitting as ft
from fitting.selectors import ByLevel


def _split(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate spike-in rows, as a caller would.

    Spike-in layout is a property of an experiment rather than of the package,
    so this lives with the test that needs it.
    """
    is_spike = counts.index.get_level_values("variant") == "spike"
    spike = (
        counts[is_spike]
        .droplevel(["variant", "barcode"])
        .groupby("replicate")
        .sum()
    )
    return counts[~is_spike], spike


def _counts_with_spikes(n: int = 30) -> pd.DataFrame:
    df, _, _ = make_dissociation(n=n)
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples(
        [(r, v, "bc") for r, v in df.index],
        names=["replicate", "variant", "barcode"],
    )
    spikes = pd.DataFrame(
        np.tile(np.linspace(900.0, 1100.0, df.shape[1]), (3, 1)),
        index=pd.MultiIndex.from_tuples(
            [(f"R{i}", "spike", "bc") for i in (1, 2, 3)],
            names=["replicate", "variant", "barcode"],
        ),
        columns=df.columns,
    )
    return pd.concat([df, spikes])


def test_spikein_log_norm_single_vector_is_relative() -> None:
    """Spikein log norm single vector is relative."""
    v = np.array([2.0, 4.0, 8.0])
    got = np.asarray(ft.spikein_log_norm(v))
    assert np.allclose(np.exp(got), v / v[-1])


def test_spikein_log_norm_multi_row_returns_selector() -> None:
    """Spikein log norm multi row returns selector."""
    frame = pd.DataFrame(
        np.array([[1.0, 2.0], [2.0, 4.0]]), index=["R1", "R2"]
    )
    got = ft.spikein_log_norm(frame, level="replicate")
    assert isinstance(got, ByLevel)


def test_spikein_log_norm_multi_row_without_level_raises() -> None:
    """Spikein log norm multi row without level raises."""
    frame = pd.DataFrame(np.ones((2, 3)), index=["R1", "R2"])
    with pytest.raises(ft.FittingError, match="level="):
        ft.spikein_log_norm(frame)


def test_fit_sequencing_end_to_end(interval_x: jnp.ndarray) -> None:
    """Fit sequencing end to end."""
    counts = _counts_with_spikes(n=30)
    samples, spike = _split(counts)
    res = ft.fit_sequencing(
        samples,
        x=interval_x,
        spikein=spike,
        spikein_level="replicate",
        progress=False,
    )
    assert res.diagnostics.converged.all()
    assert "log_k_off" in res.local.columns


def test_fit_sequencing_defaults_to_two_parameter_model(
    interval_x: jnp.ndarray,
) -> None:
    """The default must match production, and not add a background."""
    counts = _counts_with_spikes(n=10)
    samples, spike = _split(counts)
    res = ft.fit_sequencing(
        samples,
        x=interval_x,
        spikein=spike,
        spikein_level="replicate",
        progress=False,
    )
    assert isinstance(res.model, ft.SingleExponentialInterval)
    assert not isinstance(
        res.model, ft.SingleExponentialIntervalWithBackground
    )
    assert "log_bg" not in res.local.columns


def test_fit_sequencing_drops_empty_rows(interval_x: jnp.ndarray) -> None:
    """Fit sequencing drops empty rows."""
    counts = _counts_with_spikes(n=10)
    samples, spike = _split(counts)
    samples = samples.copy()
    samples.iloc[0] = 0
    res = ft.fit_sequencing(
        samples,
        x=interval_x,
        spikein=spike,
        spikein_level="replicate",
        progress=False,
    )
    assert len(res.local) == len(samples) - 1
