"""Shared fixtures."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

NDArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


@pytest.fixture
def interval_x() -> jnp.ndarray:
    """Interval right-edges with the final bound library appended."""
    return jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)


def make_dissociation(
    n: int = 40,
    seed: int = 0,
    x: NDArray | None = None,
    norm: NDArray | None = None,
) -> tuple[pd.DataFrame, NDArray, NDArray]:
    """Poisson interval counts from a known single exponential.

    Returns ``(frame, true_log_y0, true_log_k_off)``.
    """
    if x is None:
        x = np.append(np.arange(2.0, 24.1, 2.0), 24.0)
    rs = np.random.RandomState(seed)  # pylint: disable=no-member
    log_y0 = rs.normal(np.log(5000.0), 0.4, n)
    log_koff = rs.normal(-3.0, 0.35, n)
    t = np.insert(x, 0, 0.0)
    ln = np.zeros(len(x)) if norm is None else np.log(norm)
    rows = []
    for i in range(n):
        lb = log_y0[i] - t * np.exp(log_koff[i])
        b = np.exp(lb)
        counts = np.append(b[:-2] - b[1:-1], b[-1])
        rows.append(rs.poisson(counts * np.exp(ln)))
    idx = pd.MultiIndex.from_tuples(
        [(f"R{i % 3 + 1}", f"v{i}") for i in range(n)],
        names=["replicate", "variant"],
    )
    cols = [f"T{i + 1}" for i in range(len(x) - 1)] + ["Bound"]
    return pd.DataFrame(rows, index=idx, columns=cols), log_y0, log_koff
