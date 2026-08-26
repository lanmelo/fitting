"""Agreement with a reference fit on the real STAMMP-seq dataset.

Checks the package against an independently produced set of dissociation rates
for the same curves. The dataset is large and not part of the repository, so
this is skipped unless it is mounted. Run it explicitly with::

    pytest -m production
"""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import fitting as ft

BASE = Path(
    "/Users/lucas/Library/CloudStorage/GoogleDrive-lanmelo@stanford.edu/"
    "Shared drives/FordyceLab/manuscripts/Renee & Lucas: STAMMP_Seq/"
    "Analysis/sherlock_koff_fits6"
)
COUNTS = BASE / "final_kinetic_counts.csv.gz"
REFERENCE = BASE / "koffs.csv.gz"
IDX = ["replicate", "variant", "barcode", "background", "motif"]
TCOLS = [f"T{i}" for i in range(1, 13)] + ["Bound"]
N_CURVES = int(os.environ.get("FITTING_PRODUCTION_N", "5000"))

pytestmark = [
    pytest.mark.production,
    pytest.mark.skipif(
        not (COUNTS.exists() and REFERENCE.exists()),
        reason="production STAMMP-seq dataset not available",
    ),
]


def _sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_chunk = max(100, N_CURVES // 30)
    rng = np.random.RandomState(0)  # pylint: disable=no-member
    spikes, samp = [], []
    for chunk in pd.read_csv(COUNTS, chunksize=400_000):
        spikes.append(chunk[chunk.variant == "spike"])
        nz = chunk[chunk.variant != "spike"]
        nz = nz[nz[TCOLS].sum(axis=1) > 0]
        if len(nz):
            samp.append(nz.sample(n=min(per_chunk, len(nz)), random_state=rng))
    counts = pd.concat(samp + spikes).set_index(IDX)
    # Spike-ins ride in the same table, tagged variant == "spike"; the other
    # levels do not distinguish them, so sum over those.
    is_spike = counts.index.get_level_values("variant") == "spike"
    spike = (
        counts[is_spike]
        .droplevel(["variant", "barcode", "background", "motif"])
        .groupby("replicate")
        .sum()
    )
    return counts[~is_spike].iloc[:N_CURVES], spike[TCOLS]


def _reference(keys: pd.Index) -> pd.DataFrame:
    want = set(keys)
    hits = []
    for chunk in pd.read_csv(REFERENCE, chunksize=500_000):
        k = list(
            zip(
                chunk.replicate,
                chunk.variant,
                chunk.barcode,
                chunk.background,
                chunk.motif,
            )
        )
        mask = np.fromiter((z in want for z in k), bool, len(chunk))
        if mask.any():
            hits.append(chunk[mask])
    return pd.concat(hits).set_index(IDX)


def test_reproduces_production_koff_fit() -> None:
    """Fitted rates must match the reference to solver precision.

    The reference is a two-parameter single-exponential interval model with a
    Poisson likelihood and per-replicate spike-in normalisation. Agreement to
    better than 1e-6 confirms that the parameter containers, the constants
    mechanism, and the spike-in selector are all numerically transparent.
    """
    samples, spike = _sample()
    x = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)

    res = ft.fit_sequencing(
        samples,
        x=x,
        y_cols=TCOLS,
        spikein=spike,
        spikein_level="replicate",
        model=ft.SingleExponentialInterval(concat_bound=True),
        progress=False,
    )
    assert res.diagnostics.converged.all()

    joined = res.local.join(
        _reference(res.local.index)[["log_y0", "log_k_off"]],
        how="inner",
        rsuffix="_prod",
    ).dropna()
    assert len(joined) == len(
        res.local
    ), "not every curve matched the reference"

    for col in ("log_k_off", "log_y0"):
        delta = np.abs(
            joined[col].to_numpy() - joined[f"{col}_prod"].to_numpy()
        )
        assert delta.max() < 1e-6, f"{col} drifted: max|d|={delta.max():.3e}"
