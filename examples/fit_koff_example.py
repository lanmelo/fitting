"""Fit STAMMP-seq dissociation curves.

Demonstrates the sequencing entry point: spike-in normalisation, the
two-parameter interval model, and the bound library that is measured after
repeated dissociation steps rather than as a kinetic fraction.

How spike-ins are laid out is a property of this experiment, not of the
package, so ``split_spikein`` below is part of the example. Here they ride in
the same table as the samples, tagged ``variant == "spike"``.
"""

# pylint: disable=wrong-import-position

import os
import sys
from typing import Optional, Sequence

# Force CPU unless the user has asked otherwise, so this runs on nodes without
# a GPU rather than failing in CUDA initialisation.
if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)
import fitting as ft

DATA = (
    "/Users/lucas/Library/CloudStorage/GoogleDrive-lanmelo@stanford.edu/"
    "Shared drives/FordyceLab/manuscripts/Renee & Lucas: STAMMP_Seq/"
    "Analysis/sherlock_koff_fits6/final_kinetic_counts.csv.gz"
)
INDEX = ["replicate", "variant", "barcode", "background", "motif"]
COUNT_COLS = [f"T{i}" for i in range(1, 13)] + ["Bound"]
OUTFILE = "koffs.csv.gz"

#: Index levels that do not distinguish one spike-in from another, so the
#: spike-in rows are summed over them.
SPIKEIN_REDUNDANT = ["barcode", "background", "motif"]


def split_spikein(
    counts: pd.DataFrame,
    level: str = "variant",
    key: str = "spike",
    drop_levels: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate spike-in rows from sample rows.

    Args:
        counts: counts with the spike-ins carried as ordinary rows.
        level: the index level identifying spike-ins.
        key: the value of ``level`` marking a spike-in row.
        drop_levels: index levels that do not distinguish spike-ins from one
            another; the spike-in rows are summed over these.

    Returns:
        ``(samples, spikein)`` -- the non-spike-in rows, and the spike-in rows
        reduced to the levels that were kept.

    Raises:
        KeyError: if ``level`` is not an index level, or nothing matches
            ``key``.
    """
    if level not in (counts.index.names or []):
        raise KeyError(
            f"index has no level {level!r}; "
            f"available: {list(counts.index.names)}"
        )
    keys = counts.index.get_level_values(level)
    spike = counts[keys == key]
    if spike.empty:
        raise KeyError(f"no rows with {level} == {key!r}")
    samples = counts[keys != key]

    spike = spike.droplevel(level)
    if drop_levels:
        keep = [n for n in spike.index.names if n not in drop_levels]
        if keep:
            spike = spike.groupby(keep).sum()
        else:
            spike = spike.sum().to_frame().T
    return samples, spike


def main() -> int:
    """Load counts, fit dissociation curves, and write the results."""
    if not os.path.exists(DATA):
        print(f"Could not find data file at {DATA}")
        return 1

    print(f"Loading {DATA}...")
    counts = pd.read_csv(DATA, index_col=INDEX)

    samples, spikein = split_spikein(
        counts, level="variant", key="spike", drop_levels=SPIKEIN_REDUNDANT
    )

    if "--test" in sys.argv:
        print("Test mode: first 500 curves only.")
        samples = samples.iloc[:500]

    # Interval right-edges. The final 24.0 is a duplicate: with
    # concat_bound=True the last prediction is the absolute remaining bound
    # population, which is what the Bound library measures.
    x = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)

    print("Fitting dissociation curves...")
    res = ft.fit_sequencing(
        samples,
        x=x,
        y_cols=COUNT_COLS,
        spikein=spikein[COUNT_COLS],
        spikein_level="replicate",
        model=ft.SingleExponentialInterval(concat_bound=True),
        chunk_size=50_000,
    )

    failed = int((~res.diagnostics.converged).sum())
    print(f"Done. {failed} / {len(res.local)} curves did not converge.")

    print(f"Saving results to {OUTFILE}...")
    res.table.to_csv(OUTFILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
