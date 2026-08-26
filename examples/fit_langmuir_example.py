"""Fit Langmuir binding isotherms to MITOMI fluorescence by least squares.

Companion to ``fit_koff_example.py``, which fits sequencing counts.
Fluorescence is continuous, so the objective here is least squares rather
than a Poisson likelihood.

One binding curve is all the chambers holding one variant on one slide,
measured across a dilution series. Curves therefore have unequal numbers of
points (7 to 26 here) and every chamber has its own DNA concentration, hence
the NaN padding and the 2-D per-curve ``x``.

The saturation level ``ymax`` belongs to the protein prep and the detector
rather than to the DNA variant, so it is shared across the chambers of one
adapter set instead of being fitted per curve.
"""

# Each example is meant to run standalone, so the CPU-pinning and
# sys.path preamble is duplicated between them on purpose.
# pylint: disable=wrong-import-position,duplicate-code

import os
import sys

if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)
import fitting as ft

DATA = (
    "/Users/lucas/Library/CloudStorage/GoogleDrive-lanmelo@stanford.edu/"
    "Shared drives/FordyceLab_Member_Files/Lucas/Experiments/01_MAX_STAMMPseq/"
    "01-01_MAX_MITOMI_2/2025-09-12 Joint (k-)MITOMI Fitting/raw_data.csv"
)
#: One curve per (slide, variant). ``adapters`` rides along in the index so it
#: can be named as the grouping for the shared saturation level.
CURVE_KEYS = ["slide", "variant", "adapters"]
SIGNAL = "postwash_ratio"
CONCENTRATION = "log_dna_conc"
OUTFILE = "langmuir_kds.csv.gz"


def to_wide(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Reshape one-row-per-chamber into one-row-per-curve, padded with NaN.

    Curves have unequal numbers of chambers, so short rows are padded. The
    padding is NaN in the signal, which ``fit`` masks out of the loss; the
    matching positions in ``x`` are padded too, and ``fit`` replaces those
    with a finite dummy, since a NaN there would wreck convergence even at a
    zero-weight point.

    Returns:
        ``(signal, log_concentration)`` -- a frame indexed by
        :data:`CURVE_KEYS`, and a matching 2-D array of the same shape.
    """
    groups = raw.groupby(CURVE_KEYS, sort=True)
    width = int(groups.size().max())
    signal, conc, index = [], [], []
    for key, chambers in groups:
        y = chambers[SIGNAL].to_numpy(dtype=float)
        x = chambers[CONCENTRATION].to_numpy(dtype=float)
        pad = np.full(width - len(y), np.nan)
        signal.append(np.concatenate([y, pad]))
        conc.append(np.concatenate([x, pad]))
        index.append(key)
    idx = pd.MultiIndex.from_tuples(index, names=CURVE_KEYS)
    columns = pd.Index([f"chamber_{i}" for i in range(width)])
    return pd.DataFrame(signal, index=idx, columns=columns), np.stack(conc)


def main() -> int:
    """Fit every isotherm with one saturation level per adapter set."""
    if not os.path.exists(DATA):
        print(f"Could not find data file at {DATA}")
        return 1

    raw = pd.read_csv(DATA)
    raw = raw[raw["qc"]]
    signal, log_conc = to_wide(raw)
    n_variants = signal.index.get_level_values("variant").nunique()
    n_slides = signal.index.get_level_values("slide").nunique()
    print(
        f"{len(signal)} curves from {n_variants} variants on "
        f"{n_slides} slides; {np.isnan(log_conc).mean():.0%} of the padded "
        f"grid is masked"
    )

    # x is log concentration, so LogisticAffinity is the Langmuir isotherm in
    # that coordinate -- better conditioned than fitting against raw molarity.
    #
    # No `solver=` here: sharing ymax couples every curve into one
    # 200-parameter problem, and Levenberg-Marquardt stalls on it (it forms the
    # normal equations for all parameters at once, and their conditioning
    # degrades as curves are added). The default LBFGS never forms them. LM is
    # the better choice when parameters are *not* shared, where each curve is a
    # separate two-parameter solve.
    result = ft.fit(
        signal,
        model=ft.LogisticAffinity(),
        x=log_conc,
        loss=ft.SSE(),
        share={"log_ymax": ft.by_level("adapters")},
        joint_max_steps=8000,
    )

    print("\nsaturation level, one per adapter set:")
    for _, row in result.shared.iterrows():
        print(
            f"  adapters={bool(row['group'])}: "
            f"ymax = {row['linear_value']:.3f}  "
            f"(log_ymax = {row['value']:+.4f} +/- {row['se']:.4f})"
        )

    # Kd per variant, averaged over slides, with the between-slide scatter as
    # the error bar. Slides are independent measurements of the same variant,
    # so their spread is the honest uncertainty.
    log_kd = result.local["log_kd"]
    by_variant = log_kd.groupby(log_kd.index.get_level_values("variant"))
    summary = pd.DataFrame(
        {
            "n_slides": by_variant.size(),
            "log_kd": by_variant.mean(),
            "log_kd_sd": by_variant.std(),
            "kd": np.exp(by_variant.mean()),
        }
    ).sort_values("log_kd")
    print(f"\naffinity per variant (tightest first), {len(summary)} variants:")
    print(summary.head(8).to_string(float_format=lambda v: f"{v:.4g}"))
    print(
        f"\nmedian between-slide SD of log Kd: "
        f"{summary.log_kd_sd.median():.4f}"
    )

    result.table.to_csv(OUTFILE)
    print(f"\nSaved {OUTFILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
