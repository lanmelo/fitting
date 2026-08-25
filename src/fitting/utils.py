"""Utility functions for normalization mapping and sequence prediction."""

from typing import Any, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .models import BaseModel


def log1mexp(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    r"""Computes the element-wise log1mexp in a numerically stable way.

    .. math::
        \log \left( 1 - e^{-x} \right)

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    return jnp.where(x > 0.693, jnp.log1p(-jnp.exp(-x)), jnp.log(-jnp.expm1(-x)))  # type: ignore[operator]


def logsubexp(
    x: jax.typing.ArrayLike, y: jax.typing.ArrayLike
) -> jax.typing.ArrayLike:
    """Compute log(exp(x) - exp(y)) stably, assuming x > y."""
    return x + jnp.log1p(-jnp.exp(y - x))  # type: ignore[operator]


def build_normalization_array(
    index: pd.Index,
    norm_dict: Mapping[Any, Any],
    key_level: Union[int, str] = 0,
) -> np.ndarray:
    """Build a 2D normalization array mapping rows to normalization vectors.

    Args:
        index: Pandas Index or MultiIndex of the data.
        norm_dict: Dictionary mapping normalization keys to 1D arrays/lists.
        key_level: The level name or index of the MultiIndex used to look up
          the norm key.

    Returns:
        A 2D numpy array of shape (n_curves, n_timepoints).
    """
    n_curves = len(index)
    if n_curves == 0:
        return np.empty((0, 0))

    # Get the norm keys for each curve
    if isinstance(index, pd.MultiIndex):
        norm_keys = index.get_level_values(key_level)
    else:
        norm_keys = index

    # Retrieve a sample normalization vector to determine shape
    sample_key = next(iter(norm_dict.keys()))
    sample_vec = np.asarray(norm_dict[sample_key])
    n_timepoints = len(sample_vec)

    norm_arr = np.zeros((n_curves, n_timepoints))
    for i, key in enumerate(norm_keys):
        if key in norm_dict:
            norm_arr[i] = norm_dict[key]
        else:
            raise KeyError(
                f"Normalization key '{key}' not found in norm_dict."
            )

    return norm_arr


def predict_sequence(
    fitted_df: pd.DataFrame,
    model: BaseModel,
    sequence_key: tuple[Any, ...],
    x_data: jnp.ndarray,
    norm_dict: Mapping[Any, Any],
    key_level: Union[int, str] = 0,
    column_suffix: Optional[str] = None,
) -> jnp.ndarray:
    """Reconstruct predicted log-counts for a specific fitted sequence key.

    Args:
        fitted_df: DataFrame of fitted parameters (returned by fit_curves).
        model: The model instance.
        sequence_key: The index key of the sequence in fitted_df.
        x_data: The timepoints array.
        norm_dict: Dictionary of normalization vectors.
        key_level: MultiIndex level corresponding to norm keys.
        column_suffix: Optional suffix appended to parameter column names in
          fitted_df (e.g., '_single_exp' or '_double_exp').

    Returns:
        1D JAX array of log predictions.
    """
    if sequence_key not in fitted_df.index:
        raise KeyError(
            f"Sequence key {sequence_key} not found in fitted parameters."
        )

    row = fitted_df.loc[sequence_key]

    # Extract parameter values in order of model.param_names
    suffix = column_suffix or ""
    p = jnp.array([row[f"{name}{suffix}"] for name in model.param_names])

    # Resolve norm key
    if isinstance(fitted_df.index, pd.MultiIndex):
        if isinstance(key_level, str):
            level_idx = fitted_df.index.names.index(key_level)
        else:
            level_idx = key_level
        norm_key = sequence_key[level_idx]
    else:
        norm_key = sequence_key

    log_norm = jnp.log(jnp.asarray(norm_dict[norm_key]))

    return model.predict(p, log_norm, x_data)
