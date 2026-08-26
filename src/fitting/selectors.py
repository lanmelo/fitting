"""Selectors that map curves to groups, or to per-group constant values.

These exist so that "one value per replicate" can be expressed without
materialising a dense ``(n_curves, ...)`` array. A selector stores one
value per group plus an integer index per curve, and gathers on use, so
memory is O(n_groups) even at millions of curves.
"""

from typing import Any, Mapping, Optional, Sequence, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .core import FittingError

LevelSpec = Union[str, int, Sequence[Union[str, int]]]


class ByLevel(eqx.Module):  # pylint: disable=too-few-public-methods
    """One value per distinct key of one or more index levels (or columns).

    ``values`` maps a key to an array (or is a DataFrame indexed by that key).
    Used in ``share=`` it names the grouping for a shared parameter; used in
    ``curve_consts=`` it supplies a per-group constant.
    """

    level: LevelSpec = eqx.field(static=True)
    values: Optional[Any] = None
    from_columns: bool = eqx.field(static=True, default=False)


def by_level(level: LevelSpec, values: Optional[Any] = None) -> ByLevel:
    """Group by one or more index levels, e.g. ``by_level("replicate")``."""
    return ByLevel(level=level, values=values, from_columns=False)


def by_column(column: LevelSpec, values: Optional[Any] = None) -> ByLevel:
    """Group by one or more DataFrame columns."""
    return ByLevel(level=column, values=values, from_columns=True)


def _keys_for(index: pd.Index, level: LevelSpec, name: str) -> pd.Index:
    """Per-curve group keys drawn from one or more index levels."""
    levels = [level] if isinstance(level, (str, int)) else list(level)
    try:
        if isinstance(index, pd.MultiIndex):
            cols = [index.get_level_values(lv) for lv in levels]
        else:
            if len(levels) != 1:
                raise FittingError(
                    f"{name}: index is not a MultiIndex, so only a single "
                    f"level can be used, got {levels}"
                )
            cols = [index]
    except KeyError as exc:
        raise FittingError(
            f"{name}: index has no level {exc}; available levels are "
            f"{list(index.names)}"
        ) from exc
    if len(cols) == 1:
        return pd.Index(cols[0])
    return pd.MultiIndex.from_arrays(cols)


def resolve_group_index(
    spec: Any, index: pd.Index, n_curves: int, name: str
) -> Optional[jnp.ndarray]:
    """Turn a ``share=`` value into an ``(n_curves,)`` integer group index.

    ``None`` means a single global value, which is the all-zeros index.
    """
    if spec is None:
        return None
    if isinstance(spec, ByLevel):
        keys = _keys_for(index, spec.level, f"share['{name}']")
        codes, _ = pd.factorize(keys, sort=True)
        return jnp.asarray(codes, dtype=jnp.int32)
    arr = np.asarray(spec)
    if arr.shape != (n_curves,):
        raise FittingError(
            f"share['{name}'] must be None, a by_level(...) selector, or an "
            f"array of shape ({n_curves},); got shape {arr.shape}"
        )
    _, codes = np.unique(arr, return_inverse=True)
    return jnp.asarray(codes, dtype=jnp.int32)


def resolve_curve_const(
    spec: Any, index: pd.Index, _n_curves: int, name: str
) -> jnp.ndarray:
    """Turn a ``curve_consts=`` value into a per-curve-major array.

    A :class:`ByLevel` selector is expanded by gathering from its per-group
    stack, so the caller never has to build the dense array by hand.
    """
    if isinstance(spec, ByLevel):
        if spec.values is None:
            raise FittingError(
                f"curve_consts['{name}'] is a selector with no values; pass "
                f"by_level(level, values) where values maps each group key to "
                f"its constant"
            )
        keys = _keys_for(index, spec.level, f"curve_consts['{name}']")
        if isinstance(spec.values, pd.DataFrame):
            table: Mapping[Any, Any] = dict(
                zip(spec.values.index, spec.values.to_numpy())
            )
        else:
            table = dict(spec.values)
        uniq = pd.Index(pd.unique(keys))
        missing = [k for k in uniq if k not in table]
        if missing:
            raise FittingError(
                f"curve_consts['{name}']: no value supplied for group(s) "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        stack = jnp.asarray(
            np.stack([np.asarray(table[k], dtype=float) for k in uniq]),
            dtype=jnp.float64,
        )
        codes = pd.Index(uniq).get_indexer(keys)
        return stack[jnp.asarray(codes, dtype=jnp.int32)]
    return jnp.asarray(spec, dtype=jnp.float64)
