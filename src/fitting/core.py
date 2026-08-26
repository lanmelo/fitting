"""Vocabulary shared across the package.

Only definitions that more than one module needs live here: the exception type,
the array type alias, container introspection, and the parameter layout that
:mod:`fitting.fitting` and :mod:`fitting.results` both work with. Anything
belonging to a single concept lives with it -- the model base class in
:mod:`fitting.models` beside its subclasses, the objectives in
:mod:`fitting.losses`.
"""

import dataclasses
from typing import Any, Mapping, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "ArrayLike",
    "NDArray",
    "ParamLayout",
    "field_names",
    "field_defaults",
    "extend_params",
]

#: Any float ndarray. Spelled once so strict mypy stays quiet everywhere.
NDArray = np.ndarray[Any, np.dtype[Any]]


def field_names(container: type[Any]) -> tuple[str, ...]:
    """Field names of a ``Params`` or ``Consts`` container, in order.

    Containers are :class:`equinox.Module` subclasses, so a variant model's
    container may inherit its parent's fields and append its own.

    Args:
        container: a ``Params`` or ``Consts`` class.

    Returns:
        The field names, base-class fields first.
    """
    return tuple(f.name for f in dataclasses.fields(container))


def extend_params(target: type[Any], base: Any, **extra: Any) -> Any:
    """Build ``target`` from a base ``Params`` instance plus extra fields.

    Used by a variant model whose ``Params`` inherits its parent's, so that
    the parent's ``init`` can be reused and only the new fields supplied.

    Note that a base class's ``init`` must name its own container explicitly
    rather than ``self.Params``: on a subclass instance the latter resolves to
    the extended container, for which the base would supply too few arguments.

    Args:
        target: the container class to build, extending ``type(base)``.
        base: an instance of the parent container.
        **extra: values for the fields ``target`` adds.

    Returns:
        An instance of ``target``.
    """
    values = {name: getattr(base, name) for name in field_names(type(base))}
    return target(**values, **extra)


def field_defaults(container: type[Any]) -> dict[str, Any]:
    """Default values of a container's defaulted fields.

    Args:
        container: a ``Params`` or ``Consts`` class.

    Returns:
        Field name to default value, omitting fields with no default.
    """
    return {
        f.name: f.default
        for f in dataclasses.fields(container)
        if f.default is not dataclasses.MISSING
    }


class FittingError(Exception):
    """Base class for errors raised by this package."""


class ParamLayout(eqx.Module):
    """Static partition of a model's parameters into fixed / shared / local.

    Resolved once, at trace time. ``share`` maps a parameter name to an
    ``(n_curves,)`` integer group index (all zeros for a single global value);
    ``fixed`` maps a name to a scalar or ``(n_curves,)`` array.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    local: tuple[str, ...] = eqx.field(static=True)
    shared: tuple[str, ...] = eqx.field(static=True)
    fixed_names: tuple[str, ...] = eqx.field(static=True)
    group_index: dict[str, jnp.ndarray]
    group_sizes: dict[str, int] = eqx.field(static=True)
    fixed_values: dict[str, jnp.ndarray]

    @property
    def n_local(self) -> int:
        """Number of free per-curve parameters."""
        return len(self.local)

    @property
    def n_shared_total(self) -> int:
        """Total number of shared values across all shared parameters."""
        # pylint: disable=not-an-iterable,unsubscriptable-object
        return sum(self.group_sizes[k] for k in self.shared)

    @property
    def has_shared(self) -> bool:
        """True when any parameter is shared, which selects the joint solve."""
        return bool(self.shared)


def build_layout(
    params_cls: type[Any],
    n_curves: int,
    share: Optional[Mapping[str, Any]] = None,
    fixed: Optional[Mapping[str, Any]] = None,
) -> ParamLayout:
    """Partition ``params_cls`` fields using call-site ``share``/``fixed``."""
    names: tuple[str, ...] = field_names(params_cls)
    share = dict(share or {})
    fixed = dict(fixed or {})

    unknown = (set(share) | set(fixed)) - set(names)
    if unknown:
        raise FittingError(
            f"share/fixed refer to parameters {sorted(unknown)} which are not "
            f"in {params_cls.__name__}: {list(names)}"
        )
    both = set(share) & set(fixed)
    if both:
        raise FittingError(
            f"parameters {sorted(both)} are given in both share= and fixed=; "
            f"a parameter cannot be simultaneously fitted and pinned"
        )

    group_index: dict[str, jnp.ndarray] = {}
    group_sizes: dict[str, int] = {}
    for name, sel in share.items():
        idx = (
            jnp.zeros(n_curves, dtype=jnp.int32)
            if sel is None
            else jnp.asarray(sel, dtype=jnp.int32)
        )
        if idx.shape != (n_curves,):
            raise FittingError(
                f"share['{name}'] resolved to a group index of shape "
                f"{idx.shape}, expected ({n_curves},)"
            )
        group_index[name] = idx
        group_sizes[name] = int(idx.max()) + 1 if n_curves else 0

    fixed_values = {
        k: jnp.asarray(v, dtype=jnp.float64) for k, v in fixed.items()
    }
    local = tuple(n for n in names if n not in share and n not in fixed)
    if not local and not share:
        raise FittingError(
            "every parameter is pinned; there is nothing to fit"
        )

    return ParamLayout(
        names=names,
        local=local,
        shared=tuple(n for n in names if n in share),
        fixed_names=tuple(n for n in names if n in fixed),
        group_index=group_index,
        group_sizes=group_sizes,
        fixed_values=fixed_values,
    )


def assemble_params(
    params_cls: type[Any],
    layout: ParamLayout,
    local: jnp.ndarray,
    shared: Mapping[str, jnp.ndarray],
) -> Any:
    """Rebuild a ``Params`` instance from local / shared / fixed pieces.

    ``local`` is ``(n_local,)`` for one curve; ``shared`` maps each shared
    parameter name to the already-gathered scalar for this curve.
    """
    values: dict[str, Any] = {}
    for i, name in enumerate(layout.local):
        values[name] = local[i]
    for name in layout.shared:
        values[name] = shared[name]
    for name in layout.fixed_names:
        values[name] = layout.fixed_values[name]
    return params_cls(**values)
