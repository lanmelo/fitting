"""Numerically stable log-space helpers."""

import jax
import jax.numpy as jnp


def log1mexp(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    r"""Computes the element-wise log1mexp in a numerically stable way.

    .. math::
        \log \left( 1 - e^{-x} \right)

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    return jnp.where(x > 0.693, jnp.log1p(-jnp.exp(-x)), jnp.log(-jnp.expm1(-x)))  # type: ignore[operator]


def logsubexp(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute ``log(exp(x) - exp(y))`` stably for ``x > y``."""
    ok = x > y
    y_safe = jnp.where(ok, y, x - 1.0)
    out = x + jnp.log1p(-jnp.exp(y_safe - x))
    return jnp.where(ok, out, -jnp.inf)
