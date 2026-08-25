"""Loss functions for curve fitting (Poisson, MSE, Huber)."""

import jax.numpy as jnp


def poisson_loss(log_y_pred: jnp.ndarray, y_obs: jnp.ndarray) -> jnp.ndarray:
    """Poisson negative log-likelihood loss (ignores factorial term)."""
    return jnp.sum(jnp.exp(log_y_pred) - y_obs * log_y_pred)


def mse_loss(log_y_pred: jnp.ndarray, y_obs: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error in linear space (expects log predictions)."""
    return jnp.mean(jnp.square(y_obs - jnp.exp(log_y_pred)))


def mse_loss_linear(y_pred: jnp.ndarray, y_obs: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error in linear space (expects linear predictions)."""
    return jnp.mean(jnp.square(y_obs - y_pred))
