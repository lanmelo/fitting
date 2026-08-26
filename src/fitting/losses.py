"""Objective functions.

==================  ===========  ==============================
objective           predictions  use for
==================  ===========  ==============================
:class:`Poisson`    log          counts
:class:`SSE`        linear       continuous, additive noise
:class:`LogSSE`     log          continuous, relative noise
==================  ===========  ==============================

All take observations in linear space. Each sums over points rather than
averaging, so the reported loss is comparable across curves of unequal length.
"""

from typing import Any, ClassVar, override

import equinox as eqx
import jax.numpy as jnp

from .core import FittingError


class Loss(eqx.Module):
    r"""Base class for objectives.

    Subclasses declare :attr:`log_predictions` and implement :meth:`value`.

    A weight of zero excludes a point, so any transform of ``y`` must be
    guarded *before* it is applied: ``0 * NaN`` is still ``NaN``.
    """

    #: Whether :meth:`value` is passed ``log`` predictions rather than
    #: predictions themselves. Must agree with the model's.
    log_predictions: ClassVar[bool]
    #: Whether the objective is a negative log-likelihood, and so whether AIC
    #: is meaningful.
    is_nll: ClassVar[bool] = False
    #: Whether the objective requires strictly positive observations.
    requires_positive: ClassVar[bool] = False

    def check_model(self, model: Any) -> None:
        """Verify that a model produces the predictions this objective needs.

        No conversion is inserted either way, since either direction would
        silently change what is minimised.

        Args:
            model: the model about to be fitted.

        Raises:
            FittingError: if the model and objective disagree, naming an
                objective that would match.
        """
        if bool(model.log_predictions) is bool(self.log_predictions):
            return
        suits = (
            "fitting.Poisson (for counts) or fitting.LogSSE (for a "
            "continuous signal)"
            if model.log_predictions
            else "fitting.SSE"
        )
        produces = "log" if model.log_predictions else "linear"
        wants = "log" if self.log_predictions else "linear"
        raise FittingError(
            f"{type(model).__name__} produces {produces}-space predictions "
            f"but {type(self).__name__} consumes {wants}-space predictions. "
            f"Use {suits}, or change the model's `log_predictions`."
        )

    def value(
        self, pred: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate the objective for one curve.

        Args:
            pred: predictions, ``log`` if :attr:`log_predictions`, shape
                ``(n_points,)``.
            y: observations in linear space, shape ``(n_points,)``.
            w: per-point weights; zero excludes a point.

        Returns:
            The objective as a scalar.
        """
        raise NotImplementedError


class LeastSquares(Loss):
    r"""Base class for objectives that are a sum of squared residuals.

    .. math::
        L = \sum_j r_j^2

    Subclasses implement :meth:`residual`; :meth:`value` follows from it, so
    the two cannot disagree. Supplying a residual is what makes a least-squares
    solver such as ``optx.GaussNewton`` applicable.
    """

    def residual(
        self, pred: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        """Signed residuals whose sum of squares is :meth:`value`.

        Args:
            pred: predictions, ``log`` if :attr:`log_predictions`, shape
                ``(n_points,)``.
            y: observations in linear space, shape ``(n_points,)``.
            w: per-point weights; zero excludes a point.

        Returns:
            Residuals of shape ``(n_points,)``.
        """
        raise NotImplementedError

    @override
    def value(
        self, pred: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        """Sum of squared :meth:`residual` values."""
        return jnp.sum(jnp.square(self.residual(pred, y, w)))


class Poisson(Loss):
    r"""Poisson negative log-likelihood, for counts.

    .. math::
        L = \sum_j w_j \left( \hat{y}_j - y_j \log \hat{y}_j \right),
        \qquad \log \hat{y}_j \ \text{supplied}

    dropping the :math:`\log y_j!` term, which does not depend on the
    parameters. Predictions arrive as :math:`\log \hat{y}` and the rate is
    never formed in linear space, which would underflow for a fast decay.
    """

    log_predictions: ClassVar[bool] = True
    is_nll: ClassVar[bool] = True

    @override
    def value(
        self, pred: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate the negative log-likelihood; ``pred`` is ``log`` rate."""
        return jnp.sum(w * (jnp.exp(pred) - y * pred))


class SSE(LeastSquares):
    r"""Sum of squared errors, for a continuous signal with additive noise.

    .. math::
        L = \sum_j w_j \left( y_j - \hat{y}_j \right)^2

    Equivalent to mean squared error up to a per-curve constant. Appropriate
    when the noise amplitude is roughly independent of the signal.
    """

    log_predictions: ClassVar[bool] = False

    @override
    def residual(
        self, pred: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        """Weighted residuals ``sqrt(w) * (y - pred)``.

        The weight enters as its square root so that squaring and summing
        reproduces the weighted objective exactly.
        """
        return jnp.sqrt(w) * (y - pred)


class LogSSE(LeastSquares):
    r"""Sum of squared errors in log space, for relative noise.

    .. math::
        L = \sum_j w_j \left( \log y_j - \log \hat{y}_j \right)^2

    Equivalent to mean squared logarithmic error up to a per-curve constant.
    Appropriate when the *relative* error is roughly constant, as for a signal
    spanning orders of magnitude, where :class:`SSE` lets the brightest points
    dominate.

    Observations must be strictly positive, since the logarithm is taken here.
    """

    log_predictions: ClassVar[bool] = True
    requires_positive: ClassVar[bool] = True

    @override
    def residual(
        self, pred: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        """Weighted residuals between ``log y`` and ``pred``.

        Masked observations are replaced before the logarithm rather than
        after, since ``0 * NaN`` would still be ``NaN``.
        """
        return jnp.sqrt(w) * (jnp.log(jnp.where(w > 0, y, 1.0)) - pred)
