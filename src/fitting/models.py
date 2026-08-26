r"""Built-in curve models.

Notation used throughout, matching the parameter names:

.. math::
    \eta_j = e^{\mathtt{log\_norm}_j}, \quad
    y_0 = e^{\mathtt{log\_y0}}, \quad
    k_{\mathrm{off}} = e^{\mathtt{log\_k\_off}}, \quad
    b = e^{\mathtt{log\_bg}}

Rates and amplitudes are fitted in log space because they span orders of
magnitude and must stay positive; predictions for the count models are
likewise formed in log space, so no intermediate quantity is ever
exponentiated and re-logged."""

from typing import Any, ClassVar, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .core import ArrayLike, NDArray, extend_params

# Every nested Params/Consts is a pure data container, so pylint's
# minimum-public-methods rule never applies to them.
# pylint: disable=too-few-public-methods
from .utils import logsubexp


class NoConsts(eqx.Module):
    """Constants container for models that need no constants."""


class CurveModel(eqx.Module):
    r"""A curve model.

    Subclasses declare:

    * ``Params`` -- a nested container of every parameter of the model.
      Whether a parameter ends up per-curve, shared across a group of curves,
      or pinned to a constant is decided at the call site through
      ``fit(share=...)`` and ``fit(fixed=...)``, so ``predict`` always receives
      plain scalars and need not distinguish them.
    * ``Consts`` -- a nested container of the model's constants, each with a
      default, so a caller supplies only what differs.
    * ``log_predictions`` -- whether ``predict`` returns :math:`\log \hat{y}`
      rather than :math:`\hat{y}`. Must be matched by the objective.
    * ``predict`` and ``init``.

    Configuration that changes the *shape* of the computation (for example
    ``concat_bound``) belongs in an ``eqx.field(static=True)`` field, so a
    plain Python ``if`` may be used on it inside ``predict``.
    """

    Params: ClassVar[type[Any]]
    Consts: ClassVar[type[Any]] = NoConsts
    log_predictions: ClassVar[bool] = True

    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        """Predict a single curve.

        Called under ``vmap``, once per curve.

        Args:
            p: ``Params``, every field a scalar.
            x: the independent variable for this curve, shape ``(n_points,)``.
            c: ``Consts``. Shared constants keep the shape they were given;
                per-curve constants have had their leading axis removed.

        Returns:
            Predictions of shape ``(n_points,)``, ``log`` if
            :attr:`log_predictions`.
        """
        raise NotImplementedError

    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        """Choose starting values for a batch of curves.

        Unlike :meth:`predict` this is not vectorised, so it sees every curve
        at once and may derive a per-curve guess from the data.

        Args:
            y: observations, shape ``(n_curves, n_points)``.
            x: the independent variable, shared or per-curve.
            c: ``Consts``, with per-curve constants at full width.

        Returns:
            ``Params`` with every field of shape ``(n_curves,)``.
        """
        raise NotImplementedError

    def extra_columns(self, y: NDArray) -> dict[str, NDArray]:
        """Summarise the observations for the results table.

        Args:
            y: observations, shape ``(n_curves, n_points)``.

        Returns:
            Column name to per-curve value, added to
            :attr:`~fitting.results.FitResult.diagnostics`.
        """
        return {"total_count": np.sum(y, axis=1)}


# --------------------------------------------------------------------------
# Interval (sequencing) dissociation models
# --------------------------------------------------------------------------
class SingleExponentialInterval(CurveModel):
    r"""Single-exponential dissociation observed as counts per time interval.

    A single population dissociates with rate :math:`k_{\mathrm{off}}`, so the
    bound amount at time :math:`t` is

    .. math::
        B(t) = y_0 \, e^{-k_{\mathrm{off}} t}.

    ``x`` holds the right-hand edges :math:`x_1 < \dots < x_N` of the
    sampling intervals, and an implicit :math:`x_0 = 0` edge is prepended. Each
    observation is the amount released during one interval, scaled by that
    library's spike-in normalisation:

    .. math::
        \hat{y}_j = \eta_j \left[ B(x_{j-1}) - B(x_j) \right],
        \qquad j = 1, \dots, N.

    With ``concat_bound=True`` the last element of ``x`` repeats the final
    timepoint, and the last prediction is instead the **absolute remaining
    bound population** -- the library sequenced after repeated dissociation
    steps:

    .. math::
        \hat{y}_j = \begin{cases}
            \eta_j \left[ B(x_{j-1}) - B(x_j) \right]
                & j = 1, \dots, N-1 \\
            \eta_N \, B(x_N) & j = N
        \end{cases}

    That final term carries a large share of the information at low counts.
    Differences are evaluated with :func:`~fitting.utils.logsubexp`, keeping
    the whole computation in log space.
    """

    class Params(eqx.Module):
        r"""Amplitude :math:`\log y_0`, off-rate
        :math:`\log k_{\mathrm{off}}`."""

        log_y0: ArrayLike
        log_k_off: ArrayLike

    class Consts(eqx.Module):
        r"""Sequencing-depth normalisation :math:`\log \eta`, in log space.

        The default of ``0.0`` means "no normalisation" and broadcasts against
        any number of timepoints, so a model can be used with or without
        spike-ins without changing its signature.
        """

        log_norm: ArrayLike = 0.0

    log_predictions: ClassVar[bool] = True

    concat_bound: bool = eqx.field(static=True, default=True)

    def _log_bound(self, p: Any, t: jnp.ndarray) -> jnp.ndarray:
        r"""Return :math:`\log B(t)` at every interval edge in ``t``.

        Overridden by the subclasses to change :math:`B`, which is the only
        thing that distinguishes the models in this family.
        """
        return cast(jnp.ndarray, p.log_y0 - t * jnp.exp(p.log_k_off))

    def _to_counts(self, log_bound: jnp.ndarray) -> jnp.ndarray:
        r"""Turn :math:`\log B` at the edges into per-interval log counts.

        Computes :math:`\log\left[ B(x_{j-1}) - B(x_j) \right]`, appending
        :math:`\log B(x_N)` when ``concat_bound`` is set. Defined once and
        inherited, so the background variants cannot drift from the plain
        models in how they treat the final library.
        """
        if self.concat_bound:
            log_counts = logsubexp(log_bound[:-2], log_bound[1:-1])
            return jnp.concatenate([log_counts, log_bound[-1:]], axis=-1)
        return logsubexp(log_bound[:-1], log_bound[1:])

    @override
    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        t = jnp.insert(x, 0, 0.0)
        return cast(
            jnp.ndarray, c.log_norm + self._to_counts(self._log_bound(p, t))
        )

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        log_y0 = jnp.log(jnp.maximum(1.0, jnp.sum(y, axis=1)))
        return SingleExponentialInterval.Params(
            log_y0=log_y0, log_k_off=jnp.full(y.shape[0], -3.0)
        )

    @override
    def extra_columns(self, y: NDArray) -> dict[str, NDArray]:
        total = np.sum(y, axis=1)
        if self.concat_bound:
            return {
                "total_count": total,
                "kinetic_count": np.sum(y[:, :-1], axis=1),
                "bound_count": y[:, -1],
            }
        return {
            "total_count": total,
            "kinetic_count": total,
            "bound_count": np.zeros(len(y)),
        }


class SingleExponentialIntervalWithBackground(SingleExponentialInterval):
    r""":class:`SingleExponentialInterval` plus a constant background floor.

    .. math::
        B(t) = y_0 \, e^{-k_{\mathrm{off}} t} + b

    **Opt in deliberately.** A constant floor cancels out of every interval
    difference, so :math:`b` is informed only by the trailing bound
    observation -- and not at all when ``concat_bound=False``, where it has no
    effect on the predictions whatever. On low-count data it is often not
    identifiable; check its standard error before trusting it. See the
    Sequencing counts guide.
    """

    class Params(SingleExponentialInterval.Params):
        r""":class:`SingleExponentialInterval.Params` plus a floor
        :math:`\log b`."""

        log_bg: ArrayLike

    @override
    def _log_bound(self, p: Any, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.logaddexp(super()._log_bound(p, t), p.log_bg)

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        base = super().init(y, x, c)
        return extend_params(self.Params, base, log_bg=base.log_y0 - 4.0)


class DoubleExponentialInterval(SingleExponentialInterval):
    r"""Two independent dissociating populations, observed as interval counts.

    .. math::
        B(t) = y_{0,1} \, e^{-k_{\mathrm{off},1} t}
             + y_{0,2} \, e^{-k_{\mathrm{off},2} t}

    The interval differencing and the optional trailing bound term are
    inherited unchanged from :class:`SingleExponentialInterval`, so the two
    models can be compared on the same observations. The components are
    exchangeable, so the two :math:`(y_0, k_{\mathrm{off}})` pairs are only
    identified up to swapping them.
    """

    class Params(eqx.Module):
        r"""Two populations, each with :math:`\log y_0` and
        :math:`\log k_{\mathrm{off}}`."""

        log_y0_1: ArrayLike
        log_k_off_1: ArrayLike
        log_y0_2: ArrayLike
        log_k_off_2: ArrayLike

    @override
    def _log_bound(self, p: Any, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.logaddexp(
            p.log_y0_1 - t * jnp.exp(p.log_k_off_1),
            p.log_y0_2 - t * jnp.exp(p.log_k_off_2),
        )

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        log_y0 = jnp.log(jnp.maximum(1.0, jnp.sum(y, axis=1)))
        n = y.shape[0]
        return DoubleExponentialInterval.Params(
            log_y0_1=log_y0,
            log_k_off_1=jnp.full(n, -3.0),
            log_y0_2=log_y0 - 2.0,
            log_k_off_2=jnp.zeros(n),
        )

    @classmethod
    def seed_from(cls, single: Any) -> Any:
        """Seed a double-exponential fit from a converged single fit."""
        return DoubleExponentialInterval.Params(
            log_y0_1=single.log_y0,
            log_k_off_1=single.log_k_off,
            log_y0_2=single.log_y0 - 2.0,
            log_k_off_2=jnp.zeros_like(single.log_y0),
        )


class DoubleExponentialIntervalWithBackground(DoubleExponentialInterval):
    r""":class:`DoubleExponentialInterval` plus a constant background floor.

    .. math::
        B(t) = y_{0,1} \, e^{-k_{\mathrm{off},1} t}
             + y_{0,2} \, e^{-k_{\mathrm{off},2} t} + b

    The identifiability warning on
    :class:`SingleExponentialIntervalWithBackground` applies here too: the
    floor cancels from every interval difference, so it is informed only by
    the trailing bound observation, and not at all when
    ``concat_bound=False``.
    """

    class Params(DoubleExponentialInterval.Params):
        r""":class:`DoubleExponentialInterval.Params` plus a floor
        :math:`\log b`."""

        log_bg: ArrayLike

    @override
    def _log_bound(self, p: Any, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.logaddexp(super()._log_bound(p, t), p.log_bg)

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        b = super().init(y, x, c)
        return extend_params(self.Params, b, log_bg=b.log_y0_1 - 4.0)

    @classmethod
    @override
    def seed_from(cls, single: Any) -> Any:
        base = DoubleExponentialInterval.seed_from(single)
        return extend_params(
            cls.Params,
            base,
            log_bg=getattr(single, "log_bg", single.log_y0 - 4.0),
        )


# --------------------------------------------------------------------------
# Plain decay models (no interval differencing)
# --------------------------------------------------------------------------
class SingleExponentialDecay(CurveModel):
    r"""Plain exponential decay, with no interval differencing.

    .. math::
        \hat{y}_j = \eta_j \, y_0 \, e^{-k x_j}

    Unlike the interval models, each observation is the signal *at* a
    timepoint rather than the amount released between two of them, so no
    ``t = 0`` edge is prepended and ``x`` is used as given.
    """

    class Params(eqx.Module):
        r"""Amplitude :math:`\log y_0` and decay rate :math:`\log k`."""

        log_y0: ArrayLike
        log_k: ArrayLike

    Consts: ClassVar[type[Any]] = SingleExponentialInterval.Consts

    log_predictions: ClassVar[bool] = True

    def _log_signal(self, p: Any, x: jnp.ndarray) -> jnp.ndarray:
        r"""Return :math:`\log\left[ y_0 e^{-k x} \right]` at each ``x``."""
        return cast(jnp.ndarray, p.log_y0 - x * jnp.exp(p.log_k))

    @override
    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        return cast(jnp.ndarray, c.log_norm + self._log_signal(p, x))

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        return SingleExponentialDecay.Params(
            log_y0=jnp.log(jnp.maximum(1.0, y[:, 0])),
            log_k=jnp.full(y.shape[0], -3.0),
        )


class SingleExponentialDecayWithBackground(SingleExponentialDecay):
    r""":class:`SingleExponentialDecay` plus a constant background floor.

    .. math::
        \hat{y}_j = \eta_j \left[ y_0 \, e^{-k x_j} + b \right]

    Here the background *is* identifiable, unlike in the interval models:
    there is no differencing for it to cancel out of, so every observation
    constrains it. It is pinned down mainly by the late timepoints, where the
    decaying term has become small.
    """

    class Params(SingleExponentialDecay.Params):
        r""":class:`SingleExponentialDecay.Params` plus a floor
        :math:`\log b`."""

        log_bg: ArrayLike

    @override
    def _log_signal(self, p: Any, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.logaddexp(super()._log_signal(p, x), p.log_bg)

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        b = super().init(y, x, c)
        return extend_params(self.Params, b, log_bg=b.log_y0 - 4.0)


# --------------------------------------------------------------------------
# Binding isotherms (linear space)
# --------------------------------------------------------------------------
class Langmuir(CurveModel):
    r"""Langmuir binding isotherm; ``x`` is free ligand concentration.

    .. math::
        \hat{y}_j = y_{\max} \, \frac{x_j}{x_j + K_d}

    where :math:`K_d = e^{\mathtt{log\_kd}}` and
    :math:`y_{\max} = e^{\mathtt{log\_ymax}}`. Both are fitted in log space
    because they span orders of magnitude and must stay positive.

    :math:`y_{\max}` is the natural candidate for ``share=`` or ``fixed=``: it
    is identified separately from :math:`K_d` only if the titration reaches
    saturation. See the Sharing parameters guide.
    """

    class Params(eqx.Module):
        r"""Dissociation constant :math:`\log K_d` and saturation
        :math:`\log y_{\max}`."""

        log_kd: ArrayLike
        log_ymax: ArrayLike

    log_predictions: ClassVar[bool] = False

    @override
    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        return jnp.exp(p.log_ymax) * x / (x + jnp.exp(p.log_kd))

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        n = y.shape[0]
        return Langmuir.Params(
            log_kd=jnp.full(n, float(jnp.log(jnp.nanmedian(jnp.asarray(x))))),
            log_ymax=jnp.log(jnp.maximum(jnp.nanmax(y, axis=1), 1e-12)),
        )


class LangmuirWithOffset(Langmuir):
    r""":class:`Langmuir` with a non-zero baseline.

    .. math::
        \hat{y}_j = \left( y_{\max} - c \right)
                     \frac{x_j}{x_j + K_d} + c

    Parameterised so that :math:`c` is the value at :math:`x = 0` and
    :math:`y_{\max}` remains the asymptote as :math:`x \to \infty`, rather
    than an amplitude to which the baseline is added. Unlike the other
    parameters, ``offset`` is fitted directly rather than in log space,
    because a baseline may legitimately be negative.
    """

    class Params(Langmuir.Params):
        r""":class:`Langmuir.Params` plus a linear baseline :math:`c`."""

        offset: ArrayLike

    @override
    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        ymax = jnp.exp(p.log_ymax)
        return cast(
            jnp.ndarray,
            (ymax - p.offset) * x / (x + jnp.exp(p.log_kd)) + p.offset,
        )

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        b = super().init(y, x, c)
        return extend_params(self.Params, b, offset=jnp.zeros(y.shape[0]))


class LogisticAffinity(CurveModel):
    r"""Langmuir isotherm against **log** concentration.

    .. math::
        \hat{y}_j = y_{\max} \, \sigma\!\left( x_j - \log K_d \right)
        = \frac{y_{\max}}{1 + e^{-(x_j - \log K_d)}}

    where :math:`\sigma` is the logistic function and :math:`x_j` is
    :math:`\log` concentration. Substituting :math:`x_j = \log c_j` recovers
    :class:`Langmuir` exactly, so this is the same model in a better
    conditioned coordinate.
    """

    class Params(eqx.Module):
        r"""Dissociation constant :math:`\log K_d` and saturation
        :math:`\log y_{\max}`."""

        log_kd: ArrayLike
        log_ymax: ArrayLike

    log_predictions: ClassVar[bool] = False

    @override
    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        return jnp.exp(p.log_ymax) * jax.scipy.special.expit(x - p.log_kd)

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        n = y.shape[0]
        return LogisticAffinity.Params(
            log_kd=jnp.full(n, float(jnp.nanmedian(jnp.asarray(x)))),
            log_ymax=jnp.log(jnp.maximum(jnp.nanmax(y, axis=1), 1e-12)),
        )
