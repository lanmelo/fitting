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

from typing import Any, ClassVar, Optional, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .core import ArrayLike, FittingError, NDArray, ParamLayout, extend_params

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

    def check_layout(self, layout: "ParamLayout") -> None:
        """Check that this model can be fitted under ``layout``.

        Called once per :func:`~fitting.fitting.fit`, after the layout is
        resolved. The default accepts everything; a model overrides it when a
        parameter is only meaningful shared, or only vector valued.

        Args:
            layout: the resolved parameter layout.

        Raises:
            FittingError: if the layout is not usable.
        """

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


class FittedDepth(eqx.Module):
    r"""Mixin promoting an interval model's depth constant to a parameter.

    The depth :math:`\log \eta` moves out of ``Consts`` and into ``Params`` as
    a shared, vector-valued parameter, so it is estimated from the curves
    rather than supplied by spike-ins. Predictions are otherwise unchanged:

    .. math::
        \hat{y}_j = \eta_j \left[ B(x_{j-1}) - B(x_j) \right].

    Like :func:`~fitting.fitting.spikein_log_norm`, the depth is **relative to
    a reference library** ``ref``, whose entry is pinned to
    :math:`\log \eta_{\mathrm{ref}} = 0`. So ``log_eta`` holds :math:`N - 1`
    numbers, not :math:`N`, and the fitted values are directly comparable to a
    ``log_norm`` built from spike-ins with the same ``ref``.

    The pin is not cosmetic. Without it, scaling a group's :math:`\eta` and
    dividing every member's :math:`y_0` by the same factor leaves every
    prediction unchanged, so the objective has an exactly flat ridge and the
    solve does not converge.

    ``log_eta`` must also be shared: per curve it is degenerate with the
    amplitudes and rates. Share it over the level the depth varies with -- a
    replicate, say.

    Even pinned and shared, one further direction is only weakly determined:
    the *overall level* of the non-reference entries relative to the reference.
    Raising every :math:`\eta_j` for :math:`j \neq \mathrm{ref}` together is
    nearly the same as shifting every rate, and only differences in curve
    *shape* within the group tell the two apart. Two consequences:

    * A group whose curves all have the same shape -- one variant -- constrains
      it least, so a shape-diverse reference set is preferable.
    * That level can absorb any systematic failure of the curve model, and it
      will. If the fitted depth is much wider than an independent estimate,
      that is what has happened.

    Set ``pin_level`` to take the level from ``Consts.log_eta_level`` instead
    of fitting it, leaving only the depth's *shape* free. That is usually what
    you want: the shape is well determined by the curves, the level is not, and
    an independent estimate of the level already exists in the spike-ins.
    Pinning it also conditions the solve, since the near-flat direction is
    removed from the optimisation rather than from its answer.

    Either way, check the result: seed from the spike-ins, compare the fitted
    vector against that seed, and confirm the depth improves something the fit
    did not see -- agreement between groups, say. Likelihood cannot judge this,
    because every curve in a group is tied to that group's depth.

    Subclasses pair this with a base interval model and add ``log_eta`` to its
    ``Params``.
    """

    class LevelConsts(eqx.Module):
        r"""The overall depth level, used only when ``pin_level`` is set.

        The mean of :math:`\log \eta` over every observation except ``ref``.
        Give it the value an independent estimate supplies -- from spike-ins,
        via :func:`~fitting.fitting.spikein_log_norm` -- since this is the one
        direction the curves themselves cannot pin down.
        """

        log_eta_level: ArrayLike = 0.0

    Consts: ClassVar[type[Any]] = LevelConsts

    #: Which observation the depth is measured relative to. Its entry is
    #: pinned to zero and omitted from ``log_eta``.
    ref: int = eqx.field(static=True, default=-1)

    #: Hold the mean of the non-reference entries at ``log_eta_level`` rather
    #: than fitting it, so only the depth's shape is free. ``log_eta`` then
    #: holds :math:`N - 2` numbers instead of :math:`N - 1`.
    pin_level: bool = eqx.field(static=True, default=False)

    def n_free(self, n_points: int) -> int:
        """How many numbers ``log_eta`` holds for ``n_points`` observations."""
        return n_points - (2 if self.pin_level else 1)

    def _full_log_eta(
        self, log_eta: jnp.ndarray, n_points: int, level: ArrayLike
    ) -> jnp.ndarray:
        """Build the per-observation depth from the free parameters.

        Without ``pin_level`` this only re-inserts the reference zero. With it,
        ``log_eta`` holds :math:`N - 2` deviations, the last deviation is set
        to make them sum to zero, and ``level`` supplies their mean.
        """
        if not self.pin_level:
            return jnp.insert(log_eta, self.ref % n_points, 0.0)
        centred = jnp.append(log_eta, -jnp.sum(log_eta))
        return jnp.insert(level + centred, self.ref % n_points, 0.0)

    def check_layout(self, layout: ParamLayout) -> None:
        """Require ``log_eta`` to be shared and vector valued."""
        if "log_eta" not in layout.shared:
            raise FittingError(
                "log_eta must be shared: fitted per curve it is degenerate "
                "with the amplitudes and rates. Pass share={'log_eta': ...} "
                "to group the curves sequenced at a common depth."
            )
        if layout.width("log_eta") == 1:
            raise FittingError(
                "log_eta must be vector valued, with one entry per "
                "observation except the reference. Its starting values should "
                "have shape (n_curves, n_points - 1)."
            )

    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        """Predict one curve, with the depth taken from ``p`` not ``c``."""
        # The cast is for the type checker only. `base` is `self`, so both
        # calls still dispatch through the concrete model -- which is what
        # makes the double-exponential variant use its own _log_bound.
        # pylint: disable=protected-access
        base = cast(SingleExponentialInterval, self)
        t = jnp.insert(x, 0, 0.0)
        log_eta = self._full_log_eta(
            jnp.asarray(p.log_eta), x.shape[0], c.log_eta_level
        )
        counts = base._to_counts(base._log_bound(p, t))
        return log_eta + counts

    def full_depth(
        self, shared: pd.DataFrame, level: Optional[ArrayLike] = None
    ) -> NDArray:
        r"""Turn fitted ``log_eta`` into a ``log_norm``, one row per group.

        Args:
            shared: :attr:`~fitting.results.FitResult.shared`, or any frame
                with ``parameter``, ``group``, ``component`` and ``value``
                columns.
            level: the pinned level, one per group in group-index order.
                Required when ``pin_level`` is set, ignored otherwise.

        Returns:
            Array of shape ``(n_groups, n_points)``, with the reference entry
            restored as zero, ready to pass as ``log_norm``. Rows are ordered
            by group index.

        Raises:
            FittingError: if ``pin_level`` is set and ``level`` is missing.
        """
        block = (
            shared[shared["parameter"] == "log_eta"]
            .pivot(index="group", columns="component", values="value")
            .sort_index()
            .to_numpy()
        )
        if self.pin_level:
            if level is None:
                raise FittingError(
                    "pin_level is set, so full_depth needs level=, the same "
                    "log_eta_level that was passed as a constant, one value "
                    "per group"
                )
            block = np.concatenate(
                [block, -block.sum(axis=1, keepdims=True)], axis=1
            )
            block = block + np.asarray(level, dtype=float).reshape(-1, 1)
        return np.insert(block, self.ref % (block.shape[1] + 1), 0.0, axis=1)


class SingleExponentialIntervalFittedDepth(
    FittedDepth, SingleExponentialInterval
):
    """:class:`SingleExponentialInterval` with the depth fitted, not fixed.

    See :class:`FittedDepth` for what the depth means and how to check it.
    """

    class Params(SingleExponentialInterval.Params):
        r""":class:`SingleExponentialInterval.Params` plus the relative depth
        :math:`\log \eta`, one entry per observation except ``ref``."""

        log_eta: ArrayLike

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        return extend_params(
            SingleExponentialIntervalFittedDepth.Params,
            SingleExponentialInterval.init(self, y, x, c),
            log_eta=jnp.zeros((y.shape[0], self.n_free(y.shape[1]))),
        )


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


class DoubleExponentialIntervalFittedDepth(
    FittedDepth, DoubleExponentialInterval
):
    """:class:`DoubleExponentialInterval` with the depth fitted, not fixed.

    Adding a second population does not on its own make the depth better
    determined -- on STAMMP-seq data this variant drifted further than the
    single-exponential one the longer it was run, and did not converge in
    200000 LBFGS steps. Twice as many per-curve parameters give the weakly
    determined depth direction more, not less, to trade against. See
    :class:`FittedDepth`.
    """

    class Params(DoubleExponentialInterval.Params):
        r""":class:`DoubleExponentialInterval.Params` plus the relative depth
        :math:`\log \eta`, one entry per observation except ``ref``."""

        log_eta: ArrayLike

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        return extend_params(
            DoubleExponentialIntervalFittedDepth.Params,
            DoubleExponentialInterval.init(self, y, x, c),
            log_eta=jnp.zeros((y.shape[0], self.n_free(y.shape[1]))),
        )

    @classmethod
    @override
    def seed_from(cls, single: Any) -> Any:
        """Seed from a converged single-exponential fitted-depth fit."""
        return extend_params(
            DoubleExponentialIntervalFittedDepth.Params,
            DoubleExponentialInterval.seed_from(single),
            log_eta=single.log_eta,
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
