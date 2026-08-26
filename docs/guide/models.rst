Models
======

A model states what it predicts, in which space, and what its parameters and
constants are.

Built in
--------

.. list-table::
   :header-rows: 1
   :widths: 46 12 42

   * - model
     - space
     - parameters
   * - :class:`~fitting.models.SingleExponentialInterval`
     - log
     - ``log_y0``, ``log_k_off``
   * - :class:`~fitting.models.SingleExponentialIntervalWithBackground`
     - log
     - the above, plus ``log_bg``
   * - :class:`~fitting.models.DoubleExponentialInterval`
     - log
     - two amplitude and rate pairs
   * - :class:`~fitting.models.DoubleExponentialIntervalWithBackground`
     - log
     - the above, plus ``log_bg``
   * - :class:`~fitting.models.SingleExponentialDecay`
     - log
     - ``log_y0``, ``log_k``
   * - :class:`~fitting.models.SingleExponentialDecayWithBackground`
     - log
     - the above, plus ``log_bg``
   * - :class:`~fitting.models.Langmuir`
     - linear
     - ``log_kd``, ``log_ymax``
   * - :class:`~fitting.models.LangmuirWithOffset`
     - linear
     - the above, plus ``offset``
   * - :class:`~fitting.models.LogisticAffinity`
     - linear
     - ``log_kd``, ``log_ymax``

Every model documents the function it fits. The dissociation and decay families
each have a separate ``...WithBackground`` subclass adding a constant floor;
that is deliberately a different model rather than a flag, because on low-count
data the background is often not identifiable and including it shifts the fitted
rate.

Log space or linear space
-------------------------

A model's ``log_predictions`` says whether :meth:`~fitting.models.CurveModel.predict`
returns :math:`\hat{y}` or :math:`\log \hat{y}`, and it must agree with the
objective it is paired with. This is not a formality. The interval models compute differences
of exponentials, and forming the rate in linear space and taking its logarithm
again loses the result outright once the decay is fast: :math:`\hat{y}`
underflows to zero and its logarithm takes the likelihood and its gradient to
``NaN``. Staying in log space avoids the round trip.

The consequence for a caller is simply that log-space models pair with
:class:`~fitting.losses.Poisson` or :class:`~fitting.losses.LogSSE`, and
linear-space models with :class:`~fitting.losses.SSE`. A mismatch raises
:class:`~fitting.core.FittingError` and names an objective that would fit.

Writing one
-----------

Declare the parameters, whether predictions are in log space, and the two
methods:

.. code-block:: python

   from typing import ClassVar
   import equinox as eqx
   import jax.numpy as jnp
   from jax.typing import ArrayLike
   import fitting as ft

   class Hill(ft.CurveModel):
       class Params(eqx.Module):
           log_kd: ArrayLike
           log_ymax: ArrayLike
           n: ArrayLike

       log_predictions: ClassVar[bool] = False

       def predict(self, p, x, c):
           return jnp.exp(p.log_ymax) / (1.0 + (jnp.exp(p.log_kd) / x) ** p.n)

       def init(self, y, x, c):
           n_curves = y.shape[0]
           return self.Params(
               log_kd=jnp.full(n_curves, float(jnp.log(jnp.nanmedian(x)))),
               log_ymax=jnp.log(jnp.nanmax(y, axis=1)),
               n=jnp.ones(n_curves),
           )

Parameters are reached by name, never by position. ``Params`` is nested on the
model, so it is ``Hill.Params`` and does not add to the package namespace, and
it is a dataclass so a variant can inherit it:

.. code-block:: python

   class HillWithBaseline(Hill):
       class Params(Hill.Params):     # log_kd, log_ymax, n, then...
           offset: ArrayLike          # ...one more

       def predict(self, p, x, c):
           return super().predict(p, x, c) + p.offset

A base class's ``init`` must name its own container rather than ``self.Params``,
since on a variant the latter is the extended container. Use
:func:`~fitting.core.extend_params` in the variant to supply only the new
fields.

Constants
---------

Constants are declared the same way, in a nested ``Consts`` container with
defaults, and supplied either shared or per-curve:

.. code-block:: python

   ft.fit(df, model=MyModel(), x=x,
          consts={"temperature": 298.0},          # shared by all curves
          curve_consts={"conc": conc_per_curve})  # one row per curve

Configuration that changes the *shape* of the computation, such as
``concat_bound``, belongs in an ``eqx.field(static=True)`` field so that a plain
``if`` may be used on it inside ``predict``.
