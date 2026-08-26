Losses
======

.. code-block:: python

   ft.fit(df, model=..., x=x, loss=ft.Poisson())

==================================  ============  ==========================
loss                                predictions   use for
==================================  ============  ==========================
:class:`~fitting.losses.Poisson`    log           counts
:class:`~fitting.losses.SSE`        linear        continuous, additive noise
:class:`~fitting.losses.LogSSE`     log           continuous, relative noise
==================================  ============  ==========================

All three take observations in linear space, so predictions, residuals and RMSE
are always reported in the units the data was supplied in.

Choosing one
------------

:class:`~fitting.losses.Poisson` is the likelihood for counts and the default.

:class:`~fitting.losses.SSE` is ordinary least squares, for a continuous
readout whose noise amplitude does not depend much on the signal.

:class:`~fitting.losses.LogSSE` is least squares on the logarithm of the
observations,

.. math::
    L = \sum_j w_j \left( \log y_j - \log \hat{y}_j \right)^2,

which is appropriate when the *relative* error is roughly constant instead --
the usual situation for a fluorescence signal spanning orders of magnitude,
where :class:`~fitting.losses.SSE` lets the brightest points dominate. Pass the
measured values as they are; the logarithm is taken internally. They must be
strictly positive, and non-positive ones are reported as an error rather than
silently becoming ``NaN``.

Sums, not means
---------------

Each objective sums over points rather than averaging. Within one curve the two
differ only by a constant and give identical fits, but in a joint fit a sum lets
a curve with more points count for more, which a mean would not. So these are
sums of squared error rather than means; ``rmse`` is the normalised quantity to
report.

Least-squares solvers
---------------------

:class:`~fitting.losses.SSE` and :class:`~fitting.losses.LogSSE` derive from
:class:`~fitting.losses.LeastSquares`, which supplies a residual vector as well
as a scalar. That makes a least-squares solver applicable on the independent
path:

.. code-block:: python

   import optimistix as optx
   ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE(),
          solver=optx.GaussNewton(rtol=1e-9, atol=1e-9))

The default ``BFGS`` is usually the better choice, though. Gauss-Newton and
Levenberg-Marquardt approximate the Hessian by :math:`J^\top J`, dropping the
:math:`\sum_j r_j \nabla^2 r_j` term, which is only justified when the residuals
are small at the optimum; and they pay for it with a Jacobian and an
:math:`n \times n` solve every iteration. Measured on 20,000 two- and
three-parameter curves, with identical accuracy in every row:

=====================  ====================  ====================  ====================
residuals              BFGS                  GaussNewton           LevenbergMarquardt
=====================  ====================  ====================  ====================
small (noise 0.005)    **0.41 s**, 15 steps  0.50 s, **7 steps**   0.90 s, 13 steps
large (noise 0.20)     **0.45 s**, 18 steps  1.48 s, 13 steps      8.81 s, 30 steps
=====================  ====================  ====================  ====================

Gauss-Newton halves the iteration count when the fit is good, but not enough to
win on wall clock; once the residuals are large both fall behind. Reach for them
when you have few parameters, a model that genuinely fits, and you care about
iteration count.

On the joint path they scale poorly, because the Jacobian covers every parameter
at once and grows quadratically in the number of curves. A joint least-squares
solve is refused once that array would exceed
:data:`~fitting.fitting.MAX_JOINT_JACOBIAN_BYTES`.

Writing one
-----------

Subclass :class:`~fitting.losses.Loss` and implement ``value``, or subclass
:class:`~fitting.losses.LeastSquares` and implement ``residual``, in which case
``value`` follows from it:

.. code-block:: python

   from typing import ClassVar
   import jax.numpy as jnp
   import fitting as ft

   class HuberLoss(ft.Loss):
       log_predictions: ClassVar[bool] = False
       delta: float = 1.0

       def value(self, pred, y, w):
           r = jnp.abs(y - pred)
           return jnp.sum(w * jnp.where(
               r <= self.delta,
               0.5 * jnp.square(r),
               self.delta * (r - 0.5 * self.delta),
           ))

A weight of zero excludes a point, so any transform of ``y`` must be guarded
before it is applied: ``0 * NaN`` is ``NaN``.
