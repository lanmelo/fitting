Sharing parameters
==================

Some parameters belong to a group of curves rather than to one curve. A
saturation level is a property of a protein preparation and a detector, not of
the DNA variant being measured, so the curves from one device should share it.

Which parameters are shared is a property of the *fit*, not of the model, so one
model covers every variation:

.. code-block:: python

   # one value for the whole experiment
   ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE(),
          share={"log_ymax": None})

   # one value per replicate
   ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE(),
          share={"log_ymax": ft.by_level("replicate")})

   # known in advance, so not fitted at all
   ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE(),
          fixed={"log_ymax": np.log(2.28)})

Shared values are returned on their own index, with a standard error:

.. code-block:: text

     parameter  group     value        se  linear_value
   0  log_ymax      0  1.609227  0.000428      4.998946

Why it helps
------------

A titration that does not reach saturation cannot identify :math:`y_{\max}` and
:math:`K_d` separately: only their ratio is determined, so a per-curve
:math:`y_{\max}` absorbs noise and drags :math:`K_d` with it. Sharing one value
across the curves that legitimately share it removes that freedom.

On real MITOMI data, sharing one saturation level per adapter set reduced the
between-slide scatter of :math:`\log K_d` by about a fifth. On synthetic data
where no curve saturates, the improvement was an order of magnitude.

What it costs
-------------

A shared parameter enters every curve's term, so the objective no longer
separates:

.. math::
    L(\theta, \phi_1, \dots, \phi_N)
    = \sum_{i=1}^{N} \ell_i(\theta, \phi_i)

One optimiser therefore runs over the whole parameter vector. This cannot be
chunked -- all data and all parameters must be resident together -- and it is
substantially slower than the independent path. Keep sharing for parameters that
genuinely are shared.

Identifiability
---------------

Sharing only helps when the shared parameter is not collinear with a per-curve
one. A global multiplicative scale, for instance, is the *same direction* as the
per-curve amplitudes: adding one on top of :math:`N` free amplitudes
over-parameterises the model by exactly one degree of freedom, the fit fails to
converge, and the value drifts.

The reported standard error is the curvature of the objective with the local
parameters held at their fitted values. It therefore **ignores coupling with the
local parameters and is a lower bound**. It will still flag the case that
matters in practice -- a shared parameter sitting at a genuine but nearly flat
optimum -- but a small error is not on its own evidence that a shared parameter
is well determined.
