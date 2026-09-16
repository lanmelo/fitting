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

Vector-valued shared parameters
-------------------------------

A per-curve parameter is always a scalar, but a shared one may be a **vector**,
one entry per observation. That is the natural shape for a quantity that is
common to a group of curves but varies along the observation axis -- a
per-library sequencing depth, say, rather than a single number per group.

The width is read from the starting values, so it never has to be declared:
give ``init`` a ``(n_curves, width)`` array for that parameter instead of the
usual ``(n_curves,)``.

.. code-block:: python

   model = ft.SingleExponentialIntervalFittedDepth(concat_bound=True)
   guess = dataclasses.replace(
       model.init(y, x, ft.NoConsts()),
       log_eta=seed,                         # (n_curves, width)
   )
   res = ft.fit(counts, x=x, model=model, loss=ft.Poisson(),
                share={"log_eta": ft.by_level("replicate")}, init=guess)

The fitted values come back on the shared frame with a ``component`` column
indexing the entries, so one group spans ``width`` rows:

.. code-block:: text

     parameter  group  component     value        se
   0   log_eta      0          0  0.586490  0.000006
   1   log_eta      0          1  0.529294  0.000008

They are left out of :attr:`~fitting.results.FitResult.table`, which has one
row per curve and so no place to put a vector.

A per-curve parameter is always a scalar. Requesting a width for one is an
error, since there would be no way to interpret it.

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

A vector-valued shared parameter makes this easier to get wrong, because it can
be collinear with a per-curve parameter *through the model's shape*, not just
its scale. A free depth vector over :math:`N` observations is an arbitrary
:math:`N`-point rescaling of the mean curve; if every curve in the group has
nearly the same shape, that vector and the shape are indistinguishable, and it
will absorb the shape rather than the depth. What identifies it is **diversity**
among the curves sharing it: they must differ enough in form that no single
rescaling can account for all of them. Choose the group's members for that
diversity, seed from an independent estimate, and compare the fitted vector
against that estimate afterwards -- a large deviation means the parameter is
absorbing something other than what it names.

The reported standard error is the curvature of the objective with the local
parameters held at their fitted values. It therefore **ignores coupling with the
local parameters and is a lower bound**. It will still flag the case that
matters in practice -- a shared parameter sitting at a genuine but nearly flat
optimum -- but a small error is not on its own evidence that a shared parameter
is well determined.
