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

A per-curve parameter is always a scalar; a shared one may be a **vector**, one
entry per observation. That suits a quantity common to a group of curves but
varying along the observation axis, such as a per-library sequencing depth.

The width is read from the starting values: give ``init`` a
``(n_curves, width)`` array instead of the usual ``(n_curves,)``.

.. code-block:: python

   model = ft.SingleExponentialIntervalFittedDepth(concat_bound=True)
   guess = dataclasses.replace(
       model.init(y, x, model.Consts()),
       log_eta=seed,                         # (n_curves, width)
   )
   res = ft.fit(counts, x=x, model=model, loss=ft.Poisson(),
                share={"log_eta": ft.by_level("replicate")}, init=guess)

Fitted values come back on the shared frame with a ``component`` column, so one
group spans ``width`` rows:

.. code-block:: text

     parameter  group  component     value        se
   0   log_eta      0          0  0.586490  0.000006
   1   log_eta      0          1  0.529294  0.000008

They are left out of :attr:`~fitting.results.FitResult.table`, which has one row
per curve. Requesting a width for a per-curve parameter is an error.

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

A vector-valued parameter can be collinear with a per-curve one through the
model's *shape*, not only its scale: a free depth vector is an arbitrary
:math:`N`-point rescaling of the mean curve, so if every curve in the group has
nearly the same shape the two are indistinguishable. What identifies it is
diversity of curve shape within the group. Seed from an independent estimate and
compare against it afterwards; a large deviation means the parameter is
absorbing something other than what it names.

The reported standard error is the curvature of the objective with the local
parameters held fixed, so it **ignores coupling with them and is a lower
bound**. ``shared_method="profile"`` accounts for that coupling.

Choosing a method
-----------------

With anything shared, the objective no longer separates and one solver runs over
all parameters. ``shared_method`` chooses how:

``"joint"`` (default)
   Optimise every parameter at once with LBFGS. Needs all data and parameters
   resident, and its step count grows with the number of curves.

``"profile"``
   Minimise over the shared values alone, solving the curves independently at
   each step and taking the outer gradient from the envelope theorem. Reaches
   the same optimum over ``n_shared_total`` dimensions rather than that plus one
   set per curve, keeps memory bounded by ``chunk_size``, and gives standard
   errors that account for coupling with the per-curve parameters.

Prefer ``"profile"`` when few values are shared across many curves; it costs one
inner solve per outer step, so it loses when many values are shared.
