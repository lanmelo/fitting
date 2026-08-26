Quickstart
==========

Install in editable mode:

.. code-block:: bash

   pip install -e .

Every fit needs three things: a table of observations with one curve per row, a
model, and the independent variable.

.. code-block:: python

   import jax.numpy as jnp
   import pandas as pd
   import fitting as ft

   counts = pd.read_csv("counts.csv", index_col=["replicate", "variant"])
   x = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)

   res = ft.fit(counts, model=ft.SingleExponentialInterval(), x=x)

Reading the result
------------------

:class:`~fitting.results.FitResult` keeps the per-curve parameters separate from
the per-group ones, because a fit with shared parameters has two index spaces.

============================  =================================================
attribute                     contents
============================  =================================================
``res.local``                 per-curve parameters, plus a linear-space copy of
                              each ``log_``-prefixed one
``res.shared``                per-group parameters with standard errors, empty
                              when nothing is shared
``res.diagnostics``           ``loss``, ``rmse``, ``steps``, ``converged``,
                              ``status``
``res.observed``              the observations actually fitted, after column
                              selection and masking
``res.predicted``             model predictions, in observation units
``res.residuals``             ``observed - predicted``, masked points ``NaN``
``res.table``                 the flat one-row-per-curve view to write to CSV
============================  =================================================

Non-convergence is reported rather than assumed away:

.. code-block:: python

   failed = ~res.diagnostics.converged
   print(f"{failed.sum()} of {len(res.local)} curves did not converge")
   print(res.diagnostics.status.value_counts())

Scale
-----

Peak memory is set by ``chunk_size``, not by the number of curves, so the row
count is limited only by the size of the table itself:

.. code-block:: python

   res = ft.fit(counts, model=ft.SingleExponentialInterval(), x=x,
                chunk_size=50_000)

Ragged curves
-------------

Curves need not be the same length. Pad short rows with ``NaN`` and they are
masked out of the objective. When each point has its own independent variable,
pass ``x`` as a 2-D array of the same shape:

.. code-block:: python

   res = ft.fit(signal, model=ft.LogisticAffinity(), x=log_conc_2d,
                loss=ft.SSE())

Padding ``x`` with ``NaN`` as well is fine: a non-finite ``x`` at a masked point
is replaced, and one at an unmasked point raises rather than quietly spoiling
convergence.
