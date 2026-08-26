Sequencing counts
=================

:func:`~fitting.fitting.fit_sequencing` is :func:`~fitting.fitting.fit` with the
conventions of count data applied: spike-in normalisation, a Poisson
likelihood, and the interval dissociation model. :func:`~fitting.fitting.fit`
itself knows nothing about spike-ins, which is what keeps it usable for
fluorescence and other readouts.

.. code-block:: python

   import fitting as ft

   counts = pd.read_csv("counts.csv.gz",
                        index_col=["replicate", "variant", "barcode"])

   # How spike-ins are laid out is a property of the experiment. Here they
   # ride in the same table, tagged variant == "spike".
   is_spike = counts.index.get_level_values("variant") == "spike"
   spikein = (counts[is_spike]
              .droplevel(["variant", "barcode"])
              .groupby("replicate").sum())

   res = ft.fit_sequencing(counts[~is_spike], x=x, spikein=spikein,
                            spikein_level="replicate")

The package takes spike-in *counts*; extracting them from your table is your
own step, since how they are stored varies by experiment. In the STAMMP-seq
layout they are ordinary rows tagged ``variant == "spike"``, as above; see
``examples/fit_koff_example.py`` for the full version.

Spike-ins
---------

Spike-ins correct for differing sequencing depth between the libraries that
contribute to one curve. Counts are divided by the spike-in's own value in a
reference column, making the normalisation relative:

.. math::
    \eta_j = \frac{s_j}{s_{\mathrm{ref}}},
    \qquad \texttt{log\_norm}_j = \log \eta_j

Pass one vector for a single experiment, or a frame plus ``spikein_level`` for
one vector per replicate. In the second case the values are held once per group
and gathered, so memory is proportional to the number of groups rather than the
number of curves.

The normalisation enters the model as an additive offset in log space, which is
why it is a *constant* rather than a parameter. Fitting it instead is possible
in principle but rarely identifiable: a free overall scale is the same direction
as the per-curve amplitudes. See :doc:`sharing`.

Interval counts and the bound library
-------------------------------------

The interval models treat each observation as the amount released during one
interval:

.. math::
    \hat{y}_j = \eta_j \left[ B(x_{j-1}) - B(x_j) \right]

With ``concat_bound=True`` the last entry of ``x`` repeats the final timepoint
and the last prediction is instead the **absolute remaining bound population**,
matching a library sequenced after repeated dissociation steps:

.. math::
    \hat{y}_N = \eta_N \, B(x_N)

That term matters. At the low counts typical of this assay it carries a large
share of the information about the dissociation rate, and dropping it shifts the
fitted rate substantially.

Aggregation
-----------

Counts are often summed over index levels that should not distinguish curves --
barcodes, backgrounds -- before fitting. Do that with pandas and pass the
result; the package does not decide which levels are meaningful:

.. code-block:: python

   keep = [n for n in counts.index.names if n != "barcode"]
   counts = counts.groupby(keep).sum()
