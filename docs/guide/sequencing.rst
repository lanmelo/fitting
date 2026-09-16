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
why it is a *constant* by default.

Fitting the depth instead
~~~~~~~~~~~~~~~~~~~~~~~~~

The spike-ins measure the depth from one construct at one concentration, while
the samples themselves are thousands of curves spanning every library. Fitting
the depth from the samples is therefore tempting, and
:class:`~fitting.SingleExponentialIntervalFittedDepth` does it: the depth
becomes a shared, vector-valued parameter, relative to the same reference
library the spike-ins use.

The usual pattern is two stages -- fit the depth on a set of reference curves,
then hold it constant for the rest -- which keeps the coupled solve small. Two
things decide whether it is worth doing:

**Pin the reference.** With every entry free, scaling a group's :math:`\eta`
and dividing its curves' :math:`y_0` by the same factor changes no prediction.
The objective has an exactly flat ridge and the solve does not converge. The
model pins the reference entry for this reason; do not work around it.

**Give the solve enough steps.** With tens of thousands of local parameters the
coupled solve is slow: a fit with 36 shared and 56350 local parameters took
18469 LBFGS steps. The default ``joint_max_steps`` of 2000 stops well short, and
a result read off a capped solve is not a result. Check
``FitResult.info["joint_status"]``.

**Do not judge it by likelihood.** Even pinned, one direction is weakly
determined: an exponential ramp in :math:`\eta` is nearly indistinguishable from
a shift in every :math:`k_{\mathrm{off}}`, and only differences in curve shape
within the group separate them. A group whose curves all have the same shape --
one variant, say -- constrains it least.

That direction is a nuisance parameter confounded with the quantity of interest,
and adding it always lowers the objective. Worse, a held-out likelihood check
does not catch it: the depth is shared across the whole group, so no curve in
that group is independent of it. On STAMMP-seq data a depth fitted this way
improved held-out likelihood by 0.16 nats per curve while shifting one
replicate's median :math:`k_{\mathrm{off}}` twenty-fold, in the opposite
direction from another's.

**Judge it by an external criterion instead** -- something the fit does not see.
Agreement between replicates works well: if the depth is real, the same
variant's :math:`k_{\mathrm{off}}` should agree better across replicates once it
is applied. In the case above, agreement got **ten times worse** while the
within-replicate correlations barely moved, which is the signature of a
per-group scale artefact rather than a depth measurement. Report every quantity
per group; a pooled median hid a twenty-fold shift in one replicate behind an
opposite shift in another.

See :doc:`sharing` for the mechanics of vector-valued shared parameters.

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
