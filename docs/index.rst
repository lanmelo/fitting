fitting
=======

Fit large numbers of curves at once with JAX and Optimistix, optionally sharing
parameters between them.

With no shared parameters every curve is an independent problem, so the fit is a
chunked ``vmap`` whose peak memory does not grow with the number of curves.
Sharing a parameter couples the curves, and a single joint solve is used
instead.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   guide/quickstart
   guide/sharing
   guide/models
   guide/losses
   guide/sequencing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/fitting
   api/models
   api/losses
   api/results
   api/selectors
   api/core
   api/utils

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
