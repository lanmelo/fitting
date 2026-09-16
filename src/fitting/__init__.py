"""Generalised, JAX/Optimistix-accelerated curve fitting.

Fit many curves at once, optionally sharing parameters between them.

    import fitting as ft

    res = ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE)
    res = ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE,
                 share={"log_ymax": ft.by_level("replicate")})

With no shared parameters every curve is independent, and the fit is a chunked
``vmap`` whose peak memory does not grow with the number of curves. Sharing a
parameter couples the curves, so a single joint solve is used instead.
"""

from .core import FittingError
from .fitting import fit, fit_sequencing, fit_stepwise, spikein_log_norm
from .losses import SSE, LeastSquares, LogSSE, Loss, Poisson
from .models import (
    CurveModel,
    DoubleExponentialInterval,
    DoubleExponentialIntervalFittedDepth,
    DoubleExponentialIntervalWithBackground,
    FittedDepth,
    Langmuir,
    LangmuirWithOffset,
    LogisticAffinity,
    NoConsts,
    SingleExponentialDecay,
    SingleExponentialDecayWithBackground,
    SingleExponentialInterval,
    SingleExponentialIntervalFittedDepth,
    SingleExponentialIntervalWithBackground,
)
from .results import FitResult
from .selectors import ByLevel, by_column, by_level
from .utils import log1mexp, logsubexp

__all__ = [
    # entry points
    "fit",
    "fit_sequencing",
    "fit_stepwise",
    # core
    "CurveModel",
    "FitResult",
    "Loss",
    "NoConsts",
    "FittedDepth",
    "FittingError",
    # losses
    "Poisson",
    "SSE",
    "LogSSE",
    # models
    "SingleExponentialInterval",
    "SingleExponentialIntervalFittedDepth",
    "SingleExponentialIntervalWithBackground",
    "DoubleExponentialInterval",
    "DoubleExponentialIntervalFittedDepth",
    "DoubleExponentialIntervalWithBackground",
    "SingleExponentialDecay",
    "SingleExponentialDecayWithBackground",
    "Langmuir",
    "LangmuirWithOffset",
    "LogisticAffinity",
    # parameter containers are nested on their models, e.g.
    # fitting.Langmuir.Params -- see CurveModel
    # selectors and helpers
    "by_level",
    "by_column",
    "ByLevel",
    "spikein_log_norm",
    "log1mexp",
    "logsubexp",
]
