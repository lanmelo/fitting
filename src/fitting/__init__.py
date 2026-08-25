"""Fitting package providing parallel curve fitting utilities."""

from .fitter import fit_curves, fit_double_exponential_stepwise
from .losses import mse_loss, mse_loss_linear, poisson_loss
from .models import (
    BaseModel,
    CustomModel,
    DoubleExponentialDecayModel,
    DoubleExponentialIntervalModel,
    SingleExponentialDecayModel,
    SingleExponentialIntervalModel,
)
from .utils import build_normalization_array, predict_sequence

__all__ = [
    "BaseModel",
    "SingleExponentialIntervalModel",
    "DoubleExponentialIntervalModel",
    "SingleExponentialDecayModel",
    "DoubleExponentialDecayModel",
    "CustomModel",
    "poisson_loss",
    "mse_loss",
    "mse_loss_linear",
    "fit_curves",
    "fit_double_exponential_stepwise",
    "predict_sequence",
    "build_normalization_array",
]
