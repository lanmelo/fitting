"""Pre-built and custom curve models for JAX-based fitting."""

from typing import Any, Callable, Optional, override

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .utils import logsubexp


class BaseModel:
    """Abstract base class for all fitting models."""

    @property
    def param_names(self) -> list[str]:
        """Names of parameters in the order they are optimized."""
        raise NotImplementedError

    def predict(
        self,
        p: jnp.ndarray,
        log_norm: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Predict the model output for a single curve.

        Args:
            p: Parameters for the curve (1D jnp.ndarray).
            log_norm: Log-scale normalization array (1D jnp.ndarray).
            times: Timepoints array (1D jnp.ndarray).
            **kwargs: Additional model-specific arguments.

        Returns:
            log_predictions: 1D jnp.ndarray of log-predicted values.
        """
        raise NotImplementedError

    def initialize_params(
        self, y_obs: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Initialize parameters for a batch of curves.

        Args:
            y_obs: Observed data matrix of shape (n_curves, n_timepoints).
            **kwargs: Additional arguments for initialization.

        Returns:
            init_p: Stacked parameters of shape (n_curves, n_params).
        """
        raise NotImplementedError

    def postprocess_results(
        self, df_results: pd.DataFrame, y_data: np.ndarray
    ) -> pd.DataFrame:
        """Apply model-specific post-processing to the results DataFrame.

        Args:
            df_results: The fitted parameters DataFrame.
            y_data: The raw observed data array.

        Returns:
            df_results: The modified DataFrame.
        """
        # Default behavior: compute total count
        df_results["total_count"] = np.sum(y_data, axis=1)
        return df_results


class SingleExponentialIntervalModel(BaseModel):
    """Single exponential dissociation model over time intervals.

    Model parameters:
        p[0]: log_y0 (initial signal)
        p[1]: log_k_off (dissociation rate)
    """

    def __init__(self, concat_bound: bool = True):
        self.concat_bound = concat_bound

    @property
    @override
    def param_names(self) -> list[str]:
        return ["log_y0", "log_k_off", "log_bg"]

    @override
    def predict(
        self,
        p: jnp.ndarray,
        log_norm: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        # Prepend 0.0 to times
        times_arr = jnp.insert(times, 0, 0.0)
        log_bound = jnp.logaddexp(p[0] - times_arr * jnp.exp(p[1]), p[2])

        if self.concat_bound:
            log_counts = logsubexp(log_bound[:-2], log_bound[1:-1])
            log_counts = jnp.concatenate([log_counts, log_bound[-1:]], axis=-1)
        else:
            log_counts = logsubexp(log_bound[:-1], log_bound[1:])

        return log_norm + log_counts

    @override
    def initialize_params(
        self, y_obs: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        s = np.sum(y_obs, axis=1)
        log_y0_arr = np.log(np.maximum(1.0, s))
        log_koff_arr = np.full(len(y_obs), -3.0)
        log_bg_arr = log_y0_arr - 4.0
        return np.stack([log_y0_arr, log_koff_arr, log_bg_arr], axis=1)

    @override
    def postprocess_results(
        self, df_results: pd.DataFrame, y_data: np.ndarray
    ) -> pd.DataFrame:
        n_curves = len(y_data)
        df_results["total_count"] = np.sum(y_data, axis=1)
        if self.concat_bound:
            df_results["kinetic_count"] = np.sum(y_data[:, :-1], axis=1)
            df_results["bound_count"] = y_data[:, -1]
        else:
            df_results["kinetic_count"] = df_results["total_count"]
            df_results["bound_count"] = np.zeros(n_curves)
        return df_results


class DoubleExponentialIntervalModel(BaseModel):
    """Double exponential dissociation model over time intervals.

    Model parameters:
        p[0]: log_y0_1 (initial signal for first component)
        p[1]: log_k_off_1 (dissociation rate for first component)
        p[2]: log_y0_2 (initial signal for second component)
        p[3]: log_k_off_2 (dissociation rate for second component)
    """

    def __init__(self, concat_bound: bool = True):
        self.concat_bound = concat_bound

    @property
    @override
    def param_names(self) -> list[str]:
        return ["log_y0_1", "log_k_off_1", "log_y0_2", "log_k_off_2", "log_bg"]

    @override
    def predict(
        self,
        p: jnp.ndarray,
        log_norm: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        times_arr = jnp.insert(times, 0, 0.0)
        log_bound_1 = p[0] - times_arr * jnp.exp(p[1])
        log_bound_2 = p[2] - times_arr * jnp.exp(p[3])
        log_bound = jnp.logaddexp(
            jnp.logaddexp(log_bound_1, log_bound_2), p[4]
        )

        if self.concat_bound:
            log_counts = logsubexp(log_bound[:-2], log_bound[1:-1])
            log_counts = jnp.concatenate([log_counts, log_bound[-1:]], axis=-1)
        else:
            log_counts = logsubexp(log_bound[:-1], log_bound[1:])

        return log_norm + log_counts

    @override
    def initialize_params(
        self, y_obs: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        n_curves = len(y_obs)
        if "single_exp_fit" in kwargs:
            single_fit = kwargs["single_exp_fit"]
            if isinstance(single_fit, pd.DataFrame):
                log_y0 = single_fit["log_y0"].to_numpy()
                log_koff = single_fit["log_k_off"].to_numpy()
                log_bg = (
                    single_fit["log_bg"].to_numpy()
                    if "log_bg" in single_fit.columns
                    else (log_y0 - 4.0)
                )
            else:
                log_y0 = single_fit[:, 0]
                log_koff = single_fit[:, 1]
                log_bg = (
                    single_fit[:, 2]
                    if single_fit.shape[1] > 2
                    else (log_y0 - 4.0)
                )
        else:
            s = np.sum(y_obs, axis=1)
            log_y0 = np.log(np.maximum(1.0, s))
            log_koff = np.full(n_curves, -3.0)
            log_bg = log_y0 - 4.0

        # Seed double exponential from single exponential fit
        log_y0_1 = log_y0
        log_koff_1 = log_koff
        log_y0_2 = log_y0 - 2.0
        log_koff_2 = np.full(n_curves, 0.0)

        return np.stack(
            [log_y0_1, log_koff_1, log_y0_2, log_koff_2, log_bg], axis=1
        )

    @override
    def postprocess_results(
        self, df_results: pd.DataFrame, y_data: np.ndarray
    ) -> pd.DataFrame:
        n_curves = len(y_data)
        df_results["total_count"] = np.sum(y_data, axis=1)
        if self.concat_bound:
            df_results["kinetic_count"] = np.sum(y_data[:, :-1], axis=1)
            df_results["bound_count"] = y_data[:, -1]
        else:
            df_results["kinetic_count"] = df_results["total_count"]
            df_results["bound_count"] = np.zeros(n_curves)
        return df_results


class SingleExponentialDecayModel(BaseModel):
    """Standard single exponential decay model y = norm * y0 * exp(-k * t).

    Model parameters:
        p[0]: log_y0 (initial signal)
        p[1]: log_k (decay rate)
    """

    @property
    @override
    def param_names(self) -> list[str]:
        return ["log_y0", "log_k", "log_bg"]

    @override
    def predict(
        self,
        p: jnp.ndarray,
        log_norm: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        log_signal = p[0] - times * jnp.exp(p[1])
        return log_norm + jnp.logaddexp(log_signal, p[2])

    @override
    def initialize_params(
        self, y_obs: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        # Initial guess from the first timepoint
        s = y_obs[:, 0]
        log_y0_arr = np.log(np.maximum(1.0, s))
        log_k_arr = np.full(len(y_obs), -3.0)
        log_bg_arr = log_y0_arr - 4.0
        return np.stack([log_y0_arr, log_k_arr, log_bg_arr], axis=1)


class DoubleExponentialDecayModel(BaseModel):
    """Standard double exponential decay model y = norm * (components).

    Model parameters:
        p[0]: log_y0_1 (initial signal for first component)
        p[1]: log_k1 (decay rate for first component)
        p[2]: log_y0_2 (initial signal for second component)
        p[3]: log_k2 (decay rate for second component)
    """

    @property
    @override
    def param_names(self) -> list[str]:
        return ["log_y0_1", "log_k1", "log_y0_2", "log_k2", "log_bg"]

    @override
    def predict(
        self,
        p: jnp.ndarray,
        log_norm: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        log_bound_1 = p[0] - times * jnp.exp(p[1])
        log_bound_2 = p[2] - times * jnp.exp(p[3])
        log_bound = jnp.logaddexp(log_bound_1, log_bound_2)
        return log_norm + jnp.logaddexp(log_bound, p[4])

    @override
    def initialize_params(
        self, y_obs: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        n_curves = len(y_obs)
        if "single_exp_fit" in kwargs:
            single_fit = kwargs["single_exp_fit"]
            if isinstance(single_fit, pd.DataFrame):
                log_y0 = single_fit["log_y0"].to_numpy()
                log_k = single_fit["log_k"].to_numpy()
                log_bg = (
                    single_fit["log_bg"].to_numpy()
                    if "log_bg" in single_fit.columns
                    else (log_y0 - 4.0)
                )
            else:
                log_y0 = single_fit[:, 0]
                log_k = single_fit[:, 1]
                log_bg = (
                    single_fit[:, 2]
                    if single_fit.shape[1] > 2
                    else (log_y0 - 4.0)
                )
        else:
            s = y_obs[:, 0]
            log_y0 = np.log(np.maximum(1.0, s))
            log_k = np.full(n_curves, -3.0)
            log_bg = log_y0 - 4.0

        log_y0_1 = log_y0
        log_k1 = log_k
        log_y0_2 = log_y0 - 2.0
        log_k2 = np.full(n_curves, 0.0)

        return np.stack([log_y0_1, log_k1, log_y0_2, log_k2, log_bg], axis=1)


class CustomModel(BaseModel):
    """Wrapper class for user-defined fitting models."""

    def __init__(
        self,
        predict_fn: Callable[..., jnp.ndarray],
        param_names: list[str],
        init_fn: Optional[Callable[..., np.ndarray]] = None,
    ):
        """Args:

        predict_fn: A function with signature predict_fn(p, log_norm, times)
          returning log predictions.
        param_names: List of parameter names in order of optimization.
        init_fn: Optional function with signature init_fn(y_obs) returning
          initial parameters.
        """
        self._predict_fn = predict_fn
        self._param_names = param_names
        self._init_fn = init_fn

    @property
    @override
    def param_names(self) -> list[str]:
        return self._param_names

    @override
    def predict(
        self,
        p: jnp.ndarray,
        log_norm: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        return self._predict_fn(p, log_norm, times, **kwargs)

    @override
    def initialize_params(
        self, y_obs: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        if self._init_fn is not None:
            return self._init_fn(y_obs, **kwargs)
        raise NotImplementedError(
            "An initialization function was not provided for this custom "
            "model. Please provide initial parameters directly to "
            "fit_curves via `init_params`."
        )
