"""Unit test suite for parallel JAX/Optimistix curve fitting."""

# pylint: disable=redefined-outer-name

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from fitting import (
    CustomModel,
    DoubleExponentialIntervalModel,
    SingleExponentialDecayModel,
    SingleExponentialIntervalModel,
    fit_curves,
    fit_double_exponential_stepwise,
    mse_loss,
    predict_sequence,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def sample_dissociation_data() -> (
    tuple[pd.DataFrame, jnp.ndarray, dict[str, np.ndarray]]
):
    """Generate mock dissociation count data for test curves."""
    # Setup some simple mock data for 5 curves and 5 time points
    np.random.seed(42)
    times = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])

    # Let's generate dissociation count data
    # 5 replicates, 5 curves
    index = pd.MultiIndex.from_tuples(
        [
            ("R1", "var1", "bc1"),
            ("R1", "var2", "bc2"),
            ("R2", "var1", "bc1"),
            ("R2", "var2", "bc2"),
            ("R3", "var1", "bc1"),
        ],
        names=["replicate", "variant", "barcode"],
    )

    # 5 curves, 5 timepoints
    # True log_y0 and log_k_off
    # We will simulate the interval predictions manually
    y0 = np.array([1000.0, 500.0, 800.0, 600.0, 1200.0])
    k_off = np.array([0.15, 0.25, 0.1, 0.3, 0.05])

    # Normalization vectors for each replicate
    # Just 1.0 (no scaling) for simplicity
    norm_dict = {"R1": np.ones(5), "R2": np.ones(5), "R3": np.ones(5)}

    # Generate expected counts with prepended 0.0
    times_full = np.insert(times, 0, 0.0)
    data = []
    for i in range(5):
        bounds = y0[i] * np.exp(-times_full * k_off[i])
        # Concat bound prediction
        counts = bounds[:-2] - bounds[1:-1]
        counts = np.append(counts, bounds[-1])
        # Add some Poisson-like noise
        noisy_counts = np.random.poisson(counts)
        data.append(noisy_counts)

    df = pd.DataFrame(data, index=index, columns=[f"t_{t}" for t in times])
    return df, times, norm_dict


def test_single_exponential_interval_fit(
    sample_dissociation_data: tuple[
        pd.DataFrame, jnp.ndarray, dict[str, np.ndarray]
    ],
) -> None:
    """Test fitting curves with single exponential interval model."""
    df, times, norm_dict = sample_dissociation_data

    model = SingleExponentialIntervalModel(concat_bound=True)
    res_df = fit_curves(
        df=df, model=model, x_data=times, norm=norm_dict, key_level="replicate"
    )

    assert len(res_df) == 5
    assert "log_y0" in res_df.columns
    assert "log_k_off" in res_df.columns
    assert "log_bg" in res_df.columns
    assert "loss" in res_df.columns
    assert "RMSE" in res_df.columns
    assert "total_count" in res_df.columns
    assert "kinetic_count" in res_df.columns
    assert "bound_count" in res_df.columns

    # Ensure RMSE is reasonably low
    assert np.all(res_df["RMSE"].to_numpy() < 50.0)


def test_double_exponential_interval_fit(
    sample_dissociation_data: tuple[
        pd.DataFrame, jnp.ndarray, dict[str, np.ndarray]
    ],
) -> None:
    """Test fitting curves with double exponential interval model."""
    df, times, norm_dict = sample_dissociation_data

    # First get single fit for seeding
    single_model = SingleExponentialIntervalModel(concat_bound=True)
    single_res = fit_curves(
        df=df,
        model=single_model,
        x_data=times,
        norm=norm_dict,
        key_level="replicate",
    )

    model = DoubleExponentialIntervalModel(concat_bound=True)
    res_df = fit_curves(
        df=df,
        model=model,
        x_data=times,
        norm=norm_dict,
        key_level="replicate",
        init_kwargs={"single_exp_fit": single_res},
    )

    assert len(res_df) == 5
    assert "log_y0_1" in res_df.columns
    assert "log_k_off_2" in res_df.columns
    assert "log_bg" in res_df.columns
    assert "AIC" in res_df.columns


def test_decay_models() -> None:
    """Test fitting decay curves with standard decay model."""
    np.random.seed(0)
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Generate simple decay data: y = 100 * exp(-0.5 * t)
    y_true = 100.0 * np.exp(-0.5 * times)
    noisy_y = np.random.normal(y_true, 1.0)
    noisy_y = np.maximum(noisy_y, 0.1)  # prevent negative values

    df = pd.DataFrame(
        [noisy_y], index=["curve1"], columns=[f"t_{t}" for t in times]
    )

    model = SingleExponentialDecayModel()
    res_df = fit_curves(
        df=df,
        model=model,
        x_data=times,
        loss_fn=mse_loss,  # Use MSE loss for Gaussian noise
    )

    assert len(res_df) == 1
    assert "log_bg" in res_df.columns
    # Check that it fits the parameters closely
    assert np.allclose(np.exp(res_df["log_y0"].iloc[0]), 100.0, rtol=0.1)
    assert np.allclose(np.exp(res_df["log_k"].iloc[0]), 0.5, rtol=0.1)


def test_decay_linear_custom_model() -> None:
    """Test fitting linear-space decay curves with negative values via CustomModel."""
    np.random.seed(0)
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Generate simple decay data in linear space with negative values: y = -100 * exp(-0.5 * t)
    y_true = -100.0 * np.exp(-0.5 * times)
    noisy_y = np.random.normal(y_true, 1.0)

    df = pd.DataFrame(
        [noisy_y], index=["curve1"], columns=[f"t_{t}" for t in times]
    )

    # Custom model predict function in linear space:
    def linear_decay_predict(
        p: jnp.ndarray, log_norm: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        norm = jnp.exp(log_norm)
        return norm * p[0] * jnp.exp(-p[1] * t)

    def linear_decay_init(y: np.ndarray, **_kwargs: Any) -> np.ndarray:
        return np.stack([y[:, 0], np.zeros(len(y))], axis=1)

    model = CustomModel(
        predict_fn=linear_decay_predict,
        param_names=["y0", "k"],
        init_fn=linear_decay_init,
    )

    # Loss function for linear space
    def linear_mse_loss(y_pred: jnp.ndarray, y_obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.square(y_obs - y_pred))

    res_df = fit_curves(
        df=df,
        model=model,
        x_data=times,
        loss_fn=linear_mse_loss,
        is_linear=True,
    )

    assert len(res_df) == 1
    assert "y0" in res_df.columns
    assert "k" in res_df.columns
    assert "log_bg" not in res_df.columns
    # Check that it fits the parameters closely
    assert np.allclose(res_df["y0"].iloc[0], -100.0, rtol=0.1)
    assert np.allclose(res_df["k"].iloc[0], 0.5, rtol=0.1)


def test_custom_model() -> None:
    """Test fitting line data using user-defined CustomModel."""
    # Fit y = m * x + c using custom model
    times = jnp.array([1.0, 2.0, 3.0, 4.0])
    y_obs = np.array([[5.1, 7.0, 8.9, 11.2]])  # m=2, c=3 with some noise
    df = pd.DataFrame(y_obs, index=["line1"])

    def line_predict(
        p: jnp.ndarray, log_norm: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        return log_norm + jnp.log(jnp.clip(p[0] * t + p[1], min=1e-5))

    def line_init(y: np.ndarray, **_kwargs: Any) -> np.ndarray:
        # Initial guess: slope = 1.0, intercept = 1.0
        return np.ones((len(y), 2))

    model = CustomModel(
        predict_fn=line_predict,
        param_names=["slope", "intercept"],
        init_fn=line_init,
    )

    res_df = fit_curves(df=df, model=model, x_data=times, loss_fn=mse_loss)

    assert len(res_df) == 1
    assert np.allclose(res_df["slope"].iloc[0], 2.0, rtol=0.1)
    assert np.allclose(res_df["intercept"].iloc[0], 3.0, rtol=0.1)


def test_predict_sequence(
    sample_dissociation_data: tuple[
        pd.DataFrame, jnp.ndarray, dict[str, np.ndarray]
    ],
) -> None:
    """Test predicting a specific sequence curve from fitted parameters."""
    df, times, norm_dict = sample_dissociation_data
    model = SingleExponentialIntervalModel(concat_bound=True)
    res_df = fit_curves(
        df=df, model=model, x_data=times, norm=norm_dict, key_level="replicate"
    )

    key = ("R1", "var1", "bc1")
    log_pred = predict_sequence(
        fitted_df=res_df,
        model=model,
        sequence_key=key,
        x_data=times,
        norm_dict=norm_dict,
        key_level="replicate",
    )

    assert log_pred.shape == (5,)


def test_stepwise_double_exponential_fit(
    sample_dissociation_data: tuple[
        pd.DataFrame, jnp.ndarray, dict[str, np.ndarray]
    ],
) -> None:
    """Test the stepwise double exponential fitting wrapper."""
    df, times, norm_dict = sample_dissociation_data

    # Test interval style stepwise fit
    res_df = fit_double_exponential_stepwise(
        df=df,
        x_data=times,
        model_style="interval",
        norm=norm_dict,
        key_level="replicate",
    )

    assert len(res_df) == 5
    assert "log_y0_single_exp" in res_df.columns
    assert "log_k_off_single_exp" in res_df.columns
    assert "log_bg_single_exp" in res_df.columns
    assert "loss_single_exp" in res_df.columns
    assert "RMSE_single_exp" in res_df.columns
    assert "log_y0_1_double_exp" in res_df.columns
    assert "log_k_off_2_double_exp" in res_df.columns
    assert "log_bg_double_exp" in res_df.columns
    assert "loss_double_exp" in res_df.columns
    assert "RMSE_double_exp" in res_df.columns
    assert "total_count" in res_df.columns

    # Test predict_sequence with suffix
    key = ("R1", "var1", "bc1")
    single_model = SingleExponentialIntervalModel(concat_bound=True)
    log_pred_single = predict_sequence(
        fitted_df=res_df,
        model=single_model,
        sequence_key=key,
        x_data=times,
        norm_dict=norm_dict,
        key_level="replicate",
        column_suffix="_single_exp",
    )
    assert log_pred_single.shape == (5,)

    double_model = DoubleExponentialIntervalModel(concat_bound=True)
    log_pred_double = predict_sequence(
        fitted_df=res_df,
        model=double_model,
        sequence_key=key,
        x_data=times,
        norm_dict=norm_dict,
        key_level="replicate",
        column_suffix="_double_exp",
    )
    assert log_pred_double.shape == (5,)
