"""Module for parallel JAX/Optimistix curve fitting on DataFrame rows."""

from typing import Any, Callable, Mapping, Optional, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd

from .losses import poisson_loss
from .models import (
    BaseModel,
    DoubleExponentialDecayModel,
    DoubleExponentialIntervalModel,
    SingleExponentialDecayModel,
    SingleExponentialIntervalModel,
)
from .utils import build_normalization_array

# Enable JAX float64 for scientific computing
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


def fit_curves(
    df: pd.DataFrame,
    model: BaseModel,
    x_data: jnp.ndarray,
    norm: Optional[Mapping[str, Any]] = None,
    norm_arr: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    key_level: Union[int, str] = 0,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = poisson_loss,
    solver: Optional[optx.AbstractMinimiser[Any, Any, Any]] = None,
    init_params: Optional[np.ndarray] = None,
    chunk_size: int = 50_000,
    rmse_chunk_size: int = 250_000,
    solver_options: Optional[dict[str, Any]] = None,
    init_kwargs: Optional[dict[str, Any]] = None,
    is_linear: bool = False,
) -> pd.DataFrame:
    """Fit arbitrary models to curves concurrently using JAX.

    Args:
        df: Input DataFrame containing curves. Each row is a curve, and columns
          are timepoints.
        model: An instance of BaseModel (e.g. SingleExponentialIntervalModel).
        x_data: 1D array of timepoints or independent variables.
        norm: A dictionary mapping the first element of each row key to a 1D
          normalization array.
        norm_arr: A pre-computed 2D array of normalization values of shape
          (n_curves, n_timepoints). If provided, `norm` is ignored.
        key_level: The level name or index of the MultiIndex used to look
          up the norm key.
        loss_fn: Loss function to minimize, with signature loss_fn(log_y_pred,
          y_obs).
        solver: Optimistix minimizer (default: optx.BFGS(rtol=1e-9, ...)).
        init_params: 2D array of initial parameters of shape (n_curves,
          n_params). If None, model.initialize_params is called.
        chunk_size: Number of curves to optimize in a single JAX vmap batch.
        rmse_chunk_size: Number of curves to compute RMSE on in a single JAX
          vmap batch.
        solver_options: Dictionary of options for Optimistix solver (e.g.
          max_steps).
        init_kwargs: Extra keyword arguments for model.initialize_params.
        is_linear: If True, model predictions and loss are evaluated in linear
          space instead of log space.

    Returns:
        A DataFrame containing parameters, loss, RMSE, steps, AIC, and stats.
    """
    n_curves = len(df)
    if n_curves == 0:
        return pd.DataFrame()

    y_data = df.to_numpy(dtype=float)
    x_data = jnp.asarray(x_data)

    # 1. Resolve normalization array
    if norm_arr is not None:
        log_norm_arr = jnp.log(jnp.asarray(norm_arr))
    elif norm is not None:
        log_norm_arr = jnp.log(
            build_normalization_array(df.index, norm, key_level=key_level)
        )
    else:
        # Default: array of ones (zeros in log space)
        log_norm_arr = jnp.zeros_like(y_data)

    # 2. Setup Solver & Options
    if solver is None:
        solver = optx.BFGS(rtol=1e-9, atol=1e-9)

    solver_options = solver_options or {}
    max_steps = solver_options.get("max_steps", 10000)

    # 3. Setup single-curve loss function
    def single_curve_loss_fn(
        p: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        log_norm, y_obs = args
        log_y_pred = model.predict(p, log_norm, x_data)
        loss = loss_fn(log_y_pred, y_obs)
        return loss, loss

    # 4. Define and JIT compile optimization
    def optimize_single_curve(
        init_p: jnp.ndarray, log_norm: jnp.ndarray, y_obs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        res: Any = optx.minimise(
            single_curve_loss_fn,
            solver,
            init_p,
            args=(log_norm, y_obs),
            max_steps=max_steps,
            has_aux=True,
            throw=False,
        )
        return res.value, res.aux, res.stats.get("num_steps", -1)

    vmap_optimize = jax.jit(jax.vmap(optimize_single_curve, in_axes=(0, 0, 0)))

    # 5. Initialize parameters
    if init_params is not None:
        init_p_arr = np.asarray(init_params)
    else:
        init_kwargs = init_kwargs or {}
        init_p_arr = model.initialize_params(y_data, **init_kwargs)

    # 6. Run optimization loop in chunks
    n_params = len(model.param_names)
    fit_p = np.zeros((n_curves, n_params))
    fit_loss = np.zeros(n_curves)
    fit_steps = np.zeros(n_curves, dtype=int)

    log_norm_arr_np = np.asarray(log_norm_arr)

    print(
        f"Starting fit for {n_curves} curves in batches of {chunk_size}...",
        flush=True,
    )

    for start in range(0, n_curves, chunk_size):
        end = min(start + chunk_size, n_curves)

        chunk_init_p = init_p_arr[start:end]
        chunk_log_norms = log_norm_arr_np[start:end]
        chunk_y_obs = y_data[start:end]

        opt_x, opt_loss, opt_steps = vmap_optimize(
            chunk_init_p, chunk_log_norms, chunk_y_obs
        )

        fit_p[start:end] = opt_x
        fit_loss[start:end] = opt_loss
        fit_steps[start:end] = opt_steps

        print(
            f"    {end} / {n_curves} curves ({(end/n_curves)*100:.1f}%)",
            flush=True,
        )

    # 7. Compute RMSE
    def compute_single_rmse(
        p: jnp.ndarray, log_norm: jnp.ndarray, y_obs: jnp.ndarray
    ) -> jnp.ndarray:
        pred = model.predict(p, log_norm, x_data)
        y_pred = pred if is_linear else jnp.exp(pred)
        return jnp.sqrt(jnp.mean(jnp.square(y_obs - y_pred)))

    compute_rmse_batch = jax.jit(
        jax.vmap(compute_single_rmse, in_axes=(0, 0, 0))
    )

    fit_rmse = np.zeros(n_curves)
    fit_p_jnp = jnp.asarray(fit_p)

    for start in range(0, n_curves, rmse_chunk_size):
        end = min(start + rmse_chunk_size, n_curves)
        rmse_log_norms = log_norm_arr[start:end]

        chunk_rmse = compute_rmse_batch(
            fit_p_jnp[start:end], rmse_log_norms, y_data[start:end]
        )
        fit_rmse[start:end] = cast(np.ndarray, chunk_rmse)

    # 8. Calculate AIC
    # AIC = 2k + 2*loss
    aic_arr = 2 * n_params + 2 * fit_loss

    # 9. Build Results DataFrame
    results_dict = {}
    for i, name in enumerate(model.param_names):
        results_dict[name] = fit_p[:, i]

    results_dict["loss"] = fit_loss
    results_dict["RMSE"] = fit_rmse
    results_dict["steps"] = fit_steps
    results_dict["AIC"] = aic_arr

    res_df = pd.DataFrame(results_dict, index=df.index)

    # 10. Postprocess with model-specific columns
    if hasattr(model, "postprocess_results"):
        res_df = model.postprocess_results(res_df, y_data)

    return res_df


def fit_double_exponential_stepwise(
    df: pd.DataFrame,
    x_data: jnp.ndarray,
    model_style: str = "interval",
    concat_bound: bool = True,
    norm: Optional[Mapping[str, Any]] = None,
    norm_arr: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    key_level: Union[int, str] = 0,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = poisson_loss,
    solver: Optional[optx.AbstractMinimiser[Any, Any, Any]] = None,
    chunk_size: int = 50_000,
    rmse_chunk_size: int = 250_000,
    solver_options: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Perform a stepwise fit of single and double exponential models.

    Fits a single exponential model, and uses its parameters as a seed
    to fit the double exponential model.

    Returns a combined DataFrame containing columns for both fits.
    """
    if model_style == "interval":
        single_model: BaseModel = SingleExponentialIntervalModel(
            concat_bound=concat_bound
        )
        double_model: BaseModel = DoubleExponentialIntervalModel(
            concat_bound=concat_bound
        )
    elif model_style == "decay":
        single_model = SingleExponentialDecayModel()
        double_model = DoubleExponentialDecayModel()
    else:
        raise ValueError("model_style must be 'interval' or 'decay'")

    # Step 1: Fit single exponential model
    single_df = fit_curves(
        df=df,
        model=single_model,
        x_data=x_data,
        norm=norm,
        norm_arr=norm_arr,
        key_level=key_level,
        loss_fn=loss_fn,
        solver=solver,
        chunk_size=chunk_size,
        rmse_chunk_size=rmse_chunk_size,
        solver_options=solver_options,
    )

    # Step 2: Fit double exponential model (seeded from single exponential fit)
    double_df = fit_curves(
        df=df,
        model=double_model,
        x_data=x_data,
        norm=norm,
        norm_arr=norm_arr,
        key_level=key_level,
        loss_fn=loss_fn,
        solver=solver,
        chunk_size=chunk_size,
        rmse_chunk_size=rmse_chunk_size,
        solver_options=solver_options,
        init_kwargs={"single_exp_fit": single_df},
    )

    # Step 3: Combine results and suffix columns to avoid collision
    results_dict = {}

    # Rename single exp parameters and stats
    for name in single_model.param_names:
        results_dict[f"{name}_single_exp"] = single_df[name].to_numpy()
    results_dict["loss_single_exp"] = single_df["loss"].to_numpy()
    results_dict["RMSE_single_exp"] = single_df["RMSE"].to_numpy()
    results_dict["steps_single_exp"] = single_df["steps"].to_numpy()
    results_dict["AIC_single_exp"] = single_df["AIC"].to_numpy()

    # Rename double exp parameters and stats
    for name in double_model.param_names:
        results_dict[f"{name}_double_exp"] = double_df[name].to_numpy()
    results_dict["loss_double_exp"] = double_df["loss"].to_numpy()
    results_dict["RMSE_double_exp"] = double_df["RMSE"].to_numpy()
    results_dict["steps_double_exp"] = double_df["steps"].to_numpy()
    results_dict["AIC_double_exp"] = double_df["AIC"].to_numpy()

    # Keep any additional columns added by double model postprocess
    # (e.g. total_count, kinetic_count, bound_count)
    extra_cols = [
        c
        for c in double_df.columns
        if c not in double_model.param_names
        and c not in ["loss", "RMSE", "steps", "AIC"]
    ]
    for col in extra_cols:
        results_dict[col] = double_df[col].to_numpy()

    return pd.DataFrame(results_dict, index=df.index)
