# Fitting: Generalized and Accelerated Curve Fitting with JAX & Optimistix

`fitting` is a high-performance Python library designed for fitting thousands of mathematical curves concurrently using JAX-accelerated optimization and Optimistix solvers.

## Features

- **Massive Parallelism**: Vectorized optimization using `jax.vmap` and compilation using `jax.jit`.
- **Flexible Models**: Modular, class-based API to support single/double exponential dissociation, standard decay models, and custom models.
- **Customizable Losses**: Built-in support for Poisson Negative Log-Likelihood, Mean Squared Error (MSE), and Huber losses, with options for custom losses.
- **Modern Optimization**: Uses `optimistix` solvers (such as BFGS) for efficient, robust minimization.

## Installation

You can install the package locally in editable mode:

```bash
pip install -e .
```

## Quick Start

### 1. Fit Interval-Based Dissociation Curves

Here is a quick example reproducing typical koff fitting workflows:

```python
import jax.numpy as jnp
import pandas as pd
from fitting import fit_curves, SingleExponentialIntervalModel

# Load data (rows are curves, columns are timepoints)
df = pd.read_csv("data.csv", index_col=["replicate", "variant"])
times = jnp.arange(2.0, 24.1, 2.0)

# Instantiate the model
model = SingleExponentialIntervalModel(concat_bound=True)

# Run the fit
fit_results = fit_curves(
    df=df,
    model=model,
    x_data=times,
)

# Output contains parameters like log_y0, log_k_off, loss, RMSE, steps, etc.
print(fit_results.head())
```

### 2. Custom Models

You can easily define your own models by inheriting from `BaseModel`:

```python
import jax.numpy as jnp
import numpy as np
from fitting import BaseModel, fit_curves

class CustomLinearModel(BaseModel):
    @property
    def param_names(self):
        return ["slope", "intercept"]

    def predict(self, p, log_norm, times):
        # p[0] = slope, p[1] = intercept
        # Return log predicted values
        return log_norm + p[0] * times + p[1]

    def initialize_params(self, y_obs):
        # Return initial guess for each curve (shape: n_curves, n_params)
        n_curves = len(y_obs)
        return np.zeros((n_curves, 2))
```

### 3. Stepwise Double Exponential Fitting

You can perform stepwise double exponential fitting where a single exponential is fit first and its results are used to seed the double exponential fit:

```python
import jax.numpy as jnp
import pandas as pd
from fitting import fit_double_exponential_stepwise

# Load data and define timepoints
df = pd.read_csv("data.csv", index_col=["replicate", "variant"])
times = jnp.arange(2.0, 24.1, 2.0)

# Run stepwise fit (returns combined single & double exponential results)
combined_df = fit_double_exponential_stepwise(
    df=df,
    x_data=times,
    model_style="interval",  # or "decay"
)
```
