# fitting

Fit large numbers of curves at once with JAX and
[Optimistix](https://github.com/patrick-kidger/optimistix), optionally sharing
parameters between them.

- **Scales.** With no shared parameters every curve is an independent problem, so
  the fit is a chunked `vmap` whose peak memory does not grow with the number of
  curves.
- **Shares parameters.** One `Rmax` across every binding curve for a protein, or
  one per replicate, without hand-writing a joint objective or a Jacobian.
- **General.** Models declare their own parameters and constants, so the same
  package serves sequencing counts, fluorescence isotherms, and anything else.
- **Autodiff.** Gradients come from JAX; you never write a derivative.

## Install

```bash
pip install -e .
```

## Fit something

```python
import jax.numpy as jnp
import pandas as pd
import fitting as ft

counts = pd.read_csv("counts.csv", index_col=["replicate", "variant"])
x = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)

res = ft.fit(counts, model=ft.SingleExponentialInterval(), x=x)

res.table                        # one row per curve, ready for to_csv
res.diagnostics.converged.all()  # non-convergence is reported, not assumed away
```

Share a parameter across the curves that legitimately share one:

```python
res = ft.fit(df, model=ft.Langmuir(), x=conc, loss=ft.SSE(),
             share={"log_ymax": ft.by_level("replicate")})

res.shared    # one row per group, with a standard error
```

## Documentation

```bash
pip install -e '.[docs]'
make -C docs html          # or: tox -e docs
```

then open `docs/_build/html/index.html`. The guide covers getting started,
sharing parameters, writing models and objectives, and the sequencing
conventions; the API reference is generated from the docstrings.

## Examples

- `examples/fit_koff_example.py` — sequencing counts, Poisson likelihood,
  spike-in normalisation, millions of independent curves.
- `examples/fit_langmuir_example.py` — MITOMI fluorescence isotherms, least
  squares, ragged curves, and a shared saturation level.

## Development

```bash
tox                 # tests, mypy, pylint, black, isort
pytest -m production  # tests needing the real STAMMP-seq dataset
```

On the first fit in a process, JAX's CPU backend may print `Empty bitcode string
provided for eigen`. It is harmless — it comes from `jaxlib`, fires once, and
does not affect results. Silence it with `export TF_CPP_MIN_LOG_LEVEL=2`.
