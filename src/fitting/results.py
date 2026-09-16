"""The object returned by :func:`fitting.fit`.

A plain DataFrame cannot represent a fit with shared parameters honestly: there
are two index spaces, one per curve and one per group, and collapsing them
either duplicates per-group values or misreports them. ``FitResult``
keeps them separate, and offers :attr:`FitResult.table` for the flat,
one-row-per-curve view you would write to CSV.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd

from .core import NDArray, ParamLayout, build_layout
from .losses import Loss, Poisson
from .models import CurveModel


def _result_names() -> dict[int, str]:
    """Map Optimistix result codes to readable names.

    ``optx.RESULTS`` is an Equinox ``Enumeration``, whose members are not
    exposed through ``dir()``; the metaclass keeps them in ``_name_to_item``.
    """
    try:
        # Equinox Enumeration members are not exposed via dir().
        # pylint: disable=protected-access
        return {
            int(item._value): name
            for name, item in optx.RESULTS._name_to_item.items()
        }
    except AttributeError:  # pragma: no cover - defensive across versions
        return {0: "successful"}


#: Optimistix result codes, mapped back to readable names on the host.
_RESULT_NAMES = _result_names()


def result_name(code: int) -> str:
    """Readable name for an Optimistix result code."""
    return _RESULT_NAMES.get(int(code), f"code_{int(code)}")


# pylint cannot infer the type of an `eqx.field(static=True)` descriptor, so it
# believes ParamLayout's tuple fields are neither iterable nor subscriptable.
# pylint: disable=not-an-iterable,unsupported-membership-test


@dataclass(frozen=True)
class FitResult:  # pylint: disable=too-many-instance-attributes
    """Fitted parameters, diagnostics, and the data that produced them."""

    model: CurveModel
    loss: Loss
    layout: ParamLayout
    #: Per-curve free parameters, indexed like the input.
    local: pd.DataFrame
    #: Per-group shared parameters with standard errors. Empty if none.
    shared: pd.DataFrame
    #: Per-curve fit quality and solver status.
    diagnostics: pd.DataFrame
    #: The observations actually fitted, after column selection and masking.
    observed: pd.DataFrame
    #: Weight applied to each observation; 0 marks a masked point.
    mask: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)

    _x: Any = None
    _x_axis: Optional[int] = None
    _consts: Any = None
    _c_axes: Any = None

    # ---------------------------------------------------------------- views
    @property
    def table(self) -> pd.DataFrame:
        """Flat one-row-per-curve view: params, shared params, diagnostics."""
        parts = [self.local]
        if not self.shared.empty:
            parts.append(self._broadcast_shared())
        parts.append(self.diagnostics)
        return pd.concat(parts, axis=1)

    def _broadcast_shared(self) -> pd.DataFrame:
        """Repeat each shared value across the curves that share it.

        Vector-valued shared parameters are left out: a vector per curve does
        not belong in a flat table. Read them from :attr:`shared` instead.
        """
        out = {}
        for name in self.layout.shared:
            if self.layout.width(name) > 1:
                continue
            gi = np.asarray(self.layout.group_index[name])
            vals = self.shared.loc[
                self.shared["parameter"] == name, "value"
            ].to_numpy()
            out[name] = vals[gi]
        return pd.DataFrame(out, index=self.local.index)

    def params(self) -> Any:
        """The full ``Params`` pytree, per curve, re-passable as ``init=``."""
        values: dict[str, Any] = {}
        n = len(self.local)
        for name in self.layout.names:
            if name in self.layout.local:
                values[name] = jnp.asarray(self.local[name].to_numpy())
            elif name in self.layout.shared:
                gi = np.asarray(self.layout.group_index[name])
                width = self.layout.width(name)
                vals = self.shared.loc[
                    self.shared["parameter"] == name, "value"
                ].to_numpy()
                if width > 1:
                    vals = vals.reshape(-1, width)
                values[name] = jnp.asarray(vals[gi])
            else:
                values[name] = jnp.broadcast_to(
                    self.layout.fixed_values[name], (n,)
                )
        return self.model.Params(**values)

    @property
    def predicted(self) -> pd.DataFrame:
        """Predictions in observation space, aligned to :attr:`observed`."""
        p = self.params()
        model, consts = self.model, self._consts
        log_space, x, x_axis = (
            self.model.log_predictions,
            self._x,
            self._x_axis,
        )

        def one(pi: Any, x_i: Any, c_i: Any) -> jnp.ndarray:
            pred = model.predict(pi, x_i, c_i)
            return jnp.exp(pred) if log_space else pred

        out = jax.vmap(one, in_axes=(0, x_axis, self._c_axes))(p, x, consts)
        return pd.DataFrame(
            np.asarray(out),
            index=self.observed.index,
            columns=self.observed.columns,
        )

    @property
    def residuals(self) -> pd.DataFrame:
        """``observed - predicted``, with masked points set to NaN."""
        r = self.observed - self.predicted
        return r.where(self.mask.to_numpy() != 0)

    # ------------------------------------------------------------ builders
    @classmethod
    def empty(cls, model: CurveModel, index: pd.Index) -> "FitResult":
        """An empty result, for an empty input."""
        e = pd.DataFrame(index=index)
        return cls(
            model=model,
            loss=Poisson(),
            layout=build_layout(model.Params, 0),
            local=e,
            shared=pd.DataFrame(),
            diagnostics=e,
            observed=e,
            mask=e,
        )

    @classmethod
    def build(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        *,
        model: CurveModel,
        loss: Loss,
        layout: ParamLayout,
        index: pd.Index,
        local: NDArray,
        shared: Optional[Mapping[str, NDArray]],
        codes: NDArray,
        steps: NDArray,
        x: Any,
        x_axis: Optional[int],
        consts: Any,
        c_axes: Any,
        y: NDArray,
        w: NDArray,
        curve_loss: Any,
        columns: Optional[pd.Index] = None,
        progress: bool = True,
        joint_result: Optional[int] = None,
        joint_steps: Optional[int] = None,
    ) -> "FitResult":
        """Assemble a result from raw fitted arrays."""
        n_curves, n_points = y.shape
        local_df = pd.DataFrame(
            {nm: local[:, i] for i, nm in enumerate(layout.local)}, index=index
        )
        _add_linear_params(local_df)

        shared_df = _build_shared_frame(
            loss,
            layout,
            shared,
            local,
            x,
            x_axis,
            consts,
            c_axes,
            y,
            w,
            curve_loss,
        )

        gathered = (
            {
                nm: jnp.asarray(
                    np.asarray(shared[nm])[np.asarray(layout.group_index[nm])]
                )
                for nm in layout.shared
            }
            if shared
            else {}
        )

        per_curve_loss = jax.jit(
            jax.vmap(
                curve_loss,
                in_axes=(
                    0,
                    (x_axis, c_axes, 0, 0, {nm: 0 for nm in layout.shared}),
                ),
            )
        )
        losses = np.asarray(
            per_curve_loss(
                jnp.asarray(local),
                (x, consts, jnp.asarray(y), jnp.asarray(w), gathered),
            )
        )

        pred = _predict_all(
            model, layout, local, shared, x, x_axis, consts, c_axes, n_curves
        )
        # A diverged solver can leave predictions at 1e300; the resulting
        # overflow is expected and reported through `converged`, so do not let
        # numpy shout about it.
        with np.errstate(over="ignore", invalid="ignore"):
            sq = np.square(y - pred) * w
        n_used = np.maximum(w.sum(axis=1), 1.0)
        rmse = np.sqrt(sq.sum(axis=1) / n_used)

        k_local = layout.n_local
        diag = {
            "loss": losses,
            "rmse": rmse,
            "steps": steps,
            "converged": codes == 0,
            "status": pd.Categorical([result_name(int(cd)) for cd in codes]),
            "k_local": np.full(n_curves, k_local),
        }
        if loss.is_nll:
            diag["aic"] = 2 * k_local + 2 * losses
        diag.update(model.extra_columns(y))
        diag_df = pd.DataFrame(diag, index=index)

        cols = pd.RangeIndex(n_points) if columns is None else columns
        info: dict[str, Any] = {
            "n_curves": n_curves,
            "n_local_per_curve": k_local,
            "n_shared": layout.n_shared_total,
            "loss_is_nll": loss.is_nll,
        }
        if joint_result is not None:
            info["joint_converged"] = joint_result == 0
            info["joint_status"] = _RESULT_NAMES.get(
                joint_result, str(joint_result)
            )
            info["joint_steps"] = joint_steps
        if loss.is_nll:
            info["aic_total"] = float(
                2 * (n_curves * k_local + info["n_shared"]) + 2 * losses.sum()
            )

        n_bad = int((~diag_df["converged"]).sum())
        if n_bad and progress:
            if joint_result is not None:
                # In the joint path every curve carries the one solve's status,
                # so a per-curve count would misattribute a single failure.
                hint = (
                    " -- raise joint_max_steps"
                    if info.get("joint_status")
                    == "nonlinear_max_steps_reached"
                    else ""
                )
                print(
                    f"warning: the joint solve did not converge "
                    f"({info['joint_status']}){hint}",
                    flush=True,
                )
            else:
                print(
                    f"warning: {n_bad} / {n_curves} curves did not converge "
                    f"({n_bad / n_curves:.1%}); see "
                    f"FitResult.diagnostics.status",
                    flush=True,
                )

        return cls(
            model=model,
            loss=loss,
            layout=layout,
            local=local_df,
            shared=shared_df,
            diagnostics=diag_df,
            observed=pd.DataFrame(y, index=index, columns=cols),
            mask=pd.DataFrame(w, index=index, columns=cols),
            info=info,
            _x=x,
            _x_axis=x_axis,
            _consts=consts,
            _c_axes=c_axes,
        )


def _add_linear_params(df: pd.DataFrame) -> None:
    """Add ``kd`` alongside ``log_kd``, by naming convention.

    A runaway log-parameter exponentiates to ``inf``, which is the honest
    answer; numpy's overflow warning is suppressed so library code does not
    emit warnings for a value the caller can see for themselves.
    """
    for col in [c for c in df.columns if c.startswith("log_")]:
        linear = col[len("log_") :]
        if linear and linear not in df.columns:
            with np.errstate(over="ignore"):
                df[linear] = np.exp(df[col].to_numpy())


def _predict_all(
    model: CurveModel,
    layout: ParamLayout,
    local: NDArray,
    shared: Optional[Mapping[str, NDArray]],
    x: Any,
    x_axis: Optional[int],
    consts: Any,
    c_axes: Any,
    n_curves: int,
) -> NDArray:
    """Predictions in observation space for every curve."""
    values: dict[str, Any] = {}
    for i, nm in enumerate(layout.local):
        values[nm] = jnp.asarray(local[:, i])
    for nm in layout.shared:
        gi = np.asarray(layout.group_index[nm])
        values[nm] = jnp.asarray(np.asarray(shared[nm])[gi])  # type: ignore[index]
    for nm in layout.fixed_names:
        values[nm] = jnp.broadcast_to(layout.fixed_values[nm], (n_curves,))
    p = model.Params(**values)

    def one(pi: Any, x_i: Any, c_i: Any) -> jnp.ndarray:
        pred = model.predict(pi, x_i, c_i)
        return jnp.exp(pred) if model.log_predictions else pred

    return np.asarray(
        jax.jit(jax.vmap(one, in_axes=(0, x_axis, c_axes)))(p, x, consts)
    )


def _build_shared_frame(
    loss: Loss,
    layout: ParamLayout,
    shared: Optional[Mapping[str, NDArray]],
    local: NDArray,
    x: Any,
    x_axis: Optional[int],
    consts: Any,
    c_axes: Any,
    y: NDArray,
    w: NDArray,
    curve_loss: Any,
) -> pd.DataFrame:
    """Shared parameters with a curvature-based standard error.

    The error is the curvature of the total loss in the shared parameters with
    the local parameters held at their fitted values. It therefore **ignores
    coupling with the local parameters and is a lower bound** on the true
    standard error -- but it is what catches the case that matters in practice:
    a shared parameter sitting at a genuine but almost perfectly flat optimum,
    which must not be reported as a precise measurement.
    """
    if not shared:
        return pd.DataFrame(
            columns=["parameter", "group", "component", "value", "se"]
        )

    names = list(layout.shared)
    sizes = [layout.group_sizes[nm] for nm in names]
    flat = jnp.asarray(
        np.concatenate([np.asarray(shared[nm]).ravel() for nm in names])
    )

    per_curve = jax.vmap(
        curve_loss,
        in_axes=(0, (x_axis, c_axes, 0, 0, {nm: 0 for nm in names})),
    )

    def total(theta_shared: jnp.ndarray) -> jnp.ndarray:
        g, off = {}, 0
        for nm, k in zip(names, sizes):
            width = layout.width(nm)
            block = jax.lax.dynamic_slice(theta_shared, (off,), (k * width,))
            if width > 1:
                block = block.reshape(k, width)
            g[nm] = block[layout.group_index[nm]]
            off += k * width
        return jnp.sum(
            per_curve(
                jnp.asarray(local),
                (x, consts, jnp.asarray(y), jnp.asarray(w), g),
            )
        )

    se: NDArray = np.full(int(sum(sizes)), np.nan)
    try:
        hess = np.asarray(jax.hessian(total)(flat))
        cov = np.linalg.inv(hess)
        if loss.is_nll:
            # Hessian of a negative log-likelihood is the observed information.
            pass
        else:
            # Least squares: H ~ 2 J'J, so cov = sigma^2 (J'J)^-1
            # = 2 sigma^2 H^-1, with sigma^2 estimated from the residual
            # sum of squares per degree of freedom.
            n_used = float(np.sum(w))
            n_free = float(local.size + sum(sizes))
            dof = max(n_used - n_free, 1.0)
            sigma2 = float(total(flat)) / dof
            cov = 2.0 * sigma2 * cov
        d = np.diag(cov)
        # Guard inside the sqrt: a non-positive-definite Hessian means
        # the parameter is not locally identified, and the error is
        # undefined rather than zero.
        se = np.sqrt(np.where(d > 0, d, np.nan))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    rows = []
    i = 0
    for nm, k in zip(names, sizes):
        width = layout.width(nm)
        block = np.asarray(shared[nm]).reshape(k, width)
        for g in range(k):
            for component in range(width):
                rows.append(
                    {
                        "parameter": nm,
                        "group": g,
                        "component": component,
                        "value": float(block[g, component]),
                        "se": float(se[i]),
                    }
                )
                i += 1
    out = pd.DataFrame(rows)
    _add_linear_shared(out)
    return out


def _add_linear_shared(df: pd.DataFrame) -> None:
    """Linear-space value for ``log_``-prefixed shared parameters."""
    if df.empty:
        return
    with np.errstate(over="ignore"):
        df["linear_value"] = [
            np.exp(v) if str(p).startswith("log_") else np.nan
            for p, v in zip(df["parameter"], df["value"])
        ]
