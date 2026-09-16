"""Fitting entry points.

* :func:`fit` fits one model to every row of a table, optionally sharing
  parameters between rows.
* :func:`fit_stepwise` fits a sequence of models, seeding each from the
  previous one.
* :func:`fit_sequencing` applies the conventions of sequencing count data to
  :func:`fit`.

:func:`fit` takes one of two strategies. With no shared parameter the objective
separates, so every curve is solved on its own in chunked ``vmap`` batches and
peak memory is set by ``chunk_size``. A shared parameter enters every term, so
one optimiser runs over the whole parameter vector instead; that cannot be
chunked. See the Sharing parameters guide.
"""

from typing import Any, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd

from .core import (
    FittingError,
    NDArray,
    ParamLayout,
    assemble_params,
    build_layout,
    field_defaults,
    field_names,
    infer_widths,
)
from .losses import LeastSquares, Loss, Poisson
from .models import CurveModel, SingleExponentialInterval
from .results import FitResult, result_name
from .selectors import (
    ByLevel,
    by_level,
    resolve_curve_const,
    resolve_group_index,
)

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

#: Largest dense Jacobian, in bytes, that a joint least-squares solve may
#: allocate. Gauss-Newton and Levenberg-Marquardt materialise the full
#: ``n_residuals x n_parameters`` array, and with one parameter block per curve
#: that grows quadratically in the number of curves.
MAX_JOINT_JACOBIAN_BYTES = 2 * 1024**3


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def _result_code(sol: Any) -> Any:
    """Extract Optimistix's result enum as a plain integer array.

    Never return a whole ``Solution`` from a traced function: several solvers
    (``LBFGS``, ``NonlinearCG``, ``LevenbergMarquardt``, ``GaussNewton``) carry
    a ``Jaxpr`` in it, which is not a valid JAX type and fails under ``vmap``
    and ``jit``.
    """
    return sol.result._value  # pylint: disable=protected-access


def _prepare_observations(
    data: Union[pd.DataFrame, NDArray],
    y_cols: Optional[Union[Sequence[Any], slice]],
    weights: Optional[Any],
    ignore_nan: bool,
) -> tuple[NDArray, NDArray, pd.Index, pd.Index]:
    """Extract the observation matrix, its weights, and its labels.

    Args:
        data: one curve per row. A DataFrame's index identifies the curves.
        y_cols: which columns hold observations, so that metadata columns may
            stay on the frame.
        weights: optional per-point weights, broadcast against the
            observations.
        ignore_nan: give non-finite observations zero weight. Their values are
            also replaced, because a masked ``NaN`` would still propagate
            through the loss: ``0 * NaN`` is ``NaN``.

    Returns:
        ``(y, w, index, columns)``, where ``y`` and ``w`` have shape
        ``(n_curves, n_points)``.
    """
    if isinstance(data, pd.DataFrame):
        frame = (
            data
            if y_cols is None
            else (
                data.iloc[:, y_cols]
                if isinstance(y_cols, slice)
                else data[y_cols]
            )
        )
        y = frame.to_numpy(dtype=float)
        index: pd.Index = data.index
        columns: pd.Index = frame.columns
    else:
        y = np.asarray(data, dtype=float)
        index = pd.RangeIndex(len(y))
        columns = pd.RangeIndex(y.shape[1] if y.ndim > 1 else 0)

    w = np.ones_like(y)
    if ignore_nan:
        bad = ~np.isfinite(y)
        if bad.any():
            w[bad] = 0.0
            y = np.where(bad, 0.0, y)
    if weights is not None:
        w = w * np.asarray(weights, dtype=float)
    return y, w, index, columns


def _prepare_consts(
    model: CurveModel,
    n_curves: int,
    consts: Optional[Mapping[str, Any]],
    curve_consts: Optional[Mapping[str, Any]],
    index: pd.Index,
) -> tuple[Any, Any]:
    """Build the model's ``Consts`` alongside a matching tree of ``vmap`` axes.

    Args:
        model: the model whose ``Consts`` container is being filled.
        n_curves: expected leading dimension of any per-curve constant.
        consts: constants shared by every curve.
        curve_consts: constants with one row per curve. A
            :func:`~fitting.selectors.by_level` selector is expanded by
            gathering, so only one value per group is stored.
        index: curve labels, used to resolve selectors.

    Returns:
        ``(values, axes)`` -- a ``Consts`` instance, and a parallel ``Consts``
        holding ``0`` for per-curve fields and ``None`` for shared ones.
    """
    consts = dict(consts or {})
    curve_consts = dict(curve_consts or {})
    fields = field_names(model.Consts)
    defaults = field_defaults(model.Consts)

    unknown = (set(consts) | set(curve_consts)) - set(fields)
    if unknown:
        raise FittingError(
            f"consts/curve_consts refer to {sorted(unknown)}, which are not "
            f"constants of {type(model).__name__} "
            f"({model.Consts.__name__}: {list(fields)})"
        )
    both = set(consts) & set(curve_consts)
    if both:
        raise FittingError(
            f"{sorted(both)} given in both consts= (broadcast) and "
            f"curve_consts= (per-curve); pick one"
        )

    values: dict[str, Any] = {}
    axes: dict[str, Any] = {}
    for name in fields:
        if name in curve_consts:
            arr = resolve_curve_const(
                curve_consts[name], index, n_curves, name
            )
            if arr.shape[0] != n_curves:
                raise FittingError(
                    f"curve_consts['{name}'] has leading dimension "
                    f"{arr.shape[0]}, expected {n_curves} (one row per curve)"
                )
            values[name], axes[name] = arr, 0
        elif name in consts:
            values[name] = jnp.asarray(consts[name], dtype=jnp.float64)
            axes[name] = None
        elif name in defaults:
            values[name], axes[name] = defaults[name], None
        else:
            raise FittingError(
                f"constant '{name}' of {model.Consts.__name__} has no default"
                f" and was not supplied in consts= or curve_consts="
            )
    return model.Consts(**values), model.Consts(**axes)


def _prepare_x(
    x: Any, n_curves: int, w: NDArray
) -> tuple[jnp.ndarray, Optional[int]]:
    """Validate the independent variable and decide how to map it over curves.

    A 1-D ``x`` is shared by every curve; a 2-D ``x`` supplies one row each.
    Ragged curves are padded to a common width, and that padding is usually
    ``NaN`` in ``x`` as well as in the observations. Since ``0 * NaN`` is
    ``NaN``, a non-finite ``x`` would reach the residual even at a zero-weight
    point, so padding is replaced with a finite value wherever the observation
    is masked. A non-finite ``x`` at an unmasked point is an error.

    Args:
        x: the independent variable, 1-D or 2-D.
        n_curves: expected leading dimension of a 2-D ``x``.
        w: observation weights, used to tell padding from real points.

    Returns:
        ``(x, axis)`` -- the sanitised array, and the ``vmap`` axis for it
        (``None`` when shared, ``0`` when per-curve).
    """
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.ndim == 1:
        if not bool(jnp.all(jnp.isfinite(arr))):
            raise FittingError(
                "1-D x contains non-finite values. A shared x applies to "
                "every curve, so no mask could excuse them; pass a 2-D "
                "per-curve x if curves are sampled differently."
            )
        return arr, None
    if arr.ndim == 2:
        if arr.shape[0] != n_curves:
            raise FittingError(
                f"2-D x has leading dimension {arr.shape[0]}, expected "
                f"{n_curves} (one row per curve)"
            )
        bad = ~jnp.isfinite(arr)
        if bool(jnp.any(bad)) and arr.shape == w.shape:
            live = int(jnp.sum(bad & (jnp.asarray(w) != 0.0)))
            if live:
                raise FittingError(
                    f"x has {live} non-finite value(s) at points whose "
                    f"observation is not masked, so they would corrupt the "
                    f"fit. Either mask those observations (NaN with "
                    f"ignore_nan=True, or zero weight), or supply a finite x."
                )
        return jnp.where(bad, 0.0, arr), 0
    raise FittingError(
        f"x must be 1-D (shared) or 2-D (per-curve), got {arr.ndim}-D"
    )


def _init_theta(
    model: CurveModel,
    params_cls: type[Any],
    init: Optional[Any],
    y: NDArray,
    x: jnp.ndarray,
    consts: Any,
) -> Any:
    """Per-curve starting values for every parameter, as ``Params``."""
    guess = model.init(jnp.asarray(y), x, consts) if init is None else init
    missing = [
        n for n in field_names(params_cls) if getattr(guess, n, None) is None
    ]
    if missing:
        raise FittingError(f"init is missing values for {missing}")
    return guess


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def _make_curve_loss(
    model: CurveModel, loss: Loss, layout: ParamLayout
) -> Any:
    """Build the scalar loss for one curve, given its local params."""
    params_cls = model.Params

    def curve_loss(theta: jnp.ndarray, args: Any) -> jnp.ndarray:
        x_i, c_i, y_i, w_i, shared_i = args
        p = assemble_params(params_cls, layout, theta, shared_i)
        return loss.value(model.predict(p, x_i, c_i), y_i, w_i)

    return curve_loss


def _make_curve_residual(
    model: CurveModel, loss: LeastSquares, layout: ParamLayout
) -> Any:
    """Build the residual vector for one curve, for a least-squares solver.

    Defined only when the loss has a residual form; :func:`_check_solver`
    verifies that before a least-squares solver is selected.
    """
    params_cls = model.Params

    def curve_residual(theta: jnp.ndarray, args: Any) -> jnp.ndarray:
        x_i, c_i, y_i, w_i, shared_i = args
        p = assemble_params(params_cls, layout, theta, shared_i)
        res: jnp.ndarray = loss.residual(model.predict(p, x_i, c_i), y_i, w_i)
        return res

    return curve_residual


def _check_solver(solver: Any, loss: Loss) -> bool:
    """Report whether ``solver`` is least-squares, validating the loss.

    Levenberg-Marquardt and Gauss-Newton consume the residual vector rather
    than a scalar, so they require a sum-of-squares loss.
    """
    if not isinstance(solver, optx.AbstractLeastSquaresSolver):
        return False
    if not isinstance(loss, LeastSquares):
        raise FittingError(
            f"{type(solver).__name__} is a least-squares solver and needs the"
            f" residual vector, but this loss has no sum-of-squares form."
            f" Either use a least-squares loss (fitting.SSE), or a general"
            f" minimiser such as optx.BFGS."
        )
    return True


def _check_joint_jacobian(
    n_curves: int, n_points: int, layout: ParamLayout
) -> None:
    """Refuse a joint least-squares solve whose dense Jacobian is too large."""
    n_params = n_curves * layout.n_local + layout.n_shared_total
    n_residuals = n_curves * n_points
    nbytes = n_residuals * n_params * 8
    if nbytes > MAX_JOINT_JACOBIAN_BYTES:
        raise FittingError(
            f"a joint least-squares solve over {n_curves} curves would need a"
            f" dense {n_residuals} x {n_params} Jacobian"
            f" ({nbytes / 1024**3:.1f} GiB). Levenberg-Marquardt and"
            f" Gauss-Newton materialise it in full, and it grows"
            f" quadratically in the number of curves. Use a general minimiser"
            f" for the joint path (optx.LBFGS, the default), which needs"
            f" memory linear in the parameter count."
        )


# ---------------------------------------------------------------------------
# General fitting
# ---------------------------------------------------------------------------
def fit(
    data: Union[pd.DataFrame, NDArray],
    model: CurveModel,
    x: Any,
    *,
    y_cols: Optional[Union[Sequence[Any], slice]] = None,
    weights: Optional[Any] = None,
    ignore_nan: bool = True,
    consts: Optional[Mapping[str, Any]] = None,
    curve_consts: Optional[Mapping[str, Any]] = None,
    share: Optional[Mapping[str, Any]] = None,
    fixed: Optional[Mapping[str, Any]] = None,
    init: Optional[Any] = None,
    loss: Optional[Loss] = None,
    solver: Optional[Any] = None,
    max_steps: int = 10_000,
    joint_solver: Optional[Any] = None,
    joint_max_steps: int = 2_000,
    chunk_size: int = 50_000,
    progress: bool = True,
) -> FitResult:
    """Fit ``model`` to every row of ``data``.

    Args:
        data: one curve per row. A DataFrame's index identifies the curves.
        model: a :class:`~fitting.models.CurveModel`.
        x: the independent variable. 1-D is shared by every curve; 2-D gives
            each curve its own row.
        y_cols: which columns hold observations, so metadata columns may stay
            on the frame.
        weights: optional per-point weights, broadcast against the
            observations.
        ignore_nan: give non-finite observations zero weight rather than
            letting them propagate into the loss and its gradient.
        consts: model constants shared by every curve.
        curve_consts: model constants with one row per curve. May be a
            :func:`~fitting.selectors.by_level` selector, which holds one value
            per group and gathers instead of materialising a dense array.
        share: maps a parameter name to ``None`` for a single global value, or
            to a selector for one value per group. A non-empty ``share``
            selects the joint solve.
        fixed: maps a parameter name to a known value, pinning it.
        init: optional ``model.Params`` of per-curve starting values. Defaults
            to ``model.init``.
        loss: a :class:`~fitting.losses.Loss`. Defaults to
            :class:`~fitting.losses.Poisson`. Its ``log_predictions`` must
            agree with the model's.
        solver: per-curve solver for independent curves. Defaults to
            ``optx.BFGS``. A least-squares solver such as
            ``optx.LevenbergMarquardt`` switches to ``optx.least_squares`` on
            the loss's residual form, and so requires a sum-of-squares loss.
        max_steps: per-curve step limit for independent curves.
        joint_solver: solver for the joint solve. Defaults to ``optx.LBFGS``. A
            least-squares solver is accepted for modest numbers of curves but
            refused once its dense Jacobian would exceed
            :data:`MAX_JOINT_JACOBIAN_BYTES`.
        joint_max_steps: step limit for the joint solve.
        chunk_size: curves per ``vmap`` batch for independent curves. Sets peak
            memory.
        progress: print progress and a warning for curves that fail to
            converge.

    Returns:
        A :class:`~fitting.results.FitResult` holding the fitted parameters,
        per-curve diagnostics, and the observations they were fitted to.

    Raises:
        FittingError: if the inputs are inconsistent -- a model and objective
            that disagree on log space, an unknown parameter or constant name,
            a mis-shaped per-curve argument, a solver incompatible with the
            objective, or observations the objective cannot accept.
    """
    loss = Poisson() if loss is None else loss
    loss.check_model(model)
    y, w, index, columns = _prepare_observations(
        data, y_cols, weights, ignore_nan
    )
    n_curves = len(y)
    if n_curves == 0:
        return FitResult.empty(model, index)

    if loss.requires_positive:
        bad = int(np.sum((y <= 0.0) & (w != 0.0)))
        if bad:
            raise FittingError(
                f"{bad} unmasked observation(s) are <= 0, but this loss takes"
                f" the logarithm of the observations. Mask them (NaN with"
                f" ignore_nan=True, or zero weight), or use a loss that works"
                f" in linear space such as fitting.SSE."
            )

    x_arr, x_axis = _prepare_x(x, n_curves, w)
    c, c_axes = _prepare_consts(model, n_curves, consts, curve_consts, index)

    share_idx = {
        k: resolve_group_index(v, index, n_curves, k)
        for k, v in (share or {}).items()
    }
    # The starting values are needed before the layout, since a shared
    # parameter's width is read from the shape of its initial value.
    guess = _init_theta(model, model.Params, init, y, x_arr, c)
    shared_names = tuple(
        n for n in field_names(model.Params) if n in share_idx
    )
    layout = build_layout(
        model.Params,
        n_curves,
        share_idx,
        fixed,
        infer_widths(guess, shared_names, n_curves),
    )
    model.check_layout(layout)

    curve_loss = _make_curve_loss(model, loss, layout)
    y_j, w_j = jnp.asarray(y), jnp.asarray(w)

    common: dict[str, Any] = {
        "model": model,
        "loss": loss,
        "layout": layout,
        "guess": guess,
        "curve_loss": curve_loss,
        "x_arr": x_arr,
        "x_axis": x_axis,
        "c": c,
        "c_axes": c_axes,
        "y_j": y_j,
        "w_j": w_j,
        "y_np": y,
        "index": index,
        "columns": columns,
        "progress": progress,
    }
    if layout.has_shared:
        return _fit_joint(
            joint_solver=joint_solver,
            joint_max_steps=joint_max_steps,
            **common,
        )
    return _fit_local(
        solver=solver, max_steps=max_steps, chunk_size=chunk_size, **common
    )


def _slice_consts(c: Any, c_axes: Any, sl: slice) -> Any:
    """Slice only those constants that carry a per-curve leading axis."""
    return type(c)(
        **{
            name: (
                getattr(c, name)[sl]
                if getattr(c_axes, name) == 0
                else getattr(c, name)
            )
            for name in field_names(type(c))
        }
    )


def _fit_local(
    *,
    model: CurveModel,
    loss: Loss,
    layout: ParamLayout,
    guess: Any,
    curve_loss: Any,
    x_arr: jnp.ndarray,
    x_axis: Optional[int],
    c: Any,
    c_axes: Any,
    y_j: jnp.ndarray,
    w_j: jnp.ndarray,
    y_np: NDArray,
    index: pd.Index,
    columns: pd.Index,
    solver: Optional[Any],
    max_steps: int,
    chunk_size: int,
    progress: bool,
) -> FitResult:
    """Solve every curve independently, in vectorised chunks."""
    if solver is None:
        solver = optx.BFGS(rtol=1e-9, atol=1e-9)
    n_curves = len(y_np)
    theta0 = jnp.stack(
        [jnp.asarray(getattr(guess, n)).reshape(-1) for n in layout.local],
        axis=1,
    )
    use_lsq = _check_solver(solver, loss)
    curve_residual: Any = None
    if use_lsq:
        assert isinstance(loss, LeastSquares)
        curve_residual = _make_curve_residual(model, loss, layout)

    def one(theta: jnp.ndarray, x_i: Any, c_i: Any, y_i: Any, w_i: Any) -> Any:
        args: tuple[Any, Any, Any, Any, dict[str, Any]] = (
            x_i,
            c_i,
            y_i,
            w_i,
            {},
        )
        if use_lsq:
            sol: Any = optx.least_squares(
                curve_residual,
                solver,
                theta,
                args=args,
                max_steps=max_steps,
                throw=False,
            )
        else:
            sol = optx.minimise(
                curve_loss,
                solver,
                theta,
                args=args,
                max_steps=max_steps,
                throw=False,
            )
        return sol.value, _result_code(sol), sol.stats["num_steps"]

    batch = jax.jit(jax.vmap(one, in_axes=(0, x_axis, c_axes, 0, 0)))

    fit_local = np.zeros((n_curves, layout.n_local))
    codes = np.zeros(n_curves, dtype=int)
    steps = np.zeros(n_curves, dtype=int)

    if progress:
        print(
            f"Fitting {n_curves} curves in chunks of {chunk_size}...",
            flush=True,
        )
    for start in range(0, n_curves, chunk_size):
        end = min(start + chunk_size, n_curves)
        sl = slice(start, end)
        value, code, step = batch(
            theta0[sl],
            x_arr[sl] if x_axis == 0 else x_arr,
            _slice_consts(c, c_axes, sl),
            y_j[sl],
            w_j[sl],
        )
        fit_local[sl] = np.asarray(value)
        codes[sl] = np.asarray(code)
        steps[sl] = np.asarray(step)
        if progress:
            print(f"    {end} / {n_curves} ({end / n_curves:.1%})", flush=True)

    return FitResult.build(
        model=model,
        loss=loss,
        layout=layout,
        index=index,
        local=fit_local,
        shared=None,
        codes=codes,
        steps=steps,
        x=x_arr,
        x_axis=x_axis,
        consts=c,
        c_axes=c_axes,
        y=y_np,
        w=np.asarray(w_j),
        curve_loss=curve_loss,
        columns=columns,
        progress=progress,
    )


def _fit_joint(
    *,
    model: CurveModel,
    loss: Loss,
    layout: ParamLayout,
    guess: Any,
    curve_loss: Any,
    x_arr: jnp.ndarray,
    x_axis: Optional[int],
    c: Any,
    c_axes: Any,
    y_j: jnp.ndarray,
    w_j: jnp.ndarray,
    y_np: NDArray,
    index: pd.Index,
    columns: pd.Index,
    joint_solver: Optional[Any],
    joint_max_steps: int,
    progress: bool,
) -> FitResult:
    """Solve shared and local parameters together, in one pass.

    A shared parameter enters every curve's term, so the objective does not
    separate and there is nothing to chunk. ``LBFGS`` is the default because
    its iteration count is largely independent of the number of curves and it
    never forms a matrix over the whole parameter vector.
    """
    if joint_solver is None:
        joint_solver = optx.LBFGS(rtol=1e-8, atol=1e-8, history_length=10)
    n_curves, n_points = y_np.shape
    use_lsq = _check_solver(joint_solver, loss)
    if use_lsq:
        assert isinstance(loss, LeastSquares)
        _check_joint_jacobian(n_curves, n_points, layout)

    shared_names = layout.shared
    sizes = [layout.group_sizes[n] for n in shared_names]
    widths = [layout.widths[n] for n in shared_names]
    axes = (x_axis, c_axes, 0, 0, {n: 0 for n in shared_names})

    # Seed each shared value from the median of its group's per-curve guesses,
    # taken component-wise for a vector-valued parameter.
    shared0 = []
    for name, size, width in zip(shared_names, sizes, widths):
        per_curve = np.asarray(getattr(guess, name), dtype=float)
        per_curve = per_curve.reshape(len(per_curve), width)
        groups = np.asarray(layout.group_index[name])
        block = np.zeros((size, width))
        for g in range(size):
            member = groups == g
            if member.any():
                block[g] = np.median(per_curve[member], axis=0)
        shared0.append(block.ravel())
    local0 = (
        np.stack(
            [np.asarray(getattr(guess, n), dtype=float) for n in layout.local],
            axis=1,
        )
        if layout.local
        else np.zeros((n_curves, 0))
    )
    theta0 = jnp.concatenate(
        [
            jnp.asarray(np.concatenate(shared0)) if shared0 else jnp.zeros(0),
            jnp.asarray(local0.ravel()),
        ]
    )

    def unpack(
        theta: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        shared: dict[str, jnp.ndarray] = {}
        offset = 0
        for name, size, width in zip(shared_names, sizes, widths):
            block = jax.lax.dynamic_slice(theta, (offset,), (size * width,))
            shared[name] = block if width == 1 else block.reshape(size, width)
            offset += size * width
        return shared, theta[offset:].reshape(n_curves, layout.n_local)

    def gather(shared: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        # A scalar parameter gathers to (n_curves,); a vector-valued one to
        # (n_curves, width), so each curve sees its group's whole vector.
        return {n: shared[n][layout.group_index[n]] for n in shared_names}

    per_curve_loss = jax.vmap(curve_loss, in_axes=(0, axes))

    def joint_loss(theta: jnp.ndarray, _args: Any) -> jnp.ndarray:
        shared, local = unpack(theta)
        losses = per_curve_loss(local, (x_arr, c, y_j, w_j, gather(shared)))
        # Normalised, so the solver's relative tolerances remain meaningful
        # however many curves are summed over.
        total: jnp.ndarray = jnp.sum(losses) / n_curves
        return total

    def _solve_minimise(theta: jnp.ndarray) -> tuple[jnp.ndarray, Any, Any]:
        sol: Any = optx.minimise(
            joint_loss,
            joint_solver,
            theta,
            max_steps=joint_max_steps,
            throw=False,
        )
        return sol.value, _result_code(sol), sol.stats["num_steps"]

    def _solve_least_squares(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Any, Any]:
        assert isinstance(loss, LeastSquares)
        curve_residual = _make_curve_residual(model, loss, layout)
        per_curve_residual = jax.vmap(curve_residual, in_axes=(0, axes))

        def joint_residual(t: jnp.ndarray, _args: Any) -> jnp.ndarray:
            shared, local = unpack(t)
            res = per_curve_residual(
                local, (x_arr, c, y_j, w_j, gather(shared))
            )
            # sqrt, so that squaring and summing matches `joint_loss`.
            flat: jnp.ndarray = res.reshape(-1) / jnp.sqrt(n_curves)
            return flat

        sol: Any = optx.least_squares(
            joint_residual,
            joint_solver,
            theta,
            max_steps=joint_max_steps,
            throw=False,
        )
        return sol.value, _result_code(sol), sol.stats["num_steps"]

    if progress:
        print(
            f"Joint fit: {n_curves} curves, {layout.n_shared_total} shared + "
            f"{n_curves * layout.n_local} local parameters...",
            flush=True,
        )
    solve = _solve_least_squares if use_lsq else _solve_minimise
    theta, code, steps = jax.jit(solve)(theta0)
    shared_final, local_final = unpack(theta)
    if progress:
        print(
            f"    {result_name(int(code))} in {int(steps)} steps", flush=True
        )

    return FitResult.build(
        model=model,
        loss=loss,
        layout=layout,
        index=index,
        local=np.asarray(local_final),
        shared={n: np.asarray(v) for n, v in shared_final.items()},
        codes=np.full(n_curves, int(code)),
        steps=np.full(n_curves, int(steps)),
        x=x_arr,
        x_axis=x_axis,
        consts=c,
        c_axes=c_axes,
        y=y_np,
        w=np.asarray(w_j),
        curve_loss=curve_loss,
        columns=columns,
        progress=progress,
        joint_result=int(code),
        joint_steps=int(steps),
    )


# ---------------------------------------------------------------------------
# Sequential fits
# ---------------------------------------------------------------------------
def _per_step(
    value: Any, models: Sequence[CurveModel], label: str
) -> list[Any]:
    """Expand a ``share``/``fixed`` argument to one entry per model.

    A single mapping applies to every step, filtered to the parameters each
    model actually has -- models in a sequence need not share a parameter list,
    since a background variant has a ``log_bg`` the plain model does not. A
    sequence of mappings instead gives each step its own, which is how a
    parameter is fitted globally and then released::

        ft.fit_stepwise(df, [m, m], x=t,
                        share=[{"log_bg": None}, None])

    Args:
        value: one mapping, or one per model, or None.
        models: the models being fitted.
        label: ``"share"`` or ``"fixed"``, for error messages.

    Returns:
        One mapping (or None) per model.

    Raises:
        FittingError: for a sequence of the wrong length, or a key that names
            no parameter of the model it is given to.
    """
    if isinstance(value, Mapping) or value is None:
        if value:
            usable = set().union(*(set(field_names(m.Params)) for m in models))
            unknown = set(value) - usable
            if unknown:
                raise FittingError(
                    f"{label} refers to parameters {sorted(unknown)} which "
                    f"are not in any of the models given: {sorted(usable)}"
                )
        per_model = [value] * len(models)
        strict = False
    else:
        per_model = list(value)
        if len(per_model) != len(models):
            raise FittingError(
                f"{label} has {len(per_model)} entries but there are "
                f"{len(models)} models; give one mapping to apply to every "
                f"step, or exactly one per step"
            )
        strict = True

    out: list[Any] = []
    for model, mapping in zip(models, per_model):
        if not mapping:
            out.append(mapping)
            continue
        names = set(field_names(model.Params))
        unknown = set(mapping) - names
        if unknown and strict:
            # given explicitly for this step, so a mismatch is a mistake
            raise FittingError(
                f"{label} for {type(model).__name__} refers to parameters "
                f"{sorted(unknown)} which are not in "
                f"{model.Params.__name__}: {sorted(names)}"
            )
        out.append({k: v for k, v in mapping.items() if k in names})
    return out


def fit_stepwise(
    data: Any, models: Sequence[CurveModel], x: Any, **kwargs: Any
) -> list[FitResult]:
    """Fit each model in turn, seeding it from the previous fit.

    A model with more parameters often needs a good starting point, which a
    simpler model of the same family can supply. Seeding uses
    ``models[i].seed_from(previous_params)`` where the model defines it, and
    otherwise carries over any parameters the two models name identically.

    ``share`` and ``fixed`` may be given either once, applying to every step,
    or as one mapping per step. Given once, each model receives only the
    entries naming a parameter it actually has, so a background parameter can
    be shared across a sequence beginning with a model that lacks it::

        single, withbg = ft.fit_stepwise(
            df, [ft.SingleExponentialInterval(),
                 ft.SingleExponentialIntervalWithBackground()],
            x=times, share={"log_bg": None},
        )

    Given per step, each model gets exactly what it is handed -- which is how
    a parameter is fitted globally and then released to vary per curve, using
    the pooled estimate as the starting point::

        pooled, released = ft.fit_stepwise(
            df, [model, model], x=times,
            share=[{"log_bg": None}, None],
        )

    Applied once, a key usable by no model at all is an error; applied per
    step, a key the receiving model lacks is an error. Either way a misspelled
    name is caught rather than silently dropped.

    Args:
        data: one curve per row, as for :func:`fit`.
        models: the models to fit, in order.
        x: the independent variable, as for :func:`fit`.
        **kwargs: forwarded to :func:`fit`. ``share`` and ``fixed`` are
            filtered per model as described above; ``init`` may not be given,
            since seeding is what this function is for.

    Returns:
        One :class:`~fitting.results.FitResult` per model, in the same order,
        each with its own diagnostics::

            single, double = ft.fit_stepwise(df, [m1, m2], x=times)
            both = pd.concat([single.table, double.table],
                             keys=["single", "double"], axis=1)

    Raises:
        FittingError: if ``models`` is empty, if ``init`` is given, or if a
            ``share``/``fixed`` key names no parameter of any model.
    """
    if not models:
        raise FittingError("models must contain at least one model")
    if "init" in kwargs:
        raise FittingError(
            "fit_stepwise seeds each model from the previous fit, so init= "
            "would be overwritten. Call fit() directly to choose the starting "
            "values yourself."
        )

    share = _per_step(kwargs.pop("share", None), models, "share")
    fixed = _per_step(kwargs.pop("fixed", None), models, "fixed")

    results: list[FitResult] = []
    previous: Optional[Any] = None
    for step, model in enumerate(models):
        init = None
        if previous is not None:
            seed = getattr(model, "seed_from", None)
            init = (
                seed(previous)
                if callable(seed)
                else _carry_over(model, previous)
            )
        results.append(
            fit(
                data,
                model=model,
                x=x,
                init=init,
                share=share[step],
                fixed=fixed[step],
                **kwargs,
            )
        )
        previous = results[-1].params()
    return results


def _carry_over(model: CurveModel, previous: Any) -> Optional[Any]:
    """Reuse identically named parameters, if they cover the model's own."""
    wanted = field_names(model.Params)
    if set(wanted) - set(field_names(type(previous))):
        return None
    return model.Params(**{n: getattr(previous, n) for n in wanted})


# ---------------------------------------------------------------------------
# Sequencing counts
# ---------------------------------------------------------------------------
def spikein_log_norm(
    spikein: Union[pd.DataFrame, pd.Series, NDArray],
    ref: int = -1,
    level: Optional[Any] = None,
) -> Union[jnp.ndarray, ByLevel]:
    r"""Convert spike-in counts into the ``log_norm`` constant.

    Spike-ins correct for differing sequencing depth between the libraries
    contributing to one curve. Counts are divided by the spike-in's own value
    in column ``ref``, making the normalisation relative and dimensionless:

    .. math::
        \eta_j = \frac{s_j}{s_{\mathrm{ref}}},
        \qquad \texttt{log\_norm}_j = \log \eta_j.

    Args:
        spikein: spike-in counts. A 1-D array, or a single-row frame, gives one
            vector for the whole experiment. A multi-row frame gives one vector
            per row, keyed by its index.
        ref: which column to normalise against. ``-1`` is the last, which for
            interval count data is the bound library.
        level: the index level of the data that a multi-row ``spikein``'s keys
            correspond to, for example ``"replicate"``.

    Returns:
        A 1-D array to pass as ``consts={"log_norm": ...}``, or, when there is
        more than one spike-in vector, a
        :class:`~fitting.selectors.ByLevel` selector to pass as
        ``curve_consts={"log_norm": ...}``.

    Raises:
        FittingError: for a 2-D array, whose rows cannot be matched to curves,
            or for a multi-row frame given without ``level``.
    """
    if isinstance(spikein, pd.Series):
        spikein = spikein.to_frame().T
    if isinstance(spikein, np.ndarray):
        arr = np.asarray(spikein, dtype=float)
        if arr.ndim == 1:
            return jnp.log(jnp.asarray(arr / arr[ref]))
        raise FittingError(
            "a 2-D spike-in array is ambiguous; pass a DataFrame with an "
            "index and a level= so each vector can be matched to its curves"
        )

    normed = spikein.div(spikein.iloc[:, ref], axis=0)
    if len(normed) == 1:
        return jnp.log(jnp.asarray(normed.to_numpy()[0], dtype=float))
    if level is None:
        raise FittingError(
            f"spikein has {len(normed)} rows, so level= is required to say "
            f"which index level of the data they key on (e.g. 'replicate')"
        )
    return by_level(
        level,
        {
            key: np.log(row)
            for key, row in zip(normed.index, normed.to_numpy(dtype=float))
        },
    )


def fit_sequencing(
    counts: pd.DataFrame,
    x: Any,
    *,
    model: Optional[CurveModel] = None,
    y_cols: Optional[Sequence[Any]] = None,
    spikein: Optional[Union[pd.DataFrame, pd.Series, NDArray]] = None,
    spikein_level: Optional[Any] = None,
    spikein_ref: int = -1,
    drop_empty: bool = True,
    loss: Optional[Loss] = None,
    **kwargs: Any,
) -> FitResult:
    """Fit count curves, handling spike-in normalisation.

    A thin layer over :func:`fit` holding the conventions of sequencing count
    data: spike-ins become the ``log_norm`` constant, the likelihood is
    Poisson, and the model is the single-exponential interval model.

    Args:
        counts: one curve per row, with the count columns selected by
            ``y_cols``.
        x: the interval right-edges. With ``concat_bound=True`` the last entry
            repeats the final timepoint, standing for the bound library.
        model: defaults to
            :class:`~fitting.models.SingleExponentialInterval`. A background
            variant must be asked for explicitly.
        y_cols: which columns hold counts.
        spikein: spike-in counts, as accepted by :func:`spikein_log_norm`.
        spikein_level: the index level a multi-row ``spikein`` keys on.
        spikein_ref: which spike-in column to normalise against.
        drop_empty: drop rows whose selected counts are all zero, since they
            constrain nothing.
        loss: defaults to :class:`~fitting.losses.Poisson`.
        **kwargs: forwarded to :func:`fit`, for example ``share``, ``fixed`` or
            ``chunk_size``.

    Returns:
        A :class:`~fitting.results.FitResult`.
    """
    if model is None:
        model = SingleExponentialInterval(concat_bound=True)

    frame = counts if y_cols is None else counts[list(y_cols)]
    if drop_empty:
        keep = frame.to_numpy(dtype=float).sum(axis=1) > 0
        if not keep.all():
            counts = counts[keep]

    consts = dict(kwargs.pop("consts", None) or {})
    curve_consts = dict(kwargs.pop("curve_consts", None) or {})
    if spikein is not None:
        norm = spikein_log_norm(spikein, ref=spikein_ref, level=spikein_level)
        if isinstance(norm, ByLevel):
            curve_consts["log_norm"] = norm
        else:
            consts["log_norm"] = norm

    return fit(
        counts,
        model=model,
        x=x,
        y_cols=y_cols,
        loss=loss,
        consts=consts or None,
        curve_consts=curve_consts or None,
        **kwargs,
    )
