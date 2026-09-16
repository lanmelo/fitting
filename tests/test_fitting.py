"""The two fitting paths: independent per-curve, and joint with sharing."""

# Test-local model containers are small by design.
# pylint: disable=too-few-public-methods

import dataclasses
from typing import Any, ClassVar, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd
import pytest
from conftest import make_dissociation
from jax.typing import ArrayLike

import fitting as ft
import fitting.fitting


# ----------------------------------------------------------------- local path
def test_local_recovers_known_parameters(interval_x: jnp.ndarray) -> None:
    """Local recovers known parameters."""
    df, log_y0, log_koff = make_dissociation(n=60)
    res = ft.fit(
        df, model=ft.SingleExponentialInterval(), x=interval_x, progress=False
    )
    assert res.diagnostics.converged.all()
    assert np.corrcoef(res.local.log_k_off, log_koff)[0, 1] > 0.99
    assert np.median(np.abs(res.local.log_y0 - log_y0)) < 0.05


def test_local_result_is_independent_of_chunk_size(
    interval_x: jnp.ndarray,
) -> None:
    """Chunking is an execution detail and must not change the answer."""
    df, _, _ = make_dissociation(n=50)
    a = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        chunk_size=50,
        progress=False,
    )
    b = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        chunk_size=7,
        progress=False,
    )
    assert np.allclose(a.local.log_k_off, b.local.log_k_off, atol=1e-8)


def test_linear_space_parameters_are_added(interval_x: jnp.ndarray) -> None:
    """Linear space parameters are added."""
    df, _, _ = make_dissociation(n=10)
    res = ft.fit(
        df, model=ft.SingleExponentialInterval(), x=interval_x, progress=False
    )
    assert "k_off" in res.local.columns
    assert np.allclose(res.local.k_off, np.exp(res.local.log_k_off))


def test_diagnostics_report_status_and_convergence(
    interval_x: jnp.ndarray,
) -> None:
    """Diagnostics report status and convergence."""
    df, _, _ = make_dissociation(n=10)
    res = ft.fit(
        df, model=ft.SingleExponentialInterval(), x=interval_x, progress=False
    )
    assert set(res.diagnostics.status.unique()) == {"successful"}
    assert res.diagnostics.converged.all()


def test_non_convergence_is_reported_not_silent(
    interval_x: jnp.ndarray,
) -> None:
    """A step-capped curve must come back converged=False, not silently."""
    df, _, _ = make_dissociation(n=8)
    res = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        max_steps=2,
        progress=False,
    )
    assert not res.diagnostics.converged.all()
    assert "max_steps" in " ".join(res.diagnostics.status.astype(str).unique())


def test_observed_and_predicted_align(interval_x: jnp.ndarray) -> None:
    """Observed and predicted align."""
    df, _, _ = make_dissociation(n=12)
    res = ft.fit(
        df, model=ft.SingleExponentialInterval(), x=interval_x, progress=False
    )
    assert list(res.observed.columns) == list(df.columns)
    assert res.predicted.shape == df.shape
    assert res.residuals.notna().all().all()


def test_params_roundtrip_as_init(interval_x: jnp.ndarray) -> None:
    """Params roundtrip as init."""
    df, _, _ = make_dissociation(n=15)
    first = ft.fit(
        df, model=ft.SingleExponentialInterval(), x=interval_x, progress=False
    )
    second = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        init=first.params(),
        progress=False,
    )
    assert np.allclose(
        first.local.log_k_off, second.local.log_k_off, atol=1e-6
    )


# --------------------------------------------------------- the Bound library
def test_bound_library_constrains_k_off(interval_x: jnp.ndarray) -> None:
    """Dropping the final bound library must visibly change fitted rates.

    That library holds the population remaining after repeated dissociation,
    and carries a large share of the information in a low-count fit.
    """
    df, _, _ = make_dissociation(n=60)
    withbound = ft.fit(
        df,
        model=ft.SingleExponentialInterval(concat_bound=True),
        x=interval_x,
        progress=False,
    )
    kinetic_only = ft.fit(
        df.iloc[:, :-1],
        model=ft.SingleExponentialInterval(concat_bound=False),
        x=interval_x[:-1],
        progress=False,
    )
    shift = np.abs(
        withbound.local.log_k_off.to_numpy()
        - kinetic_only.local.log_k_off.to_numpy()
    )
    assert np.median(shift) > 1e-3, "dropping Bound should change the fit"


# ------------------------------------------------------------------ joint path
def _langmuir_data(
    n_per: int = 80, ymax: dict[str, float] | None = None
) -> tuple[
    pd.DataFrame,
    jnp.ndarray,
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    dict[str, float],
]:
    ymax = ymax or {"R1": 5.0, "R2": 12.0, "R3": 30.0}
    conc = jnp.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    rs = np.random.RandomState(0)  # pylint: disable=no-member
    rows, idx, kd = [], [], []
    for rep, ym in ymax.items():
        for i in range(n_per):
            k = np.exp(rs.normal(0.0, 0.6))
            kd.append(k)
            rows.append(
                ym * np.asarray(conc) / (np.asarray(conc) + k)
                + rs.normal(0, 0.02, len(conc))
            )
            idx.append((rep, f"v{i}"))
    df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["replicate", "variant"]),
    )
    return df, conc, np.array(kd), ymax


def test_global_sharing_recovers_one_value() -> None:
    """Global sharing recovers one value."""
    df, conc, kd, _ = _langmuir_data(n_per=60, ymax={"R1": 8.0})
    res = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": None},
        progress=False,
    )
    assert len(res.shared) == 1
    assert np.isclose(float(res.shared.linear_value.iloc[0]), 8.0, rtol=1e-3)
    assert np.corrcoef(res.local.kd, kd)[0, 1] > 0.99


def test_grouped_sharing_recovers_each_group() -> None:
    """Grouped sharing recovers each group."""
    df, conc, kd, ymax_true = _langmuir_data()
    res = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": ft.by_level("replicate")},
        progress=False,
    )
    assert len(res.shared) == 3
    got = res.shared.sort_values("group").linear_value.to_numpy()
    assert np.allclose(got, list(ymax_true.values()), rtol=1e-3)
    assert np.corrcoef(res.local.kd, kd)[0, 1] > 0.99


def test_shared_result_is_invariant_to_curve_order() -> None:
    """Row order must not move the shared optimum."""
    df, conc, _, _ = _langmuir_data(n_per=40)
    a = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": ft.by_level("replicate")},
        progress=False,
    )
    shuffled = df.sample(frac=1.0, random_state=1)
    b = ft.fit(
        shuffled,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": ft.by_level("replicate")},
        progress=False,
    )
    av = a.shared.sort_values("group").value.to_numpy()
    bv = b.shared.sort_values("group").value.to_numpy()
    assert np.allclose(av, bv, atol=1e-4)


def test_shared_parameters_get_standard_errors() -> None:
    """Shared parameters get standard errors."""
    df, conc, _, _ = _langmuir_data(n_per=50)
    res = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": ft.by_level("replicate")},
        progress=False,
    )
    assert res.shared.se.notna().all()
    assert (res.shared.se > 0).all()


def test_shared_values_broadcast_into_table() -> None:
    """Shared values broadcast into table."""
    df, conc, _, _ = _langmuir_data(n_per=20)
    res = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": ft.by_level("replicate")},
        progress=False,
    )
    assert "log_ymax" in res.table.columns
    per_rep = res.table.groupby(res.table.index.get_level_values("replicate"))
    assert per_rep.log_ymax.nunique().eq(1).all()


# -------------------------------------------------------------- fixed / errors
def test_fixed_pins_a_parameter() -> None:
    """Fixed pins a parameter."""
    df, conc, _, _ = _langmuir_data(n_per=30, ymax={"R1": 9.0})
    res = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        fixed={"log_ymax": float(np.log(9.0))},
        progress=False,
    )
    assert "log_ymax" not in res.local.columns
    assert res.layout.fixed_names == ("log_ymax",)


def test_space_mismatch_raises_rather_than_nan() -> None:
    """A linear-space model with a log-space loss must fail loudly."""
    df, conc, _, _ = _langmuir_data(n_per=5)
    with pytest.raises(ft.FittingError, match="log-space"):
        ft.fit(
            df, model=ft.Langmuir(), x=conc, loss=ft.Poisson(), progress=False
        )


def test_space_mismatch_names_a_working_loss(interval_x: jnp.ndarray) -> None:
    """A mismatch is refused, and the message names a loss that would fit."""
    df, _, _ = make_dissociation(n=5)
    with pytest.raises(ft.FittingError, match="LogSSE"):
        ft.fit(
            df,
            model=ft.SingleExponentialInterval(),
            x=interval_x,
            loss=ft.SSE(),
            progress=False,
        )


def test_unknown_share_parameter_raises() -> None:
    """Unknown share parameter raises."""
    df, conc, _, _ = _langmuir_data(n_per=3)
    with pytest.raises(ft.FittingError, match="not in"):
        ft.fit(
            df,
            model=ft.Langmuir(),
            x=conc,
            loss=ft.SSE(),
            share={"nonexistent": None},
            progress=False,
        )


def test_share_and_fixed_conflict_raises() -> None:
    """Share and fixed conflict raises."""
    df, conc, _, _ = _langmuir_data(n_per=3)
    with pytest.raises(ft.FittingError, match="both"):
        ft.fit(
            df,
            model=ft.Langmuir(),
            x=conc,
            loss=ft.SSE(),
            share={"log_ymax": None},
            fixed={"log_ymax": 1.0},
            progress=False,
        )


def test_unknown_const_raises(interval_x: jnp.ndarray) -> None:
    """Unknown const raises."""
    df, _, _ = make_dissociation(n=3)
    with pytest.raises(ft.FittingError, match="not constants"):
        ft.fit(
            df,
            model=ft.SingleExponentialInterval(),
            x=interval_x,
            consts={"bogus": 1.0},
            progress=False,
        )


def test_curve_const_wrong_length_raises(interval_x: jnp.ndarray) -> None:
    """Curve const wrong length raises."""
    df, _, _ = make_dissociation(n=6)
    with pytest.raises(ft.FittingError, match="leading dimension"):
        ft.fit(
            df,
            model=ft.SingleExponentialInterval(),
            x=interval_x,
            curve_consts={"log_norm": np.zeros((3, len(interval_x)))},
            progress=False,
        )


# ------------------------------------------------------------------ misc
def test_y_cols_keeps_metadata_columns(interval_x: jnp.ndarray) -> None:
    """Metadata columns may stay on the frame, selected out by ``y_cols``."""
    df, _, _ = make_dissociation(n=10)
    df = df.assign(low_GFP=False, note="x")
    res = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        y_cols=[c for c in df.columns if c not in ("low_GFP", "note")],
        progress=False,
    )
    assert res.observed.shape[1] == len(interval_x)


def test_nan_observations_are_masked(interval_x: jnp.ndarray) -> None:
    """Nan observations are masked."""
    df, _, _ = make_dissociation(n=12)
    dirty = df.astype(float).copy()
    dirty.iloc[0, 3] = np.nan
    res = ft.fit(
        dirty,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        progress=False,
    )
    assert res.diagnostics.converged.all()
    assert res.mask.to_numpy()[0, 3] == 0.0
    assert np.isfinite(res.local.log_k_off.to_numpy()).all()


def test_per_curve_x_is_accepted() -> None:
    """Per curve x is accepted."""
    df, conc, _, _ = _langmuir_data(n_per=10, ymax={"R1": 6.0})
    x2d = np.tile(np.asarray(conc), (len(df), 1))
    res = ft.fit(df, model=ft.Langmuir(), x=x2d, loss=ft.SSE(), progress=False)
    assert res.diagnostics.converged.all()


def test_empty_input_returns_empty_result(interval_x: jnp.ndarray) -> None:
    """Empty input returns empty result."""
    df, _, _ = make_dissociation(n=4)
    res = ft.fit(
        df.iloc[:0],
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        progress=False,
    )
    assert len(res.local) == 0


def test_custom_solver_is_used(interval_x: jnp.ndarray) -> None:
    """Custom solver is used."""
    df, _, _ = make_dissociation(n=10)
    res = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        solver=optx.BFGS(rtol=1e-6, atol=1e-6),
        progress=False,
    )
    assert res.diagnostics.converged.all()


def test_stepwise_seeds_double_from_single(interval_x: jnp.ndarray) -> None:
    """Stepwise seeds double from single."""
    df, _, _ = make_dissociation(n=25)
    results = ft.fit_stepwise(
        df,
        [ft.SingleExponentialInterval(), ft.DoubleExponentialInterval()],
        x=interval_x,
        progress=False,
    )
    assert len(results) == 2
    single, double = results[0], results[1]
    assert "log_k_off" in single.local.columns
    assert "log_k_off_1" in double.local.columns
    # the richer model cannot fit worse at the same optimum
    assert (
        double.diagnostics.loss.sum() <= single.diagnostics.loss.sum() + 1e-6
    )


# ------------------------------------------------------- least-squares solvers
def test_least_squares_solvers_agree_with_minimiser() -> None:
    """LM and Gauss-Newton must reach the same optimum as BFGS.

    They minimise the residual vector rather than the scalar loss, so agreement
    confirms the residual form really is the same objective.
    """
    df, conc, _, _ = _langmuir_data(n_per=60, ymax={"R1": 4.0})
    ref = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        solver=optx.BFGS(rtol=1e-11, atol=1e-11),
        progress=False,
    )
    solvers: list[Any] = [
        optx.LevenbergMarquardt(rtol=1e-11, atol=1e-11),
        optx.GaussNewton(rtol=1e-11, atol=1e-11),
    ]
    for solver in solvers:
        got = ft.fit(
            df,
            model=ft.Langmuir(),
            x=conc,
            loss=ft.SSE(),
            solver=solver,
            progress=False,
        )
        assert got.diagnostics.converged.all()
        assert np.allclose(got.local.log_kd, ref.local.log_kd, atol=1e-5)


def test_least_squares_value_follows_from_residual() -> None:
    """``value`` is derived from ``residual``, so the two cannot disagree."""
    rs = np.random.RandomState(0)  # pylint: disable=no-member
    pred = jnp.asarray(rs.normal(size=20))
    y = jnp.asarray(rs.normal(size=20))
    w = jnp.asarray(rs.randint(0, 2, 20).astype(float))
    for loss in (ft.SSE(), ft.LogSSE()):
        target = jnp.abs(y) + 0.5 if loss.requires_positive else y
        squares = float(jnp.sum(jnp.square(loss.residual(pred, target, w))))
        assert np.isclose(
            squares, float(loss.value(pred, target, w)), rtol=1e-12
        )


def test_least_squares_solver_rejects_non_least_squares_loss(
    interval_x: jnp.ndarray,
) -> None:
    """Poisson has no residual form, so LM cannot be used with it."""
    df, _, _ = make_dissociation(n=5)
    with pytest.raises(ft.FittingError, match="residual"):
        ft.fit(
            df,
            model=ft.SingleExponentialInterval(),
            x=interval_x,
            loss=ft.Poisson(),
            solver=optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
            progress=False,
        )


def test_joint_least_squares_refused_when_jacobian_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A joint LM solve is allowed only while its dense Jacobian is reasonable.

    Levenberg-Marquardt materialises an (n_residuals x n_parameters) Jacobian,
    and with one parameter set per curve that grows quadratically, so it is
    fine for hundreds of curves and hopeless for millions.
    """
    df, conc, _, _ = _langmuir_data(n_per=10, ymax={"R1": 3.0})
    monkeypatch.setattr(fitting.fitting, "MAX_JOINT_JACOBIAN_BYTES", 1)
    with pytest.raises(ft.FittingError, match="dense"):
        ft.fit(
            df,
            model=ft.Langmuir(),
            x=conc,
            loss=ft.SSE(),
            share={"log_ymax": None},
            joint_solver=optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
            progress=False,
        )


def test_joint_least_squares_allowed_when_small() -> None:
    """At modest size it runs; it just tends not to converge as well as LBFGS.

    Kept available rather than refused: the joint objective has one parameter
    block per curve, which Gauss-Newton-style steps handle poorly, but that is
    reported through ``info["joint_status"]`` rather than hidden.
    """
    df, conc, _, _ = _langmuir_data(n_per=10, ymax={"R1": 3.0})
    res = ft.fit(
        df,
        model=ft.Langmuir(),
        x=conc,
        loss=ft.SSE(),
        share={"log_ymax": None},
        joint_solver=optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
        joint_max_steps=200,
        progress=False,
    )
    assert "joint_status" in res.info


# ------------------------------------------------------------- ragged curves
def _ragged() -> (
    tuple[pd.DataFrame, np.ndarray[tuple[int, ...], np.dtype[np.float64]]]
):
    """Curves of unequal length, padded with NaN in both y and x."""
    rs = np.random.RandomState(0)  # pylint: disable=no-member
    width, n = 8, 40
    rows_y, rows_x = [], []
    for i in range(n):
        used = 4 + (i % 5)
        conc = np.geomspace(0.1, 100.0, used)
        kd = np.exp(rs.normal(0.0, 0.5))
        y = 3.0 * conc / (conc + kd) + rs.normal(0, 0.01, used)
        pad = np.full(width - used, np.nan)
        rows_y.append(np.concatenate([y, pad]))
        rows_x.append(np.concatenate([conc, pad]))
    return pd.DataFrame(rows_y), np.stack(rows_x)


def test_ragged_curves_with_nan_padded_x_converge() -> None:
    """NaN padding in ``x`` must not reach masked points.

    ``0 * NaN`` is ``NaN``, so a non-finite ``x`` would enter the residual even
    where the observation carries no weight.
    """
    y, x = _ragged()
    res = ft.fit(y, model=ft.Langmuir(), x=x, loss=ft.SSE(), progress=False)
    assert res.diagnostics.converged.all()
    assert np.isfinite(res.local.log_kd.to_numpy()).all()


def test_nonfinite_x_at_a_live_point_raises() -> None:
    """A non-finite x where the observation is *not* masked is an error."""
    y, x = _ragged()
    x[0, 0] = np.nan  # y[0, 0] is a real measurement
    with pytest.raises(ft.FittingError, match="not masked"):
        ft.fit(y, model=ft.Langmuir(), x=x, loss=ft.SSE(), progress=False)


def test_nonfinite_shared_x_raises(interval_x: jnp.ndarray) -> None:
    """A shared 1-D x applies to every curve, so NaN can never be excused."""
    df, _, _ = make_dissociation(n=4)
    bad = np.asarray(interval_x, dtype=float).copy()
    bad[0] = np.nan

    with pytest.raises(ft.FittingError, match="non-finite"):
        ft.fit(df, model=ft.SingleExponentialInterval(), x=bad, progress=False)


def test_progress_false_silences_convergence_warning(
    interval_x: jnp.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    """``progress=False`` must print nothing, while still recording status."""
    df, _, _ = make_dissociation(n=6)
    res = ft.fit(
        df,
        model=ft.SingleExponentialInterval(),
        x=interval_x,
        max_steps=2,
        progress=False,
    )
    assert capsys.readouterr().out == ""
    assert not res.diagnostics.converged.any()


# ---------------------------------------------------- log-space least squares
def _multiplicative_decay(
    n: int = 200, noise: float = 0.10, seed: int = 0
) -> tuple[
    pd.DataFrame,
    jnp.ndarray,
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
]:
    """Exponential decay with lognormal (multiplicative) noise."""
    rs = np.random.RandomState(seed)  # pylint: disable=no-member
    x = jnp.asarray(np.arange(0.0, 13.0, 2.0))
    y0 = np.exp(rs.normal(np.log(5000.0), 0.5, n))
    k = np.exp(rs.normal(-1.5, 0.3, n))
    clean = y0[:, None] * np.exp(-k[:, None] * np.asarray(x))
    return pd.DataFrame(clean * np.exp(rs.normal(0, noise, clean.shape))), x, k


class _LinearDecay(ft.CurveModel):
    """Exponential decay predicting in linear space, to pair with ``SSE``."""

    class Params(eqx.Module):
        """Amplitude and rate, both in log space."""

        log_y0: ArrayLike
        log_k: ArrayLike

    log_predictions: ClassVar[bool] = False

    @override
    def predict(self, p: Any, x: jnp.ndarray, c: Any) -> jnp.ndarray:
        return jnp.exp(p.log_y0 - x * jnp.exp(p.log_k))

    @override
    def init(self, y: jnp.ndarray, x: jnp.ndarray, c: Any) -> Any:
        return _LinearDecay.Params(
            log_y0=jnp.log(jnp.maximum(1.0, y[:, 0])),
            log_k=jnp.full(y.shape[0], -1.0),
        )


def test_log_sse_beats_linear_sse_on_multiplicative_noise() -> None:
    """Least squares in log space is the right objective for relative error.

    With multiplicative noise the residual variance grows with the signal, so
    linear least squares over-weights the bright points. Comparing in log
    space equalises them.
    """
    obs, x, true_k = _multiplicative_decay()
    errs = {}
    for name, model, loss in (
        ("linear", _LinearDecay(), ft.SSE()),
        ("log", ft.SingleExponentialDecay(), ft.LogSSE()),
    ):
        res = ft.fit(obs, model=model, x=x, loss=loss, progress=False)
        errs[name] = float(
            np.median(np.abs(res.local.log_k.to_numpy() - np.log(true_k)))
        )
    assert errs["log"] < errs["linear"]


def test_log_sse_reports_in_observation_units() -> None:
    """``predicted`` and ``rmse`` must be in the same units as ``observed``.

    ``LogSSE`` consumes log-space predictions but linear observations, so
    reporting has to convert back rather than expose the log-space values.
    """
    obs, x, _ = _multiplicative_decay(n=50)
    res = ft.fit(
        obs,
        model=ft.SingleExponentialDecay(),
        x=x,
        loss=ft.LogSSE(),
        progress=False,
    )
    observed, predicted = res.observed.to_numpy(), res.predicted.to_numpy()
    # both in intensity units: same order of magnitude, and residuals are small
    assert np.median(predicted) > 100.0
    assert np.median(np.abs(observed - predicted) / observed) < 0.25
    assert res.diagnostics.rmse.median() < np.median(observed)


def test_log_sse_works_with_a_least_squares_solver() -> None:
    """``LOG_SSE`` has a residual form, so Gauss-Newton applies."""
    obs, x, _ = _multiplicative_decay(n=100)
    res = ft.fit(
        obs,
        model=ft.SingleExponentialDecay(),
        x=x,
        loss=ft.LogSSE(),
        solver=optx.GaussNewton(rtol=1e-10, atol=1e-10),
        progress=False,
    )
    assert res.diagnostics.converged.all()


def test_log_sse_refuses_non_positive_observations() -> None:
    """It takes log(y), so a non-positive unmasked observation is an error."""
    obs, x, _ = _multiplicative_decay(n=20)
    obs.iloc[0, 0] = -1.0
    with pytest.raises(ft.FittingError, match="logarithm"):
        ft.fit(
            obs,
            model=ft.SingleExponentialDecay(),
            x=x,
            loss=ft.LogSSE(),
            progress=False,
        )


def test_log_sse_tolerates_masked_non_positive_observations() -> None:
    """A masked non-positive point is fine, and must not leak NaN."""
    obs, x, _ = _multiplicative_decay(n=20)
    obs.iloc[0, 0] = np.nan  # masked by ignore_nan
    res = ft.fit(
        obs,
        model=ft.SingleExponentialDecay(),
        x=x,
        loss=ft.LogSSE(),
        progress=False,
    )
    assert res.diagnostics.converged.all()
    assert np.isfinite(res.local.log_k.to_numpy()).all()


def test_vector_shared_recovers_depth() -> None:
    """A shared, vector-valued depth is recovered from diverse curves."""
    rng = np.random.default_rng(0)
    n_per, n_points = 200, 13
    x = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)
    # Relative to the last library, as spikein_log_norm and the model do.
    true_eta = np.stack(
        [
            np.log(np.linspace(3.0, 1.0, n_points)),
            np.log(np.linspace(1.0, 2.5, n_points)) + 0.3,
        ]
    )
    true_eta -= true_eta[:, -1:]
    groups = np.repeat([0, 1], n_per)
    n = len(groups)

    plain = ft.SingleExponentialInterval(concat_bound=True)
    mu = jax.vmap(plain.predict, in_axes=(0, None, 0))(
        plain.Params(
            log_y0=jnp.asarray(np.log(rng.uniform(2e5, 2e6, n))),
            log_k_off=jnp.asarray(np.log(rng.uniform(0.02, 0.4, n))),
        ),
        x,
        plain.Consts(log_norm=jnp.asarray(true_eta[groups])),
    )
    y = rng.poisson(np.exp(np.asarray(mu)))

    model = ft.SingleExponentialIntervalFittedDepth(concat_bound=True)
    seed = true_eta[groups][:, :-1] + rng.normal(0, 0.3, (n, n_points - 1))
    guess = dataclasses.replace(
        model.init(jnp.asarray(y, float), x, model.Consts()),
        log_eta=jnp.asarray(seed),
    )
    res = ft.fit(
        y,
        x=x,
        model=model,
        loss=ft.Poisson(),
        share={"log_eta": jnp.asarray(groups)},
        init=guess,
        progress=False,
    )

    # No re-anchoring: pinning the reference fixes the gauge, so the fitted
    # depth is directly comparable to the truth.
    fitted = model.full_depth(res.shared)
    assert fitted.shape == (2, n_points)
    assert np.allclose(fitted[:, -1], 0.0)
    assert np.max(np.abs(fitted - true_eta)) < 0.02


def test_double_fitted_depth_shares_the_depth_logic() -> None:
    """The double-exponential variant fits the same shared depth vector."""
    rng = np.random.default_rng(3)
    n_points = 13
    x = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)
    true_eta = np.log(np.linspace(2.0, 1.0, n_points))
    true_eta -= true_eta[-1]
    n = 400

    plain = ft.DoubleExponentialInterval(concat_bound=True)
    y0 = np.log(rng.uniform(2e5, 2e6, n))
    mu = jax.vmap(plain.predict, in_axes=(0, None, None))(
        plain.Params(
            log_y0_1=jnp.asarray(y0),
            log_k_off_1=jnp.asarray(np.log(rng.uniform(0.02, 0.1, n))),
            log_y0_2=jnp.asarray(y0 - 1.0),
            log_k_off_2=jnp.asarray(np.log(rng.uniform(0.5, 2.0, n))),
        ),
        x,
        plain.Consts(log_norm=jnp.asarray(true_eta)),
    )
    y = rng.poisson(np.exp(np.asarray(mu)))

    model = ft.DoubleExponentialIntervalFittedDepth(concat_bound=True)
    guess = dataclasses.replace(
        model.init(jnp.asarray(y, float), x, model.Consts()),
        log_eta=jnp.asarray(
            np.broadcast_to(true_eta[:-1], (n, n_points - 1))
            + rng.normal(0, 0.2, (n, n_points - 1))
        ),
    )
    res = ft.fit(
        y,
        x=x,
        model=model,
        loss=ft.Poisson(),
        share={"log_eta": None},
        init=guess,
        progress=False,
    )
    fitted = model.full_depth(res.shared)
    assert fitted.shape == (1, n_points)
    assert fitted[0, -1] == 0.0
    # The first interval is the least constrained: the fast component releases
    # most of its amplitude there, so eta_1 trades against log_y0_2.
    assert np.max(np.abs(fitted[0] - true_eta)) < 0.15
    assert np.max(np.abs(fitted[0, 1:] - true_eta[1:])) < 0.05


@pytest.mark.parametrize(
    "cls",
    [
        ft.SingleExponentialIntervalFittedDepth,
        ft.DoubleExponentialIntervalFittedDepth,
    ],
)
def test_fitted_depth_ref_is_pinned(cls: Any) -> None:
    """Whichever library is the reference, its depth entry stays at zero."""
    x = jnp.arange(1.0, 6.0)
    model = cls(concat_bound=False, ref=0)
    n_free = x.shape[0] - 1
    p = model.Params(
        **{
            n: jnp.asarray(0.0 if n != "log_eta" else jnp.full(n_free, 3.0))
            for n in model.Params.__dataclass_fields__
        }
    )
    pred = model.predict(p, x, model.Consts())
    base = cls.__mro__[2](concat_bound=False)
    plain = base.predict(
        p, x, base.Consts(log_norm=jnp.insert(jnp.full(n_free, 3.0), 0, 0.0))
    )
    assert np.allclose(np.asarray(pred), np.asarray(plain))


def test_fitted_depth_requires_sharing() -> None:
    """The fitted-depth model rejects a per-curve or scalar depth."""
    x = jnp.arange(1.0, 5.0)
    y = np.full((6, 4), 10.0)
    model = ft.SingleExponentialIntervalFittedDepth(concat_bound=False)
    with pytest.raises(ft.FittingError, match="log_eta must be shared"):
        ft.fit(y, x=x, model=model, loss=ft.Poisson(), progress=False)

    guess = dataclasses.replace(
        model.init(jnp.asarray(y), x, model.Consts()), log_eta=jnp.zeros(6)
    )
    with pytest.raises(ft.FittingError, match="must be vector valued"):
        ft.fit(
            y,
            x=x,
            model=model,
            loss=ft.Poisson(),
            share={"log_eta": jnp.zeros(6, int)},
            init=guess,
            progress=False,
        )
