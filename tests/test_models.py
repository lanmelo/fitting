"""Model math, and the separation between plain and background variants."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import fitting as ft
from fitting.core import field_names


def test_interval_shapes_match_x(interval_x: jnp.ndarray) -> None:
    """concat_bound consumes the duplicated final timepoint."""
    m = ft.SingleExponentialInterval(concat_bound=True)
    p = ft.SingleExponentialInterval.Params(jnp.log(1000.0), -3.0)
    out = m.predict(p, interval_x, ft.SingleExponentialInterval.Consts())
    assert out.shape == (len(interval_x),)


def test_final_element_is_bound_population(interval_x: jnp.ndarray) -> None:
    """With concat_bound the last prediction is the absolute remaining bound
    signal, not an interval difference."""
    m = ft.SingleExponentialInterval(concat_bound=True)
    log_y0, log_koff = jnp.log(1000.0), -2.0
    out = m.predict(
        ft.SingleExponentialInterval.Params(log_y0, log_koff),
        interval_x,
        ft.SingleExponentialInterval.Consts(),
    )
    expected = log_y0 - interval_x[-1] * jnp.exp(log_koff)
    assert np.isclose(float(out[-1]), float(expected))


def test_interval_counts_sum_to_total_decay(interval_x: jnp.ndarray) -> None:
    """The kinetic intervals plus the bound remainder reconstruct y0."""
    m = ft.SingleExponentialInterval(concat_bound=True)
    p = ft.SingleExponentialInterval.Params(jnp.log(1000.0), -2.0)
    y = np.exp(
        np.asarray(
            m.predict(p, interval_x, ft.SingleExponentialInterval.Consts())
        )
    )
    assert np.isclose(y.sum(), 1000.0, rtol=1e-9)


def test_background_reduces_to_plain_model(interval_x: jnp.ndarray) -> None:
    """The background variant must be a strict generalisation of the plain
    model, reducing to it exactly as the background vanishes."""
    plain = ft.SingleExponentialInterval(concat_bound=True)
    withbg = ft.SingleExponentialIntervalWithBackground(concat_bound=True)
    a = plain.predict(
        ft.SingleExponentialInterval.Params(8.0, -3.0),
        interval_x,
        ft.SingleExponentialInterval.Consts(),
    )
    b = withbg.predict(
        ft.SingleExponentialIntervalWithBackground.Params(8.0, -3.0, -jnp.inf),
        interval_x,
        ft.SingleExponentialInterval.Consts(),
    )
    assert np.allclose(np.asarray(a), np.asarray(b), equal_nan=True)


def test_background_actually_changes_predictions(
    interval_x: jnp.ndarray,
) -> None:
    """Background actually changes predictions."""
    withbg = ft.SingleExponentialIntervalWithBackground()
    a = withbg.predict(
        ft.SingleExponentialIntervalWithBackground.Params(8.0, -3.0, -jnp.inf),
        interval_x,
        ft.SingleExponentialInterval.Consts(),
    )
    b = withbg.predict(
        ft.SingleExponentialIntervalWithBackground.Params(8.0, -3.0, 2.0),
        interval_x,
        ft.SingleExponentialInterval.Consts(),
    )
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_background_params_extend_rather_than_duplicate() -> None:
    """A variant's ``Params`` extends its parent's fields rather than
    restating them."""
    plain = ft.SingleExponentialInterval.Params
    withbg = ft.SingleExponentialIntervalWithBackground.Params
    assert issubclass(withbg, plain)
    assert field_names(plain) == ("log_y0", "log_k_off")
    assert field_names(withbg) == ("log_y0", "log_k_off", "log_bg")


def test_param_containers_are_scoped_to_their_models() -> None:
    """Containers are nested on the model, not package-level names."""
    assert ft.Langmuir.Params.__qualname__ == "Langmuir.Params"
    assert not hasattr(ft, "LangmuirParams")
    assert not hasattr(ft, "DissocParams")
    assert issubclass(ft.LangmuirWithOffset.Params, ft.Langmuir.Params)
    assert field_names(ft.LangmuirWithOffset.Params) == (
        "log_kd",
        "log_ymax",
        "offset",
    )


def test_concat_bound_static_field_is_inherited() -> None:
    """Concat bound static field is inherited."""
    m = ft.SingleExponentialIntervalWithBackground(concat_bound=False)
    assert m.concat_bound is False


def test_log_norm_default_is_no_normalisation(interval_x: jnp.ndarray) -> None:
    """Log norm default is no normalisation."""
    m = ft.SingleExponentialInterval()
    p = ft.SingleExponentialInterval.Params(8.0, -3.0)
    a = m.predict(p, interval_x, ft.SingleExponentialInterval.Consts())
    b = m.predict(
        p,
        interval_x,
        ft.SingleExponentialInterval.Consts(
            log_norm=jnp.zeros(len(interval_x))
        ),
    )
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_logsubexp_guard_returns_neg_inf_not_nan() -> None:
    """x <= y has no logarithm; it must not produce NaN."""
    out = np.asarray(
        ft.logsubexp(jnp.array([1.0, 1.0]), jnp.array([2.0, 0.0]))
    )
    assert out[0] == -np.inf
    assert np.isfinite(out[1])


def test_logsubexp_matches_unguarded_where_valid() -> None:
    """Logsubexp matches unguarded where valid."""
    x, y = jnp.array([3.0, 5.0, 2.5]), jnp.array([1.0, 4.0, 2.4])
    got = np.asarray(ft.logsubexp(x, y))
    want = np.asarray(x + jnp.log1p(-jnp.exp(y - x)))
    assert np.array_equal(got, want)


def test_prediction_space_declared() -> None:
    """Count models work in log space; isotherms in linear space."""
    assert ft.SingleExponentialInterval.log_predictions
    assert ft.SingleExponentialDecay.log_predictions
    assert not ft.Langmuir.log_predictions
    assert not ft.LogisticAffinity.log_predictions


@pytest.mark.parametrize(
    "model",
    [
        ft.SingleExponentialInterval(),
        ft.SingleExponentialIntervalWithBackground(),
        ft.DoubleExponentialInterval(),
        ft.DoubleExponentialIntervalWithBackground(),
        ft.SingleExponentialDecay(),
        ft.SingleExponentialDecayWithBackground(),
        ft.Langmuir(),
        ft.LangmuirWithOffset(),
        ft.LogisticAffinity(),
    ],
    ids=lambda m: type(m).__name__,
)
def test_every_model_fits_from_its_own_default_init(
    model: ft.CurveModel, interval_x: jnp.ndarray
) -> None:
    """Every built-in model must be fittable with no explicit ``init``.

    A variant model inherits its parent's ``init``, so the parent has to build
    a container the variant can extend rather than one it must replace.
    """
    rs = np.random.RandomState(0)  # pylint: disable=no-member
    y = pd.DataFrame(
        rs.uniform(0.5, 5.0, (6, len(interval_x))),
        columns=[f"c{i}" for i in range(len(interval_x))],
    )
    guess = model.init(jnp.asarray(y.to_numpy()), interval_x, model.Consts())
    assert set(field_names(type(guess))) == set(field_names(model.Params))

    loss = ft.Poisson() if model.log_predictions else ft.SSE()
    res = ft.fit(y, model=model, x=interval_x, loss=loss, progress=False)
    assert len(res.local) == len(y)
    assert np.isfinite(res.diagnostics.loss.to_numpy()).all()
