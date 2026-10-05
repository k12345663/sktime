#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ARAR forecaster."""

__author__ = ["Akai01"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.arar import ARARForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_simple():
    """Test ARAR forecaster with simple time series."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster
    forecaster = ARARForecaster()
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3, 4, 5]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_with_params():
    """Test ARAR forecaster with custom parameters."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster with custom parameters
    forecaster = ARARForecaster(max_ar_depth=10, max_lag=15)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_prediction_intervals():
    """Test ARAR forecaster prediction intervals."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster
    forecaster = ARARForecaster()
    forecaster.fit(y)

    # Make predictions with intervals
    fh = [1, 2, 3]
    coverage = [0.80, 0.90]
    pred_int = forecaster.predict_interval(fh=fh, coverage=coverage)

    # Check prediction intervals
    assert isinstance(pred_int, pd.DataFrame)
    assert pred_int.shape[0] == len(fh)
    assert pred_int.shape[1] == len(coverage) * 2  # lower and upper for each coverage

    # Check that lower bounds are less than upper bounds
    for cov in coverage:
        assert all(pred_int[(0, cov, "lower")] <= pred_int[(0, cov, "upper")])


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_quantiles():
    """Test ARAR forecaster quantile predictions."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster
    forecaster = ARARForecaster()
    forecaster.fit(y)

    # Make quantile predictions
    fh = [1, 2, 3]
    alpha = [0.05, 0.5, 0.95]
    quantiles = forecaster.predict_quantiles(fh=fh, alpha=alpha)

    # Check quantiles
    assert isinstance(quantiles, pd.DataFrame)
    assert quantiles.shape[0] == len(fh)

    # Check that quantiles are ordered (0.05 <= 0.5 <= 0.95)
    var_name = y.name if y.name is not None else 0
    assert all(quantiles[(var_name, 0.05)] <= quantiles[(var_name, 0.5)])
    assert all(quantiles[(var_name, 0.5)] <= quantiles[(var_name, 0.95)])


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_short_series():
    """Test ARAR forecaster with short time series."""
    # Create a very short time series (should trigger warning)
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Fit the forecaster (should use safe mode and return mean fallback)
    forecaster = ARARForecaster(safe=True)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_airline_data():
    """Test ARAR forecaster with airline data."""
    from sktime.datasets import load_airline

    y = load_airline()

    # Fit the forecaster
    forecaster = ARARForecaster()
    forecaster.fit(y)

    # Make predictions
    fh = list(range(1, 13))  # 12 months ahead
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))

    # Check that predictions are reasonable (positive for airline data)
    assert all(y_pred > 0)


def _select_ar_lags_reference(gamma, max_ar_depth, max_lag):
    """Loop-based reference for ``_select_ar_lags``, one lstsq per lag triple."""
    best_sigma2, best_phi, best_lag = np.inf, np.zeros(4), (1, 0, 0, 0)
    for i in range(2, max_ar_depth - 1):
        for j in range(i + 1, max_ar_depth):
            for k in range(j + 1, max_ar_depth + 1):
                if k > max_lag:
                    continue
                A = np.full((4, 4), gamma[0])
                A[0, 1] = A[1, 0] = gamma[i - 1]
                A[0, 2] = A[2, 0] = gamma[j - 1]
                A[1, 2] = A[2, 1] = gamma[j - i]
                A[0, 3] = A[3, 0] = gamma[k - 1]
                A[1, 3] = A[3, 1] = gamma[k - i]
                A[2, 3] = A[3, 2] = gamma[k - j]
                b = np.array([gamma[1], gamma[i], gamma[j], gamma[k]])
                phi, *_ = np.linalg.lstsq(A, b, rcond=None)
                sigma2 = float(gamma[0] - np.dot(phi, b))
                if np.isfinite(sigma2) and sigma2 < best_sigma2:
                    best_sigma2, best_phi, best_lag = sigma2, phi, (1, i, j, k)
    return best_sigma2, best_phi, best_lag


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n", [15, 40, 100, 500])
@pytest.mark.parametrize("max_ar_depth, max_lag", [(4, 10), (13, 13), (26, 40)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_select_ar_lags_matches_loop_reference(n, max_ar_depth, max_lag, seed):
    """Vectorized lag selection gives the same result as the loop-based search."""
    from sktime.forecasting.arar._arar_forecaster import _select_ar_lags

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x = x - x.mean()
    # biased autocovariances, as computed in _fit_arar
    gamma = np.array(
        [np.sum(x[: n - lag] * x[lag:]) / n if lag < n else 0.0 for lag in range(41)]
    )

    sigma2, phi, lag = _select_ar_lags(gamma, max_ar_depth, max_lag)
    sigma2_ref, phi_ref, lag_ref = _select_ar_lags_reference(
        gamma, max_ar_depth, max_lag
    )

    assert lag == lag_ref
    np.testing.assert_allclose(sigma2, sigma2_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-8, atol=1e-10)
