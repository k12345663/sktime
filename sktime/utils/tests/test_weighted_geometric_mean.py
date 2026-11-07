# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for weighted geometric mean axis alignment."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sktime.utils.stats import _weighted_geometric_mean


@pytest.mark.parametrize("shape", [(3, 2), (2, 2), (3, 1), (3,)])
@pytest.mark.parametrize("axis", [0, -1])
def test_weighted_geometric_mean_axis(shape, axis):
    """One-dimensional weights apply along the requested axis."""
    y = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape)
    weights = np.arange(1, y.shape[axis] + 1, dtype=float)
    expected = np.exp(np.average(np.log(y), weights=weights, axis=axis))

    actual = _weighted_geometric_mean(y, weights=weights, axis=axis)

    assert_allclose(actual, expected)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_weighted_geometric_mean_elementwise_weights(axis):
    """Same-shaped weights retain their elementwise meaning."""
    y = np.array([[1.0, 2.0], [3.0, 8.0], [4.0, 16.0]])
    weights = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])
    expected = np.exp(np.average(np.log(y), weights=weights, axis=axis))

    actual = _weighted_geometric_mean(y, weights=weights, axis=axis)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "metric_name",
    [
        "geometric_mean_absolute_error",
        "geometric_mean_squared_error",
        "geometric_mean_relative_absolute_error",
        "geometric_mean_relative_squared_error",
    ],
)
@pytest.mark.parametrize("n_samples", [2, 3])
def test_geometric_metric_multivariate_horizon_weights(metric_name, n_samples):
    """Public metrics weight time points, independently for each output."""
    from sktime.performance_metrics import forecasting

    metric = getattr(forecasting, metric_name)
    y_true = np.zeros((n_samples, 2))
    y_pred = np.arange(1, n_samples * 2 + 1, dtype=float).reshape(n_samples, 2)
    weights = np.arange(1, n_samples + 1, dtype=float)
    kwargs = {}
    errors = y_pred.copy()
    if "relative" in metric_name:
        kwargs["y_pred_benchmark"] = np.full_like(y_pred, 2.0)
        errors /= 2.0
    if "squared" in metric_name:
        errors **= 2
    expected = np.exp(np.average(np.log(errors), weights=weights, axis=0))

    actual = metric(
        y_true, y_pred, horizon_weight=weights, multioutput="raw_values", **kwargs
    )

    assert_allclose(actual, expected)
