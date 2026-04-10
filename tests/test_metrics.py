import numpy as np
import pytest

from src.common.metrics import compute_metrics, compute_metrics_per_horizon


def test_compute_metrics_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


def test_compute_metrics_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.5
    assert metrics["RMSE"] == pytest.approx(0.5, abs=0.001)
    assert "R2" in metrics
    assert "MAPE" in metrics
    assert "sMAPE" in metrics


def test_compute_metrics_keys():
    y_true = np.random.randn(50)
    y_pred = np.random.randn(50)
    metrics = compute_metrics(y_true, y_pred)
    expected_keys = {"MAE", "RMSE", "MAPE", "sMAPE", "R2"}
    assert set(metrics.keys()) == expected_keys


def test_compute_metrics_per_horizon():
    y_true = np.random.randn(50, 10)
    y_pred = np.random.randn(50, 10)
    metrics_list = compute_metrics_per_horizon(y_true, y_pred)
    assert len(metrics_list) == 10
    assert metrics_list[0]["horizon_step"] == 1
    assert metrics_list[9]["horizon_step"] == 10
    for m in metrics_list:
        assert "MAE" in m
        assert "RMSE" in m
