from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data import (
    create_sequences,
    create_tabular_features,
    generate_synthetic_data,
    prepare_features,
    split_data,
)


def test_generate_synthetic_data(tmp_path):
    output = tmp_path / "test_data.csv"
    df = generate_synthetic_data(n_days=30, output_path=str(output))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30 * 24
    assert "price" in df.columns
    assert "timestamp" in df.columns
    assert "temperature" in df.columns
    assert "demand" in df.columns
    assert output.exists()


def test_generate_synthetic_data_columns():
    df = generate_synthetic_data(n_days=7, output_path="data/test_tmp.csv")
    expected_cols = [
        "timestamp", "price", "temperature", "demand",
        "wind_generation", "solar_generation", "gas_price",
        "hour", "day_of_week", "month", "is_weekend",
    ]
    for col in expected_cols:
        assert col in df.columns
    Path("data/test_tmp.csv").unlink(missing_ok=True)


def test_generate_synthetic_data_values():
    df = generate_synthetic_data(n_days=7, output_path="data/test_tmp2.csv")
    assert (df["price"] >= 5.0).all()
    assert df["hour"].min() == 0
    assert df["hour"].max() == 23
    assert df["is_weekend"].isin([0, 1]).all()
    Path("data/test_tmp2.csv").unlink(missing_ok=True)


def test_prepare_features():
    df = generate_synthetic_data(n_days=10, output_path="data/test_tmp3.csv")
    data_scaled, target_scaled, scaler_X, scaler_y, feature_cols = prepare_features(df)
    assert data_scaled.shape[0] == len(df)
    assert len(feature_cols) == 10
    assert abs(data_scaled.mean(axis=0)).max() < 1.0
    Path("data/test_tmp3.csv").unlink(missing_ok=True)


def test_create_sequences():
    n = 200
    n_features = 5
    data = np.random.randn(n, n_features)
    target = np.random.randn(n)
    seq_len = 24
    horizon = 12

    X, y = create_sequences(data, target, seq_len, horizon)
    assert X.shape[1] == seq_len
    assert X.shape[2] == n_features
    assert y.shape[1] == horizon
    assert len(X) == len(y)
    assert len(X) == n - seq_len - horizon + 1


def test_create_tabular_features():
    n = 200
    n_features = 5
    data = np.random.randn(n, n_features)
    target = np.random.randn(n)
    seq_len = 48
    horizon = 12

    X, y = create_tabular_features(data, target, seq_len, horizon)
    assert X.ndim == 2
    assert y.shape[1] == horizon
    assert len(X) == len(y)


def test_split_data():
    X = np.random.randn(100, 10)
    y = np.random.randn(100, 5)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, 0.2, 0.1)
    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(X_test) == 20
    assert len(X_val) == 10
    assert len(X_train) == 70
