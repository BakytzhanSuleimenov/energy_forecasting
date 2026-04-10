import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("energy_forecasting")


def load_config(config_path="config/default.yml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_data(
    n_days=730,
    freq="h",
    seed=42,
    output_path="data/energy_prices.csv",
):
    rng = np.random.default_rng(seed=seed)
    periods = n_days * 24
    dates = pd.date_range(start="2022-01-01", periods=periods, freq=freq)

    hours = np.arange(periods) % 24
    day_of_week = np.array([d.dayofweek for d in dates])
    month = np.array([d.month for d in dates])

    base_price = 50.0
    yearly_pattern = 15 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))
    daily_pattern = 10 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly_pattern = 5 * np.where(day_of_week >= 5, -1, 1)
    trend = 0.002 * np.arange(periods)
    noise = rng.normal(0, 3, periods)
    spikes = np.zeros(periods)
    spike_indices = rng.choice(periods, size=int(periods * 0.01), replace=False)
    spikes[spike_indices] = rng.uniform(20, 60, len(spike_indices))

    price = (
        base_price + yearly_pattern + daily_pattern + weekly_pattern + trend + noise + spikes
    )
    price = np.maximum(price, 5.0)

    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24) - np.pi / 2)
    temperature += rng.normal(0, 2, periods)

    demand = 30000 + 5000 * np.sin(2 * np.pi * (hours - 8) / 24)
    demand += 3000 * np.where(day_of_week >= 5, -1, 1)
    demand += 2000 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))
    demand += rng.normal(0, 500, periods)

    wind_generation = 2000 + 1500 * np.abs(np.sin(2 * np.pi * np.arange(periods) / (7 * 24)))
    wind_generation += rng.normal(0, 300, periods)
    wind_generation = np.maximum(wind_generation, 0)

    solar_generation = np.maximum(
        3000 * np.sin(np.pi * np.clip((hours - 6) / 12, 0, 1)) *
        (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))),
        0,
    )
    solar_generation += rng.normal(0, 100, periods)
    solar_generation = np.maximum(solar_generation, 0)

    gas_price = 25 + 5 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))
    gas_price += rng.normal(0, 1, periods)
    gas_price += 0.001 * np.arange(periods)

    df = pd.DataFrame({
        "timestamp": dates,
        "price": np.round(price, 2),
        "temperature": np.round(temperature, 2),
        "demand": np.round(demand, 2),
        "wind_generation": np.round(wind_generation, 2),
        "solar_generation": np.round(solar_generation, 2),
        "gas_price": np.round(gas_price, 2),
        "hour": hours,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": (day_of_week >= 5).astype(int),
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated synthetic dataset with %d samples -> %s", len(df), output_path)
    return df


def load_data(data_path="data/energy_prices.csv"):
    if not Path(data_path).exists():
        logger.warning("Dataset not found at %s, generating synthetic data...", data_path)
        return generate_synthetic_data(output_path=data_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    logger.info("Loaded dataset with %d samples from %s", len(df), data_path)
    return df


def prepare_features(df, sequence_length=72, forecast_horizon=24):
    feature_cols = [
        "price", "temperature", "demand", "wind_generation",
        "solar_generation", "gas_price", "hour", "day_of_week",
        "month", "is_weekend",
    ]
    target_col = "price"

    data = df[feature_cols].values
    target = df[target_col].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    data_scaled = scaler_X.fit_transform(data)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()

    return data_scaled, target_scaled, scaler_X, scaler_y, feature_cols


def create_sequences(data, target, sequence_length=72, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i : i + sequence_length])
        y.append(target[i + sequence_length : i + sequence_length + forecast_horizon])
    return np.array(X), np.array(y)


def create_tabular_features(data, target, sequence_length=72, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq = data[i : i + sequence_length]
        features = np.concatenate([
            seq[-1],
            seq.mean(axis=0),
            seq.std(axis=0),
            seq[-1] - seq[-24] if sequence_length >= 24 else np.zeros(seq.shape[1]),
            seq[-1] - seq[-48] if sequence_length >= 48 else np.zeros(seq.shape[1]),
        ])
        X.append(features)
        y.append(target[i + sequence_length : i + sequence_length + forecast_horizon])
    return np.array(X), np.array(y)


def split_data(X, y, test_ratio=0.2, validation_ratio=0.1):
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * validation_ratio)
    train_size = n - test_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]
    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    return X_train, y_train, X_val, y_val, X_test, y_test
