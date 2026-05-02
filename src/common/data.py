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
    include_legacy_columns=True,
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

    wind_speed = 6 + 3 * np.abs(np.sin(2 * np.pi * np.arange(periods) / (7 * 24)))
    wind_speed += rng.normal(0, 1.2, periods)
    wind_speed = np.maximum(wind_speed, 0)

    solar_irradiation = np.maximum(
        500 * np.sin(np.pi * np.clip((hours - 6) / 12, 0, 1))
        * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))),
        0,
    )
    solar_irradiation += rng.normal(0, 20, periods)
    solar_irradiation = np.maximum(solar_irradiation, 0)

    gas_price = 25 + 5 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))
    gas_price += rng.normal(0, 1, periods)
    gas_price += 0.001 * np.arange(periods)

    df = pd.DataFrame({
        "timestamp": dates,
        "price": np.round(price, 2),
        "temperature": np.round(temperature, 2),
        "demand": np.round(demand, 2),
        "wind_speed": np.round(wind_speed, 2),
        "solar_irradiation": np.round(solar_irradiation, 2),
        "gas_price": np.round(gas_price, 2),
        "hour": hours,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": (day_of_week >= 5).astype(int),
    })
    if include_legacy_columns:
        df["wind_generation"] = np.round(df["wind_speed"] * 300, 2)
        df["solar_generation"] = np.round(df["solar_irradiation"] * 6, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated synthetic dataset with %d samples -> %s", len(df), output_path)
    return df


def fetch_weather_data(start_date: str, end_date: str) -> pd.DataFrame:
    import requests as req

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude=53.35&longitude=-6.26"
        f"&hourly=temperature_2m,wind_speed_10m,shortwave_radiation"
        f"&start_date={start_date}&end_date={end_date}"
        "&timezone=UTC"
    )
    logger.info("Fetching Dublin weather from Open-Meteo (%s to %s)...", start_date, end_date)
    resp = req.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    times = pd.to_datetime(data["hourly"]["time"]).tz_localize("UTC")
    weather = pd.DataFrame(
        {
            "temperature": data["hourly"].get("temperature_2m", []),
            "wind_speed": data["hourly"].get("wind_speed_10m", []),
            "solar_irradiation": data["hourly"].get("shortwave_radiation", []),
        },
        index=times,
    )
    return weather


def fetch_real_data(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    output_path: str = "data/energy_prices.csv",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch real Ireland electricity market data and save to CSV.

    Sources:
    - ENTSOE Transparency Platform (entsoe-py): day-ahead prices, system load,
      wind + solar generation for the IE bidding zone.
    - Open-Meteo Historical Weather API: hourly temperature, wind speed,
      and shortwave radiation for Dublin (free, no key).
    - Synthetic gas price proxy correlated with electricity price (no public IE TTF API).

    Args:
        start_date: First date to fetch (YYYY-MM-DD, inclusive).
        end_date:   Last date to fetch  (YYYY-MM-DD, inclusive).
        output_path: Destination CSV path.
        api_key: ENTSOE REST API key.  Falls back to ENTSOE_API_KEY env var.

    Returns:
        DataFrame with the same schema as generate_synthetic_data().

    Raises:
        ValueError: If no ENTSOE API key is available.
    """
    import os

    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError("entsoe-py is not installed. Run: uv add entsoe-py") from exc

    api_key = api_key or os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "ENTSOE API key required. Set the ENTSOE_API_KEY environment variable.\n"
            "Register at https://transparency.entsoe.eu and email transparency@entsoe.eu "
            "with subject 'Restful API access'."
        )

    client = EntsoePandasClient(api_key=api_key)
    country = "IE_SEM"
    start = pd.Timestamp(start_date, tz="Europe/Dublin")
    end = pd.Timestamp(end_date, tz="Europe/Dublin") + pd.Timedelta(days=1)

    prices: pd.Series | None = None
    logger.info("Fetching ENTSOE day-ahead prices for Ireland (%s – %s)...", start_date, end_date)
    try:
        prices = client.query_day_ahead_prices(country, start=start, end=end).tz_convert("UTC")
        prices.name = "price"
        logger.info("Fetched %d price records from ENTSOE.", len(prices))
    except Exception as exc:
        logger.warning("Could not fetch day-ahead prices (%s) – will use synthetic price proxy.", exc)

    load_series: pd.Series | None = None
    try:
        logger.info("Fetching ENTSOE actual load for Ireland...")
        raw_load = client.query_load(country, start=start, end=end)
        raw_load = raw_load.tz_convert("UTC")
        load_series = raw_load.iloc[:, 0] if isinstance(raw_load, pd.DataFrame) else raw_load
        load_series.name = "demand"
    except Exception as exc:
        logger.warning("Could not fetch load data (%s) – using synthetic proxy.", exc)

    wind_series: pd.Series | None = None
    solar_series: pd.Series | None = None
    try:
        logger.info("Fetching ENTSOE actual generation for Ireland...")
        gen = client.query_generation(country, start=start, end=end).tz_convert("UTC")
        wind_cols = [c for c in gen.columns if "Wind" in str(c)]
        solar_cols = [c for c in gen.columns if "Solar" in str(c)]
        if wind_cols:
            wind_series = gen[wind_cols].sum(axis=1)
            wind_series.name = "wind_generation"
        if solar_cols:
            solar_series = gen[solar_cols].sum(axis=1)
            solar_series.name = "solar_generation"
    except Exception as exc:
        logger.warning("Could not fetch generation data (%s) – trying wind/solar forecast.", exc)
        try:
            ws = client.query_wind_and_solar_forecast(country, start=start, end=end).tz_convert("UTC")
            w_cols = [c for c in ws.columns if "Wind" in str(c)]
            s_cols = [c for c in ws.columns if "Solar" in str(c)]
            if w_cols:
                wind_series = ws[w_cols[0]]
                wind_series.name = "wind_generation"
            if s_cols:
                solar_series = ws[s_cols[0]]
                solar_series.name = "solar_generation"
        except Exception as exc2:
            logger.warning("Could not fetch wind/solar forecast (%s) – using synthetic proxy.", exc2)

    weather_df = fetch_weather_data(start_date, end_date)

    hourly_index = pd.date_range(
        start=pd.Timestamp(start_date, tz="UTC"),
        end=pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        freq="h",
    )

    def _to_hourly(s: pd.Series) -> pd.Series:
        """Resample any series to clean hourly UTC, handling sub-hourly resolution and DST dupes."""
        s = s[~s.index.duplicated(keep="first")]
        return s.resample("h").mean().reindex(hourly_index)

    if prices is not None:
        df = pd.DataFrame({"price": _to_hourly(prices)})
    else:
        df = pd.DataFrame(index=hourly_index)

    if weather_df is not None and not weather_df.empty:
        for weather_col in ["temperature", "wind_speed", "solar_irradiation"]:
            if weather_col in weather_df.columns:
                weather_series = weather_df[weather_col]
                weather_series.name = weather_col
                df[weather_col] = _to_hourly(weather_series)

    for series in [load_series, wind_series, solar_series]:
        if series is not None:
            df[series.name] = _to_hourly(series)

    n = len(df)
    rng = np.random.default_rng(42)
    t = np.arange(n)
    hours = df.index.hour.values
    dow = df.index.dayofweek.values
    months = df.index.month.values

    if "demand" not in df.columns or df["demand"].isna().all():
        logger.info("Using synthetic proxy for demand (Ireland ~3–4 GW mean).")
        df["demand"] = np.round(
            3500
            + 500 * np.sin(2 * np.pi * (hours - 8) / 24)
            + 300 * np.where(dow >= 5, -1, 1)
            + 2000 * np.sin(2 * np.pi * t / (365.25 * 24))
            + rng.normal(0, 100, n),
            2,
        )

    if "wind_speed" not in df.columns or df["wind_speed"].isna().all():
        logger.info("Using synthetic proxy for wind speed.")
        df["wind_speed"] = np.round(
            np.maximum(6 + 3 * np.abs(np.sin(2 * np.pi * t / (7 * 24))) + rng.normal(0, 1.2, n), 0),
            2,
        )

    if "solar_irradiation" not in df.columns or df["solar_irradiation"].isna().all():
        logger.info("Using synthetic proxy for solar irradiation.")
        df["solar_irradiation"] = np.round(
            np.maximum(
                500 * np.sin(np.pi * np.clip((hours - 6) / 12, 0, 1))
                * (0.5 + 0.5 * np.sin(2 * np.pi * (months - 1) / 12)),
                0,
            )
            + rng.normal(0, 20, n),
            2,
        )
        df["solar_irradiation"] = np.maximum(df["solar_irradiation"], 0)

    if "temperature" not in df.columns or df["temperature"].isna().all():
        logger.info("Using synthetic proxy for temperature.")
        df["temperature"] = np.round(
            10 + 7 * np.sin(2 * np.pi * t / (365.25 * 24) - np.pi / 2) + rng.normal(0, 2, n), 2
        )

    if "wind_generation" not in df.columns:
        df["wind_generation"] = np.round(df["wind_speed"] * 300, 2)

    if "solar_generation" not in df.columns:
        df["solar_generation"] = np.round(df["solar_irradiation"] * 6, 2)

    if "price" not in df.columns or df["price"].isna().all():
        logger.info("Using synthetic proxy for electricity price (IE ~€60-120/MWh).")
        df["price"] = np.round(
            np.maximum(
                70
                + 30 * np.sin(2 * np.pi * (hours - 8) / 24)
                + 20 * np.sin(2 * np.pi * t / (365.25 * 24))
                + rng.normal(0, 15, n),
                0.0,
            ),
            2,
        )

    price_vals = df["price"].fillna(df["price"].median()).values
    gas = (
        30.0
        + 10 * np.sin(2 * np.pi * t / (365.25 * 24) + np.pi)
        + rng.normal(0, 2, n)
        + 0.1 * (price_vals - price_vals.mean())
    )
    df["gas_price"] = np.round(np.maximum(gas, 5.0), 2)

    df["timestamp"] = df.index
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.dropna(subset=["price"])
    df = df.ffill().bfill()
    df = df.reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved real Ireland dataset with %d samples -> %s", len(df), output_path)
    return df


def load_data(data_path="data/energy_prices.csv"):
    if not Path(data_path).exists():
        logger.warning("Dataset not found at %s, generating synthetic data...", data_path)
        return generate_synthetic_data(output_path=data_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    if "wind_speed" not in df.columns and "wind_generation" in df.columns:
        df["wind_speed"] = np.round(df["wind_generation"] / 300, 2)
    if "solar_irradiation" not in df.columns and "solar_generation" in df.columns:
        df["solar_irradiation"] = np.round(df["solar_generation"] / 6, 2)
    logger.info("Loaded dataset with %d samples from %s", len(df), data_path)
    return df


BASE_FEATURE_COLUMNS = [
    "price",
    "demand",
    "wind_speed",
    "solar_irradiation",
    "gas_price",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS


def prepare_features(df, sequence_length=72, forecast_horizon=24):
    feature_cols = BASE_FEATURE_COLUMNS
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


def prepare_data_pipeline(
    start_date="2022-01-01",
    end_date="2024-12-31",
    output_path="data/prepared_energy_data.csv",
    use_real_data=True,
):
    if use_real_data:
        try:
            return fetch_real_data(start_date=start_date, end_date=end_date, output_path=output_path)
        except ValueError:
            logger.warning("ENTSOE key unavailable, falling back to synthetic data for preparation pipeline")
    return generate_synthetic_data(output_path=output_path)


def feature_engineering_pipeline(
    data_path="data/prepared_energy_data.csv",
    output_path="data/feature_engineered_energy_data.csv",
):
    df = load_data(data_path)
    required = [
        "timestamp",
        "price",
        "demand",
        "wind_speed",
        "solar_irradiation",
        "gas_price",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    engineered = df[required].copy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(output_path, index=False)
    logger.info("Saved feature-engineered dataset with %d samples -> %s", len(engineered), output_path)
    return engineered
