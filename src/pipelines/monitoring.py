import argparse
import json
import logging
import logging.config
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.metrics import compute_metrics
from src.inference.predictor import EnergyForecaster

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")


def load_dataset(path):
    columns = pd.read_csv(path, nrows=0).columns
    if "timestamp" in columns:
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.read_csv(path)


def validate_monitoring_frame(df, required_columns):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for monitoring: {missing_columns}")


def build_feature_drift_report(reference_df, current_df, feature_cols):
    report = []
    for column in feature_cols:
        reference_values = reference_df[column].astype(float)
        current_values = current_df[column].astype(float)
        reference_std = max(float(reference_values.std(ddof=0)), 1e-8)
        current_std = float(current_values.std(ddof=0))
        mean_shift = abs(float(current_values.mean()) - float(reference_values.mean())) / reference_std
        std_ratio = current_std / reference_std
        report.append(
            {
                "feature": column,
                "reference_mean": round(float(reference_values.mean()), 4),
                "current_mean": round(float(current_values.mean()), 4),
                "reference_std": round(reference_std, 4),
                "current_std": round(current_std, 4),
                "mean_shift_zscore": round(mean_shift, 4),
                "std_ratio": round(std_ratio, 4),
                "drift_detected": bool(mean_shift > 0.5 or std_ratio > 1.5 or std_ratio < 0.67),
            }
        )
    return report


def build_error_report(forecaster, df, max_windows=48):
    total_windows = len(df) - forecaster.sequence_length - forecaster.forecast_horizon + 1
    if total_windows < 1:
        raise ValueError("Dataset does not contain enough rows for monitoring windows")
    if total_windows <= max_windows:
        window_starts = range(total_windows)
    else:
        window_starts = np.linspace(0, total_windows - 1, num=max_windows, dtype=int)
    actuals = []
    predictions = []
    for start in window_starts:
        history = df[forecaster.feature_cols].iloc[start : start + forecaster.sequence_length]
        actual = df["price"].iloc[
            start + forecaster.sequence_length : start + forecaster.sequence_length + forecaster.forecast_horizon
        ]
        predictions.append(forecaster.predict(history))
        actuals.append(actual.to_list())
    metrics = compute_metrics(np.array(actuals), np.array(predictions))
    metrics["window_count"] = len(actuals)
    metrics["total_window_count"] = total_windows
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run lightweight monitoring for the saved best model")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/best_model")
    parser.add_argument("--reference-csv", type=str, default="data/energy_prices.csv")
    parser.add_argument("--current-csv", type=str, default="data/energy_prices.csv")
    parser.add_argument("--output-dir", type=str, default="results/monitoring")
    parser.add_argument("--max-windows", type=int, default=48)
    args = parser.parse_args()

    forecaster = EnergyForecaster(args.artifacts_dir)
    reference_df = load_dataset(args.reference_csv)
    current_df = load_dataset(args.current_csv)
    validate_monitoring_frame(reference_df, [*forecaster.feature_cols, "price"])
    validate_monitoring_frame(current_df, [*forecaster.feature_cols, "price"])
    drift_report = build_feature_drift_report(reference_df, current_df, forecaster.feature_cols)
    error_report = {
        "reference": build_error_report(forecaster, reference_df, max_windows=args.max_windows),
        "current": build_error_report(forecaster, current_df, max_windows=args.max_windows),
    }
    summary = {
        "model_name": forecaster.model_name,
        "drifted_features": [item["feature"] for item in drift_report if item["drift_detected"]],
        "reference_rmse": error_report["reference"]["RMSE"],
        "current_rmse": error_report["current"]["RMSE"],
        "rmse_delta": round(error_report["current"]["RMSE"] - error_report["reference"]["RMSE"], 4),
        "evaluated_windows": error_report["current"]["window_count"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_drift.json").write_text(json.dumps(drift_report, indent=2))
    (output_dir / "error_report.json").write_text(json.dumps(error_report, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Monitoring reports saved to %s", output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
