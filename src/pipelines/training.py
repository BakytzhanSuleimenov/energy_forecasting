import argparse
import json
import logging
import logging.config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from src.common.data import (
    create_sequences,
    create_tabular_features,
    load_config,
    load_data,
    prepare_features,
    split_data,
)
from src.common.metrics import compute_metrics, compute_metrics_per_horizon
from src.models.dnn import DNNModel
from src.models.lstm import LSTMModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "dnn": DNNModel,
    "lstm": LSTMModel,
}


def train_single_model(model_name, config, data_scaled, target_scaled, seq_len, horizon, test_ratio, val_ratio):
    logger.info("=" * 60)
    logger.info("Training model: %s", model_name)
    logger.info("=" * 60)

    model_config = config.get("models", {}).get(model_name, {})
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(config=model_config)

    if model_name in ("random_forest", "xgboost"):
        X, y = create_tabular_features(data_scaled, target_scaled, seq_len, horizon)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_ratio, val_ratio)
        model.build(input_shape=X_train.shape[1], output_shape=horizon)
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
    else:
        X, y = create_sequences(data_scaled, target_scaled, seq_len, horizon)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_ratio, val_ratio)
        if model_name == "lstm":
            model.build(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=horizon)
            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_test)
        else:
            flat_shape = X_train.shape[1] * X_train.shape[2]
            model.build(input_shape=flat_shape, output_shape=horizon)
            model.fit(
                X_train.reshape(X_train.shape[0], -1), y_train,
                X_val.reshape(X_val.shape[0], -1) if X_val is not None else None, y_val,
            )
            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

    overall_metrics = compute_metrics(y_test, y_pred)
    horizon_metrics = compute_metrics_per_horizon(y_test, y_pred)
    overall_metrics["training_time"] = round(model.get_training_time(), 2)

    logger.info("Results for %s:", model_name)
    for k, v in overall_metrics.items():
        logger.info("  %s: %s", k, v)

    return {
        "model_name": model_name,
        "overall_metrics": overall_metrics,
        "horizon_metrics": horizon_metrics,
        "predictions": y_pred.tolist(),
        "actuals": y_test.tolist(),
        "training_history": (
            model.get_training_history()
            if hasattr(model, "get_training_history") else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Energy Price Forecasting Training")
    parser.add_argument("--model", type=str, default=None, help="Train specific model")
    parser.add_argument("--config", type=str, default="config/default.yml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    seq_len = data_config.get("sequence_length", 72)
    horizon = data_config.get("forecast_horizon", 24)
    test_ratio = data_config.get("test_ratio", 0.2)
    val_ratio = data_config.get("validation_ratio", 0.1)

    logger.info("Loading energy price data...")
    df = load_data()
    data_scaled, target_scaled, scaler_X, scaler_y, feature_cols = prepare_features(
        df, seq_len, horizon,
    )
    logger.info("Data shape: %s, Features: %d", data_scaled.shape, len(feature_cols))

    models_to_train = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    results = []

    for model_name in models_to_train:
        if model_name not in MODEL_REGISTRY:
            logger.error("Unknown model: %s. Available: %s", model_name, list(MODEL_REGISTRY.keys()))
            continue
        result = train_single_model(
            model_name, config, data_scaled, target_scaled,
            seq_len, horizon, test_ratio, val_ratio,
        )
        results.append(result)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    serializable_results = []
    for r in results:
        sr = {
            "model_name": r["model_name"],
            "overall_metrics": r["overall_metrics"],
            "horizon_metrics": r["horizon_metrics"],
            "predictions": r["predictions"],
            "actuals": r["actuals"],
        }
        if r.get("training_history"):
            sr["training_history"] = {
                k: [float(v) for v in vals]
                for k, vals in r["training_history"].items()
            }
        serializable_results.append(sr)

    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info("Results saved to results/benchmark_results.json")

    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    header = f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R2':<10} {'Time(s)':<10}"
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        m = r["overall_metrics"]
        row = (
            f"{r['model_name']:<20} {m['MAE']:<10} {m['RMSE']:<10} "
            f"{m['MAPE']:<10} {m['R2']:<10} {m['training_time']:<10}"
        )
        logger.info(row)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
