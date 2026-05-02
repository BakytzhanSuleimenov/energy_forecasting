import argparse
import contextlib
import itertools
import json
import logging
import logging.config
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.data import (
    feature_engineering_pipeline,
    load_config,
    load_data,
    prepare_data_pipeline,
    prepare_features,
    split_data,
)
from src.common.metrics import compute_metrics, compute_metrics_per_horizon
from src.common.mlflow_utils import (
    is_mlflow_enabled,
    log_training_run,
    promote_best_to_staging,
    register_best_model,
    setup_experiment,
)
from src.inference import EnergyForecaster
from src.inference.artifacts import save_inference_artifacts
from src.models.dnn import DNNModel
from src.models.lstm import LSTMModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.pipelines.benchmarking import generate_comparison_table, generate_horizon_comparison, rank_models
from src.pipelines.experiment_templates import (
    get_tuning_defaults,
    get_tuning_grid,
    resolve_experiment_template,
)
from src.pipelines.monitoring import (
    build_error_report,
    build_feature_drift_report,
    load_dataset,
    validate_monitoring_frame,
)

try:
    from metaflow import FlowSpec, Parameter, step

    METAFLOW_AVAILABLE = True
except ImportError:
    FlowSpec = object

    def Parameter(*args, **kwargs):
        return None

    def step(func):
        return func

    METAFLOW_AVAILABLE = False

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "dnn": DNNModel,
    "lstm": LSTMModel,
}


def _noop_parameter(*args, **kwargs):
    return None


def _build_sequence_data(model_name, data_scaled, target_scaled, seq_len, horizon, test_ratio, val_ratio):
    from src.common.data import create_sequences, create_tabular_features

    if model_name in ("random_forest", "xgboost"):
        X, y = create_tabular_features(data_scaled, target_scaled, seq_len, horizon)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_ratio, val_ratio)
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "model_input_shape": X_train.shape[1],
            "reshape_for_dnn": False,
        }

    X, y = create_sequences(data_scaled, target_scaled, seq_len, horizon)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_ratio, val_ratio)
    if model_name == "lstm":
        model_input_shape = (X_train.shape[1], X_train.shape[2])
        reshape_for_dnn = False
    else:
        model_input_shape = X_train.shape[1] * X_train.shape[2]
        reshape_for_dnn = True

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "model_input_shape": model_input_shape,
        "reshape_for_dnn": reshape_for_dnn,
    }


def _coerce_json_dict(raw_value, fallback=None):
    if fallback is None:
        fallback = {}
    if not raw_value:
        return fallback
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def _merge_dict(base, override):
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged


def _iter_tuning_configs(base_config, grid_config, max_trials):
    if not grid_config:
        return [base_config]
    keys = list(grid_config.keys())
    values = [grid_config[k] for k in keys]
    combinations = []
    for combo in itertools.product(*values):
        trial = dict(base_config)
        for key, value in zip(keys, combo, strict=False):
            trial[key] = value
        combinations.append(trial)
    return combinations[:max_trials]


def _train_once(model_name, model_config, sequence_data, horizon):
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(config=model_config)
    model.build(input_shape=sequence_data["model_input_shape"], output_shape=horizon)

    if sequence_data["reshape_for_dnn"]:
        X_val_flat = (
            sequence_data["X_val"].reshape(sequence_data["X_val"].shape[0], -1)
            if sequence_data["X_val"] is not None
            else None
        )
        model.fit(
            sequence_data["X_train"].reshape(sequence_data["X_train"].shape[0], -1),
            sequence_data["y_train"],
            X_val_flat,
            sequence_data["y_val"],
        )
        y_pred = model.predict(sequence_data["X_test"].reshape(sequence_data["X_test"].shape[0], -1))
    else:
        model.fit(
            sequence_data["X_train"],
            sequence_data["y_train"],
            sequence_data["X_val"],
            sequence_data["y_val"],
        )
        y_pred = model.predict(sequence_data["X_test"])

    y_test = sequence_data["y_test"]
    overall_metrics = compute_metrics(y_test, y_pred)
    horizon_metrics = compute_metrics_per_horizon(y_test, y_pred)
    overall_metrics["training_time"] = round(model.get_training_time(), 2)

    return {
        "model": model,
        "predictions": y_pred,
        "actuals": y_test,
        "overall_metrics": overall_metrics,
        "horizon_metrics": horizon_metrics,
    }


def train_single_model(
    model_name,
    config,
    data_scaled,
    target_scaled,
    seq_len,
    horizon,
    test_ratio,
    val_ratio,
    model_config,
    tuning_enabled,
    tuning_grid,
    max_trials_per_model,
):
    logger.info("=" * 60)
    logger.info("Training model: %s", model_name)
    logger.info("=" * 60)

    sequence_data = _build_sequence_data(
        model_name,
        data_scaled,
        target_scaled,
        seq_len,
        horizon,
        test_ratio,
        val_ratio,
    )

    candidate_configs = [model_config]
    if tuning_enabled:
        candidate_configs = _iter_tuning_configs(model_config, tuning_grid, max_trials_per_model)

    best_trial = None
    all_trials = []

    for trial_index, trial_config in enumerate(candidate_configs, start=1):
        trial_name = f"{model_name}-trial-{trial_index}"
        logger.info("Running %s with config: %s", trial_name, trial_config)
        trial_result = _train_once(model_name, trial_config, sequence_data, horizon)
        trial_result["trial_name"] = trial_name
        trial_result["model_config"] = trial_config
        all_trials.append(trial_result)

        log_training_run(
            trial_result["model"],
            model_name,
            _merge_dict(config.get("models", {}).get(model_name, {}), {"trial_name": trial_name, **trial_config}),
            seq_len,
            horizon,
            test_ratio,
            val_ratio,
            trial_result["overall_metrics"],
            trial_result["horizon_metrics"],
        )

        if best_trial is None or trial_result["overall_metrics"]["RMSE"] < best_trial["overall_metrics"]["RMSE"]:
            best_trial = trial_result

    overall_metrics = best_trial["overall_metrics"]
    logger.info("Best trial for %s: %s", model_name, best_trial["trial_name"])
    for key, value in overall_metrics.items():
        logger.info("  %s: %s", key, value)

    return {
        "model_name": model_name,
        "model_config": best_trial["model_config"],
        "overall_metrics": best_trial["overall_metrics"],
        "horizon_metrics": best_trial["horizon_metrics"],
        "predictions": best_trial["predictions"].tolist(),
        "actuals": best_trial["actuals"].tolist(),
        "training_history": (
            best_trial["model"].get_training_history()
            if hasattr(best_trial["model"], "get_training_history")
            else None
        ),
        "model_obj": best_trial["model"],
        "selected_trial": best_trial["trial_name"],
        "trial_count": len(all_trials),
    }


def _serialize_results(results):
    serializable_results = []
    for result in results:
        serialized = {
            "model_name": result["model_name"],
            "overall_metrics": result["overall_metrics"],
            "horizon_metrics": result["horizon_metrics"],
            "predictions": result["predictions"],
            "actuals": result["actuals"],
            "selected_trial": result.get("selected_trial"),
            "trial_count": result.get("trial_count"),
        }
        if result.get("training_history"):
            serialized["training_history"] = {
                key: [float(value) for value in values]
                for key, values in result["training_history"].items()
            }
        serializable_results.append(serialized)
    return serializable_results


def _save_benchmark_artifacts(results, results_dir):
    results_dir.mkdir(exist_ok=True)
    serializable_results = _serialize_results(results)

    with open(results_dir / "benchmark_results.json", "w") as handle:
        json.dump(serializable_results, handle, indent=2)

    comparison_df = generate_comparison_table(serializable_results)
    rankings, overall_ranking = rank_models(serializable_results)
    horizon_data = generate_horizon_comparison(serializable_results)

    horizon_rows = []
    for model_name, metrics_list in horizon_data.items():
        for metrics in metrics_list:
            row = {"Model": model_name}
            row.update(metrics)
            horizon_rows.append(row)

    comparison_df.to_csv(results_dir / "comparison_table.csv", index=False)

    import pandas as pd

    pd.DataFrame(horizon_rows).to_csv(results_dir / "horizon_comparison.csv", index=False)

    summary = {
        "rankings": rankings,
        "overall_ranking": overall_ranking,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }
    with open(results_dir / "benchmark_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _run_monitoring(reference_csv, current_csv, artifacts_dir, output_dir, max_windows=48):
    forecaster = EnergyForecaster(artifacts_dir)
    reference_df = load_dataset(reference_csv)
    current_df = load_dataset(current_csv)
    validate_monitoring_frame(reference_df, [*forecaster.feature_cols, "price"])
    validate_monitoring_frame(current_df, [*forecaster.feature_cols, "price"])

    drift_report = build_feature_drift_report(reference_df, current_df, forecaster.feature_cols)
    error_report = {
        "reference": build_error_report(forecaster, reference_df, max_windows=max_windows),
        "current": build_error_report(forecaster, current_df, max_windows=max_windows),
    }
    summary = {
        "model_name": forecaster.model_name,
        "drifted_features": [item["feature"] for item in drift_report if item["drift_detected"]],
        "reference_rmse": error_report["reference"]["RMSE"],
        "current_rmse": error_report["current"]["RMSE"],
        "rmse_delta": round(error_report["current"]["RMSE"] - error_report["reference"]["RMSE"], 4),
        "evaluated_windows": error_report["current"]["window_count"],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "feature_drift.json").write_text(json.dumps(drift_report, indent=2))
    (output_path / "error_report.json").write_text(json.dumps(error_report, indent=2))
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_full_pipeline(
    config_path,
    selected_model,
    template_name,
    tuning_override,
    max_trials_override,
    start_date,
    end_date,
    prepared_output,
    engineered_output,
):
    config = load_config(config_path)
    data_config = config.get("data", {})
    seq_len = data_config.get("sequence_length", 72)
    horizon = data_config.get("forecast_horizon", 24)
    test_ratio = data_config.get("test_ratio", 0.2)
    val_ratio = data_config.get("validation_ratio", 0.1)

    template = resolve_experiment_template(config, template_name)
    tuning_defaults = get_tuning_defaults(config)
    tuning_enabled = tuning_override if tuning_override is not None else (
        template["tuning_enabled"] or tuning_defaults["enabled"]
    )
    max_trials_per_model = max_trials_override if max_trials_override is not None else tuning_defaults[
        "max_trials_per_model"
    ]

    setup_experiment(f"energy-forecasting-{template['name']}")

    use_real = bool(os.environ.get("ENTSOE_API_KEY"))
    prepare_data_pipeline(
        start_date=start_date,
        end_date=end_date,
        output_path=prepared_output,
        use_real_data=use_real,
    )
    feature_engineering_pipeline(data_path=prepared_output, output_path=engineered_output)

    df = load_data(engineered_output)
    data_scaled, target_scaled, scaler_X, scaler_y, feature_cols = prepare_features(df, seq_len, horizon)
    logger.info("Data shape: %s, Features: %d", data_scaled.shape, len(feature_cols))

    available_models = list(MODEL_REGISTRY.keys())
    template_models = [m for m in template["models"] if m in available_models]
    if selected_model:
        models_to_train = [selected_model] if selected_model in template_models else []
    else:
        models_to_train = template_models

    tuning_grids = get_tuning_grid(config)
    results = []

    for model_name in models_to_train:
        model_config = template["model_configs"].get(model_name, config.get("models", {}).get(model_name, {}))
        tuning_grid = tuning_grids.get(model_name, {}) if tuning_enabled else {}

        if is_mlflow_enabled():
            import mlflow

            run_ctx = mlflow.start_run(run_name=f"{model_name}-{template['name']}")
        else:
            run_ctx = contextlib.nullcontext()

        with run_ctx:
            result = train_single_model(
                model_name=model_name,
                config=config,
                data_scaled=data_scaled,
                target_scaled=target_scaled,
                seq_len=seq_len,
                horizon=horizon,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                model_config=model_config,
                tuning_enabled=tuning_enabled,
                tuning_grid=tuning_grid,
                max_trials_per_model=max_trials_per_model,
            )
        results.append(result)

    results_dir = Path("results")
    benchmark_summary = _save_benchmark_artifacts(results, results_dir)

    registry_summary = {}
    monitoring_summary = {}
    serving_summary = {}

    if results:
        best_result = min(results, key=lambda result: result["overall_metrics"]["RMSE"])
        artifacts_dir = save_inference_artifacts(
            best_result["model_obj"],
            best_result["model_name"],
            feature_cols,
            scaler_X,
            scaler_y,
            seq_len,
            horizon,
        )
        register_best_model(results)
        promote_best_to_staging(results)
        registry_summary = {
            "selected_model": best_result["model_name"],
            "selected_trial": best_result.get("selected_trial"),
            "rmse": best_result["overall_metrics"]["RMSE"],
            "artifacts_dir": str(artifacts_dir),
        }

        monitoring_summary = _run_monitoring(
            reference_csv=prepared_output,
            current_csv=engineered_output,
            artifacts_dir=str(artifacts_dir),
            output_dir="results/monitoring",
        )

        forecaster = EnergyForecaster(str(artifacts_dir))
        serving_summary = {
            "model_name": forecaster.model_name,
            "forecast_horizon": forecaster.forecast_horizon,
            "schema": forecaster.describe_schema(),
        }

    pipeline_summary = {
        "template": template["name"],
        "models": models_to_train,
        "tuning_enabled": tuning_enabled,
        "max_trials_per_model": max_trials_per_model,
        "benchmark": benchmark_summary,
        "registry": registry_summary,
        "serving": serving_summary,
        "monitoring": monitoring_summary,
        "prepared_data": prepared_output,
        "feature_data": engineered_output,
    }

    with open(results_dir / "pipeline_summary.json", "w") as handle:
        json.dump(pipeline_summary, handle, indent=2)

    return pipeline_summary


class EnergyForecastingFlow(FlowSpec):
    config = (
        Parameter("config", default="config/default.yml") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    template = (
        Parameter("template", default="baseline") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    model = (
        Parameter("model", default="") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    tuning = (
        Parameter("tuning", default="") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    max_trials = (
        Parameter("max_trials", default=0) if METAFLOW_AVAILABLE else _noop_parameter()
    )
    start_date = (
        Parameter("start_date", default="2022-01-01") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    end_date = (
        Parameter("end_date", default="2024-12-31") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    prepared_output = (
        Parameter("prepared_output", default="data/prepared_energy_data.csv")
        if METAFLOW_AVAILABLE
        else _noop_parameter()
    )
    engineered_output = (
        Parameter("engineered_output", default="data/feature_engineered_energy_data.csv")
        if METAFLOW_AVAILABLE
        else _noop_parameter()
    )

    @step
    def start(self):
        tuning_raw = str(self.tuning).strip().lower()
        if tuning_raw in {"true", "1", "yes"}:
            tuning_override = True
        elif tuning_raw in {"false", "0", "no"}:
            tuning_override = False
        else:
            tuning_override = None

        max_trials_override = int(self.max_trials) if int(self.max_trials) > 0 else None
        selected_model = self.model or None

        self.summary = run_full_pipeline(
            config_path=self.config,
            selected_model=selected_model,
            template_name=self.template,
            tuning_override=tuning_override,
            max_trials_override=max_trials_override,
            start_date=self.start_date,
            end_date=self.end_date,
            prepared_output=self.prepared_output,
            engineered_output=self.engineered_output,
        )
        self.next(self.end)

    @step
    def end(self):
        logger.info("Flow completed for template=%s", self.summary["template"])


def main():
    parser = argparse.ArgumentParser(description="Energy Forecasting End-to-End Training Pipeline")
    parser.add_argument("--model", type=str, default=None, help="Train specific model")
    parser.add_argument("--config", type=str, default="config/default.yml")
    parser.add_argument("--template", type=str, default="baseline")
    parser.add_argument("--tuning", type=str, default="")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--prepared-output", type=str, default="data/prepared_energy_data.csv")
    parser.add_argument("--engineered-output", type=str, default="data/feature_engineered_energy_data.csv")
    args = parser.parse_args()

    tuning_raw = str(args.tuning).strip().lower()
    if tuning_raw in {"true", "1", "yes"}:
        tuning_override = True
    elif tuning_raw in {"false", "0", "no"}:
        tuning_override = False
    else:
        tuning_override = None

    run_full_pipeline(
        config_path=args.config,
        selected_model=args.model,
        template_name=args.template,
        tuning_override=tuning_override,
        max_trials_override=args.max_trials if args.max_trials > 0 else None,
        start_date=args.start_date,
        end_date=args.end_date,
        prepared_output=args.prepared_output,
        engineered_output=args.engineered_output,
    )


if __name__ == "__main__":
    if any(arg in sys.argv for arg in ["run", "resume", "show"]):
        if not METAFLOW_AVAILABLE:
            raise ImportError("metaflow is not installed. Install with: uv add metaflow")
        EnergyForecastingFlow()
    else:
        main()
