import logging
import os
import socket
from urllib.parse import urlparse

logger = logging.getLogger("energy_forecasting")


def is_mlflow_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _is_http_server_reachable(uri: str, timeout: float = 2.0) -> bool:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_experiment(experiment_name: str = "energy-forecasting") -> str | None:
    if not is_mlflow_enabled():
        logger.info("MLFLOW_TRACKING_URI not set – MLflow tracking disabled")
        return None
    try:
        import mlflow

        uri = os.environ["MLFLOW_TRACKING_URI"]
        if uri.startswith(("http://", "https://")) and not _is_http_server_reachable(uri):
            logger.warning(
                "MLflow server at %s is not reachable – skipping MLflow tracking", uri
            )
            return None

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            "MLflow experiment '%s' ready at %s",
            experiment_name,
            uri,
        )
        return experiment_name
    except Exception as exc:
        logger.warning("MLflow setup failed (non-fatal): %s", exc)
        return None


def log_training_run(
    model_obj,
    model_name: str,
    model_config: dict,
    seq_len: int,
    horizon: int,
    test_ratio: float,
    val_ratio: float,
    overall_metrics: dict,
    horizon_metrics: list,
) -> None:
    if not is_mlflow_enabled():
        return
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        mlflow.log_params(
            {
                "model_name": model_name,
                "seq_len": seq_len,
                "horizon": horizon,
                "test_ratio": test_ratio,
                "val_ratio": val_ratio,
                **{k: v for k, v in model_config.items() if isinstance(v, (int, float, str, bool))},
            }
        )

        mlflow.log_metrics({k: float(v) for k, v in overall_metrics.items()})

        for step, hm in enumerate(horizon_metrics):
            mlflow.log_metrics(
                {"horizon_MAE": float(hm["MAE"]), "horizon_RMSE": float(hm["RMSE"])},
                step=step + 1,
            )

        _log_model_artifact(model_obj, model_name)

    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)


def _log_model_artifact(model_obj, model_name: str) -> None:
    try:
        import mlflow

        inner = getattr(model_obj, "model", None)
        if inner is None:
            return
        registered_name = f"energy-forecasting-{model_name.lower().replace('_', '-')}"
        if model_name in ("random_forest", "xgboost"):
            mlflow.sklearn.log_model(
                inner,
                name="model",
                registered_model_name=registered_name,
            )
        else:
            mlflow.tensorflow.log_model(
                inner,
                name="model",
                registered_model_name=registered_name,
            )
        logger.info("MLflow: model '%s' logged and registered", registered_name)
    except Exception as exc:
        logger.warning("MLflow model artifact logging failed (non-fatal): %s", exc)


def promote_best_to_staging(results: list) -> None:
    if not is_mlflow_enabled() or not results:
        return
    try:
        from mlflow import MlflowClient

        best = min(results, key=lambda r: r["overall_metrics"].get("RMSE", float("inf")))
        best_name = best["model_name"]
        registered_name = f"energy-forecasting-{best_name.lower().replace('_', '-')}"

        client = MlflowClient()
        versions = client.get_latest_versions(registered_name, stages=["None"])
        if versions:
            latest_version = versions[0].version
            client.transition_model_version_stage(
                name=registered_name,
                version=latest_version,
                stage="Staging",
                archive_existing_versions=True,
            )
            logger.info(
                "MLflow: '%s' v%s promoted to Staging (best RMSE: %s)",
                registered_name,
                latest_version,
                best["overall_metrics"].get("RMSE"),
            )
    except Exception as exc:
        logger.warning("MLflow promotion failed (non-fatal): %s", exc)
