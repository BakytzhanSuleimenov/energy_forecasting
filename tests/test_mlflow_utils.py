
import pytest


def test_is_mlflow_enabled_false_when_no_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from src.common.mlflow_utils import is_mlflow_enabled

    assert is_mlflow_enabled() is False


def test_is_mlflow_enabled_true_when_set(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    import importlib

    from src.common import mlflow_utils
    importlib.reload(mlflow_utils)
    from src.common.mlflow_utils import is_mlflow_enabled

    assert is_mlflow_enabled() is True


def test_setup_experiment_noop_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from src.common.mlflow_utils import setup_experiment

    result = setup_experiment("test-experiment")
    assert result is None


def test_log_training_run_noop_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from src.common.mlflow_utils import log_training_run

    log_training_run(
        model_obj=None,
        model_name="random_forest",
        model_config={"n_estimators": 100},
        seq_len=72,
        horizon=24,
        test_ratio=0.2,
        val_ratio=0.1,
        overall_metrics={"MAE": 1.0, "RMSE": 1.5, "MAPE": 0.05, "R2": 0.9, "training_time": 1.0},
        horizon_metrics=[{"MAE": 1.0, "RMSE": 1.5}],
    )


def test_register_best_model_noop_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from src.common.mlflow_utils import register_best_model

    register_best_model([
        {
            "model_name": "random_forest",
            "model_config": {"n_estimators": 10},
            "overall_metrics": {"RMSE": 1.0},
            "model_obj": None,
        }
    ])


def test_register_best_model_logs_hyperparameters(monkeypatch, tmp_path):
    tracking_uri = tmp_path.joinpath("mlflow.db").as_uri().replace("file:///", "sqlite:///")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    from src.common.mlflow_utils import register_best_model, setup_experiment
    from src.models.random_forest import RandomForestModel

    setup_experiment("test-experiment")

    model = RandomForestModel(config={"n_estimators": 5, "max_depth": 3})
    model.build(input_shape=4, output_shape=2)

    register_best_model([
        {
            "model_name": "random_forest",
            "model_config": {"n_estimators": 5, "max_depth": 3},
            "overall_metrics": {"RMSE": 0.8, "MAE": 0.4},
            "model_obj": model,
        },
        {
            "model_name": "xgboost",
            "model_config": {"n_estimators": 20},
            "overall_metrics": {"RMSE": 1.2},
            "model_obj": None,
        },
    ])

    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    assert runs
    run_data = runs[0].data
    assert run_data.params["model_name"] == "random_forest"
    assert run_data.params["registered_model_name"] == "energy-forecasting-random-forest"
    assert run_data.params["hp_n_estimators"] == "5"
    assert run_data.params["hp_max_depth"] == "3"
    assert float(run_data.metrics["RMSE"]) == pytest.approx(0.8)


def test_promote_best_to_staging_noop_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from src.common.mlflow_utils import promote_best_to_staging

    promote_best_to_staging([
        {"model_name": "random_forest", "overall_metrics": {"RMSE": 1.5}},
        {"model_name": "xgboost", "overall_metrics": {"RMSE": 1.2}},
    ])


def test_setup_experiment_with_local_mlruns(monkeypatch, tmp_path):
    tracking_uri = tmp_path.joinpath("mlflow.db").as_uri().replace("file:///", "sqlite:///")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    from src.common.mlflow_utils import setup_experiment

    result = setup_experiment("test-experiment")
    assert result == "test-experiment"


def test_log_training_run_with_active_run(monkeypatch, tmp_path):
    tracking_uri = tmp_path.joinpath("mlflow.db").as_uri().replace("file:///", "sqlite:///")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    import mlflow

    from src.common.mlflow_utils import log_training_run, setup_experiment

    setup_experiment("test-experiment")

    with mlflow.start_run():
        log_training_run(
            model_obj=None,
            model_name="random_forest",
            model_config={"n_estimators": 100},
            seq_len=72,
            horizon=24,
            test_ratio=0.2,
            val_ratio=0.1,
            overall_metrics={"MAE": 1.0, "RMSE": 1.5, "MAPE": 0.05, "R2": 0.9, "training_time": 1.0},
            horizon_metrics=[{"MAE": 1.0, "RMSE": 1.5}, {"MAE": 1.1, "RMSE": 1.6}],
        )
        run = mlflow.active_run()
        run_id = run.info.run_id

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run_data = client.get_run(run_id).data
    assert run_data.params["model_name"] == "random_forest"
    assert run_data.params["seq_len"] == "72"
    assert float(run_data.metrics["MAE"]) == pytest.approx(1.0)
    assert float(run_data.metrics["RMSE"]) == pytest.approx(1.5)
