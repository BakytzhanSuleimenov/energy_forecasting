import json

from src.common.data import create_tabular_features, generate_synthetic_data, prepare_features
from src.inference.artifacts import save_inference_artifacts
from src.models.random_forest import RandomForestModel
from src.pipelines.predict import main


def prepare_artifacts(tmp_path):
    df = generate_synthetic_data(n_days=15, output_path=str(tmp_path / "energy.csv"))
    data_scaled, target_scaled, scaler_X, scaler_y, feature_cols = prepare_features(df, 24, 12)
    X, y = create_tabular_features(data_scaled, target_scaled, 24, 12)

    model = RandomForestModel(config={"n_estimators": 5, "max_depth": 3})
    model.build(input_shape=X.shape[1], output_shape=12)
    model.fit(X[:80], y[:80])

    artifacts_dir = tmp_path / "artifacts"
    save_inference_artifacts(
        model,
        "random_forest",
        feature_cols,
        scaler_X,
        scaler_y,
        24,
        12,
        output_dir=artifacts_dir,
    )
    return df, artifacts_dir


def test_predict_cli_schema_output_without_artifacts(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "--schema"],
    )
    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["sequence_length"] == 72
    assert payload["input_type"] == "history_window"


def test_predict_cli_schema_output(tmp_path, monkeypatch, capsys):
    _, artifacts_dir = prepare_artifacts(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "--artifacts-dir", str(artifacts_dir), "--schema"],
    )
    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["sequence_length"] == 24
    assert payload["input_type"] == "history_window"


def test_predict_cli_prediction_output(tmp_path, monkeypatch, capsys):
    df, artifacts_dir = prepare_artifacts(tmp_path)
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "--artifacts-dir", str(artifacts_dir), "--input-csv", str(input_csv)],
    )
    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["model_name"] == "random_forest"
    assert payload["forecast_horizon"] == 12
    assert len(payload["prediction"]) == 12
