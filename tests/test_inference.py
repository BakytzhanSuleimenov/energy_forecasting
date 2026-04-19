import pandas as pd
import pytest

from src.common.data import create_tabular_features, generate_synthetic_data, prepare_features
from src.inference.artifacts import save_inference_artifacts
from src.inference.predictor import EnergyForecaster
from src.inference.schema import SchemaValidationError, build_input_schema, validate_history_frame
from src.models.random_forest import RandomForestModel


def test_build_input_schema():
    schema = build_input_schema(["price", "temperature"], 24)
    assert schema["required_columns"] == ["price", "temperature"]
    assert schema["sequence_length"] == 24
    assert schema["input_type"] == "history_window"


def test_validate_history_frame_missing_column():
    history = pd.DataFrame({"price": [1.0] * 24})
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_history_frame(history, ["price", "temperature"], 24)


def test_validate_history_frame_invalid_values():
    history = pd.DataFrame({
        "price": [1.0] * 24,
        "temperature": ["bad"] * 24,
    })
    with pytest.raises(SchemaValidationError, match="Invalid or missing values"):
        validate_history_frame(history, ["price", "temperature"], 24)


def test_energy_forecaster_predicts_from_saved_artifacts(tmp_path):
    df = generate_synthetic_data(n_days=15, output_path=str(tmp_path / "energy.csv"))
    data_scaled, target_scaled, scaler_X, scaler_y, feature_cols = prepare_features(df, 24, 12)
    X, y = create_tabular_features(data_scaled, target_scaled, 24, 12)

    model = RandomForestModel(config={"n_estimators": 5, "max_depth": 3})
    model.build(input_shape=X.shape[1], output_shape=12)
    model.fit(X[:80], y[:80])

    save_inference_artifacts(
        model,
        "random_forest",
        feature_cols,
        scaler_X,
        scaler_y,
        24,
        12,
        output_dir=tmp_path / "artifacts",
    )

    forecaster = EnergyForecaster(artifacts_dir=tmp_path / "artifacts")
    history = df[feature_cols].tail(24)
    prediction = forecaster.predict(history)

    assert len(prediction) == 12
    assert all(isinstance(value, float) for value in prediction)
    assert forecaster.describe_schema()["required_columns"] == feature_cols
