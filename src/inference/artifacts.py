from json import dump
from pathlib import Path

import joblib

KERAS_MODELS = {"dnn", "lstm"}


def save_inference_artifacts(
    model_obj,
    model_name,
    feature_cols,
    scaler_X,
    scaler_y,
    sequence_length,
    forecast_horizon,
    output_dir="artifacts/best_model",
):
    artifacts_dir = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_filename = "model.keras" if model_name in KERAS_MODELS else "model.joblib"

    if model_name in KERAS_MODELS:
        model_obj.model.save(artifacts_dir / model_filename)
    else:
        joblib.dump(model_obj.model, artifacts_dir / model_filename)

    joblib.dump(scaler_X, artifacts_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, artifacts_dir / "scaler_y.joblib")

    metadata = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "sequence_length": sequence_length,
        "forecast_horizon": forecast_horizon,
        "model_filename": model_filename,
        "scaler_X_filename": "scaler_X.joblib",
        "scaler_y_filename": "scaler_y.joblib",
    }

    with open(artifacts_dir / "metadata.json", "w") as f:
        dump(metadata, f, indent=2)

    return artifacts_dir
