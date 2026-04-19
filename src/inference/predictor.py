import json
from pathlib import Path

import joblib
import numpy as np

from src.inference.schema import build_input_schema, validate_history_frame


class EnergyForecaster:
    def __init__(self, artifacts_dir="artifacts/best_model"):
        self.artifacts_dir = Path(artifacts_dir)
        with open(self.artifacts_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.model_name = self.metadata["model_name"]
        self.feature_cols = self.metadata["feature_cols"]
        self.sequence_length = self.metadata["sequence_length"]
        self.forecast_horizon = self.metadata["forecast_horizon"]
        self.schema = build_input_schema(self.feature_cols, self.sequence_length)

        self.scaler_X = joblib.load(self.artifacts_dir / self.metadata["scaler_X_filename"])
        self.scaler_y = joblib.load(self.artifacts_dir / self.metadata["scaler_y_filename"])
        self.model = self._load_model()

    def _load_model(self):
        model_path = self.artifacts_dir / self.metadata["model_filename"]
        if self.model_name in {"dnn", "lstm"}:
            from keras.models import load_model

            return load_model(model_path)
        return joblib.load(model_path)

    def predict(self, history):
        validated = validate_history_frame(history, self.feature_cols, self.sequence_length)
        scaled_history = self.scaler_X.transform(validated.to_numpy())
        model_input = self._build_model_input(scaled_history)
        if self.model_name in {"dnn", "lstm"}:
            raw_prediction = self.model.predict(model_input, verbose=0)
        else:
            raw_prediction = self.model.predict(model_input)
        prediction_scaled = np.asarray(raw_prediction).reshape(1, -1)
        prediction = self.scaler_y.inverse_transform(prediction_scaled).flatten()
        return prediction.tolist()

    def describe_schema(self):
        return self.schema

    def _build_model_input(self, scaled_history):
        if self.model_name == "lstm":
            return scaled_history.reshape(1, self.sequence_length, len(self.feature_cols))
        if self.model_name == "dnn":
            return scaled_history.reshape(1, -1)
        return self._build_tabular_features(scaled_history).reshape(1, -1)

    def _build_tabular_features(self, scaled_history):
        last_row = scaled_history[-1]
        mean_row = scaled_history.mean(axis=0)
        std_row = scaled_history.std(axis=0)
        daily_delta = (
            scaled_history[-1] - scaled_history[-24]
            if self.sequence_length >= 24
            else np.zeros(scaled_history.shape[1])
        )
        two_day_delta = (
            scaled_history[-1] - scaled_history[-48]
            if self.sequence_length >= 48
            else np.zeros(scaled_history.shape[1])
        )
        return np.concatenate([last_row, mean_row, std_row, daily_delta, two_day_delta])
