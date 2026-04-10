import numpy as np
import pytest

from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel


@pytest.fixture
def sample_tabular_data():
    np.random.seed(42)
    X_train = np.random.randn(200, 50)
    y_train = np.random.randn(200, 12)
    X_val = np.random.randn(30, 50)
    y_val = np.random.randn(30, 12)
    X_test = np.random.randn(50, 50)
    y_test = np.random.randn(50, 12)
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_random_forest_build():
    model = RandomForestModel(config={"n_estimators": 10, "max_depth": 5})
    model.build(input_shape=50, output_shape=12)
    assert model.model is not None


def test_random_forest_train_predict(sample_tabular_data):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_tabular_data
    model = RandomForestModel(config={"n_estimators": 10, "max_depth": 5})
    model.build(input_shape=50, output_shape=12)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (50, 12)
    assert model.get_training_time() > 0


def test_xgboost_build():
    model = XGBoostModel(config={"n_estimators": 10, "max_depth": 3})
    model.build(input_shape=50, output_shape=12)
    assert model.model is not None


def test_xgboost_train_predict(sample_tabular_data):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_tabular_data
    model = XGBoostModel(config={"n_estimators": 10, "max_depth": 3})
    model.build(input_shape=50, output_shape=12)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (50, 12)
    assert model.get_training_time() > 0


def test_dnn_build():
    from src.models.dnn import DNNModel
    model = DNNModel(config={"hidden_layers": [32, 16], "epochs": 2, "batch_size": 32})
    model.build(input_shape=50, output_shape=12)
    assert model.model is not None


def test_dnn_train_predict(sample_tabular_data):
    from src.models.dnn import DNNModel
    X_train, y_train, X_val, y_val, X_test, y_test = sample_tabular_data
    model = DNNModel(config={"hidden_layers": [32, 16], "epochs": 2, "batch_size": 32})
    model.build(input_shape=50, output_shape=12)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    assert preds.shape == (50, 12)
    assert model.get_training_time() > 0


def test_lstm_build():
    from src.models.lstm import LSTMModel
    model = LSTMModel(config={"units": [16, 8], "epochs": 2, "batch_size": 32})
    model.build(input_shape=(24, 10), output_shape=12)
    assert model.model is not None


def test_lstm_train_predict():
    from src.models.lstm import LSTMModel
    np.random.seed(42)
    X_train = np.random.randn(100, 24, 10)
    y_train = np.random.randn(100, 12)
    X_val = np.random.randn(20, 24, 10)
    y_val = np.random.randn(20, 12)
    X_test = np.random.randn(30, 24, 10)

    model = LSTMModel(config={"units": [16, 8], "epochs": 2, "batch_size": 32})
    model.build(input_shape=(24, 10), output_shape=12)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    assert preds.shape == (30, 12)
    assert model.get_training_time() > 0
