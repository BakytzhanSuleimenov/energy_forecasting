from fastapi.testclient import TestClient

from src.serving.app import create_app
from tests.test_predict_cli import prepare_artifacts


def test_serving_endpoints_return_schema_and_prediction(tmp_path):
    df, artifacts_dir = prepare_artifacts(tmp_path)
    app = create_app(artifacts_dir=str(artifacts_dir))

    with TestClient(app) as client:
        health_response = client.get("/health")
        schema_response = client.get("/schema")
        history = df.drop(columns=["timestamp"]).tail(24).to_dict(orient="records")
        prediction_response = client.post("/predict", json={"history": history})

    assert health_response.status_code == 200
    assert health_response.json()["model_name"] == "random_forest"
    assert schema_response.status_code == 200
    assert schema_response.json()["sequence_length"] == 24
    assert prediction_response.status_code == 200
    assert len(prediction_response.json()["prediction"]) == 12


def test_serving_batch_prediction_returns_multiple_forecasts(tmp_path):
    df, artifacts_dir = prepare_artifacts(tmp_path)
    app = create_app(artifacts_dir=str(artifacts_dir))
    features = df.drop(columns=["timestamp"])
    first_history = features.iloc[-30:-6].to_dict(orient="records")
    second_history = features.tail(24).to_dict(orient="records")

    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"histories": [first_history, second_history]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "random_forest"
    assert len(payload["predictions"]) == 2
    assert all(len(prediction) == 12 for prediction in payload["predictions"])


def test_serving_predict_rejects_invalid_payload(tmp_path):
    _, artifacts_dir = prepare_artifacts(tmp_path)
    app = create_app(artifacts_dir=str(artifacts_dir))

    with TestClient(app) as client:
        response = client.post("/predict", json={"history": [{"price": 1.0}] * 24})

    assert response.status_code == 422
