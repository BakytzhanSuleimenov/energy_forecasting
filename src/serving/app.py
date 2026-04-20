from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import EnergyForecaster, SchemaValidationError


class PredictionRequest(BaseModel):
    history: list[dict[str, object]]


class PredictionResponse(BaseModel):
    model_name: str
    forecast_horizon: int
    prediction: list[float]


def create_app(artifacts_dir="artifacts/best_model"):
    app = FastAPI(title="Energy Forecasting API", version="0.1.0")
    app.state.artifacts_dir = artifacts_dir
    app.state.forecaster = None

    @app.on_event("startup")
    def startup():
        app.state.forecaster = EnergyForecaster(artifacts_dir=app.state.artifacts_dir)

    def get_forecaster():
        forecaster = app.state.forecaster
        if forecaster is None:
            forecaster = EnergyForecaster(artifacts_dir=app.state.artifacts_dir)
            app.state.forecaster = forecaster
        return forecaster

    @app.get("/health")
    def health():
        forecaster = get_forecaster()
        return {
            "status": "ok",
            "model_name": forecaster.model_name,
            "forecast_horizon": forecaster.forecast_horizon,
        }

    @app.get("/schema")
    def schema():
        return get_forecaster().describe_schema()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        forecaster = get_forecaster()
        try:
            prediction = forecaster.predict(request.history)
        except SchemaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(
            model_name=forecaster.model_name,
            forecast_horizon=forecaster.forecast_horizon,
            prediction=prediction,
        )

    return app


app = create_app()
