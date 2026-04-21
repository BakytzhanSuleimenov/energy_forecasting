import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.inference import EnergyForecaster, SchemaValidationError

logger = logging.getLogger("energy_forecasting")


class PredictionRequest(BaseModel):
    history: list[dict[str, object]] = Field(min_length=1, max_length=1000)


class BatchPredictionRequest(BaseModel):
    histories: list[list[dict[str, object]]] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    model_name: str
    forecast_horizon: int
    prediction: list[float]


class BatchPredictionResponse(BaseModel):
    model_name: str
    forecast_horizon: int
    predictions: list[list[float]]


def create_app(artifacts_dir: str = "artifacts/best_model") -> FastAPI:
    max_request_bytes = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    expected_api_key = os.getenv("API_KEY")
    request_log: defaultdict[str, deque[float]] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifacts_dir = artifacts_dir
        app.state.forecaster = EnergyForecaster(artifacts_dir=artifacts_dir)
        yield

    app = FastAPI(title="Energy Forecasting API", version="0.1.0", lifespan=lifespan)

    def get_forecaster() -> EnergyForecaster:
        forecaster = app.state.forecaster
        if forecaster is None:
            forecaster = EnergyForecaster(artifacts_dir=app.state.artifacts_dir)
            app.state.forecaster = forecaster
        return forecaster

    def enforce_rate_limit(client_id: str) -> None:
        now = time.monotonic()
        window_start = now - rate_limit_window_seconds
        events = request_log[client_id]
        while events and events[0] < window_start:
            events.popleft()
        if len(events) >= rate_limit_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        events.append(now)

    def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if expected_api_key and x_api_key != expected_api_key:
            raise HTTPException(status_code=403, detail="Invalid API key")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.middleware("http")
    async def request_size_limit_middleware(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > max_request_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request too large"})
        return await call_next(request)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/predict"):
            client_host = request.client.host if request.client else "unknown"
            enforce_rate_limit(client_host)
        return await call_next(request)

    @app.exception_handler(SchemaValidationError)
    async def schema_validation_handler(request: Request, exc: SchemaValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc), "request_id": request.state.request_id})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request.state.request_id},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        forecaster = get_forecaster()
        return {
            "status": "ok",
            "model_name": forecaster.model_name,
            "forecast_horizon": forecaster.forecast_horizon,
        }

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return get_forecaster().describe_schema()

    @app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
    def predict(request: PredictionRequest) -> PredictionResponse:
        forecaster = get_forecaster()
        prediction = forecaster.predict(request.history)
        return PredictionResponse(
            model_name=forecaster.model_name,
            forecast_horizon=forecaster.forecast_horizon,
            prediction=prediction,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        forecaster = get_forecaster()
        predictions = [forecaster.predict(history) for history in request.histories]
        return BatchPredictionResponse(
            model_name=forecaster.model_name,
            forecast_horizon=forecaster.forecast_horizon,
            predictions=predictions,
        )

    return app


app = create_app(artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts/best_model"))
