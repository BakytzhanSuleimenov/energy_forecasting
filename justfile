set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set dotenv-load

MLFLOW_TRACKING_URI := env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

default:
    @just --list

@test:
    uv run pytest

@lint:
    uv run ruff check src/ tests/

@typecheck:
    uv run mypy src/

@test-cov:
    uv run pytest --cov=src --cov-report=term-missing

@security-audit:
    uv run pip-audit

@format:
    uv run ruff format src/ tests/

[group('data')]
@generate-data:
    uv run python src/pipelines/generate_data.py

[group('data')]
@feature-engineer:
    uv run python src/pipelines/feature_engineering.py

[group('inference')]
@predict-schema:
    uv run python src/pipelines/predict.py --schema

[group('inference')]
@predict:
    uv run python src/pipelines/predict.py

[group('inference')]
@monitor:
    uv run python src/pipelines/monitoring.py

[group('inference')]
@serve:
    uv run uvicorn src.serving.app:app --host 127.0.0.1 --port 8000

[group('deployment')]
@docker-build:
    docker build -t energy-forecasting-api .

[group('deployment')]
@docker-run:
    docker run --rm -p 8000:8000 -e ARTIFACTS_DIR=/app/artifacts/best_model energy-forecasting-api

[group('training')]
@train:
    uv run python src/pipelines/training.py --template baseline
@train-model model:
    uv run python src/pipelines/training.py --template baseline --model {{model}}
@train-template template:
    uv run python src/pipelines/training.py --template {{template}}
@train-tuned template="fast_iteration" max_trials="8":
    uv run python src/pipelines/training.py --template {{template}} --tuning true --max-trials {{max_trials}}
@train-metaflow template="baseline" tuning="" max_trials="0":
    uv run python src/pipelines/training.py run --template {{template}} --tuning {{tuning}} --max-trials {{max_trials}}

[group('benchmarking')]
@benchmark:
    uv run python src/pipelines/benchmarking.py

[group('dashboard')]
@dashboard:
    uv run streamlit run src/dashboard/app.py

[group('setup')]
@mlflow:
    uv run mlflow server --host 127.0.0.1 --port 5000

[group('setup')]
@mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

[group('setup')]
@dependencies:
    uv run python -c "import sklearn; import tensorflow; import xgboost; import keras; print(f'sklearn: {sklearn.__version__}'); print(f'tensorflow: {tensorflow.__version__}'); print(f'xgboost: {xgboost.__version__}'); print(f'keras: {keras.__version__}')"