set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set dotenv-load

MLFLOW_TRACKING_URI := env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

default:
    @just --list

# Run project unit tests
@test:
    uv run pytest

# Lint the codebase
@lint:
    uv run ruff check src/ tests/

# Format the codebase
@format:
    uv run ruff format src/ tests/

# Generate synthetic energy price dataset
[group('data')]
@generate-data:
    uv run python src/pipelines/generate_data.py

# Train all models and benchmark them
[group('training')]
@train:
    uv run python src/pipelines/training.py

# Train a specific model (random_forest, lstm, dnn, xgboost)
[group('training')]
@train-model model:
    uv run python src/pipelines/training.py --model {{model}}

# Run model benchmarking comparison
[group('benchmarking')]
@benchmark:
    uv run python src/pipelines/benchmarking.py

# Launch the visualization dashboard
[group('dashboard')]
@dashboard:
    uv run streamlit run src/dashboard/app.py

# Run MLflow server
[group('setup')]
@mlflow:
    uv run mlflow server --host 127.0.0.1 --port 5000

# Display versions of required dependencies
[group('setup')]
@dependencies:
    uv run python -c "import sklearn; import tensorflow; import xgboost; import keras; print(f'sklearn: {sklearn.__version__}'); print(f'tensorflow: {tensorflow.__version__}'); print(f'xgboost: {xgboost.__version__}'); print(f'keras: {keras.__version__}')"
