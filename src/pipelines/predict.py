import argparse
import json
import logging
import logging.config
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.data import FEATURE_COLUMNS, load_config
from src.inference.predictor import EnergyForecaster
from src.inference.schema import build_input_schema

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")


def main():
    parser = argparse.ArgumentParser(description="Run inference using the best saved forecasting model")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/best_model")
    parser.add_argument("--input-csv", type=str, default="data/energy_prices.csv")
    parser.add_argument("--config", type=str, default="config/default.yml")
    parser.add_argument("--schema", action="store_true", help="Print the expected input schema and exit")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if args.schema and not (artifacts_dir / "metadata.json").exists():
        config = load_config(args.config)
        sequence_length = config.get("data", {}).get("sequence_length", 72)
        print(json.dumps(build_input_schema(FEATURE_COLUMNS, sequence_length), indent=2))
        return

    forecaster = EnergyForecaster(args.artifacts_dir)

    if args.schema:
        print(json.dumps(forecaster.describe_schema(), indent=2))
        return

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    history = df[forecaster.feature_cols].tail(forecaster.sequence_length)
    prediction = forecaster.predict(history)

    output = {
        "model_name": forecaster.model_name,
        "forecast_horizon": forecaster.forecast_horizon,
        "prediction": prediction,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
