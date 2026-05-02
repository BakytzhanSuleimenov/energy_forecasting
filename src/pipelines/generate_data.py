import argparse
import logging
import logging.config
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.data import load_config, prepare_data_pipeline

try:
    from metaflow import FlowSpec, Parameter, step

    METAFLOW_AVAILABLE = True
except ImportError:
    FlowSpec = object

    def Parameter(*args, **kwargs):
        return None

    def step(func):
        return func

    METAFLOW_AVAILABLE = False

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")


def _noop_parameter(*args, **kwargs):
    return None


class DataPreparationFlow(FlowSpec):
    start_date = (
        Parameter("start_date", default="2022-01-01") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    end_date = (
        Parameter("end_date", default="2024-12-31") if METAFLOW_AVAILABLE else _noop_parameter()
    )
    output_path = (
        Parameter("output_path", default="data/prepared_energy_data.csv")
        if METAFLOW_AVAILABLE
        else _noop_parameter()
    )

    @step
    def start(self):
        load_config()
        use_real = bool(os.environ.get("ENTSOE_API_KEY"))
        self.df = prepare_data_pipeline(
            start_date=self.start_date,
            end_date=self.end_date,
            output_path=self.output_path,
            use_real_data=use_real,
        )
        self.next(self.end)

    @step
    def end(self):
        logger.info("Dataset shape: %s", self.df.shape)
        logger.info("Date range: %s to %s", self.df["timestamp"].min(), self.df["timestamp"].max())


def main():
    parser = argparse.ArgumentParser(description="Energy Data Preparation Pipeline")
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--output", type=str, default="data/prepared_energy_data.csv")
    args = parser.parse_args()

    load_config()
    use_real = bool(os.environ.get("ENTSOE_API_KEY"))
    df = prepare_data_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        use_real_data=use_real,
    )
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Date range: %s to %s", df["timestamp"].min(), df["timestamp"].max())


if __name__ == "__main__":
    if any(arg in sys.argv for arg in ["run", "resume", "show"]):
        if not METAFLOW_AVAILABLE:
            raise ImportError("metaflow is not installed. Install with: uv add metaflow")
        DataPreparationFlow()
    else:
        main()
