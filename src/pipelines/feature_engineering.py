import argparse
import logging
import logging.config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.data import feature_engineering_pipeline

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


class FeatureEngineeringFlow(FlowSpec):
    input_path = (
        Parameter("input_path", default="data/prepared_energy_data.csv")
        if METAFLOW_AVAILABLE
        else _noop_parameter()
    )
    output_path = (
        Parameter("output_path", default="data/feature_engineered_energy_data.csv")
        if METAFLOW_AVAILABLE
        else _noop_parameter()
    )

    @step
    def start(self):
        self.df = feature_engineering_pipeline(data_path=self.input_path, output_path=self.output_path)
        self.next(self.end)

    @step
    def end(self):
        logger.info("Feature dataset shape: %s", self.df.shape)


def main():
    parser = argparse.ArgumentParser(description="Energy Feature Engineering Pipeline")
    parser.add_argument("--input", type=str, default="data/prepared_energy_data.csv")
    parser.add_argument("--output", type=str, default="data/feature_engineered_energy_data.csv")
    args = parser.parse_args()

    df = feature_engineering_pipeline(data_path=args.input, output_path=args.output)
    logger.info("Feature dataset shape: %s", df.shape)


if __name__ == "__main__":
    if any(arg in sys.argv for arg in ["run", "resume", "show"]):
        if not METAFLOW_AVAILABLE:
            raise ImportError("metaflow is not installed. Install with: uv add metaflow")
        FeatureEngineeringFlow()
    else:
        main()
