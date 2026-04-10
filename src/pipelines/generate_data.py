import logging
import logging.config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.data import generate_synthetic_data, load_config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")


def main():
    load_config()
    logger.info("Generating synthetic energy price dataset...")
    df = generate_synthetic_data()
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Date range: %s to %s", df["timestamp"].min(), df["timestamp"].max())
    logger.info("Price stats:\n%s", df["price"].describe())


if __name__ == "__main__":
    main()
