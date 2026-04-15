import logging
import logging.config
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.data import fetch_real_data, generate_synthetic_data, load_config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")


def main():
    load_config()
    api_key = os.environ.get("ENTSOE_API_KEY")
    if api_key:
        logger.info("ENTSOE_API_KEY found – fetching real Ireland energy market data...")
        df = fetch_real_data(
            start_date="2022-01-01",
            end_date="2024-12-31",
        )
    else:
        logger.info("ENTSOE_API_KEY not set – generating synthetic energy price dataset...")
        df = generate_synthetic_data()
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Date range: %s to %s", df["timestamp"].min(), df["timestamp"].max())
    logger.info("Price stats:\n%s", df["price"].describe())


if __name__ == "__main__":
    main()
