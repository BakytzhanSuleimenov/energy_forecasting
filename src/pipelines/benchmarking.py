import json
import logging
import logging.config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("energy_forecasting")


def load_results(results_path="results/benchmark_results.json"):
    with open(results_path) as f:
        return json.load(f)


def generate_comparison_table(results):
    rows = []
    for r in results:
        row = {"Model": r["model_name"]}
        row.update(r["overall_metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def rank_models(results):
    df = generate_comparison_table(results)
    rankings = {}
    rankings["MAE"] = df.sort_values("MAE")["Model"].tolist()
    rankings["RMSE"] = df.sort_values("RMSE")["Model"].tolist()
    rankings["MAPE"] = df.sort_values("MAPE")["Model"].tolist()
    rankings["R2"] = df.sort_values("R2", ascending=False)["Model"].tolist()
    rankings["training_time"] = df.sort_values("training_time")["Model"].tolist()

    model_scores = {}
    for model in df["Model"]:
        score = 0
        for metric, ranked in rankings.items():
            score += ranked.index(model)
        model_scores[model] = score

    overall_ranking = sorted(model_scores.items(), key=lambda x: x[1])
    return rankings, overall_ranking


def generate_horizon_comparison(results):
    horizon_data = {}
    for r in results:
        model_name = r["model_name"]
        horizon_data[model_name] = r["horizon_metrics"]
    return horizon_data


def main():
    results_path = Path("results/benchmark_results.json")
    if not results_path.exists():
        logger.error("No benchmark results found. Run training first: python src/pipelines/training.py")
        sys.exit(1)

    results = load_results()
    logger.info("Loaded results for %d models", len(results))

    comparison_df = generate_comparison_table(results)
    logger.info("\nModel Comparison:\n%s", comparison_df.to_string(index=False))

    rankings, overall = rank_models(results)

    logger.info("\nRankings by metric:")
    for metric, ranked in rankings.items():
        logger.info("  %s: %s", metric, " > ".join(ranked))

    logger.info("\nOverall ranking (lower score = better):")
    for i, (model, score) in enumerate(overall, 1):
        logger.info("  %d. %s (score: %d)", i, model, score)

    horizon_data = generate_horizon_comparison(results)
    horizon_df_rows = []
    for model_name, metrics_list in horizon_data.items():
        for m in metrics_list:
            row = {"Model": model_name}
            row.update(m)
            horizon_df_rows.append(row)
    horizon_df = pd.DataFrame(horizon_df_rows)

    results_dir = Path("results")
    comparison_df.to_csv(results_dir / "comparison_table.csv", index=False)
    horizon_df.to_csv(results_dir / "horizon_comparison.csv", index=False)

    summary = {
        "rankings": rankings,
        "overall_ranking": overall,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }
    with open(results_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nBenchmark analysis saved to results/")


if __name__ == "__main__":
    main()
