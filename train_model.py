"""
Amazon Supply Chain Intelligence — Training Entry Point
Usage: python train_model.py --data_path data/supply_chain_data.csv
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(__file__))

from logger import logger
from src.pipeline.train_pipeline import TrainPipeline


def print_metrics_table(metrics: dict):
    """Print a formatted summary table of model metrics."""
    print("\n" + "=" * 65)
    print("  AMAZON SUPPLY CHAIN INTELLIGENCE — TRAINING RESULTS")
    print("=" * 65)
    print(f"  Best Model : {metrics['best_model']}")
    print(f"  Weighted F1: {metrics['best_weighted_f1']:.4f}")
    print("-" * 65)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Wt-F1':>10} {'Macro-F1':>10} {'Kappa':>10}")
    print("-" * 65)

    comparison = metrics.get("model_comparison", {})
    best = metrics["best_model"]

    for name, m in comparison.items():
        marker = " ★" if name == best else "  "
        print(
            f"  {name + marker:<22} "
            f"{m['accuracy']*100:>8.2f}%  "
            f"{m['weighted_f1']*100:>8.2f}%  "
            f"{m['macro_f1']*100:>8.2f}%  "
            f"{m['cohen_kappa']:>8.3f}"
        )

    print("=" * 65)
    print("  Artifacts saved to: ./artifacts/")
    print("  Start app: python application.py")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Amazon Supply Chain Intelligence ML models"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/supply_chain_data.csv",
        help="Path to the supply chain CSV dataset",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"\n❌ Dataset not found at: {args.data_path}")
        print("  Download the DataCo Supply Chain dataset and place it at the path above.")
        print("  Dataset source: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")
        sys.exit(1)

    logger.info(f"Starting training with dataset: {args.data_path}")

    pipeline = TrainPipeline()
    metrics = pipeline.run_pipeline(data_path=args.data_path)

    print_metrics_table(metrics)

    # Save metrics JSON
    metrics_path = os.path.join("artifacts", "metrics.json")
    logger.info(f"Metrics saved at: {metrics_path}")


if __name__ == "__main__":
    main()
