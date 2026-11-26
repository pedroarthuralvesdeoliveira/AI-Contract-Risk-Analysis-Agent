"""CLI entrypoint to download public datasets, train V2 ML models, and log metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.train import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, train_all_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 Contract Risk ML classifiers.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    print("Training V2 ML classifiers (CUAD + ContractNLI + LEDGAR)...")
    metrics = train_all_models(
        raw_dir=args.raw_dir,
        models_dir=args.models_dir,
        force_download=args.force_download,
    )

    summary = {
        name: {
            "accuracy": metrics[name]["accuracy"],
            "macro_f1": metrics[name]["macro_f1"],
            "weighted_f1": metrics[name]["weighted_f1"],
            "train_samples": metrics[name]["train_samples"],
            "test_samples": metrics[name]["test_samples"],
        }
        for name in (
            "contract_type_classifier",
            "clause_risk_classifier",
            "clause_category_classifier",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
