"""Reproducible training and evaluation pipeline for V2 ML classifiers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.ml.dataset import load_clause_risk_datasets, load_contract_type_dataset

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def build_text_classifier(
    max_features: int = 25000,
    ngram_range: tuple[int, int] = (1, 2),
    c_value: float = 2.0,
) -> Pipeline:
    """Create a TF-IDF + balanced LogisticRegression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _evaluate_predictions(y_true, y_pred) -> Dict[str, Any]:
    """Compute accuracy, macro F1, weighted F1, and per-class metrics."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(vals["precision"]), 4),
                "recall": round(float(vals["recall"]), 4),
                "f1": round(float(vals["f1-score"]), 4),
                "support": int(vals["support"]),
            }
            for label, vals in report.items()
            if isinstance(vals, dict) and label not in ("macro avg", "weighted avg")
        },
    }


def train_all_models(
    raw_dir: Path = DEFAULT_DATA_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    force_download: bool = False,
) -> Dict[str, Any]:
    """Download datasets, train all V2 classifiers, evaluate them, and save artifacts."""
    models_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Contract Type Classifier (CUAD + ContractNLI)
    df_types = load_contract_type_dataset(raw_dir=raw_dir, force_download=force_download)
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(
        df_types["text"],
        df_types["contract_type"],
        test_size=0.2,
        random_state=42,
        stratify=df_types["contract_type"],
    )
    contract_type_pipe = build_text_classifier(max_features=20000, ngram_range=(1, 2), c_value=3.0)
    contract_type_pipe.fit(x_train_t, y_train_t)
    y_pred_t = contract_type_pipe.predict(x_test_t)
    type_metrics = _evaluate_predictions(y_test_t, y_pred_t)
    type_metrics["train_samples"] = int(len(x_train_t))
    type_metrics["test_samples"] = int(len(x_test_t))
    type_metrics["classes"] = sorted(df_types["contract_type"].unique().tolist())

    joblib.dump(
        contract_type_pipe,
        models_dir / "contract_type_classifier.joblib",
        compress=3,
    )

    # 2. Clause Risk Level & Risk Category Classifiers (LEDGAR + CUAD)
    train_clauses, test_clauses = load_clause_risk_datasets(
        raw_dir=raw_dir, force_download=force_download
    )

    risk_level_pipe = build_text_classifier(max_features=30000, ngram_range=(1, 2), c_value=2.0)
    risk_level_pipe.fit(train_clauses["text"], train_clauses["risk_level"])
    y_pred_level = risk_level_pipe.predict(test_clauses["text"])
    level_metrics = _evaluate_predictions(test_clauses["risk_level"], y_pred_level)
    level_metrics["train_samples"] = int(len(train_clauses))
    level_metrics["test_samples"] = int(len(test_clauses))
    level_metrics["classes"] = sorted(train_clauses["risk_level"].unique().tolist())

    joblib.dump(
        risk_level_pipe,
        models_dir / "clause_risk_classifier.joblib",
        compress=3,
    )

    risk_cat_pipe = build_text_classifier(max_features=30000, ngram_range=(1, 2), c_value=2.5)
    risk_cat_pipe.fit(train_clauses["text"], train_clauses["risk_category"])
    y_pred_cat = risk_cat_pipe.predict(test_clauses["text"])
    cat_metrics = _evaluate_predictions(test_clauses["risk_category"], y_pred_cat)
    cat_metrics["train_samples"] = int(len(train_clauses))
    cat_metrics["test_samples"] = int(len(test_clauses))
    cat_metrics["classes"] = sorted(train_clauses["risk_category"].unique().tolist())

    joblib.dump(
        risk_cat_pipe,
        models_dir / "clause_category_classifier.joblib",
        compress=3,
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "contract_type": ["CUAD v1 (510 contracts)", "ContractNLI (NDAs)"],
            "clause_risk": ["LEDGAR / LexGLUE (80k provisions)", "CUAD v1 (extracted clauses)"],
        },
        "contract_type_classifier": type_metrics,
        "clause_risk_classifier": level_metrics,
        "clause_category_classifier": cat_metrics,
    }

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
