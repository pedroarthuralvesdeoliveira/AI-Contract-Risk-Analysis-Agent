"""Local ML inference for contract type and clause-level risk classification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.models import ClauseRiskPrediction, ContractTypePrediction

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def split_into_clauses(contract_text: str, min_length: int = 35) -> List[str]:
    """Split raw contract text into meaningful clause/paragraph segments."""
    if not contract_text or not contract_text.strip():
        return []

    normalized = re.sub(r"\r\n?", "\n", contract_text).strip()
    # Split on blank lines first; if a single block has numbered sections, split there too
    raw_blocks = re.split(r"\n\s*\n+", normalized)
    segments: List[str] = []
    for block in raw_blocks:
        sub_blocks = re.split(r"\n(?=(?:\d+\.|Section\s+\d+|[A-Z]\.)\s+[A-Za-z])", block)
        for sub in sub_blocks:
            cleaned = re.sub(r"\s+", " ", sub).strip()
            if cleaned:
                segments.append(cleaned)

    # Merge short headings (e.g., "1. Scope") into the following clause
    clauses: List[str] = []
    pending_header = ""
    for seg in segments:
        if len(seg) < min_length:
            pending_header = f"{pending_header} {seg}".strip() if pending_header else seg
            continue
        if pending_header:
            seg = f"{pending_header}: {seg}"
            pending_header = ""
        clauses.append(seg)

    if pending_header and not clauses:
        clauses.append(pending_header)

    return clauses


class ContractMLPredictor:
    """Loads trained scikit-learn pipelines and runs local contract & clause inference."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        type_path = self.models_dir / "contract_type_classifier.joblib"
        risk_path = self.models_dir / "clause_risk_classifier.joblib"
        cat_path = self.models_dir / "clause_category_classifier.joblib"

        missing = [p.name for p in (type_path, risk_path, cat_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing trained ML model artifacts in {self.models_dir}: {missing}. "
                "Run `uv run python scripts/train_models.py` first."
            )

        self.contract_type_model = joblib.load(type_path)
        self.clause_risk_model = joblib.load(risk_path)
        self.clause_category_model = joblib.load(cat_path)
        self.metrics_path = self.models_dir / "metrics.json"

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Load saved evaluation metrics if available."""
        if not self.metrics_path.exists():
            return None
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict_contract_type(
        self, contract_text: str, max_chars: int = 12000
    ) -> ContractTypePrediction:
        """Predict the contract type and class probabilities from document text."""
        snippet = (contract_text or "").strip()[:max_chars]
        if not snippet:
            return ContractTypePrediction(
                contract_type="Unknown",
                confidence=0.0,
                probabilities={},
            )

        probs = self.contract_type_model.predict_proba([snippet])[0]
        classes = self.contract_type_model.classes_
        prob_map = {
            str(cls): round(float(prob), 4)
            for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        }
        top_class = next(iter(prob_map))
        return ContractTypePrediction(
            contract_type=top_class,
            confidence=prob_map[top_class],
            probabilities=prob_map,
        )

    def predict_clause_risks(self, contract_text: str) -> List[ClauseRiskPrediction]:
        """Classify every extracted clause by risk level and risk category."""
        clauses = split_into_clauses(contract_text)
        if not clauses:
            return []

        risk_probs = self.clause_risk_model.predict_proba(clauses)
        risk_classes = self.clause_risk_model.classes_

        cat_probs = self.clause_category_model.predict_proba(clauses)
        cat_classes = self.clause_category_model.classes_

        predictions: List[ClauseRiskPrediction] = []
        for idx, clause in enumerate(clauses, start=1):
            r_idx = int(risk_probs[idx - 1].argmax())
            c_idx = int(cat_probs[idx - 1].argmax())

            predictions.append(
                ClauseRiskPrediction(
                    clause_index=idx,
                    clause_text=clause,
                    risk_level=str(risk_classes[r_idx]),
                    risk_category=str(cat_classes[c_idx]),
                    confidence=round(float(risk_probs[idx - 1][r_idx]), 4),
                    category_confidence=round(float(cat_probs[idx - 1][c_idx]), 4),
                )
            )

        return predictions
