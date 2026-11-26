import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.analyzer import analyze_contract_hybrid
from src.ml.dataset import map_cuad_contract_type, map_ledgar_risk
from src.ml.predictor import ContractMLPredictor, split_into_clauses
from src.models import ContractAnalysys
from src.parser import extract_text_from_pdf


def test_split_into_clauses_merges_short_headings():
    text = (
        "1. Scope\n\n"
        "Contractor will perform software development services for the Client platform.\n\n"
        "2. Liability\n\n"
        "Contractor shall be liable for any and all damages without any monetary cap."
    )
    clauses = split_into_clauses(text)
    assert len(clauses) == 2
    assert clauses[0].startswith("1. Scope:")
    assert "software development" in clauses[0]
    assert clauses[1].startswith("2. Liability:")


def test_dataset_mappers():
    assert map_cuad_contract_type("ACME_2020_SERVICE AGREEMENT") == "Service / Consulting Agreement"
    assert map_cuad_contract_type("XYZ_SOFTWARE_LICENSE_AGREEMENT") == "Software / IP Licensing"
    assert map_cuad_contract_type("MUTUAL_NON-DISCLOSURE_AGREEMENT") == "Non-Disclosure Agreement (NDA)"

    assert map_ledgar_risk("Indemnifications") == ("High", "Broad Indemnification")
    assert map_ledgar_risk("Terminations") == ("Medium", "Termination Risk")
    assert map_ledgar_risk("Governing Laws") == ("Low", "Standard / Low-Risk Clause")


def test_ml_predictor_on_sample_assets():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "contract_type_classifier.joblib").exists():
        pytest.skip("Trained models not present in models/")

    predictor = ContractMLPredictor(models_dir=models_dir)

    nda_path = Path(__file__).resolve().parent.parent / "assets" / "Non-Disclosure Agreement.pdf"
    nda_text = extract_text_from_pdf(nda_path)
    nda_pred = predictor.predict_contract_type(nda_text)
    assert nda_pred.contract_type == "Non-Disclosure Agreement (NDA)"
    assert nda_pred.confidence > 0.3

    lic_path = Path(__file__).resolve().parent.parent / "assets" / "Software Licensing Agreement.pdf"
    lic_text = extract_text_from_pdf(lic_path)
    lic_pred = predictor.predict_contract_type(lic_text)
    assert lic_pred.contract_type == "Software / IP Licensing"

    clause_preds = predictor.predict_clause_risks(lic_text)
    assert len(clause_preds) >= 2
    assert all(c.risk_level in ("High", "Medium", "Low") for c in clause_preds)


def test_analyze_contract_hybrid_ml_only():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "contract_type_classifier.joblib").exists():
        pytest.skip("Trained models not present in models/")

    sample_text = (
        "NON-DISCLOSURE AGREEMENT\n\n"
        "1. Confidentiality\n"
        "Recipient agrees not to disclose any confidential information.\n\n"
        "2. Indemnification\n"
        "Recipient shall indemnify, defend, and hold harmless Discloser from any and all claims, "
        "damages, losses, and liabilities without limitation.\n\n"
        "3. Governing Law\n"
        "This Agreement shall be governed by the laws of the State of Delaware."
    )

    result = analyze_contract_hybrid(contract_text=sample_text, api_key=None)
    assert isinstance(result, ContractAnalysys)
    assert result.analysis_mode == "ml_only"
    assert result.contract_type is not None
    assert result.contract_type.contract_type == "Non-Disclosure Agreement (NDA)"
    assert len(result.ml_clause_predictions) == 3
    assert any(r.source == "ml" for r in result.risks)


@patch("src.analyzer.genai.Client")
def test_analyze_contract_hybrid_with_gemini(mock_client):
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "contract_type_classifier.joblib").exists():
        pytest.skip("Trained models not present in models/")

    class MockResp:
        def __init__(self, text):
            self.text = text

    mock_payload = {
        "risks": [
            {
                "risk_type": "Broad Indemnification",
                "clause_text": "Recipient shall indemnify, defend, and hold harmless Discloser from any and all claims.",
                "explanation": "Broad unilateral indemnity.",
                "remediation_suggestion": "Limit to gross negligence or willful misconduct.",
            }
        ]
    }
    mock_client.return_value.models.generate_content.return_value = MockResp(json.dumps(mock_payload))

    sample_text = (
        "SERVICE AGREEMENT\n\n"
        "1. Indemnification\n"
        "Recipient shall indemnify, defend, and hold harmless Discloser from any and all claims, "
        "damages, and liabilities."
    )
    result = analyze_contract_hybrid(contract_text=sample_text, api_key="fake_key")
    assert result.analysis_mode == "hybrid"
    assert result.contract_type is not None
    assert len(result.risks) == 1
    assert result.risks[0].source == "hybrid"
