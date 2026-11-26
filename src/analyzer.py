import json
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

try:
    from src.ml.predictor import ContractMLPredictor
    from src.models import (
        ClauseRiskPrediction,
        ContractAnalysys,
        ContractRisk,
        ContractTypePrediction,
    )
except ImportError:
    from ml.predictor import ContractMLPredictor
    from models import (
        ClauseRiskPrediction,
        ContractAnalysys,
        ContractRisk,
        ContractTypePrediction,
    )

DEFAULT_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """
You are a highly experienced and meticulous Senior Contract Risk Analyst, specializing in complex Construction and Enterprise IT agreements. Your tone must be authoritative, objective, and detailed.

Your core mission is to critically review the provided contract and identify EVERY instance of an unfavorable or ambiguous term. Focus on the financial and legal exposure for the client.

TARGET RISKS:
- Vague payment terms (Lacking specific dates, milestones, or conditions)
- Uncapped liability (Any "any and all damages" clauses)
- Ambiguous scope of work (Subject to change, undefined deliverables)
- Missing termination terms (No "with or without cause" options)
- Missing insurance requirements (Critical in Construction Subcontracts)
- Broad indemnification clauses (Transferring excessive risk)
- Overly unilateral terms (Heavily favoring one party, e.g., in termination or payment)

CRITICAL OUTPUT GUIDELINES:
1. CLAUSE TEXT: The 'clause_text' field MUST be the precise, verbatim text block from the contract that contains the risk.
2. REMEDIATION: The 'remediation_suggestion' field MUST provide a concrete, actionable revision that mitigates the risk.
3. SCHEMA: You MUST respond ONLY with a valid JSON object strictly matching the required schema. The root object MUST contain the key 'risks'. If NO significant risks are identified, return {"risks": []}.
"""

_ML_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "Uncapped Liability": (
        "ML classifier flagged this clause under Uncapped Liability, which may expose a party to unlimited financial damages.",
        "Cap aggregate liability to a fixed dollar amount or fees paid in the preceding 12 months, and exclude consequential damages.",
    ),
    "Broad Indemnification": (
        "ML classifier detected broad indemnification/release language that may shift disproportionate third-party or legal liability.",
        "Limit indemnification to third-party claims arising from gross negligence, willful misconduct, or material breach, subject to notice and control of defense.",
    ),
    "Termination Risk": (
        "ML classifier flagged termination or renewal terms that may create lock-in, unilateral exit rights, or insufficient notice.",
        "Add mutual termination for convenience with a clear notice period (e.g., 30 days) and clarify post-termination payment/transition obligations.",
    ),
    "Payment & Fee Terms": (
        "ML classifier flagged payment, fee, or withholding language that may contain ambiguous milestones or unfavorable payment timing.",
        "Specify exact net payment deadlines (e.g., Net 30), objective acceptance milestones, and dispute resolution procedures for contested invoices.",
    ),
    "Insurance Requirements": (
        "ML classifier flagged insurance provisions that may omit minimum coverage limits or mutual proof-of-insurance obligations.",
        "Define minimum required coverage types and limits (e.g., Commercial General Liability, Cyber/E&O) and require certificates of insurance prior to work.",
    ),
    "IP & Licensing Risk": (
        "ML classifier flagged intellectual property or licensing terms that may grant overly broad, perpetual, or irrevocable rights.",
        "Clarify ownership of background vs. foreground IP and restrict license scope, territory, and duration to the agreed business purpose.",
    ),
    "Restrictive Covenants": (
        "ML classifier flagged restrictive covenant language (e.g., exclusivity, non-compete, or non-solicit) that may limit future business.",
        "Narrow the scope, duration, and geographic reach of any restrictive covenant, or remove unilateral exclusivity requirements.",
    ),
    "Liability & Dispute Terms": (
        "ML classifier flagged liability, warranty, or dispute resolution terms that may waive key legal rights or remedies.",
        "Ensure balanced warranty periods, mutual dispute resolution procedures, and proportional remedies.",
    ),
    "Unilateral / Assignment Terms": (
        "ML classifier flagged assignment or change-of-control provisions that may allow unilateral transfer or restrict restructuring.",
        "Require prior written consent for assignment (not to be unreasonably withheld), except in connection with a merger or acquisition.",
    ),
    "Confidentiality & Compliance": (
        "ML classifier flagged confidentiality or representation obligations that may be asymmetric or overly broad.",
        "Make confidentiality obligations mutual with standard exclusions (public domain, prior knowledge, legal compulsion) and a defined survival period.",
    ),
}


def _strip_code_fences(response_text: str) -> str:
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def analyze_contract_with_gemini(
    contract_text: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    ml_context: Optional[str] = None,
) -> ContractAnalysys:
    """Analyze contract using Google Gemini API with structured output enforced by Pydantic."""
    client = genai.Client(api_key=api_key)
    schema_dict = ContractAnalysys.model_json_schema()

    context_block = f"\n\nLOCAL ML TRIAGE CONTEXT:\n{ml_context}\n" if ml_context else ""
    user_prompt = (
        f"Analyze this contract and identify all significant risks."
        f"{context_block}\nCONTRACT TEXT:\n\n{contract_text}"
    )

    response_text = ""

    try:
        response = client.models.generate_content(
            model=model,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=schema_dict,
            ),
        )

        response_text = _strip_code_fences(response.text)
        parsed_json = json.loads(response_text)
        final_data_for_pydantic = parsed_json

        if isinstance(parsed_json, dict):
            if len(parsed_json) == 1:
                root_key = next(iter(parsed_json.keys()))
                if isinstance(parsed_json[root_key], dict) and "risks" in parsed_json[root_key]:
                    final_data_for_pydantic = parsed_json[root_key]

            if "risks" in final_data_for_pydantic and isinstance(final_data_for_pydantic["risks"], list):
                final_data_for_pydantic = {"risks": final_data_for_pydantic["risks"]}
            elif isinstance(final_data_for_pydantic, list):
                final_data_for_pydantic = {"risks": final_data_for_pydantic}

        return ContractAnalysys(**final_data_for_pydantic)

    except json.JSONDecodeError as e:
        return ContractAnalysys(
            risks=[
                ContractRisk(
                    risk_type="Analysis Error (JSON Decode)",
                    clause_text=f"Raw Response Start: {response_text[:100]}...",
                    explanation=f"Failed to analyze LLM response. The model did not return valid JSON. Error: {str(e)}",
                    remediation_suggestion="Try again or use a 'Pro' model.",
                )
            ]
        )

    except Exception as e:
        error_explanation = str(e)
        if "API key not valid" in error_explanation or "UNAUTHENTICATED" in error_explanation:
            error_explanation = "Invalid or missing Google AI API key."
        elif "validation error for ContractAnalysys" in error_explanation:
            error_explanation = (
                "Pydantic validation error: JSON structure does not match expected schema. "
                f"{error_explanation}"
            )

        return ContractAnalysys(
            risks=[
                ContractRisk(
                    risk_type="System/Validation Error",
                    clause_text="N/A",
                    explanation=f"Error during analysis: {error_explanation}",
                    remediation_suggestion="Verify API key and connection. If LLM fails, try a more robust model (1.5 Pro).",
                )
            ]
        )


def analyze_contract_with_gemini_pro(
    contract_text: str,
    api_key: str,
) -> ContractAnalysys:
    """Analyze contract using Gemini 1.5 Pro for more thorough analysis."""
    return analyze_contract_with_gemini(
        contract_text=contract_text,
        api_key=api_key,
        model="gemini-1.5-pro-latest",
    )


def _ml_predictions_to_risks(clause_preds: List[ClauseRiskPrediction]) -> List[ContractRisk]:
    """Convert flagged ML clause predictions into ContractRisk items."""
    flagged = [c for c in clause_preds if c.risk_level in ("High", "Medium")]
    flagged.sort(key=lambda c: (0 if c.risk_level == "High" else 1, -c.confidence))

    risks: List[ContractRisk] = []
    for pred in flagged:
        explanation, remediation = _ML_TEMPLATES.get(
            pred.risk_category,
            (
                f"ML classifier flagged this clause as {pred.risk_level} risk ({pred.risk_category}).",
                "Review this clause with legal counsel to ensure balanced obligations and clear limits.",
            ),
        )
        risks.append(
            ContractRisk(
                risk_type=pred.risk_category,
                clause_text=pred.clause_text,
                explanation=f"{explanation} (ML confidence: {pred.confidence:.0%})",
                remediation_suggestion=remediation,
                risk_level=pred.risk_level,
                ml_confidence=pred.confidence,
                source="ml",
            )
        )
    return risks


def _match_ml_metadata(
    clause_text: str, clause_preds: List[ClauseRiskPrediction]
) -> Optional[ClauseRiskPrediction]:
    """Find the best matching ML clause prediction for a Gemini-extracted clause."""
    needle = clause_text.strip().lower()[:80]
    if not needle:
        return None
    for pred in clause_preds:
        haystack = pred.clause_text.lower()
        if needle in haystack or haystack[:80] in clause_text.lower():
            return pred
    return None


def analyze_contract_hybrid(
    contract_text: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    predictor: Optional[ContractMLPredictor] = None,
) -> ContractAnalysys:
    """Run V2 hybrid pipeline: local ML classification + optional Gemini deep analysis."""
    ml_predictor = predictor or ContractMLPredictor()
    contract_type: ContractTypePrediction = ml_predictor.predict_contract_type(contract_text)
    clause_preds: List[ClauseRiskPrediction] = ml_predictor.predict_clause_risks(contract_text)

    if not api_key or not api_key.strip():
        return ContractAnalysys(
            risks=_ml_predictions_to_risks(clause_preds),
            contract_type=contract_type,
            ml_clause_predictions=clause_preds,
            analysis_mode="ml_only",
        )

    flagged = [c for c in clause_preds if c.risk_level in ("High", "Medium")]
    triage_lines = [
        f"Predicted Contract Type: {contract_type.contract_type} (confidence: {contract_type.confidence:.0%})",
        f"ML Flagged Clauses ({len(flagged)}/{len(clause_preds)}):",
    ]
    for c in flagged[:12]:
        triage_lines.append(
            f"- [Clause #{c.clause_index} | {c.risk_level} | {c.risk_category} | conf={c.confidence:.2f}]: {c.clause_text[:240]}"
        )

    gemini_result = analyze_contract_with_gemini(
        contract_text=contract_text,
        api_key=api_key,
        model=model,
        ml_context="\n".join(triage_lines),
    )

    enriched_risks: List[ContractRisk] = []
    for risk in gemini_result.risks:
        matched = _match_ml_metadata(risk.clause_text, clause_preds)
        enriched_risks.append(
            ContractRisk(
                risk_type=risk.risk_type,
                clause_text=risk.clause_text,
                explanation=risk.explanation,
                remediation_suggestion=risk.remediation_suggestion,
                risk_level=matched.risk_level if matched else risk.risk_level,
                ml_confidence=matched.confidence if matched else risk.ml_confidence,
                source="hybrid",
            )
        )

    return ContractAnalysys(
        risks=enriched_risks,
        contract_type=contract_type,
        ml_clause_predictions=clause_preds,
        analysis_mode="hybrid",
    )
