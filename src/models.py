from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class ContractTypePrediction(BaseModel):
    """ML prediction for the overall contract type."""

    contract_type: str = Field(
        description="Predicted contract category (e.g., 'Service / Consulting Agreement', 'Non-Disclosure Agreement (NDA)')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability for the top contract type",
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Class probability distribution across all contract types",
    )


class ClauseRiskPrediction(BaseModel):
    """ML prediction for a single contract clause."""

    clause_index: int = Field(description="1-based clause index in the document")
    clause_text: str = Field(description="Extracted clause text")
    risk_level: Literal["High", "Medium", "Low"] = Field(
        description="Predicted risk severity level"
    )
    risk_category: str = Field(
        description="Predicted legal/risk category for the clause"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted risk_level",
    )
    category_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted risk_category",
    )


class ContractRisk(BaseModel):
    """Schema for a single identified contract risk."""

    risk_type: str = Field(
        description="Type of risk (e.g., 'Uncapped Liability', 'Vague Payment Terms')"
    )
    clause_text: str = Field(
        description="Text of the clause that contains the risk"
    )
    explanation: str = Field(
        description="Why this is a risk and its potential legal/financial impact"
    )
    remediation_suggestion: str = Field(
        description="How the clause could be modified to mitigate the risk"
    )
    risk_level: Optional[str] = Field(
        default=None,
        description="Optional severity level (High, Medium, Low)",
    )
    ml_confidence: Optional[float] = Field(
        default=None,
        description="Optional ML classifier confidence score",
    )
    source: str = Field(
        default="gemini",
        description="Origin of the finding ('gemini', 'ml', or 'hybrid')",
    )


class ContractAnalysys(BaseModel):
    """Root schema containing contract classification, ML triage, and identified risks."""

    risks: List[ContractRisk] = Field(
        description="List of all identified risks"
    )
    contract_type: Optional[ContractTypePrediction] = Field(
        default=None,
        description="ML-predicted contract type",
    )
    ml_clause_predictions: List[ClauseRiskPrediction] = Field(
        default_factory=list,
        description="Clause-level ML risk triage predictions",
    )
    analysis_mode: str = Field(
        default="gemini",
        description="Execution mode used ('gemini', 'ml_only', or 'hybrid')",
    )

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


ContractAnalysis = ContractAnalysys
