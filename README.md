# AI Contract Risk Analysis Agent (V2)

Production-ready hybrid AI pipeline for **contract type classification** and **clause-level legal/financial risk detection**, combining local **Machine Learning (scikit-learn)** trained on public legal datasets with **Google Gemini** for deep legal explanations and remediation suggestions.

## 🎯 What's New in V2

- **Contract Type Classification (Local ML)**: Automatically classifies uploaded contracts into 7 categories (`Service / Consulting Agreement`, `Non-Disclosure Agreement (NDA)`, `Software / IP Licensing`, `Distribution / Marketing Agreement`, `Supply / Purchase Agreement`, `Joint Venture / Partnership`, `Other Commercial Agreement`).
- **Clause-Level Risk Triage (Local ML)**: Splits contracts into clauses and classifies every clause by **Risk Level** (`High`, `Medium`, `Low`) and **Risk Category** (11 legal/risk categories such as `Uncapped Liability`, `Broad Indemnification`, `Termination Risk`, `Payment & Fee Terms`, `Insurance Requirements`, `IP & Licensing Risk`, `Restrictive Covenants`, etc.).
- **3 Analysis Modes in Streamlit**:
  - **Hybrid (ML + Gemini)**: Local ML classifies contract type and flags risky clauses first, then Gemini provides contextual explanations and concrete remediations.
  - **ML Only (Offline / Local)**: 100% local classification & template-based remediation with **no API key required**.
  - **Gemini Only (V1)**: Classic end-to-end LLM analysis.
- **Reproducible Training Pipeline**: Automated script (`scripts/train_models.py`) to download public datasets, train, evaluate, and export `.joblib` models + `models/metrics.json`.

## 🏗️ V2 Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────────┐
│  PDF / TXT  │ -> │ PyMuPDF / OCR    │ -> │ Local ML (TF-IDF + LogReg)     │
│   Upload    │    │ Text Extraction  │    │ 1. Contract Type Classifier    │
└─────────────┘    └──────────────────┘    │ 2. Clause Risk & Category ML   │
                                           └────────────────────────────────┘
                                                            │
                                                            v
                                           ┌────────────────────────────────┐
                                           │ Optional Gemini Enrichment     │
                                           │ (Explanations & Remediations)  │
                                           └────────────────────────────────┘
                                                            │
                                                            v
                                           ┌────────────────────────────────┐
                                           │ Pydantic V2 + Streamlit UI     │
                                           └────────────────────────────────┘
```

## 📊 Public Datasets & ML Benchmarks

Models are trained using `TfidfVectorizer` (unigrams + bigrams, `sublinear_tf=True`) + balanced `LogisticRegression`:

| Classifier | Public Datasets | Train / Test Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| **Contract Type** (`contract_type_classifier.joblib`) | CUAD v1 (510 contracts) + ContractNLI (150 NDAs) | 528 / 132 | **79.55%** | **76.83%** | **79.54%** |
| **Clause Risk Level** (`clause_risk_classifier.joblib`) | LEDGAR (LexGLUE) + CUAD v1 clauses | 42,728 / 11,732 | **90.94%** | **85.73%** | **91.07%** |
| **Clause Risk Category** (`clause_category_classifier.joblib`) | LEDGAR (LexGLUE) + CUAD v1 clauses | 42,728 / 11,732 | **87.97%** | **79.47%** | **88.48%** |

Full per-class metrics are stored in [`models/metrics.json`](models/metrics.json).

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/pedroarthuralvesdeoliveira/AI-Contract-Risk-Analysis-Agent.git
cd AI-Contract-Risk-Analysis-Agent
uv sync
```

### Train / Reproduce ML Models (Optional — pre-trained artifacts are included in `models/`)

```bash
uv run python scripts/train_models.py
```

This downloads CUAD v1, ContractNLI, and LEDGAR into `data/raw/`, trains all three classifiers, and saves the `.joblib` artifacts and `metrics.json` into `models/`.

### Run Streamlit App

```bash
# Optional for Hybrid / Gemini Only modes:
export GOOGLE_API_KEY="your_api_key_here"

uv run streamlit run src/app.py
```

### Run Tests

```bash
uv run pytest -v
```

## 📁 Project Structure

```
AI-Contract-Risk-Analysis-Agent/
├── models/                               # Pre-trained V2 ML artifacts & metrics
│   ├── contract_type_classifier.joblib
│   ├── clause_risk_classifier.joblib
│   ├── clause_category_classifier.joblib
│   └── metrics.json
├── scripts/
│   └── train_models.py                   # Reproducible dataset download & training CLI
├── src/
│   ├── ml/
│   │   ├── dataset.py                    # CUAD, ContractNLI & LEDGAR loaders
│   │   ├── train.py                      # Training & evaluation pipeline
│   │   └── predictor.py                  # Clause splitting & local ML inference
│   ├── models.py                         # Pydantic V2 schemas (V1-compatible + V2 ML fields)
│   ├── parser.py                         # PyMuPDF + lazy PaddleOCR text extraction
│   ├── analyzer.py                       # Hybrid ML + Gemini & V1 Gemini pipelines
│   └── app.py                            # Streamlit V2 UI
├── tests/
│   ├── test_analyzer.py
│   ├── test_ml.py
│   └── test_parser.py
├── assets/                               # Sample contract PDFs
├── pyproject.toml                        # Dependencies & pytest config (managed by uv)
└── README.md
```

## 📄 License

MIT License
