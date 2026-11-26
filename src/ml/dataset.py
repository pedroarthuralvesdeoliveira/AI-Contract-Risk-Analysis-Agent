"""Public dataset download and preparation for V2 contract ML models.

Uses three open legal NLP datasets:
1. CUAD v1 (The Atticus Project) - commercial contracts & 41 clause categories
2. ContractNLI (kiddothe2b/contract-nli) - Non-Disclosure Agreements (NDAs)
3. LEDGAR (coastalcph/lex_glue) - 80,000 labeled contract provisions across 100 types
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

CUAD_ZIP_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
CONTRACT_NLI_URL = (
    "https://huggingface.co/api/datasets/kiddothe2b/contract-nli/"
    "parquet/contractnli_b/train/0.parquet"
)
LEDGAR_URLS: Dict[str, str] = {
    "train": "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/ledgar/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/ledgar/validation/0.parquet",
    "test": "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/ledgar/test/0.parquet",
}

LEDGAR_LABEL_NAMES = [
    "Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws",
    "Approvals", "Arbitration", "Assignments", "Assigns", "Authority",
    "Authorizations", "Base Salary", "Benefits", "Binding Effects", "Books",
    "Brokers", "Capitalization", "Change In Control", "Closings", "Compliance With Laws",
    "Confidentiality", "Consent To Jurisdiction", "Consents", "Construction", "Cooperation",
    "Costs", "Counterparts", "Death", "Defined Terms", "Definitions",
    "Disability", "Disclosures", "Duties", "Effective Dates", "Effectiveness",
    "Employment", "Enforceability", "Enforcements", "Entire Agreements", "Erisa",
    "Existence", "Expenses", "Fees", "Financial Statements", "Forfeitures",
    "Further Assurances", "General", "Governing Laws", "Headings", "Indemnifications",
    "Indemnity", "Insurances", "Integration", "Intellectual Property", "Interests",
    "Interpretations", "Jurisdictions", "Liens", "Litigations", "Miscellaneous",
    "Modifications", "No Conflicts", "No Defaults", "No Waivers", "Non-Disparagement",
    "Notices", "Organizations", "Participations", "Payments", "Positions",
    "Powers", "Publicity", "Qualifications", "Records", "Releases",
    "Remedies", "Representations", "Sales", "Sanctions", "Severability",
    "Solvency", "Specific Performance", "Submission To Jurisdiction", "Subsidiaries", "Successors",
    "Survival", "Tax Withholdings", "Taxes", "Terminations", "Terms",
    "Titles", "Transactions With Affiliates", "Use Of Proceeds", "Vacations", "Venues",
    "Vesting", "Waiver Of Jury Trials", "Waivers", "Warranties", "Withholdings",
]

# Mapping from CUAD / LEDGAR fine-grained labels to (risk_level, risk_category).
CUAD_RISK_MAP: Dict[str, Tuple[str, str]] = {
    "Uncapped Liability": ("High", "Uncapped Liability"),
    "Liquidated Damages": ("High", "Uncapped Liability"),
    "Non-Compete": ("High", "Restrictive Covenants"),
    "Exclusivity": ("High", "Restrictive Covenants"),
    "Most Favored Nation": ("High", "Restrictive Covenants"),
    "No-Solicit Of Customers": ("Medium", "Restrictive Covenants"),
    "No-Solicit Of Employees": ("Medium", "Restrictive Covenants"),
    "Non-Disparagement": ("Medium", "Restrictive Covenants"),
    "Termination For Convenience": ("High", "Termination Risk"),
    "Renewal Term": ("Medium", "Termination Risk"),
    "Notice Period To Terminate Renewal": ("Medium", "Termination Risk"),
    "Post-Termination Services": ("Medium", "Termination Risk"),
    "Ip Ownership Assignment": ("High", "IP & Licensing Risk"),
    "Irrevocable Or Perpetual License": ("High", "IP & Licensing Risk"),
    "Unlimited/All-You-Can-Eat-License": ("High", "IP & Licensing Risk"),
    "Joint Ip Ownership": ("Medium", "IP & Licensing Risk"),
    "Source Code Escrow": ("Medium", "IP & Licensing Risk"),
    "License Grant": ("Medium", "IP & Licensing Risk"),
    "Non-Transferable License": ("Medium", "IP & Licensing Risk"),
    "Covenant Not To Sue": ("High", "Liability & Dispute Terms"),
    "Cap On Liability": ("Medium", "Liability & Dispute Terms"),
    "Warranty Duration": ("Medium", "Liability & Dispute Terms"),
    "Audit Rights": ("Medium", "Liability & Dispute Terms"),
    "Insurance": ("Medium", "Insurance Requirements"),
    "Minimum Commitment": ("Medium", "Payment & Fee Terms"),
    "Price Restrictions": ("Medium", "Payment & Fee Terms"),
    "Revenue/Profit Sharing": ("Medium", "Payment & Fee Terms"),
    "Change Of Control": ("Medium", "Unilateral / Assignment Terms"),
    "Anti-Assignment": ("Medium", "Unilateral / Assignment Terms"),
    "Rofr/Rofo/Rofn": ("Medium", "Unilateral / Assignment Terms"),
    "Volume Restriction": ("Medium", "Unilateral / Assignment Terms"),
    "Governing Law": ("Low", "Standard / Low-Risk Clause"),
}

LEDGAR_RISK_MAP: Dict[str, Tuple[str, str]] = {
    "Indemnifications": ("High", "Broad Indemnification"),
    "Indemnity": ("High", "Broad Indemnification"),
    "Releases": ("High", "Broad Indemnification"),
    "Waiver Of Jury Trials": ("High", "Liability & Dispute Terms"),
    "Forfeitures": ("High", "Liability & Dispute Terms"),
    "Liens": ("High", "Liability & Dispute Terms"),
    "Sanctions": ("High", "Liability & Dispute Terms"),
    "Warranties": ("Medium", "Liability & Dispute Terms"),
    "Remedies": ("Medium", "Liability & Dispute Terms"),
    "Arbitration": ("Medium", "Liability & Dispute Terms"),
    "Terminations": ("Medium", "Termination Risk"),
    "Insurances": ("Medium", "Insurance Requirements"),
    "Payments": ("Medium", "Payment & Fee Terms"),
    "Fees": ("Medium", "Payment & Fee Terms"),
    "Expenses": ("Medium", "Payment & Fee Terms"),
    "Withholdings": ("Medium", "Payment & Fee Terms"),
    "Tax Withholdings": ("Medium", "Payment & Fee Terms"),
    "Intellectual Property": ("Medium", "IP & Licensing Risk"),
    "Confidentiality": ("Medium", "Confidentiality & Compliance"),
    "Representations": ("Medium", "Confidentiality & Compliance"),
    "Disclosures": ("Medium", "Confidentiality & Compliance"),
    "Duties": ("Medium", "Confidentiality & Compliance"),
    "Change In Control": ("Medium", "Unilateral / Assignment Terms"),
    "Assignments": ("Medium", "Unilateral / Assignment Terms"),
    "Assigns": ("Medium", "Unilateral / Assignment Terms"),
    "Non-Disparagement": ("Medium", "Restrictive Covenants"),
}


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file if it is not already cached on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


def map_cuad_contract_type(title: str) -> str:
    """Normalize a CUAD contract title into a canonical contract type."""
    t = title.upper()
    if any(k in t for k in ("NON-DISCLOSURE", "NDA", "CONFIDENTIALITY")):
        return "Non-Disclosure Agreement (NDA)"
    if any(k in t for k in ("LICENSE", "LICENSING", "INTELLECTUAL PROPERTY", "SOFTWARE", "HOSTING", "SAAS", "TECHNOLOGY")):
        return "Software / IP Licensing"
    if any(k in t for k in ("SERVICE", "MAINTENANCE", "OUTSOURCING", "CONSULTING", "DEVELOPMENT", "MANUFACTURING", "CONTRACTOR", "SUBCONTRACT")):
        return "Service / Consulting Agreement"
    if any(k in t for k in ("DISTRIBUT", "RESELLER", "MARKETING", "PROMOTION", "SPONSORSHIP", "ENDORSEMENT", "FRANCHISE", "AGENCY")):
        return "Distribution / Marketing Agreement"
    if any(k in t for k in ("SUPPLY", "PURCHASE", "PROCUREMENT")):
        return "Supply / Purchase Agreement"
    if any(k in t for k in ("JOINT VENTURE", "STRATEGIC ALLIANCE", "PARTNERSHIP", "COLLABORATION", "CO-OPERATION", "COOPERATION")):
        return "Joint Venture / Partnership"
    return "Other Commercial Agreement"


def map_ledgar_risk(label_name: str) -> Tuple[str, str]:
    """Map a LEDGAR provision label to (risk_level, risk_category)."""
    return LEDGAR_RISK_MAP.get(label_name, ("Low", "Standard / Low-Risk Clause"))


def load_contract_type_dataset(
    raw_dir: Path,
    max_nda_samples: int = 150,
    max_text_chars: int = 12000,
    force_download: bool = False,
) -> pd.DataFrame:
    """Build contract-type classification DataFrame from CUAD v1 + ContractNLI."""
    cuad_zip = download_file(CUAD_ZIP_URL, raw_dir / "cuad_data.zip", force=force_download)
    nli_parquet = download_file(
        CONTRACT_NLI_URL, raw_dir / "contractnli_b_train.parquet", force=force_download
    )

    rows = []
    with zipfile.ZipFile(cuad_zip) as zf:
        with zf.open("CUADv1.json") as f:
            cuad = json.load(f)

    for entry in cuad["data"]:
        title = entry.get("title", "")
        paragraphs = entry.get("paragraphs", [])
        if not paragraphs:
            continue
        context = paragraphs[0].get("context", "").strip()
        if len(context) < 100:
            continue
        rows.append(
            {
                "text": context[:max_text_chars],
                "contract_type": map_cuad_contract_type(title),
                "source": "cuad",
            }
        )

    df_nli = pd.read_parquet(nli_parquet)
    unique_ndas = (
        df_nli["premise"]
        .dropna()
        .drop_duplicates()
        .iloc[:max_nda_samples]
        .tolist()
    )
    for nda_text in unique_ndas:
        text = str(nda_text).strip()
        if len(text) >= 100:
            rows.append(
                {
                    "text": text[:max_text_chars],
                    "contract_type": "Non-Disclosure Agreement (NDA)",
                    "source": "contract_nli",
                }
            )

    return pd.DataFrame(rows)


def load_clause_risk_datasets(
    raw_dir: Path,
    max_train_per_ledgar_class: int = 400,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build train and test DataFrames for clause risk classification from LEDGAR + CUAD."""
    cuad_zip = download_file(CUAD_ZIP_URL, raw_dir / "cuad_data.zip", force=force_download)
    ledgar_files = {
        split: download_file(url, raw_dir / f"ledgar_{split}.parquet", force=force_download)
        for split, url in LEDGAR_URLS.items()
    }

    def _convert_ledgar(df: pd.DataFrame, cap_per_label: int | None = None) -> pd.DataFrame:
        if cap_per_label is not None:
            df = (
                df.groupby("label", group_keys=False)
                .head(cap_per_label)
                .reset_index(drop=True)
            )
        records = []
        for text, label_idx in zip(df["text"], df["label"]):
            clean_text = str(text).strip()
            if len(clean_text) < 20:
                continue
            label_name = LEDGAR_LABEL_NAMES[int(label_idx)]
            risk_level, risk_category = map_ledgar_risk(label_name)
            records.append(
                {
                    "text": clean_text,
                    "risk_level": risk_level,
                    "risk_category": risk_category,
                    "raw_label": label_name,
                    "source": "ledgar",
                }
            )
        return pd.DataFrame(records)

    df_ledgar_train = pd.concat(
        [
            pd.read_parquet(ledgar_files["train"]),
            pd.read_parquet(ledgar_files["validation"]),
        ],
        ignore_index=True,
    )
    df_ledgar_test = pd.read_parquet(ledgar_files["test"])

    train_df = _convert_ledgar(df_ledgar_train, cap_per_label=max_train_per_ledgar_class)
    test_df = _convert_ledgar(df_ledgar_test, cap_per_label=None)

    # Extract CUAD risky clauses and split 80/20 deterministically by contract index
    with zipfile.ZipFile(cuad_zip) as zf:
        with zf.open("CUADv1.json") as f:
            cuad = json.load(f)

    cuad_train_rows = []
    cuad_test_rows = []
    for idx, entry in enumerate(cuad["data"]):
        target = cuad_test_rows if idx % 5 == 0 else cuad_train_rows
        for para in entry.get("paragraphs", []):
            for qa in para.get("qas", []):
                cat = qa["id"].split("__")[-1]
                if cat not in CUAD_RISK_MAP:
                    continue
                risk_level, risk_category = CUAD_RISK_MAP[cat]
                for ans in qa.get("answers", []):
                    ans_text = ans.get("text", "").strip()
                    if len(ans_text) < 25:
                        continue
                    target.append(
                        {
                            "text": ans_text,
                            "risk_level": risk_level,
                            "risk_category": risk_category,
                            "raw_label": cat,
                            "source": "cuad",
                        }
                    )

    if cuad_train_rows:
        train_df = pd.concat([train_df, pd.DataFrame(cuad_train_rows)], ignore_index=True)
    if cuad_test_rows:
        test_df = pd.concat([test_df, pd.DataFrame(cuad_test_rows)], ignore_index=True)

    return train_df, test_df
