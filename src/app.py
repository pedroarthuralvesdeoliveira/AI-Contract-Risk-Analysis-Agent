import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from src.analyzer import (
        analyze_contract_hybrid,
        analyze_contract_with_gemini,
        analyze_contract_with_gemini_pro,
    )
    from src.ml.predictor import ContractMLPredictor
    from src.parser import extract_text_from_pdf
except ImportError:
    from analyzer import (
        analyze_contract_hybrid,
        analyze_contract_with_gemini,
        analyze_contract_with_gemini_pro,
    )
    from ml.predictor import ContractMLPredictor
    from parser import extract_text_from_pdf

st.set_page_config(
    page_title="Contract Risk Analyzer V2",
    page_icon="📋",
    layout="wide",
)

st.title("📋 AI Contract Risk Analysis (V2)")
st.markdown(
    "Classify contract types and clause-level risks with **local Machine Learning** "
    "(trained on CUAD, ContractNLI & LEDGAR), optionally enriched with **Google Gemini**."
)


@st.cache_resource
def load_ml_predictor():
    try:
        return ContractMLPredictor(), None
    except FileNotFoundError as exc:
        return None, str(exc)


predictor, ml_load_error = load_ml_predictor()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    analysis_mode = st.radio(
        "Analysis Mode",
        [
            "Hybrid (ML + Gemini)",
            "ML Only (Offline / Local)",
            "Gemini Only (V1)",
        ],
        help="Hybrid runs local ML classification first and enriches flagged clauses with Gemini.",
    )

    api_key = st.text_input(
        "Google AI API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        disabled=(analysis_mode == "ML Only (Offline / Local)"),
        help="Required for Hybrid and Gemini Only modes. Get a free key at https://aistudio.google.com/apikey",
    )

    model_choice = st.selectbox(
        "Gemini Model",
        [
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
        ],
        disabled=(analysis_mode == "ML Only (Offline / Local)"),
    )

    st.divider()
    st.markdown("### 🧠 Local ML Models (V2)")
    if predictor is not None:
        st.success("✅ ML models loaded")
        metrics = predictor.get_metrics()
        if metrics:
            ct = metrics.get("contract_type_classifier", {})
            cr = metrics.get("clause_risk_classifier", {})
            cc = metrics.get("clause_category_classifier", {})
            st.caption(
                f"**Contract Type**: Acc {ct.get('accuracy', 0):.1%} | F1 {ct.get('macro_f1', 0):.1%}\n\n"
                f"**Clause Risk Level**: Acc {cr.get('accuracy', 0):.1%} | F1 {cr.get('macro_f1', 0):.1%}\n\n"
                f"**Clause Category**: Acc {cc.get('accuracy', 0):.1%} | F1 {cc.get('macro_f1', 0):.1%}"
            )
    else:
        st.warning(f"⚠️ ML models unavailable: {ml_load_error}")

    st.divider()
    st.markdown("### 📊 Session Stats")
    if "total_analyzed" not in st.session_state:
        st.session_state.total_analyzed = 0
    if "total_risks" not in st.session_state:
        st.session_state.total_risks = 0

    st.metric("Contracts Analyzed", st.session_state.total_analyzed)
    st.metric("Total Risks Found", st.session_state.total_risks)

# Main content
uploaded_file = st.file_uploader(
    "Upload Contract (PDF)",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    help="Upload a contract document for classification and risk analysis",
)

if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()
    temp_path = Path(f"temp_contract.{ext}")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📄 Size: {uploaded_file.size / 1024:.1f} KB")

        with col2:
            analyze_btn = st.button(
                f"🔍 Analyze Contract ({analysis_mode})",
                type="primary",
                use_container_width=True,
            )

            if analyze_btn:
                needs_key = analysis_mode in ("Hybrid (ML + Gemini)", "Gemini Only (V1)")
                if needs_key and not api_key:
                    st.error("⚠️ Please enter your Google AI API key in the sidebar, or switch to 'ML Only (Offline / Local)'.")
                elif analysis_mode != "Gemini Only (V1)" and predictor is None:
                    st.error("⚠️ ML models are not trained yet. Run `uv run python scripts/train_models.py`.")
                else:
                    with st.spinner("📄 Extracting text from file..."):
                        try:
                            if ext == "txt":
                                contract_text = temp_path.read_text(encoding="utf-8", errors="ignore")
                            else:
                                contract_text = extract_text_from_pdf(temp_path)

                            if len(contract_text.strip()) < 50:
                                st.warning("⚠️ Very little text extracted. The file might be scanned or image-based.")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Characters", f"{len(contract_text):,}")
                            with col_b:
                                st.metric("Words", f"{len(contract_text.split()):,}")
                        except Exception as exc:
                            st.error(f"❌ Error extracting text: {exc}")
                            contract_text = None

                    if contract_text:
                        with st.spinner(f"🤖 Running {analysis_mode}..."):
                            if analysis_mode == "ML Only (Offline / Local)":
                                analysis = analyze_contract_hybrid(
                                    contract_text=contract_text,
                                    api_key=None,
                                    predictor=predictor,
                                )
                            elif analysis_mode == "Hybrid (ML + Gemini)":
                                analysis = analyze_contract_hybrid(
                                    contract_text=contract_text,
                                    api_key=api_key,
                                    model=model_choice,
                                    predictor=predictor,
                                )
                            else:
                                if "pro" in model_choice:
                                    analysis = analyze_contract_with_gemini_pro(contract_text, api_key)
                                else:
                                    analysis = analyze_contract_with_gemini(contract_text, api_key, model_choice)

                            st.session_state.total_analyzed += 1
                            st.session_state.total_risks += len(analysis.risks)

                            st.divider()
                            st.header("📊 Analysis Results")

                            # Contract Type Classification (V2)
                            if analysis.contract_type is not None:
                                ct = analysis.contract_type
                                ct_col1, ct_col2 = st.columns([2, 1])
                                with ct_col1:
                                    st.subheader(f"📑 Contract Type: {ct.contract_type}")
                                with ct_col2:
                                    st.metric("ML Confidence", f"{ct.confidence:.1%}")

                                with st.expander("📈 Contract Type Probability Breakdown"):
                                    prob_df = pd.DataFrame(
                                        [
                                            {"Contract Type": k, "Probability": v}
                                            for k, v in ct.probabilities.items()
                                        ]
                                    )
                                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                            # Clause-level ML Triage (V2)
                            if analysis.ml_clause_predictions:
                                high_cnt = sum(1 for c in analysis.ml_clause_predictions if c.risk_level == "High")
                                med_cnt = sum(1 for c in analysis.ml_clause_predictions if c.risk_level == "Medium")
                                low_cnt = sum(1 for c in analysis.ml_clause_predictions if c.risk_level == "Low")

                                st.subheader("🧪 ML Clause Triage")
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Clauses Scanned", len(analysis.ml_clause_predictions))
                                m2.metric("🔴 High Risk", high_cnt)
                                m3.metric("🟠 Medium Risk", med_cnt)
                                m4.metric("🟢 Low Risk", low_cnt)

                                with st.expander("🔍 View All Clause ML Predictions"):
                                    clause_df = pd.DataFrame(
                                        [
                                            {
                                                "#": c.clause_index,
                                                "Risk Level": c.risk_level,
                                                "Category": c.risk_category,
                                                "Confidence": f"{c.confidence:.1%}",
                                                "Clause Preview": c.clause_text[:140] + ("..." if len(c.clause_text) > 140 else ""),
                                            }
                                            for c in analysis.ml_clause_predictions
                                        ]
                                    )
                                    st.dataframe(clause_df, use_container_width=True, hide_index=True)

                            st.divider()
                            st.subheader("🚨 Detailed Risk Findings")
                            if not analysis.risks:
                                st.success("✅ No significant risks identified!")
                            else:
                                st.warning(f"⚠️ {len(analysis.risks)} risk(s) identified")
                                for idx, risk in enumerate(analysis.risks, 1):
                                    badge = ""
                                    if risk.risk_level == "High":
                                        badge = "🔴 High | "
                                    elif risk.risk_level == "Medium":
                                        badge = "🟠 Medium | "
                                    elif risk.risk_level == "Low":
                                        badge = "🟢 Low | "

                                    with st.expander(
                                        f"{badge}Risk #{idx}: {risk.risk_type} ({risk.source})",
                                        expanded=(idx <= 3),
                                    ):
                                        st.markdown("**📝 Clause Text:**")
                                        st.info(risk.clause_text)
                                        st.markdown("**⚠️ Explanation:**")
                                        st.write(risk.explanation)
                                        st.markdown("**💡 Remediation Suggestion:**")
                                        st.success(risk.remediation_suggestion)

                            # Export
                            st.divider()
                            st.subheader("📥 Export Results")
                            json_str = json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)

                            ec1, ec2, ec3 = st.columns(3)
                            with ec1:
                                st.download_button(
                                    "💾 Download JSON",
                                    data=json_str,
                                    file_name=f"analysis_{uploaded_file.name}.json",
                                    mime="application/json",
                                    use_container_width=True,
                                )
                            with ec2:
                                md_lines = [
                                    "# Contract Risk Analysis (V2)",
                                    f"**File:** {uploaded_file.name}",
                                    f"**Mode:** {analysis.analysis_mode}",
                                ]
                                if analysis.contract_type:
                                    md_lines.append(
                                        f"**Contract Type:** {analysis.contract_type.contract_type} "
                                        f"({analysis.contract_type.confidence:.1%})"
                                    )
                                md_lines.append(f"**Risks Found:** {len(analysis.risks)}\n")
                                for idx, risk in enumerate(analysis.risks, 1):
                                    md_lines.append(f"## Risk {idx}: {risk.risk_type}")
                                    if risk.risk_level:
                                        md_lines.append(f"**Severity:** {risk.risk_level}")
                                    md_lines.append(f"**Clause:** {risk.clause_text}\n")
                                    md_lines.append(f"**Explanation:** {risk.explanation}\n")
                                    md_lines.append(f"**Remediation:** {risk.remediation_suggestion}\n\n---\n")

                                st.download_button(
                                    "📄 Download Report (MD)",
                                    data="\n".join(md_lines),
                                    file_name=f"report_{uploaded_file.name}.md",
                                    mime="text/markdown",
                                    use_container_width=True,
                                )
                            with ec3:
                                with st.popover("👁️ View Raw JSON"):
                                    st.json(analysis.to_dict())
    finally:
        if temp_path.exists():
            temp_path.unlink()
else:
    st.info("👆 Upload a contract PDF to get started (try any sample in `assets/`!)")
