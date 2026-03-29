"""
RCA Viewer Page — Generate and view Root Cause Analysis reports.
"""
import streamlit as st
import pandas as pd
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import PROCESSED_DATA_DIR

st.set_page_config(page_title="RCA Viewer", page_icon="🔍", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none !important; }
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .page-header h2 { color: white; margin: 0; }
    .page-header p { color: #e0e0e0; margin: 0.3rem 0 0 0; }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .severity-high { color: #dc3545; font-weight: bold; }
    .severity-medium { color: #ffc107; font-weight: bold; }
    .severity-low { color: #28a745; font-weight: bold; }
    .rca-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📡 Telecom RCA")
    st.markdown("**Multi-Agent RAG System**")
    st.caption("Autonomous Root Cause Analysis for Billing Anomalies")
    st.markdown("---")
    st.page_link("app.py", label="🏠  Home")
    st.page_link("pages/1_📊_Upload_Detect.py", label="📊  Upload & Detect")
    st.page_link("pages/2_🔍_RCA_Viewer.py", label="🔍  RCA Viewer")
    st.page_link("pages/3_📚_Knowledge_Base.py", label="📚  Knowledge Base")
    st.markdown("---")
    st.caption("MTech Thesis — Tatsat Pandey | 2026")

st.markdown("""
<div class="page-header">
    <h2>🔍 Root Cause Analysis Viewer</h2>
    <p>Select an anomaly and generate a detailed RCA report using the multi-agent pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Load Anomalies ──
anomalies_df = st.session_state.get("detected_anomalies", None)

if anomalies_df is None:
    labeled_path = PROCESSED_DATA_DIR / "anomalies_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
        anomalies_df = df[df["is_anomaly"] == 1] if "is_anomaly" in df.columns else df.head(20)
        st.info("Loaded anomalies from labeled dataset. Run detection on **Upload & Detect** for live results.")
    else:
        st.warning("No anomalies available. Go to **Upload & Detect** first.")
        st.stop()

st.markdown(f"**{len(anomalies_df):,} anomalies** available for analysis")

# ── Filters ──
st.markdown("---")
f1, f2 = st.columns([1, 1])
with f1:
    if "anomaly_type" in anomalies_df.columns:
        types = ["All"] + sorted([t for t in anomalies_df["anomaly_type"].unique() if t != "normal"])
        selected_type = st.selectbox("Filter by Type", types)
        if selected_type != "All":
            anomalies_df = anomalies_df[anomalies_df["anomaly_type"] == selected_type]
with f2:
    max_display = st.slider("Max records", 5, 50, 20)

# Display table
display_cols = ["customerID", "MonthlyCharges", "TotalCharges", "tenure"]
if "anomaly_type" in anomalies_df.columns:
    display_cols.append("anomaly_type")
if "anomaly_confidence" in anomalies_df.columns:
    display_cols.append("anomaly_confidence")
available_cols = [c for c in display_cols if c in anomalies_df.columns]
st.dataframe(anomalies_df[available_cols].head(max_display), width='stretch')

# ── RCA Generation ──
st.markdown("---")
st.markdown("### Generate RCA Report")

anomaly_options = []
for idx, row in anomalies_df.head(max_display).iterrows():
    cid = row.get("customerID", f"Row-{idx}")
    atype = row.get("anomaly_type", "unknown")
    charges = row.get("MonthlyCharges", 0)
    anomaly_options.append(f"{cid} | {atype} | ${charges:.2f}/mo")

if anomaly_options:
    sel_col, btn_col = st.columns([3, 1])
    with sel_col:
        selected = st.selectbox("Select anomaly to investigate", anomaly_options)
    with btn_col:
        st.markdown("")
        st.markdown("")
        generate_btn = st.button("🚀  Generate RCA", type="primary", use_container_width=True)

    selected_idx = anomaly_options.index(selected)
    selected_row = anomalies_df.head(max_display).iloc[selected_idx]

    # Selected anomaly details
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Customer", selected_row.get("customerID", "N/A"))
    with d2:
        st.metric("Type", selected_row.get("anomaly_type", "unknown"))
    with d3:
        st.metric("Monthly Charges", f"${selected_row.get('MonthlyCharges', 0):.2f}")
    with d4:
        st.metric("Tenure", f"{selected_row.get('tenure', 0)} mo")

    if generate_btn:
        with st.spinner("Running multi-agent RCA pipeline..."):
            start_time = time.time()

            anomaly_record = {
                "account_id": str(selected_row.get("customerID", f"ROW-{selected_idx}")),
                "anomaly_type": str(selected_row.get("anomaly_type",
                                    selected_row.get("estimated_type", "unknown"))),
                "confidence": float(selected_row.get("anomaly_confidence", 0.5)),
                "monthly_charges": float(selected_row.get("MonthlyCharges", 0)),
                "total_charges": float(selected_row.get("TotalCharges", 0))
                                 if pd.notna(selected_row.get("TotalCharges")) else 0.0,
                "tenure": int(selected_row.get("tenure", 0)),
                "features": {
                    col: str(selected_row[col]) for col in ["Contract", "InternetService", "PaymentMethod"]
                    if col in selected_row.index
                },
            }

            progress = st.progress(0)
            status = st.empty()

            status.text("🔍 Investigator querying knowledge base...")
            progress.progress(20)

            try:
                from src.agents.graph import run_pipeline
                result = run_pipeline(anomaly_record)

                progress.progress(100)
                status.text("✅ Pipeline complete!")
                elapsed = (time.time() - start_time) * 1000

                rca = result.get("rca_report", {})

                if rca:
                    st.success(f"RCA generated in {elapsed:.0f}ms")

                    st.markdown("---")
                    st.markdown("## 📋 RCA Report")

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.metric("Anomaly ID", rca.get("anomaly_id", "N/A"))
                    with r2:
                        st.metric("Type", rca.get("anomaly_type", "N/A"))
                    with r3:
                        severity = rca.get("severity", "N/A")
                        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
                        st.metric("Severity", f"{icon} {severity}")
                    with r4:
                        st.metric("Confidence", f"{rca.get('confidence_score', 0):.0%}")

                    st.markdown("#### Root Cause")
                    st.info(rca.get("root_cause", "No root cause determined."))

                    st.markdown("#### Summary")
                    st.write(rca.get("summary", "No summary available."))

                    col_ev, col_act = st.columns(2)
                    with col_ev:
                        st.markdown("#### Supporting Evidence")
                        for i, ev in enumerate(rca.get("supporting_evidence", []), 1):
                            st.markdown(f"{i}. {ev}")
                    with col_act:
                        st.markdown("#### Recommended Actions")
                        for i, act in enumerate(rca.get("recommended_actions", []), 1):
                            st.markdown(f"{i}. {act}")

                    # Retrieved docs
                    docs = result.get("retrieved_docs", [])
                    if docs:
                        st.markdown("#### Retrieved Documents")
                        for doc in docs:
                            with st.expander(f"📄 {doc.get('source', 'Unknown')} — relevance {doc.get('relevance_score', 0):.2f}"):
                                st.write(doc.get("text", ""))

                    # Raw JSON & Metadata
                    tab_json, tab_meta = st.tabs(["Raw JSON", "Pipeline Metadata"])
                    with tab_json:
                        st.json(rca)
                    with tab_meta:
                        st.write(f"**Status:** {result.get('pipeline_status', 'N/A')}")
                        st.write(f"**Latency:** {result.get('latency_ms', 0):.0f}ms")
                        st.write(f"**Retrieval Query:** {result.get('retrieval_query', 'N/A')}")
                        st.write(f"**Documents Retrieved:** {result.get('retrieval_count', 0)}")

                    st.session_state["last_rca_result"] = result
                else:
                    st.error("Pipeline completed but no RCA report was generated.")

            except Exception as e:
                progress.progress(100)
                st.error(f"Pipeline error: {str(e)}")
                st.info("Make sure the knowledge base is built. Run: `python src/cli.py --setup`")

    # ── Batch RCA ──
    st.markdown("---")
    st.markdown("### Batch RCA Generation")

    b1, b2 = st.columns([1, 2])
    with b1:
        batch_limit = st.number_input("Anomalies to process", min_value=1, max_value=50, value=5)
    with b2:
        st.markdown("")
        st.markdown("")
        batch_btn = st.button("📦  Run Batch RCA", type="primary", use_container_width=True)

    if batch_btn:
        with st.spinner(f"Processing {batch_limit} anomalies..."):
            from src.agents.graph import run_pipeline

            batch_results = []
            progress = st.progress(0)

            for i, (idx, row) in enumerate(anomalies_df.head(batch_limit).iterrows()):
                progress.progress((i + 1) / batch_limit)
                record = {
                    "account_id": str(row.get("customerID", f"ROW-{idx}")),
                    "anomaly_type": str(row.get("anomaly_type", "unknown")),
                    "confidence": float(row.get("anomaly_confidence", 0.5)),
                    "monthly_charges": float(row.get("MonthlyCharges", 0)),
                    "total_charges": float(row.get("TotalCharges", 0)) if pd.notna(row.get("TotalCharges")) else 0.0,
                    "tenure": int(row.get("tenure", 0)),
                    "features": {},
                }
                try:
                    result = run_pipeline(record)
                    rca = result.get("rca_report", {})
                    batch_results.append({
                        "Account": record["account_id"],
                        "Type": record["anomaly_type"],
                        "Severity": rca.get("severity", "N/A"),
                        "Root Cause": rca.get("root_cause", "N/A")[:100],
                        "Latency": f"{result.get('latency_ms', 0):.0f}ms",
                    })
                except Exception as e:
                    batch_results.append({
                        "Account": record["account_id"],
                        "Type": record["anomaly_type"],
                        "Severity": "ERROR",
                        "Root Cause": str(e)[:100],
                        "Latency": "—",
                    })

            st.success(f"Processed {len(batch_results)} anomalies.")
            st.dataframe(pd.DataFrame(batch_results), width='stretch')
