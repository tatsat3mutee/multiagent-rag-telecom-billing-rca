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
st.title("🔍 Root Cause Analysis Viewer")
st.markdown("Select an anomaly to generate a detailed RCA report using the multi-agent pipeline.")

# ── Load Anomalies ──
anomalies_df = st.session_state.get("detected_anomalies", None)

if anomalies_df is None:
    # Try loading from labeled dataset
    labeled_path = PROCESSED_DATA_DIR / "anomalies_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
        anomalies_df = df[df["is_anomaly"] == 1] if "is_anomaly" in df.columns else df.head(20)
        st.info("Loaded anomalies from pre-labeled dataset. Run detection on Upload & Detect page for live results.")
    else:
        st.warning("No anomalies detected yet. Go to **Upload & Detect** page first.")
        st.stop()

st.markdown(f"**{len(anomalies_df)} anomalies available for analysis**")

# ── Anomaly Selection ──
st.markdown("---")
st.markdown("### Select Anomaly for RCA")

# Filters
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    if "anomaly_type" in anomalies_df.columns:
        types = ["All"] + [t for t in anomalies_df["anomaly_type"].unique() if t != "normal"]
        selected_type = st.selectbox("Filter by Type", types)
        if selected_type != "All":
            anomalies_df = anomalies_df[anomalies_df["anomaly_type"] == selected_type]

with filter_col2:
    max_display = st.slider("Max records to show", 5, 50, 20)

# Display anomalies table
display_cols = ["customerID", "MonthlyCharges", "TotalCharges", "tenure"]
if "anomaly_type" in anomalies_df.columns:
    display_cols.append("anomaly_type")
if "anomaly_confidence" in anomalies_df.columns:
    display_cols.append("anomaly_confidence")

available_cols = [c for c in display_cols if c in anomalies_df.columns]
display_df = anomalies_df[available_cols].head(max_display)

st.dataframe(display_df, use_container_width=True)

# ── RCA Generation ──
st.markdown("---")
st.markdown("### 🤖 Generate RCA Report")

# Select a specific anomaly
anomaly_options = []
for idx, row in anomalies_df.head(max_display).iterrows():
    cid = row.get("customerID", f"Row-{idx}")
    atype = row.get("anomaly_type", "unknown")
    charges = row.get("MonthlyCharges", 0)
    anomaly_options.append(f"{cid} | Type: {atype} | Charges: ${charges:.2f}")

if anomaly_options:
    selected = st.selectbox("Select anomaly to investigate", anomaly_options)
    selected_idx = anomaly_options.index(selected)
    selected_row = anomalies_df.head(max_display).iloc[selected_idx]

    # Show selected anomaly details
    with st.expander("Selected Anomaly Details", expanded=True):
        detail_cols = st.columns(4)
        with detail_cols[0]:
            st.metric("Customer ID", selected_row.get("customerID", "N/A"))
        with detail_cols[1]:
            st.metric("Anomaly Type", selected_row.get("anomaly_type", "unknown"))
        with detail_cols[2]:
            st.metric("Monthly Charges", f"${selected_row.get('MonthlyCharges', 0):.2f}")
        with detail_cols[3]:
            st.metric("Tenure", f"{selected_row.get('tenure', 0)} months")

    # Generate RCA button
    if st.button("🚀 Generate RCA Report", type="primary"):
        with st.spinner("Running multi-agent RCA pipeline..."):
            start_time = time.time()

            # Build anomaly record
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

            # Progress tracking
            progress = st.progress(0)
            status = st.empty()

            status.text("🔍 Investigator Agent: Querying knowledge base...")
            progress.progress(20)

            try:
                from src.agents.graph import run_pipeline
                result = run_pipeline(anomaly_record)

                progress.progress(100)
                status.text("✅ Pipeline complete!")
                elapsed = (time.time() - start_time) * 1000

                # Display results
                st.success(f"RCA generated in {elapsed:.0f}ms")

                rca = result.get("rca_report", {})

                if rca:
                    st.markdown("---")
                    st.markdown("## 📋 Root Cause Analysis Report")

                    # Header metrics
                    rca_cols = st.columns(4)
                    with rca_cols[0]:
                        st.metric("Anomaly ID", rca.get("anomaly_id", "N/A"))
                    with rca_cols[1]:
                        st.metric("Type", rca.get("anomaly_type", "N/A"))
                    with rca_cols[2]:
                        severity = rca.get("severity", "N/A")
                        severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
                        st.metric("Severity", f"{severity_color} {severity}")
                    with rca_cols[3]:
                        st.metric("Confidence", f"{rca.get('confidence_score', 0):.0%}")

                    # Root Cause
                    st.markdown("### 🎯 Root Cause")
                    st.info(rca.get("root_cause", "No root cause determined."))

                    # Executive Summary
                    st.markdown("### 📝 Summary")
                    st.write(rca.get("summary", "No summary available."))

                    # Evidence
                    st.markdown("### 📚 Supporting Evidence")
                    for i, evidence in enumerate(rca.get("supporting_evidence", []), 1):
                        st.markdown(f"{i}. {evidence}")

                    # Actions
                    st.markdown("### ✅ Recommended Actions")
                    for i, action in enumerate(rca.get("recommended_actions", []), 1):
                        st.markdown(f"{i}. {action}")

                    # Retrieved Documents
                    st.markdown("### 📄 Retrieved Documents")
                    docs = result.get("retrieved_docs", [])
                    for doc in docs:
                        with st.expander(f"📄 {doc.get('source', 'Unknown')} (Relevance: {doc.get('relevance_score', 0):.2f})"):
                            st.write(doc.get("text", ""))

                    # Raw JSON
                    with st.expander("📄 Raw RCA JSON"):
                        st.json(rca)

                    # Pipeline metadata
                    with st.expander("⚙️ Pipeline Metadata"):
                        st.write(f"**Status:** {result.get('pipeline_status', 'N/A')}")
                        st.write(f"**Latency:** {result.get('latency_ms', 0):.0f}ms")
                        st.write(f"**Retrieval Query:** {result.get('retrieval_query', 'N/A')}")
                        st.write(f"**Documents Retrieved:** {result.get('retrieval_count', 0)}")

                    # Store in session
                    st.session_state["last_rca_result"] = result

                else:
                    st.error("Pipeline completed but no RCA report was generated.")

            except Exception as e:
                progress.progress(100)
                st.error(f"Pipeline error: {str(e)}")
                st.info("Make sure the knowledge base is built. Run: `python src/cli.py --setup`")

    # ── Batch RCA ──
    st.markdown("---")
    st.markdown("### 📦 Batch RCA Generation")
    batch_limit = st.number_input("Number of anomalies to process", min_value=1, max_value=50, value=5)

    if st.button("Generate Batch RCA"):
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
                        "Account ID": record["account_id"],
                        "Type": record["anomaly_type"],
                        "Severity": rca.get("severity", "N/A"),
                        "Root Cause": rca.get("root_cause", "N/A")[:100],
                        "Latency (ms)": f"{result.get('latency_ms', 0):.0f}",
                    })
                except Exception as e:
                    batch_results.append({
                        "Account ID": record["account_id"],
                        "Type": record["anomaly_type"],
                        "Severity": "ERROR",
                        "Root Cause": str(e)[:100],
                        "Latency (ms)": "N/A",
                    })

            st.success(f"Processed {len(batch_results)} anomalies!")
            st.dataframe(pd.DataFrame(batch_results), use_container_width=True)
