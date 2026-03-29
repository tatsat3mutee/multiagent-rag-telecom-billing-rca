"""
Upload & Detect Page — Upload billing CSV and detect anomalies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

st.set_page_config(page_title="Upload & Detect", page_icon="📊", layout="wide")
st.title("📊 Upload & Detect Anomalies")
st.markdown("Upload a billing CSV or use the pre-loaded dataset to detect anomalies.")

# ── Data Source Selection ──
data_source = st.radio(
    "Select Data Source",
    ["Use Pre-loaded Dataset", "Upload CSV File"],
    horizontal=True,
)

df = None

if data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload billing CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} records from uploaded file.")
else:
    labeled_path = PROCESSED_DATA_DIR / "anomalies_labeled.csv"
    raw_path = RAW_DATA_DIR / "ibm_telco_churn.csv"

    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
        st.success(f"Loaded {len(df)} records from pre-loaded labeled dataset.")
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
        st.info(f"Loaded {len(df)} records from raw dataset. Click 'Inject Anomalies' to create labeled data.")
    else:
        st.warning("No dataset found. Click below to generate synthetic datasets.")
        if st.button("Generate Datasets"):
            with st.spinner("Generating datasets..."):
                from scripts.download_datasets import download_ibm_telco, download_maven_telecom
                download_ibm_telco()
                download_maven_telecom()
                st.success("Datasets generated! Refresh the page.")
                st.rerun()

if df is not None:
    st.markdown("---")

    # ── Data Overview ──
    st.markdown("### 📋 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        if "is_anomaly" in df.columns:
            st.metric("Anomalies", int(df["is_anomaly"].sum()))
        else:
            st.metric("Anomalies", "Not labeled")
    with col4:
        if "anomaly_type" in df.columns:
            st.metric("Anomaly Types", df["anomaly_type"].nunique() - (1 if "normal" in df["anomaly_type"].values else 0))
        else:
            st.metric("Anomaly Types", "N/A")

    # Show data sample
    with st.expander("View Data Sample", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # ── Anomaly Injection (if not already labeled) ──
    if "is_anomaly" not in df.columns:
        st.markdown("---")
        st.markdown("### 💉 Anomaly Injection")
        if st.button("Inject Synthetic Anomalies"):
            with st.spinner("Injecting anomalies..."):
                from src.data.anomaly_injector import inject_all_anomalies
                df = inject_all_anomalies(df)
                df.to_csv(PROCESSED_DATA_DIR / "anomalies_labeled.csv", index=False)
                st.success(f"Injected anomalies! Total anomalies: {df['is_anomaly'].sum()}")
                st.rerun()

    # ── Anomaly Detection ──
    st.markdown("---")
    st.markdown("### 🔍 Anomaly Detection")

    detect_col1, detect_col2 = st.columns(2)
    with detect_col1:
        method = st.selectbox("Detection Method", ["isolation_forest", "dbscan"])
    with detect_col2:
        detect_button = st.button("🚀 Run Anomaly Detection", type="primary")

    if detect_button:
        with st.spinner("Running anomaly detection..."):
            from src.detection.detector import BillingAnomalyDetector

            detector = BillingAnomalyDetector(method=method)

            # Check required columns
            required_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                detector.fit(df)
                result_df = detector.predict(df)
                detector.save()

                # Show results
                detected_count = result_df["predicted_anomaly"].sum()
                st.success(f"Detection complete! Found {detected_count} anomalies.")

                # Metrics if ground truth available
                if "is_anomaly" in df.columns:
                    metrics = detector.evaluate(df)
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Precision", f"{metrics['precision']:.3f}")
                    with metric_cols[1]:
                        st.metric("Recall", f"{metrics['recall']:.3f}")
                    with metric_cols[2]:
                        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                    with metric_cols[3]:
                        if "roc_auc" in metrics:
                            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

                # Show detected anomalies
                st.markdown("#### Detected Anomalies")
                anomalies_df = result_df[result_df["predicted_anomaly"] == 1]

                if "anomaly_type" in anomalies_df.columns:
                    type_dist = anomalies_df["anomaly_type"].value_counts()
                    st.bar_chart(type_dist)

                st.dataframe(
                    anomalies_df[["customerID", "MonthlyCharges", "TotalCharges", "tenure",
                                  "anomaly_confidence", "predicted_anomaly"] +
                                 (["anomaly_type"] if "anomaly_type" in anomalies_df.columns else [])
                    ].head(50).sort_values("anomaly_confidence", ascending=False),
                    use_container_width=True,
                )

                # Store in session for RCA Viewer
                st.session_state["detected_anomalies"] = anomalies_df
                st.session_state["full_dataset"] = result_df

    # ── Anomaly Distribution (if labeled) ──
    if "anomaly_type" in df.columns:
        st.markdown("---")
        st.markdown("### 📊 Anomaly Distribution")
        type_counts = df["anomaly_type"].value_counts()
        st.bar_chart(type_counts)

        # Per-type statistics
        with st.expander("Per-Type Statistics"):
            for atype in df["anomaly_type"].unique():
                if atype == "normal":
                    continue
                subset = df[df["anomaly_type"] == atype]
                st.markdown(f"**{atype}** ({len(subset)} records)")
                st.dataframe(subset[["MonthlyCharges", "TotalCharges", "tenure"]].describe().round(2))
