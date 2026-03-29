"""
Streamlit Dashboard — Main Entry Point
Multi-Agent RAG System for Telecom Billing RCA
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

st.set_page_config(
    page_title="Telecom Billing RCA System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/signal-tower.png", width=80)
    st.title("Telecom RCA")
    st.markdown("---")
    st.markdown("""
    **Multi-Agent RAG System**  
    Autonomous Root Cause Analysis  
    for Billing Anomalies
    """)
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 📊 **Upload & Detect** — Upload billing CSV
    - 🔍 **RCA Viewer** — Generate RCA reports
    - 📚 **Knowledge Base** — Browse RAG corpus
    """)
    st.markdown("---")
    st.caption("MTech Thesis — Tatsat Pandey")

# ── Main Page ──
st.title("📡 Multi-Agent RAG System for Telecom Billing RCA")
st.markdown("### Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks")

st.markdown("---")

# System Architecture
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔍 Investigator Agent")
    st.markdown("""
    Receives anomaly context and queries the 
    RAG knowledge base to retrieve relevant 
    SLA documents, RCA playbooks, and 
    incident reports.
    """)

with col2:
    st.markdown("#### 🧠 Reasoning Agent")
    st.markdown("""
    Synthesizes anomaly data with retrieved 
    documentation to generate structured 
    root cause hypotheses with evidence.
    """)

with col3:
    st.markdown("#### 📋 Reporter Agent")
    st.markdown("""
    Produces JSON-schema-validated RCA 
    reports with recommended corrective 
    actions and severity assessment.
    """)

st.markdown("---")

# Quick Stats
st.markdown("### System Status")
col1, col2, col3, col4 = st.columns(4)

# Check system status
try:
    from src.rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb_count = kb.count
except Exception:
    kb_count = 0

try:
    import pandas as pd
    from config import PROCESSED_DATA_DIR
    labeled_path = PROCESSED_DATA_DIR / "anomalies_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
        total_records = len(df)
        anomaly_count = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
    else:
        total_records = 0
        anomaly_count = 0
except Exception:
    total_records = 0
    anomaly_count = 0

try:
    from config import MODELS_DIR
    model_exists = (MODELS_DIR / "isolation_forest_model.joblib").exists()
except Exception:
    model_exists = False

with col1:
    st.metric("📊 KB Documents", kb_count)
with col2:
    st.metric("📁 Dataset Records", total_records)
with col3:
    st.metric("⚠️ Anomalies", int(anomaly_count))
with col4:
    st.metric("🤖 Detector", "Ready" if model_exists else "Not Trained")

st.markdown("---")

# Quick Start
st.markdown("### 🚀 Quick Start")
st.markdown("""
1. **Navigate to Upload & Detect** to upload a billing CSV or use the pre-loaded dataset
2. **View detected anomalies** and click on any to trigger the multi-agent RCA pipeline
3. **Read the generated RCA report** with root cause, evidence, and recommended actions
4. **Browse the Knowledge Base** to see indexed domain documents
""")

# Architecture Diagram (text-based)
with st.expander("📐 System Architecture", expanded=False):
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: UI & Monitoring                                       │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  Streamlit    │  │   MLflow     │                             │
│  │  (Dashboard)  │  │  (Tracking)  │                             │
│  └──────┬───────┘  └──────┬───────┘                             │
├─────────┼──────────────────┼────────────────────────────────────┤
│  Layer 4: Agent Orchestration (LangGraph)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │Investi-  │→ │Reasoning │→ │Reporter  │                      │
│  │gator     │  │Agent     │  │Agent     │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: RAG Engine                                             │
│  ChromaDB + sentence-transformers + PyMuPDF                      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Anomaly Detection (scikit-learn)                       │
│  IsolationForest / DBSCAN                                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Data Ingestion (Pandas + NumPy)                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 0: LLM Backend (Groq — Llama 3.3 70B)                         │
└─────────────────────────────────────────────────────────────────┘
    """, language="text")

# Tech Stack
with st.expander("🛠️ Technology Stack", expanded=False):
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
        | Component | Technology |
        |-----------|------------|
        | LLM | Groq (Llama 3.3 70B Versatile) |
        | Agent Orchestration | LangGraph |
        | Vector Database | ChromaDB |
        | Embeddings | sentence-transformers (MiniLM) |
        | Anomaly Detection | scikit-learn |
        """)
    with tech_col2:
        st.markdown("""
        | Component | Technology |
        |-----------|------------|
        | Experiment Tracking | MLflow |
        | UI | Streamlit |
        | Data Processing | Pandas, NumPy |
        | PDF Parsing | PyMuPDF |
        | Evaluation | BERTScore, ROUGE-L |
        """)

# Ablation Results
with st.expander("📊 Ablation Study Results", expanded=False):
    try:
        import json as _json
        ablation_path = Path(__file__).parent / "ablation_results.json"
        if ablation_path.exists():
            with open(ablation_path) as f:
                abl = _json.load(f)
            st.markdown(f"**Model:** `{abl.get('model', 'N/A')}`")
            rows = []
            for cfg_key, cfg_data in abl["configs"].items():
                m = cfg_data["metrics"]
                rows.append({
                    "Config": cfg_data["description"],
                    "ROUGE-L": f"{m['rouge_l_f1']:.4f}",
                    "BERTScore": f"{m['bert_score_f1']:.4f}",
                    "Type Accuracy": f"{m['type_accuracy']:.0%}",
                    "Avg Latency": f"{m['avg_latency_ms']:.0f}ms",
                })
            import pandas as _pd
            st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

            a_m = abl["configs"]["A_no_rag"]["metrics"]
            d_m = abl["configs"]["D_multi_agent_rag"]["metrics"]
            rouge_imp = ((d_m["rouge_l_f1"] - a_m["rouge_l_f1"]) / max(a_m["rouge_l_f1"], 0.001)) * 100
            bert_imp = ((d_m["bert_score_f1"] - a_m["bert_score_f1"]) / max(a_m["bert_score_f1"], 0.001)) * 100
            st.success(f"Multi-Agent RAG vs No-RAG: ROUGE-L **{rouge_imp:+.1f}%** | BERTScore **{bert_imp:+.1f}%**")
        else:
            st.info("Run `python run_ablation.py` to generate ablation results.")
    except Exception as e:
        st.warning(f"Could not load ablation results: {e}")
