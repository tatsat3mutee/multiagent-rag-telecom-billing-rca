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

# ── Custom CSS ──
st.markdown("""
<style>
    /* Hide default Streamlit auto-generated sidebar nav */
    [data-testid="stSidebarNav"] { display: none !important; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 1.1rem; }
    .agent-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid;
        height: 100%;
    }
    .agent-card h4 { margin-top: 0; }
    .investigator { border-left-color: #667eea; }
    .reasoner { border-left-color: #f7971e; }
    .reporter { border-left-color: #56ab2f; }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stat-card .number { font-size: 2rem; font-weight: 700; color: #667eea; }
    .stat-card .label { font-size: 0.85rem; color: #888; }
    .step-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        width: 28px; height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 8px;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📡 Telecom RCA")
    st.markdown("**Multi-Agent RAG System**")
    st.caption("Autonomous Root Cause Analysis for Billing Anomalies")
    st.markdown("---")
    st.page_link("app.py", label="🏠  Home", icon=None)
    st.page_link("pages/1_📊_Upload_Detect.py", label="📊  Upload & Detect")
    st.page_link("pages/2_🔍_RCA_Viewer.py", label="🔍  RCA Viewer")
    st.page_link("pages/3_📚_Knowledge_Base.py", label="📚  Knowledge Base")
    st.markdown("---")
    st.caption("MTech Thesis — Tatsat Pandey | 2026")

# ── Hero Section ──
st.markdown("""
<div class="main-header">
    <h1>📡 Multi-Agent RAG System</h1>
    <p>Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks</p>
</div>
""", unsafe_allow_html=True)

# ── Agent Pipeline Cards ──
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="agent-card investigator">
        <h4>🔍 Investigator Agent</h4>
        <p>Queries the knowledge base to retrieve relevant SLA documents, RCA playbooks, and incident reports for the anomaly.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="agent-card reasoner">
        <h4>🧠 Reasoning Agent</h4>
        <p>Synthesizes anomaly data with retrieved context to generate structured root cause hypotheses with evidence chains.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="agent-card reporter">
        <h4>📋 Reporter Agent</h4>
        <p>Produces validated RCA reports with corrective actions, severity assessment, and confidence scoring.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── System Status ──
st.markdown("### System Status")

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
        anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
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

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("KB Chunks", f"{kb_count}")
with c2:
    st.metric("Dataset Records", f"{total_records:,}")
with c3:
    st.metric("Anomalies Detected", f"{anomaly_count:,}")
with c4:
    st.metric("Detector Status", "✅ Ready" if model_exists else "⚠️ Not Trained")

st.markdown("---")

# ── Getting Started ──
st.markdown("### Getting Started")
st.markdown("""
<span class="step-badge">1</span> Go to **Upload & Detect** → load billing data and run anomaly detection  
<span class="step-badge">2</span> Go to **RCA Viewer** → select any anomaly → generate an AI-powered root cause report  
<span class="step-badge">3</span> Browse the **Knowledge Base** to explore indexed telecom domain documents
""", unsafe_allow_html=True)

st.markdown("---")

# Architecture & Tech Stack in tabs
tab1, tab2, tab3 = st.tabs(["📐 Architecture", "🛠️ Tech Stack", "📊 Ablation Results"])

with tab1:
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: UI & Monitoring                                       │
│  Streamlit Dashboard  +  MLflow Tracking                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Agent Orchestration (LangGraph)                        │
│  Investigator  →  Reasoner  →  Reporter                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: RAG Engine — ChromaDB + sentence-transformers          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Anomaly Detection — IsolationForest / DBSCAN           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Data Ingestion — Pandas + NumPy                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 0: LLM Backend — Groq (Llama 3.3 70B Versatile)          │
└─────────────────────────────────────────────────────────────────┘
    """, language="text")

with tab2:
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        | Component | Technology |
        |-----------|------------|
        | **LLM** | Groq — Llama 3.3 70B |
        | **Agents** | LangGraph StateGraph |
        | **Vector DB** | ChromaDB |
        | **Embeddings** | all-MiniLM-L6-v2 |
        | **Detection** | scikit-learn |
        """)
    with t2:
        st.markdown("""
        | Component | Technology |
        |-----------|------------|
        | **Tracking** | MLflow |
        | **UI** | Streamlit |
        | **Evaluation** | BERTScore, ROUGE-L |
        | **Stats** | Wilcoxon signed-rank |
        | **Data** | IBM Telco (35K augmented) |
        """)

with tab3:
    try:
        import json as _json
        ablation_path = Path(__file__).parent / "ablation_results.json"
        if ablation_path.exists():
            with open(ablation_path) as f:
                abl = _json.load(f)
            st.markdown(f"**Model:** `{abl.get('model', 'N/A')}` | **Test Set:** 15 anomalies (3 per type × 5 types)")
            rows = []
            for cfg_key, cfg_data in abl["configs"].items():
                m = cfg_data["metrics"]
                rows.append({
                    "Config": cfg_data["description"],
                    "ROUGE-L": f"{m['rouge_l_f1']:.3f}",
                    "BERTScore": f"{m['bert_score_f1']:.3f}",
                    "Type Accuracy": f"{m['type_accuracy']:.0%}",
                    "Avg Latency": f"{m['avg_latency_ms']:.0f}ms",
                })
            import pandas as _pd
            st.dataframe(_pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("Ablation results not available. Run `python run_ablation.py` to generate.")
    except Exception as e:
        st.warning(f"Could not load ablation results: {e}")
