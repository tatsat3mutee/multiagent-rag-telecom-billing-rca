"""
Central configuration for the Multi-Agent RAG System.
"""
import os
from pathlib import Path

# ── Project Root ──
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Data Paths ──
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "corpus"
RCA_PLAYBOOKS_DIR = CORPUS_DIR / "rca_playbooks"
EVAL_DIR = DATA_DIR / "eval"
GROUND_TRUTH_DIR = EVAL_DIR / "ground_truth_rca"

# ── Model Paths ──
MODELS_DIR = PROJECT_ROOT / "models"

# ── ChromaDB ──
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "telecom_billing_kb"

# ── Embedding Model ──
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ── Chunking ──
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ── Retrieval ──
TOP_K = 5

# ── LLM (Groq) ──
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

# ── Ablation Study Configurations ──
ABLATION_CONFIGS = {
    "no_rag": {
        "use_rag": False,
        "use_agents": False,
        "description": "Config A: Direct LLM — no RAG, no agents",
    },
    "rag_only": {
        "use_rag": True,
        "use_agents": False,
        "description": "Config B: RAG + LLM — no agent decomposition",
    },
    "single_agent_rag": {
        "use_rag": True,
        "use_agents": True,
        "single_agent": True,
        "description": "Config C: Single Agent + RAG",
    },
    "multi_agent_rag": {
        "use_rag": True,
        "use_agents": True,
        "single_agent": False,
        "description": "Config D: Multi-Agent + RAG (proposed system)",
    },
}

# ── Anomaly Detection ──
RANDOM_SEED = 42
ANOMALY_RATIOS = {
    "zero_billing": 0.03,
    "duplicate_charge": 0.02,
    "usage_spike": 0.03,
    "cdr_failure": 0.015,
    "sla_breach": 0.02,
}
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.1,
    "max_features": 0.8,
    "random_state": RANDOM_SEED,
}

# ── MLflow ──
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT_NAME = "telecom_billing_rca"

# ── Streamlit ──
STREAMLIT_PAGE_TITLE = "Telecom Billing RCA System"
STREAMLIT_PAGE_ICON = "📡"

# ── Create directories ──
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CORPUS_DIR, RCA_PLAYBOOKS_DIR,
          EVAL_DIR, GROUND_TRUTH_DIR, MODELS_DIR, CHROMA_PERSIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)
