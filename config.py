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

# ── LLM (Groq preferred, Kimi fallback — both via OpenAI-compatible API) ──
#
# Priority order (auto-detected at import):
#   1. If GROQ_API_KEY is set    → use Groq    (free tier, fast)
#   2. Elif KIMI_API_KEY is set  → use Kimi K2 (~₹17 / full ablation)
#   3. Elif LLM_API_KEY is set   → use whatever LLM_BASE_URL points at
#
# Explicit overrides (LLM_API_KEY + LLM_BASE_URL + LLM_MODEL) always win.

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")

# Explicit overrides
_explicit_key = os.environ.get("LLM_API_KEY", "")
_explicit_base = os.environ.get("LLM_BASE_URL", "")
_explicit_model = os.environ.get("LLM_MODEL", "")

if _explicit_key:
    LLM_API_KEY = _explicit_key
    LLM_BASE_URL = _explicit_base
    LLM_MODEL = _explicit_model or "kimi-k2-0711-preview"
    LLM_PROVIDER = "custom"
elif GROQ_API_KEY:
    LLM_API_KEY = GROQ_API_KEY
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
    LLM_MODEL = _explicit_model or "llama-3.3-70b-versatile"
    LLM_PROVIDER = "groq"
elif KIMI_API_KEY:
    LLM_API_KEY = KIMI_API_KEY
    LLM_BASE_URL = "https://api.moonshot.ai/v1"
    LLM_MODEL = _explicit_model or "kimi-k2-0711-preview"
    LLM_PROVIDER = "kimi"
else:
    LLM_API_KEY = ""
    LLM_BASE_URL = ""
    LLM_MODEL = _explicit_model or "kimi-k2-0711-preview"
    LLM_PROVIDER = "none"

LLM_TEMPERATURE = 0.1

# ── Judge LLM (evaluation) ──
# Defaults to the same provider/model as the generator. Optionally override
# JUDGE_API_KEY + JUDGE_BASE_URL + JUDGE_MODEL to use a different family
# (cross-family judge = stronger bias-mitigation story, e.g. Groq gen + Kimi judge).
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", LLM_API_KEY)
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", LLM_BASE_URL)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL)
JUDGE_TEMPERATURE = 0.0  # deterministic scoring
JUDGE_FALLBACK_MODEL = LLM_MODEL

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
    "graph_rag": {
        "use_rag": True,
        "use_agents": True,
        "single_agent": False,
        "use_graph_rag": True,
        "description": "Config E: Multi-Agent + GraphRAG (headline novelty)",
    },
}

# ── Anomaly Detection ──
RANDOM_SEED = 42
AUGMENTED_TARGET_SIZE = 35_000  # Augmented dataset size (ROSE-style oversampling)
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
