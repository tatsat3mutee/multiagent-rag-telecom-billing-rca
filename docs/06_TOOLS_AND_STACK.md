# Tools & Technology Stack

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

**Stack Philosophy:** 100% Open Source — Zero API Cost (Groq Free Tier) — Fully Reproducible

---

## 1. Architecture Layer Map

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: UI & Monitoring                                       │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  Streamlit    │  │   MLflow     │                             │
│  │  (Dashboard)  │  │  (Tracking)  │                             │
│  └──────┬───────┘  └──────┬───────┘                             │
├─────────┼──────────────────┼────────────────────────────────────┤
│  Layer 4: Agent Orchestration                                    │
│  ┌──────────────────────────────────────────────────────┐       │
│  │               LangGraph (StateGraph)                  │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │       │
│  │  │Investi-  │→ │Reasoning │→ │Reporter  │           │       │
│  │  │gator     │  │Agent     │  │Agent     │           │       │
│  │  └──────────┘  └──────────┘  └──────────┘           │       │
│  └──────────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: RAG Engine                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │  ChromaDB    │  │  sentence-   │  │   PyMuPDF        │      │
│  │  (Vectors)   │  │  transformers│  │   (PDF Parser)   │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Anomaly Detection                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │           scikit-learn (IsolationForest, DBSCAN)      │       │
│  └──────────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Data Ingestion                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │   Pandas     │  │   NumPy      │  │   SQLite         │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 0: LLM Backend                                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │     Groq API (Llama 3.3 70B Versatile — Cloud)        │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core AI/ML Libraries

### 2.1 Agent Orchestration — LangGraph

| Property | Value |
|----------|-------|
| **Package** | `langgraph` |
| **Version** | >= 0.2.x |
| **Purpose** | Define stateful multi-agent workflows as directed graphs |
| **Key Feature** | Explicit state graph with typed state passing between agent nodes |
| **Why Chosen** | Deterministic, debuggable, auditable agent transitions; better than AutoGen's conversational loops for production-grade traceability |
| **Install** | `pip install langgraph` |

**Usage in project:**
- `StateGraph` defines the agent execution DAG
- Each agent (Investigator, Reasoner, Reporter) is a graph node
- State (anomaly data, retrieved docs, hypothesis, RCA) flows through typed edges
- Conditional routing handles retrieval failures

### 2.2 RAG Framework — LangChain + LlamaIndex

| Property | Value |
|----------|-------|
| **Packages** | `langchain`, `langchain-community`, `llama-index` |
| **Purpose** | Document loading, chunking, embedding pipeline, LLM integration |
| **LlamaIndex Role** | Primary RAG pipeline (document ingestion → retrieval) |
| **LangChain Role** | Agent tool integration, prompt templates, LLM wrappers |
| **Install** | `pip install langchain langchain-community llama-index` |

### 2.3 Vector Store — ChromaDB

| Property | Value |
|----------|-------|
| **Package** | `chromadb` |
| **Version** | >= 0.5.x |
| **Purpose** | Embedded vector database for document chunk storage and retrieval |
| **Key Feature** | No external server required; runs in-process; persistent storage |
| **Capacity** | Handles 100K+ documents comfortably on local machine |
| **Install** | `pip install chromadb` |

**Configuration:**
- Collection: `telecom_billing_kb`
- Distance metric: cosine similarity
- Metadata: source document, page number, chunk index, document type

### 2.4 Embedding Model — sentence-transformers

| Property | Value |
|----------|-------|
| **Package** | `sentence-transformers` |
| **Model** | `all-MiniLM-L6-v2` |
| **Dimensions** | 384 |
| **Speed** | ~14,000 sentences/sec on CPU |
| **Size** | ~80MB |
| **Purpose** | Convert document chunks and queries to dense vectors for semantic search |
| **Install** | `pip install sentence-transformers` |

**Why this model:** Fast, accurate, runs on CPU, well-benchmarked, widely used in production RAG systems.

### 2.5 LLM Backend — Groq

| Property | Value |
|----------|-------|
| **Service** | Groq Cloud API |
| **Purpose** | Fast LLM inference via cloud API; free tier available |
| **Model** | `llama-3.3-70b-versatile` |
| **API** | REST API compatible with OpenAI client format |
| **Speed** | ~1-5s per call (cloud inference) |
| **Install** | `pip install langchain-groq` |

**Setup:**
```bash
# Get free API key at https://console.groq.com
# Add to .env file:
GROQ_API_KEY=your_key_here
```

### 2.6 Anomaly Detection — scikit-learn

| Property | Value |
|----------|-------|
| **Package** | `scikit-learn` |
| **Algorithms** | `IsolationForest` (primary), `DBSCAN` (secondary) |
| **Purpose** | Flag anomalous billing events with confidence scores |
| **Install** | `pip install scikit-learn` |

**IsolationForest config:**
```python
from sklearn.ensemble import IsolationForest
model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    max_features=0.8,
    random_state=42
)
```

---

## 3. Data & Infrastructure

| Tool | Package | Purpose | Install |
|------|---------|---------|---------|
| **Pandas** | `pandas` | Data manipulation, anomaly injection scripting | `pip install pandas` |
| **NumPy** | `numpy` | Numerical operations, seed-controlled randomization | `pip install numpy` |
| **PyMuPDF** | `pymupdf` | PDF text extraction for knowledge base documents | `pip install pymupdf` |
| **SQLite** | Built-in (`sqlite3`) | Lightweight local database for billing records | Built into Python |
| **DVC** | `dvc` | Data version control — track dataset versions | `pip install dvc` |
| **Git** | System tool | Code version control | https://git-scm.com |

---

## 4. Experiment Tracking — MLflow

| Property | Value |
|----------|-------|
| **Package** | `mlflow` |
| **Purpose** | Log every agent run: parameters, metrics, artifacts |
| **Install** | `pip install mlflow` |

**What gets logged per run:**
- **Parameters:** anomaly_id, anomaly_type, model_name, top_k, temperature
- **Metrics:** retrieval_score, faithfulness, bertscore, rouge_l, latency_ms
- **Artifacts:** RCA report (markdown), retrieved doc IDs, agent state snapshots

```bash
mlflow ui  # Launch tracking UI on localhost:5000
```

---

## 5. User Interface — Streamlit

| Property | Value |
|----------|-------|
| **Package** | `streamlit` |
| **Purpose** | Web dashboard — upload CSV, view anomalies, trigger RCA, display reports |
| **Install** | `pip install streamlit` |

**Dashboard pages:**
1. **Home** — Project overview, system status
2. **Upload & Detect** — CSV upload → anomaly detection → results table
3. **RCA Viewer** — Click anomaly → trigger agent pipeline → view RCA report
4. **Knowledge Base** — Browse indexed documents, search corpus, view chunks

```bash
streamlit run app.py  # Launch on localhost:8501
```

---

## 6. Evaluation Libraries

| Tool | Package | Purpose | Install |
|------|---------|---------|---------|
| **RAGAS** | `ragas` | RAG evaluation: faithfulness, answer relevancy, context recall/precision | `pip install ragas` |
| **BERTScore** | `bert-score` | Semantic similarity between generated and reference RCA | `pip install bert-score` |
| **ROUGE** | `rouge-score` | Text overlap (ROUGE-L) between generated and reference RCA | `pip install rouge-score` |
| **SciPy** | `scipy` | Statistical tests: Wilcoxon signed-rank, Mann-Whitney U | `pip install scipy` |
| **Matplotlib** | `matplotlib` | Charts, confusion matrices, evaluation plots | `pip install matplotlib` |
| **Seaborn** | `seaborn` | Statistical visualizations | `pip install seaborn` |

---

## 7. Visualization & Reporting

| Tool | Purpose |
|------|---------|
| **Matplotlib + Seaborn** | EDA plots, evaluation charts, confusion matrices |
| **Plotly** | Interactive visualizations in Streamlit dashboard |
| **MLflow UI** | Experiment comparison dashboards |

---

## 8. Complete `requirements.txt`

```txt
# Core AI/ML
langgraph>=0.2.0
langchain>=0.2.0
langchain-community>=0.2.0
llama-index>=0.10.0
chromadb>=0.5.0
sentence-transformers>=3.0.0

# Data
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pymupdf>=1.24.0

# LLM
langchain-groq>=1.0.0

# Tracking & UI
mlflow>=2.10.0
streamlit>=1.30.0

# Evaluation
ragas>=0.1.0
bert-score>=0.3.13
rouge-score>=0.1.2
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
tqdm>=4.65.0

# Data versioning
dvc>=3.30.0
```

---

## 9. Development Tools

| Tool | Purpose |
|------|---------|
| **VS Code** | Primary IDE |
| **Jupyter Notebooks** | EDA, experimentation, prototyping |
| **Black** | Python code formatter |
| **pytest** | Unit testing framework |
| **Git** | Version control |
| **DVC** | Data version control |

---

## 10. Hardware Requirements

### Minimum (Development)

| Component | Specification |
|-----------|-------------|
| **CPU** | 8+ cores (Intel i7 / AMD Ryzen 7) |
| **RAM** | 16 GB |
| **GPU** | Optional (LLM runs on Groq cloud) |
| **Storage** | 20 GB free (vector DB + datasets) |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

### Recommended (Comfortable Development)

| Component | Specification |
|-----------|-------------|
| **CPU** | 12+ cores |
| **RAM** | 32 GB |
| **GPU** | Optional (not required — Groq handles LLM inference) |
| **Storage** | 100 GB SSD |

### No GPU Fallback

Since LLM inference runs on Groq cloud, no local GPU is required.
- Groq free tier: 30 requests/min, 14,400 requests/day
- Expect occasional rate-limiting on free tier during heavy evaluation runs
- Fallback templates still work if API is unreachable

---

## 11. Project Directory Structure

```
RAGML/
├── .venv/                          # Python virtual environment
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── raw/                        # Original datasets (DVC-tracked)
│   ├── processed/                  # Labeled anomaly datasets
│   ├── corpus/                     # RAG knowledge base documents
│   │   ├── standards/
│   │   ├── regulatory/
│   │   ├── incidents/
│   │   └── rca_playbooks/
│   └── eval/                       # Evaluation ground truth
│       ├── ground_truth_rca/
│       └── test_anomalies.csv
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Dataset loading & schema validation
│   │   └── injector.py             # Anomaly injection scripts
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py             # IsolationForest / DBSCAN
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py              # Document chunking pipeline
│   │   ├── embedder.py             # Embedding generation
│   │   └── retriever.py            # ChromaDB query interface
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── investigator.py         # Investigator Agent
│   │   ├── reasoner.py             # Reasoning Agent
│   │   ├── reporter.py             # Reporter Agent
│   │   ├── graph.py                # LangGraph StateGraph assembly
│   │   └── prompts/                # System prompts for each agent
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # RAGAS, BERTScore, ROUGE
│   │   └── ablation.py             # Ablation study runner
│   └── cli.py                      # Command-line interface
│
├── app/
│   ├── app.py                      # Streamlit main app
│   └── pages/                      # Multi-page Streamlit app
│       ├── 1_upload_detect.py
│       ├── 2_rca_viewer.py
│       └── 3_knowledge_base.py
│
├── notebooks/
│   ├── 01_eda_ibm.ipynb
│   ├── 02_eda_maven.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_rag_experiments.ipynb
│   └── 05_agent_prototyping.ipynb
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_injector.py
│   ├── test_detector.py
│   ├── test_retriever.py
│   └── test_agents.py
│
├── docs/                           # Documentation (these files)
│   ├── 01_THESIS_STRUCTURE.md
│   ├── 02_ABSTRACT.md
│   ├── 03_WEEK_WISE_IMPLEMENTATION.md
│   ├── 04_DATA_SOURCES.md
│   ├── 05_REFERENCES.md
│   ├── 06_TOOLS_AND_STACK.md
│   ├── 07_COST_ANALYSIS.md
│   └── corpus_manifest.csv
│
├── models/                         # Saved model artifacts
│   └── detector/
│
├── mlruns/                         # MLflow experiment data
│
└── chroma_db/                      # ChromaDB persistent storage
```
