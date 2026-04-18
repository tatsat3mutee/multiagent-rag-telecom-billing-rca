# Multi-Agent RAG System for Telecom Billing RCA

**A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks**

MTech Thesis — Data Science & Engineering | Tatsat Pandey | 2026

---

## Overview

This system uses a multi-agent Retrieval-Augmented Generation (RAG) architecture to autonomously investigate billing anomalies in telecom networks and generate structured Root Cause Analysis (RCA) reports.

Three specialized LLM-powered agents are orchestrated via LangGraph:
1. **Investigator Agent** — Queries the RAG knowledge base to retrieve relevant documentation
2. **Reasoning Agent** — Synthesizes anomaly data with retrieved context to generate root cause hypotheses
3. **Reporter Agent** — Produces JSON-schema-validated RCA reports with recommended actions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard + MLflow Tracking                       │
├─────────────────────────────────────────────────────────────┤
│  LangGraph Agent Pipeline                                    │
│  Investigator → Reasoner → Reporter                          │
├─────────────────────────────────────────────────────────────┤
│  RAG Engine: ChromaDB + sentence-transformers                │
├─────────────────────────────────────────────────────────────┤
│  Anomaly Detection: IsolationForest / DBSCAN                 │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline: Pandas + Synthetic Anomaly Injection         │
├─────────────────────────────────────────────────────────────┤
│  LLM Backend: OpenAI (gpt-4o-mini generator, gpt-4o judge)   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

Sign up at https://platform.openai.com and create an API key, then create a `.env` file:
```
OPENAI_API_KEY=sk-...
# optional overrides
LLM_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
```

### 3. Run Full Pipeline

```bash
python run_pipeline.py
```

This will:
- Generate synthetic telecom billing datasets
- Inject 5 types of billing anomalies
- Train IsolationForest and DBSCAN detectors
- Build the RAG knowledge base (ChromaDB)
- Run the multi-agent RCA pipeline on detected anomalies
- Evaluate results and log to MLflow

### 4. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

### 5. CLI Usage

```bash
# Setup system
python src/cli.py --setup

# Process anomalies from CSV  
python src/cli.py --csv data/processed/anomalies_labeled.csv --limit 5

# Single anomaly
python src/cli.py --input '{"account_id":"CUST-001","anomaly_type":"zero_billing","confidence":0.95,"monthly_charges":0.0,"total_charges":2500.0,"tenure":36,"features":{}}'
```

### 6. MLflow Dashboard

```bash
mlflow ui
```
Open http://localhost:5000

## Project Structure

```
RAGML/
├── app.py                          # Streamlit main dashboard
├── run_pipeline.py                 # End-to-end pipeline runner
├── config.py                       # Central configuration
├── requirements.txt                # Python dependencies
├── src/
│   ├── data/
│   │   ├── loader.py               # Dataset loading
│   │   └── anomaly_injector.py     # Synthetic anomaly injection
│   ├── detection/
│   │   └── detector.py             # IsolationForest, DBSCAN
│   ├── rag/
│   │   ├── chunker.py              # Document chunking
│   │   ├── embedder.py             # Embedding pipeline
│   │   └── knowledge_base.py       # ChromaDB management
│   ├── agents/
│   │   ├── state.py                # Agent state schema
│   │   ├── prompts.py              # Agent prompt templates
│   │   ├── investigator.py         # Investigator agent
│   │   ├── reasoner.py             # Reasoning agent
│   │   ├── reporter.py             # Reporter agent
│   │   └── graph.py                # LangGraph StateGraph
│   ├── evaluation/
│   │   └── metrics.py              # Evaluation metrics
│   ├── mlflow_tracking.py          # MLflow integration
│   └── cli.py                      # Command-line interface
├── data/
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Labeled datasets
│   ├── corpus/rca_playbooks/       # RCA knowledge base
│   └── eval/ground_truth_rca/      # Ground truth for evaluation
├── pages/                          # Streamlit pages
│   ├── 1_Upload_Detect.py
│   ├── 2_RCA_Viewer.py
│   └── 3_Knowledge_Base.py
├── scripts/
│   └── download_datasets.py        # Dataset generation
└── docs/                           # Thesis documentation
```

## Anomaly Types

| Type | Description | Detection Method |
|------|-------------|-----------------|
| Zero Billing | Active customer charged $0 | MonthlyCharges = 0 with active services |
| Duplicate Charge | Same usage billed twice | Doubled charge amounts |
| Usage Spike | 10x usage increase | Multiplied usage features |
| CDR Failure | Missing billing data | NULL values in critical fields |
| SLA Breach | Charges exceed contract cap | Charges > 95th percentile × 1.5 |

## Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI (gpt-4o-mini generator, gpt-4o judge) |
| Agent Orchestration | LangGraph |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Anomaly Detection | scikit-learn (IsolationForest, DBSCAN) |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit |
| Evaluation | ROUGE, BERTScore |

## Evaluation

The system is evaluated across three dimensions:
1. **Anomaly Detection**: Precision, Recall, F1-Score, ROC-AUC
2. **RAG Retrieval Quality**: Context Recall, Context Precision, MRR@5
3. **RCA Output Quality**: BERTScore, ROUGE-L, Anomaly Type Match accuracy

## License

This project is for academic purposes (MTech Thesis).
