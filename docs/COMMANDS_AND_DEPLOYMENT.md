# Commands & Deployment — Quick Reference

All commands assume PowerShell from the repo root with `.venv` activated.

---

## 1. One-Time Setup

```powershell
# Clone and enter workspace
cd "C:\Users\TatsatPandey\Documents\Personal\Personal\Mtech DSE\Sem 4\RAGML"

# Create venv and activate
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Configure LLM

Create `.env` in the repo root:

```env
# Preferred provider (pick one)
GROQ_API_KEY=gsk_...

# Fallback provider
KIMI_API_KEY=...

# Custom OpenAI-compatible endpoint (optional)
LLM_API_KEY=...
LLM_BASE_URL=https://example.com/v1
LLM_MODEL=your-model-name

# Independent judge (optional — defaults to generator)
JUDGE_API_KEY=...
JUDGE_BASE_URL=https://example.com/v1
JUDGE_MODEL=your-judge-model
```

### Verify LLM

```powershell
python test_llm.py
# Expect: "SUCCESS: ..."
```

---

## 2. Build Knowledge Base

```powershell
# Index 8 telecom RCA playbooks into ChromaDB
python -c "from src.rag.knowledge_base import build_knowledge_base; build_knowledge_base(force_rebuild=True)"
```

---

## 3. Build GraphRAG Graph

```powershell
# Offline / deterministic (no LLM cost)
python scripts\build_graph_rag.py --offline

# Full LLM-extracted graph (richer, recommended for final eval)
python scripts\build_graph_rag.py
```

---

## 4. Run Full Pipeline

```powershell
python run_pipeline.py
```

This: loads data → injects anomalies → trains IsolationForest + DBSCAN → indexes playbooks → runs multi-agent pipeline on a sample → logs to MLflow.

---

## 5. Run Ablation Study

```powershell
# All 5 configs, 60 anomalies, with LLM-as-Judge
python run_ablation.py --n 60 --configs A,B,C,D,E --judge

# Quick smoke test (10 items, no judge)
python run_ablation.py --n 10 --configs A,D
```

Output: `ablation_results.json` + printed significance table.

### Configs

| Config | Description |
|---|---|
| A | No RAG — direct LLM generation |
| B | RAG-only — retrieve + generate in single prompt |
| C | Single-agent + RAG |
| D | Multi-agent + RAG (proposed 4-agent pipeline) |
| E | Multi-agent + GraphRAG (headline novelty) |

---

## 6. Run Tests

```powershell
# Quick
python -m pytest tests/ -q

# With coverage
python -m pytest tests/ --cov=src --cov-report=term
```

---

## 7. Launch Streamlit UI

```powershell
streamlit run app.py
# Opens http://localhost:8501
```

Pages:
- **Home** — architecture overview
- **Upload & Detect** — upload CSV, run anomaly detection
- **RCA Viewer** — select anomaly → run agent pipeline → view RCA
- **Knowledge Base** — browse/search telecom playbooks

---

## 8. MLflow UI

```powershell
mlflow ui
# Opens http://localhost:5000
```

---

## 9. CLI

```powershell
# Setup system
python src/cli.py --setup

# Process anomalies from CSV
python src/cli.py --csv data/processed/anomalies_labeled.csv --limit 5

# Single anomaly JSON
python src/cli.py --input '{"account_id":"TEST","anomaly_type":"zero_billing","confidence":0.9,"monthly_charges":0.0,"total_charges":1000,"tenure":12,"features":{}}'
```

---

## 10. Docker Deployment

```powershell
docker build -t telecom-rca .
docker run -p 8501:8501 -e GROQ_API_KEY=$env:GROQ_API_KEY telecom-rca
# Opens http://localhost:8501
```

---

## 11. Streamlit Cloud Deployment

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `app.py` on branch `main`
4. Set secrets: `GROQ_API_KEY` (or `KIMI_API_KEY`)
5. Runtime uses Python 3.11 via `runtime.txt`

---

## 12. Plot Results

```powershell
python scripts\plot_results.py --metric rouge_l
# PNGs saved to docs/diagrams/
```
