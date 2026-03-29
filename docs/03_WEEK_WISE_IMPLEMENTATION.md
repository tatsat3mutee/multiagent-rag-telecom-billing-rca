# Week-by-Week Implementation Plan

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

**Total Duration:** 16 Weeks  
**Start Date:** ___________  
**End Date:** ___________  

---

## Phase 1: Foundation (Weeks 1–2)

### Week 1 — Environment Setup & Stack Validation

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Project scaffolding | Create Git repo, folder structure, `.gitignore`, `requirements.txt` | GitHub repo initialized |
| 2 | Python environment | Set up venv/conda, install core packages (pandas, numpy, scikit-learn, matplotlib) | Working Python env |
| 3 | Set up Groq API | Get Groq API key, install `langchain-groq`, test inference with `llama-3.3-70b-versatile` | `test_llm.py` returns output |
| 4 | Install ChromaDB | `pip install chromadb`, create test collection, insert/query test embeddings | ChromaDB hello-world script |
| 5 | Install LangGraph | `pip install langgraph`, build minimal 2-node StateGraph workflow | LangGraph hello-world script |

**Milestone Check:** LLM inference (Groq API) + vector DB + agent framework all operational.

### Week 2 — RAG Hello World + MLflow/Streamlit Setup

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Install sentence-transformers | `pip install sentence-transformers`, test `all-MiniLM-L6-v2` embedding generation | Embedding test script |
| 2 | Hello World RAG | Index 3 sample PDF pages → embed → store in ChromaDB → query → retrieve → pass to LLM | Working RAG query (end-to-end) |
| 3 | Install MLflow | `pip install mlflow`, log a dummy experiment run with parameters and metrics | MLflow UI accessible |
| 4 | Install Streamlit | `pip install streamlit`, build minimal UI: text input → display LLM response | Streamlit hello-world app |
| 5 | DVC + documentation | Install DVC for data versioning, write `README.md`, document setup steps | Project README complete |

**Deliverable:** Working dev environment with validated RAG query pipeline.

---

## Phase 2: Data Pipeline (Weeks 3–4)

### Week 3 — Dataset Loading & EDA

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Download datasets | IBM Telco Churn (Kaggle), Maven Telecom Churn | Raw CSVs in `data/raw/` |
| 2 | Data loading scripts | Pandas loaders with schema validation, dtype enforcement | `src/data/loader.py` |
| 3 | EDA — IBM Telco | Distribution plots, missing values, correlation matrix, billing feature analysis | `notebooks/01_eda_ibm.ipynb` |
| 4 | EDA — Maven Telecom | Same analysis for Maven dataset, cross-dataset feature comparison | `notebooks/02_eda_maven.ipynb` |
| 5 | Feature engineering | Create derived billing features: charges_per_month, usage_ratio, tenure_bucket | Feature engineering script |

### Week 4 — Synthetic Anomaly Injection

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Zero-billing injection | Set `MonthlyCharges = 0` for random active customers (seed-controlled) | Injection script + labeled CSV |
| 2 | Duplicate charge injection | Duplicate CDR records within billing cycle with matching timestamps | Injection script |
| 3 | Usage spike simulation | Multiply data/voice usage by 10x for random accounts | Injection script |
| 4 | CDR failure + SLA breach | Introduce null fields (CDR failure); generate SLA-violating usage patterns | Injection scripts |
| 5 | Final labeled dataset | Merge all anomaly types, create `anomaly_type` and `is_anomaly` columns, DVC track | `data/processed/anomalies_labeled.csv` |

**Deliverable:** Labeled anomaly dataset with 5 anomaly types + EDA notebooks + DVC tracking.

**Supervisor Checkpoint M1:** Show Jupyter notebooks with visualizations + labeled CSV.

---

## Phase 3: Anomaly Detection (Weeks 5–6)

### Week 5 — Model Training

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Train/test split | Stratified split preserving anomaly type distribution (80/20) | Split script |
| 2 | IsolationForest training | Train on billing features, default params, evaluate baseline | Baseline model |
| 3 | Hyperparameter tuning | Grid search: `n_estimators`, `contamination`, `max_features` | Tuned IsolationForest |
| 4 | DBSCAN training | DBSCAN as alternative detector, tune `eps` and `min_samples` | Trained DBSCAN |
| 5 | Feature importance | SHAP values or feature contribution analysis for IsolationForest | Feature importance plots |

### Week 6 — Evaluation & Integration

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Metrics computation | Precision, Recall, F1, ROC-AUC, Average Precision for both models | Evaluation report |
| 2 | Confusion matrices | Per-anomaly-type confusion matrices, error analysis | Confusion matrix visualizations |
| 3 | Model comparison | IsolationForest vs DBSCAN comparison table with statistical tests | Comparison report |
| 4 | Integration interface | Define output schema: `{account_id, anomaly_type, confidence, features}` | `src/detection/detector.py` |
| 5 | MLflow logging | Log training runs, hyperparams, metrics, model artifacts to MLflow | MLflow experiment runs |

**Deliverable:** Trained anomaly detector with F1 > 0.80, evaluation report, MLflow logs.

**Supervisor Checkpoint M2:** Evaluation report + confusion matrix + MLflow UI screenshot.

---

## Phase 4: RAG Knowledge Base (Weeks 7–8)

### Week 7 — Document Collection & Processing

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Corpus manifest | Create spreadsheet: document name, source URL, format, relevant sections, page range | `docs/corpus_manifest.csv` |
| 2 | Download documents | ETSI/ITU-T SLA templates, 3GPP TS 32.240, FCC complaint data, TRAI regulations | PDFs in `data/corpus/` |
| 3 | PDF parsing | PyMuPDF extraction, section-level splitting, metadata tagging | Parsed text files |
| 4 | Write RCA playbooks | Create 15–20 synthetic RCA documents for known anomaly types (gold retrieval targets) | `data/corpus/rca_playbooks/` |
| 5 | Write ground-truth RCAs | 3–5 reference RCA documents per anomaly type (evaluation gold standard) | `data/eval/ground_truth_rca/` |

### Week 8 — Embedding, Indexing & Retrieval Testing

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Chunking pipeline | Recursive text splitting (chunk_size=512, overlap=64), metadata preservation | `src/rag/chunker.py` |
| 2 | Embedding pipeline | Batch embed all chunks using `all-MiniLM-L6-v2`, handle large corpus | `src/rag/embedder.py` |
| 3 | ChromaDB indexing | Create collection, insert embeddings with metadata (source, page, type) | Populated ChromaDB |
| 4 | Retrieval testing | Test queries for each anomaly type, check top-5 results, compute MRR@5 | Retrieval quality report |
| 5 | Retrieval tuning | Adjust chunk size, overlap, top-k, consider re-ranking strategies | Optimized retrieval config |

**Deliverable:** Populated ChromaDB + retrieval quality score + ground-truth RCA documents.

**Supervisor Checkpoint M3:** Demo: query → retrieved docs screenshot + retrieval metrics.

---

## Phase 5: Agent Development (Weeks 9–10)

### Week 9 — Individual Agent Implementation

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | State schema definition | Define `AgentState` TypedDict: anomaly_data, retrieved_docs, hypothesis, rca_report | `src/agents/state.py` |
| 2 | Investigator Agent | LangGraph node: receives anomaly → formulates query → retrieves top-k docs from ChromaDB | `src/agents/investigator.py` |
| 3 | Reasoning Agent | LangGraph node: receives anomaly + docs → generates structured root cause hypothesis | `src/agents/reasoner.py` |
| 4 | Reporter Agent | LangGraph node: receives hypothesis + evidence → produces JSON-schema-validated RCA | `src/agents/reporter.py` |
| 5 | Unit tests | Test each agent independently with mock inputs, verify output schema compliance | `tests/test_agents.py` |

### Week 10 — StateGraph Assembly & CLI Pipeline

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | LangGraph StateGraph | Wire Investigator → Reasoner → Reporter as graph nodes with typed edges | `src/agents/graph.py` |
| 2 | Conditional routing | Add error handling: if retrieval returns < threshold docs, request broader query | Routing logic |
| 3 | Prompt engineering | Refine system prompts: grounding instructions, output format, few-shot examples | `src/agents/prompts/` |
| 4 | CLI interface | Command-line tool: input anomaly record → output RCA markdown | `src/cli.py` |
| 5 | End-to-end test | Run 10 anomaly cases through full pipeline, verify outputs manually | E2E test results log |

**Deliverable:** Working 3-agent pipeline (CLI) producing RCA reports.

**Supervisor Checkpoint M4:** Screen recording of pipeline run (anomaly input → RCA output).

---

## Phase 6: Integration & Testing (Weeks 11–12)

### Week 11 — Full System Integration

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Detector → Agent pipeline | Connect anomaly detector output as agent pipeline trigger | Integrated pipeline |
| 2 | Batch processing | Process multiple anomalies sequentially, handle errors gracefully | Batch runner script |
| 3 | MLflow integration | Log per-run: anomaly input, retrieved doc IDs, RCA output, latency, token count | MLflow experiment runs |
| 4 | Basic Streamlit UI | File upload → anomaly detection → trigger RCA → display report | Minimal Streamlit app |
| 5 | Integration tests | End-to-end tests: CSV upload → detection → RCA → report output | Integration test suite |

### Week 12 — Prompt Tuning & Optimization

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Error analysis | Review first 50 RCA outputs, categorize failures (retrieval miss, reasoning error, format error) | Error analysis report |
| 2 | Prompt iteration | Refine prompts based on error analysis, add few-shot examples for failure cases | Updated prompts v2 |
| 3 | Retrieval tuning | Adjust top-k, try hybrid search (keyword + semantic), test re-ranking | Tuned retrieval config |
| 4 | Latency optimization | Profile pipeline, identify bottlenecks, optimize where possible | Performance profile |
| 5 | Regression testing | Re-run full test suite with updated prompts/retrieval, compare metrics | Before/after comparison |

**Deliverable:** Full integrated system with MLflow logging, tuned prompts.

**Supervisor Checkpoint M5:** MLflow UI screenshot + experiment run comparison.

---

## Phase 7: Evaluation (Week 13)

### Week 13 — Comprehensive Evaluation & Ablation

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Evaluation dataset | Prepare 100+ anomaly test cases across all 5 types with ground-truth RCAs | `data/eval/test_set.csv` |
| 2 | Multi-agent evaluation | Run full proposed system on test set, compute all metrics | Multi-agent results |
| 3 | Ablation runs | Run 3 comparison configs: No RAG, RAG Only, Single Agent + RAG | Ablation results |
| 4 | Statistical testing | Wilcoxon signed-rank test for paired comparisons, compute confidence intervals | Statistical significance results |
| 5 | Results compilation | Create results tables, comparison charts, per-anomaly-type breakdown | Evaluation results package |

**Deliverable:** Complete evaluation results + ablation study + statistical significance tests.

**Supervisor Checkpoint M6:** Results tables + comparison charts.

---

## Phase 8: UI & Polish (Week 14)

### Week 14 — Streamlit Dashboard & Demo-Ready System

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Dashboard layout | Multi-page Streamlit: Home, Upload & Detect, RCA Viewer, Knowledge Base Browser | UI wireframe |
| 2 | Upload & Detect page | CSV upload → table view → anomaly detection → highlight flagged rows | Working page |
| 3 | RCA Viewer page | Click anomaly → trigger agent pipeline → display RCA report with citations | Working page |
| 4 | Knowledge Base browser | Browse indexed documents, search knowledge base, view chunk details | Working page |
| 5 | Demo polish | Add loading spinners, error messages, export RCA as PDF/markdown | Demo-ready application |

**Deliverable:** Live Streamlit application.

**Supervisor Checkpoint M7:** Live demo to supervisor.

---

## Phase 9: Thesis Writing (Weeks 14–15)

> **Note:** Begin writing Introduction and Literature Review sections as early as Week 10. Do not leave all writing for the final weeks.

### Week 14 (parallel with UI)
- Draft Chapter 1: Introduction (leverage proposal document)
- Draft Chapter 2: Literature Review (expand from notes collected throughout project)

### Week 15 — Full Thesis Draft

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Chapter 3: Methodology | Architecture diagrams, data collection details, agent design rationale | Chapter 3 draft |
| 2 | Chapter 4: Implementation | Code walkthroughs, screenshots, configuration details | Chapter 4 draft |
| 3 | Chapter 5: Results | Tables, charts, ablation analysis, qualitative case studies | Chapter 5 draft |
| 4 | Chapter 6: Discussion + Chapter 7: Conclusion | Interpret results, limitations, future work | Chapters 6–7 draft |
| 5 | Abstract + References + Appendices | Finalize abstract, compile all references (BibTeX), format appendices | Full thesis draft v1 |

**Deliverable:** Complete thesis draft.

**Supervisor Checkpoint M8:** Full document submitted for review.

---

## Phase 10: Defense Preparation (Week 16)

### Week 16 — Presentation & Rehearsal

| Day | Task | Details | Deliverable |
|-----|------|---------|-------------|
| 1 | Slide deck creation | 20–25 slides: problem, architecture, methodology, results, demo, conclusion | Presentation v1 |
| 2 | Demo script | Step-by-step demo: upload CSV → detect anomalies → generate RCA → show report | Demo script document |
| 3 | Q&A preparation | Review defense Q&A guide, prepare answers for anticipated committee questions | Q&A preparation document |
| 4 | Mock defense #1 | Practice with peers/colleagues, time the presentation (15–20 min), get feedback | Feedback notes |
| 5 | Final revisions | Incorporate thesis reviewer feedback, finalize slides, rehearse demo | Final thesis + slides |

**Deliverable:** Final thesis + defense presentation + demo script.

**Supervisor Checkpoint M9:** Defense presentation rehearsal.

---

## Milestone Summary

| Milestone | Week | Deliverable | Status |
|-----------|------|-------------|--------|
| M1: Data Ready | 4 | Labeled anomaly dataset + EDA notebooks | ⬜ |
| M2: Detector Trained | 6 | IsolationForest with F1 > 0.80 + evaluation report | ⬜ |
| M3: RAG KB Live | 8 | Populated ChromaDB + retrieval quality metrics | ⬜ |
| M4: Agent Pipeline Working | 10 | End-to-end CLI pipeline producing RCA | ⬜ |
| M5: Full System Integrated | 12 | MLflow tracking + tuned prompts + integration tests | ⬜ |
| M6: Evaluation Complete | 13 | All metrics + ablation study + statistical tests | ⬜ |
| M7: Streamlit App Live | 14 | Demo-ready Streamlit application | ⬜ |
| M8: Thesis Submitted | 15 | Full thesis draft | ⬜ |
| M9: Defense Ready | 16 | Final thesis + slides + demo | ⬜ |

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM inference too slow on local hardware | HIGH | MEDIUM | Use quantized models (Q4); fallback to Groq free tier for eval runs |
| RAG retrieval quality insufficient | HIGH | MEDIUM | Invest in RCA playbook documents as gold retrieval targets; tune chunking |
| Prompt engineering requires many iterations | MEDIUM | HIGH | Start simple, iterate weekly; maintain prompt version log |
| Document corpus difficult to collect | MEDIUM | MEDIUM | Start collection in Week 1; prioritize RCA playbooks over raw standards |
| Evaluation metrics below target thresholds | MEDIUM | MEDIUM | Lower targets are still publishable; focus on ablation (relative gains) |
| Thesis writing compressed | HIGH | MEDIUM | Begin Intro/Lit Review by Week 10; write methodology alongside implementation |
| Scope creep | MEDIUM | HIGH | Strictly follow milestone deliverables; defer nice-to-haves to future work |
