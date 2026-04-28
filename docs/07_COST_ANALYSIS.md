# Cost Analysis

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

**Key Principle:** 100% open-source stack — zero recurring API costs.

---

## 1. Cost Summary

| Category | One-Time Cost | Monthly Recurring | Notes |
|----------|--------------|-------------------|-------|
| Software & Tools | **$0** | **$0** | All open-source |
| LLM Inference | **$0** | **$0** | Groq free tier (Llama 3.3 70B) |
| Cloud/API Services | **$0** | **$0** | No cloud dependency |
| Datasets | **$0** | **$0** | All public/open |
| Hardware | Existing | — | Uses personal/lab machine |
| **Total Project Cost** | **$0** | **$0** | |

---

## 2. Software Cost Breakdown

### 2.1 Core Tools — All Free & Open Source

| Tool | License | Cost | Commercial Alternative | Alternative Cost |
|------|---------|------|----------------------|------------------|
| **Groq API** (LLM serving) | Free tier | $0 | Commercial LLM API | $0.50–$15/1M tokens |
| **Llama 3.3 70B** (via Groq) | Meta Llama License | $0 | GPT-4 | $30/1M tokens |
| **LangGraph** | MIT | $0 | LangSmith (managed) | $39/month |
| **ChromaDB** | Apache 2.0 | $0 | Pinecone | $70+/month |
| **sentence-transformers** | Apache 2.0 | $0 | OpenAI Embeddings | $0.13/1M tokens |
| **scikit-learn** | BSD | $0 | DataRobot | $10K+/year |
| **MLflow** | Apache 2.0 | $0 | Weights & Biases | $50+/month |
| **Streamlit** | Apache 2.0 | $0 | — | — |
| **Python** | PSF | $0 | — | — |

### 2.2 Evaluation Tools — Free

| Tool | License | Cost |
|------|---------|------|
| RAGAS | Apache 2.0 | $0 |
| BERTScore | MIT | $0 |
| rouge-score | Apache 2.0 | $0 |
| SciPy | BSD | $0 |

### 2.3 Development Tools — Free

| Tool | License | Cost |
|------|---------|------|
| VS Code | MIT | $0 |
| Git | GPL-2.0 | $0 |
| DVC | Apache 2.0 | $0 |
| Jupyter | BSD | $0 |

---

## 3. LLM Inference Cost Analysis

### 3.1 Groq Cloud Inference (Chosen Approach) — $0

| Model | Provider | Speed | Cost per Query | Cost per 1000 Queries |
|-------|----------|-------|----|---|
| Llama 3.3 70B Versatile | Groq (free tier) | ~1-5s/call | $0 | $0 |

**Per pipeline run:** 3 agent calls × ~500 tokens each = ~1,500 tokens generated  
**For 100-case evaluation:** 100 × 1,500 = 150,000 tokens generated = **$0**

### 3.2 What It Would Cost with Commercial APIs (For Comparison)

| Provider | Model | Input Cost | Output Cost | Per Pipeline Run | 100 Cases | 1000 Cases |
|----------|-------|-----------|-------------|-----------------|----------|-----------|
| OpenAI | GPT-4o | $2.50/1M | $10.00/1M | ~$0.02 | ~$2.00 | ~$20.00 |
| OpenAI | GPT-4 | $30.00/1M | $60.00/1M | ~$0.14 | ~$14.00 | ~$140.00 |
| Anthropic | Claude 3.5 Sonnet | $3.00/1M | $15.00/1M | ~$0.03 | ~$3.00 | ~$30.00 |
| Groq | Llama 3.3 70B | Free tier | Free tier | $0 | $0 | $0* |

*Groq free tier: 30 requests/min, 14,400 requests/day — sufficient for evaluation and demo.

### 3.3 Savings Over Commercial Stack

| Scenario | Commercial Cost (GPT-4o) | This Project | Savings |
|----------|--------------------------|--------------|---------|
| Development (500 pipeline runs) | ~$10 | $0 | $10 |
| Evaluation (100 test cases × 4 ablation configs) | ~$8 | $0 | $8 |
| Prompt iteration (200 experiments) | ~$4 | $0 | $4 |
| Total project | ~$22 | $0 | **$22** |

> With GPT-4 (non-mini): total would be ~$150+. With Groq free tier: $0.

---

## 4. Dataset Cost

| Dataset | Source | Cost | Access Method |
|---------|--------|------|---------------|
| IBM Telco Customer Churn | Kaggle | $0 | Free download (Kaggle account) |
| Maven Telecom Churn | Maven Analytics | $0 | Free download |
| FCC Consumer Complaints | FCC.gov | $0 | Public open data |
| ETSI Standards (public) | ETSI.org | $0 | Free public access |
| ITU-T Recommendations (free subset) | ITU.int | $0 | Free public access |
| 3GPP Specifications | 3GPP.org | $0 | Free public access |
| TRAI Regulations | TRAI.gov.in | $0 | Free public access |
| AT&T SEC Filings | SEC EDGAR | $0 | Free public access |
| GitHub Postmortems | GitHub | $0 | Open source |

**Total dataset cost: $0**

---

## 5. Hardware Cost

### 5.1 Using Existing Hardware (Recommended) — $0

Most MTech students already have a laptop or access to university lab machines that meet minimum requirements:
- 16 GB RAM
- 8 GB VRAM GPU (or CPU-only with quantized models)
- 50 GB free storage

### 5.2 If GPU Upgrade Needed (Optional)

| Option | Cost | Benefit |
|--------|------|---------|
| NVIDIA RTX 4060 (8GB) | ~$300 | Comfortable for 7B models Q4 |
| NVIDIA RTX 4060 Ti (16GB) | ~$400 | Full FP16 inference for 7B-8B models |
| Google Colab Pro | $10/month | T4/A100 GPU access |
| AWS g4dn.xlarge (spot) | ~$0.16/hr | On-demand GPU for evaluation runs |

### 5.3 Cloud GPU — Only If Local Hardware Insufficient

| Provider | Instance | GPU | Cost/Hour | Estimated Hours | Total |
|----------|----------|-----|-----------|----------------|-------|
| Google Colab Pro | — | T4 / A100 | $10/month flat | — | $10–$20 |
| AWS SageMaker (spot) | ml.g4dn.xlarge | T4 | ~$0.16/hr | 20 hrs | ~$3.20 |
| Vast.ai | RTX 3090 | 24GB | ~$0.20/hr | 20 hrs | ~$4.00 |
| Lambda Labs | A10G | 24GB | $0.75/hr | 10 hrs | ~$7.50 |

> **Recommendation:** Use Groq free tier for LLM inference. Only consider cloud GPU if self-hosting LLMs becomes necessary.

---

## 6. Hosting and Deployment Cost

### 6.1 Development — $0

Most services run locally, with LLM inference on Groq cloud:
- Groq API: cloud (free tier)
- ChromaDB: in-process
- MLflow: localhost:5000
- Streamlit: localhost:8501

### 6.2 Demo Deployment (Optional)

| Platform | Cost | Purpose |
|---------|------|---------|
| Streamlit Community Cloud | $0 | Host Streamlit app (no GPU — UI only) |
| HuggingFace Spaces | $0 | Host demo app (limited compute) |
| GitHub Pages | $0 | Host project documentation |
| Render (free tier) | $0 | Backend API hosting |

> **Note:** For thesis defense, demo from local machine. No cloud deployment required.

---

## 7. Cost Comparison: This Project vs. Enterprise Alternatives

| Component | Enterprise Stack | Cost/Year | This Project | Cost |
|-----------|-----------------|-----------|--------------|------|
| LLM API | OpenAI GPT-4 | $5,000–$50,000 | Groq free tier | $0 |
| Vector DB | Pinecone | $840–$3,600 | ChromaDB | $0 |
| Agent Platform | LangSmith | $468 | LangGraph (OSS) | $0 |
| Embeddings | OpenAI Embeddings | $500–$2,000 | sentence-transformers | $0 |
| ML Platform | Databricks MLflow | $5,000+ | MLflow (OSS) | $0 |
| Dashboard | Tableau | $840 | Streamlit | $0 |
| **Total** | | **$12,648–$56,908** | | **$0** |

---

## 8. Time Investment (Non-Monetary Cost)

| Phase | Estimated Hours | Effort Level |
|-------|----------------|--------------|
| Environment setup | 10–15 hrs | Low |
| Data pipeline + EDA | 15–20 hrs | Medium |
| Anomaly detection | 15–20 hrs | Medium |
| RAG knowledge base | 25–30 hrs | High (document curation is manual) |
| Agent development | 30–40 hrs | High (prompt engineering is iterative) |
| Integration + testing | 20–25 hrs | Medium |
| Evaluation | 15–20 hrs | Medium |
| Streamlit UI | 10–15 hrs | Low |
| Thesis writing | 40–60 hrs | High |
| Defense preparation | 10–15 hrs | Medium |
| **Total** | **190–260 hrs** | ~16 weeks × 12–16 hrs/week |

---

## 9. Risk: Hidden Costs

| Risk | Potential Cost | Mitigation |
|------|---------------|------------|
| Need Groq API for faster eval | $0 (free tier sufficient) | Use free tier: 14,400 req/day |
| Google Colab for GPU access | $10–$20/month | Only if local GPU insufficient |
| Premium datasets needed | $0 | All datasets are free/open |
| Domain expert consultation | $0 | Rely on public documents + literature |
| Printing thesis | $20–$50 | University may cover this |

---

## 10. Bottom Line

| Item | Cost |
|------|------|
| Software | $0 |
| LLM inference | $0 |
| Datasets | $0 |
| APIs | $0 |
| Hardware (existing) | $0 |
| Cloud (optional) | $0–$20 |
| **Total Project Budget** | **$0–$20** |

This is a **zero-cost project** using Groq free tier and open-source tools. The entire stack is open-source, all data is publicly available, and LLM inference runs via Groq’s free API tier. This is a significant academic and practical advantage — the system can be replicated by any researcher or telecom operator without licensing barriers.
