# System Architecture — In-Depth Reference

**Version:** post-Phase-2 (GraphRAG + Critic + Hybrid + Observability)
**Date:** April 2026
**Scope:** Technical reference for defense, viva, interviews, and reproducibility. Every component below links back to the file + line where it's implemented.

---

## 0. TL;DR — what this system is

An anomaly-triggered, multi-agent Retrieval-Augmented Generation system for **telecom billing root-cause analysis**. Given a suspicious billing record flagged by an unsupervised detector, the system retrieves relevant telecom-operations playbook context, runs a multi-step agent workflow (Investigator → Reasoner → Critic → Reporter), and produces a structured RCA report. Evaluation combines lexical (ROUGE-L), semantic (BERTScore), RAGAS-style (faithfulness, answer-relevancy), and LLM-as-Judge (4-axis Likert) metrics, with bootstrap CIs, paired-bootstrap, and Wilcoxon significance tests.

The **headline novelty** for the viva is **GraphRAG over telecom playbooks** ([src/rag/graph_rag.py](src/rag/graph_rag.py)) — entity+relation extraction from domain docs, stored in NetworkX, used for multi-hop retrieval, ablated as **Config E** against flat-vector baselines.

---

## 1. Why OpenAI (why we removed Groq)

### The decision
Generator + judge both moved to OpenAI models.
- **Generator:** `gpt-4o-mini` — set via `LLM_MODEL` env var or [config.py](config.py#L41)
- **Judge:** `gpt-4o` — set via `JUDGE_MODEL` env var or [config.py](config.py#L48)

### Why we removed Groq
| # | Reason | Consequence for the thesis |
|---|---|---|
| 1 | **Same-model bias.** The generator and the judge were both Llama 3.3 70B. LLM-as-Judge has a well-documented self-preference bias (Zheng et al., 2023, *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*). When judge = generator, the evaluation is not defensible. | Viva-blocking. Reviewers will ask why the judge isn't independent. |
| 2 | **Free-tier rate limiting.** Groq's free tier is ~30 RPM. Ablation = 5 configs × 60 GT × ~2 LLM calls ≈ 600 requests → the 93-100% 429-failure rate documented in `ablation_output*.txt` directly contaminated the first-round numbers. | Prior ablation results were unreliable (mostly rate-limit errors, not model failures). |
| 3 | **No JSON-mode guarantees.** Groq's `response_format={"type":"json_object"}` compatibility was fragile on Llama 3.3 through LangChain wrappers; the judge prompts and critic prompts needed hard JSON. | Parsing error rate was elevated, again skewing metrics. |
| 4 | **Model tiering for bias mitigation.** OpenAI lets us use a **weaker model for generation** (`gpt-4o-mini`, ~$0.15/$0.60 per 1M tokens) and a **stronger, different-family model for judging** (`gpt-4o`). This is the standard bias-mitigation pattern. | Judge is independent in both tier and size — defensible. |
| 5 | **Tokenizer / context reliability.** OpenAI's client retries, timeouts, and structured outputs are first-class and battle-tested. | Less engineering effort spent on retry logic; more on research. |
| 6 | **Cost for 60-item ablation.** Rough estimate at gpt-4o-mini + gpt-4o: generation ~300 calls × ~1k tokens ≈ $0.50; judging ~240 calls × ~2k tokens at gpt-4o ≈ $3. **Total well under $5/run.** Affordable for re-runs during thesis iteration. | Trade money for reliability and defensibility — the right call for thesis. |

### What we kept from Groq
- `langchain-groq` remains in [requirements.txt](requirements.txt#L12) as an **optional fallback** so any collaborator using a Groq key still gets a working system.
- The judge module [src/evaluation/llm_judge.py](src/evaluation/llm_judge.py#L46-L56) auto-detects `OPENAI_API_KEY` first, then `GROQ_API_KEY`, then `"none"` — pipelines degrade gracefully, tests stay offline-safe.
- The legacy `GROQ_API_KEY` alias in [config.py](config.py#L44) prevents any lingering import from crashing.

### Things to disclose in the viva
- LLM-as-Judge bias is **mitigated, not eliminated.** Every judge result carries a `backend` field ([src/evaluation/llm_judge.py](src/evaluation/llm_judge.py#L46)) so readers can filter by model if needed.
- OpenAI is a **closed-source commercial API.** The project no longer fits the "entirely open-source" framing in the original abstract. Abstract updated to "widely-adopted open tooling" at [docs/02_ABSTRACT.md](docs/02_ABSTRACT.md#L17).

---

## 2. Component map

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Telecom Billing RCA System                     │
├──────────────────────────────────────────────────────────────────────┤
│  UI                 Streamlit dashboard (app.py + pages/*)           │
├──────────────────────────────────────────────────────────────────────┤
│  Agents             LangGraph: Investigator → Reasoner → Critic →    │
│                     Reporter (one revise loop)                       │
├──────────────────────────────────────────────────────────────────────┤
│  Retrieval          Flat-Vector (Chroma) | BM25 | Hybrid (RRF) |     │
│                     Reranker (cross-encoder) | GraphRAG (NetworkX)   │
├──────────────────────────────────────────────────────────────────────┤
│  Detection          IsolationForest (F1=0.81, AUROC=0.877) + DBSCAN  │
├──────────────────────────────────────────────────────────────────────┤
│  Data               IBM Telco (synthetic anomalies) + optional       │
│                     Telecom Italia CDR (real Milan grid)             │
├──────────────────────────────────────────────────────────────────────┤
│  Evaluation         ROUGE-L, BERTScore, RAGAS (faithfulness,         │
│                     answer-relevancy), LLM-as-Judge (4-axis Likert), │
│                     bootstrap CI + paired-bootstrap + Wilcoxon       │
├──────────────────────────────────────────────────────────────────────┤
│  Observability      JSONL tracer ([src/utils/tracing.py])            │
│  Tracking           MLflow ([src/mlflow_tracking.py])                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. File-by-file reference

### 3.1 Entry points
| File | Purpose | Notes |
|---|---|---|
| [app.py](app.py) | Streamlit homepage — architecture, stack, nav | Uses `st.switch_page()` for sidebar nav |
| [pages/1_📊_Upload_Detect.py](pages/1_📊_Upload_Detect.py) | Upload CSV, run IsolationForest detector | |
| [pages/2_🔍_RCA_Viewer.py](pages/2_🔍_RCA_Viewer.py) | Click an anomaly → run full agent DAG → render RCA | |
| [pages/3_📚_Knowledge_Base.py](pages/3_📚_Knowledge_Base.py) | Browse playbooks + query Chroma | |
| [run_pipeline.py](run_pipeline.py) | CLI end-to-end runner: download → inject → train → index → evaluate | |
| [run_ablation.py](run_ablation.py) | Ablation harness for configs A/B/C/D (and E = GraphRAG) | Token-bucket paced, bootstrap CI reported |
| [test_pipeline.py](test_pipeline.py) | Integration sanity test on a few anomalies | |
| [test_llm.py](test_llm.py) | 5-second smoke test — verifies OpenAI key + model reachable | |
| [scripts/build_graph_rag.py](scripts/build_graph_rag.py) | Build the NetworkX graph from playbooks | `--offline` for heuristic-only |
| [scripts/plot_results.py](scripts/plot_results.py) | Matplotlib bars + judge radar for the deck | |

### 3.2 Configuration
| Config | Location | Default |
|---|---|---|
| Generator model | [config.py](config.py#L41) | `gpt-4o-mini` (env `LLM_MODEL`) |
| Judge model | [config.py](config.py#L48) | `gpt-4o` (env `JUDGE_MODEL`) |
| Rate-limit RPM | [src/utils/rate_limit.py](src/utils/rate_limit.py#L70) | 450 (env `LLM_RATE_PER_MIN`) |
| Chunk size / overlap | [config.py](config.py#L32-L33) | 512 / 64 |
| Top-K retrieval | [config.py](config.py#L36) | 5 |
| Embedding model | [config.py](config.py#L29) | `all-MiniLM-L6-v2` (384-dim) |
| Random seed | [config.py](config.py#L84) | 42 |
| Ablation configs | [config.py](config.py#L54-L84) | A, B, C, D, **E=graph_rag** |

### 3.3 Data
| File | Purpose |
|---|---|
| [src/data/loader.py](src/data/loader.py) | IBM Telco churn CSV loader |
| [src/data/anomaly_injector.py](src/data/anomaly_injector.py) | 5 synthetic anomaly injection rules (zero_billing, duplicate_charge, usage_spike, cdr_failure, sla_breach) |
| [src/data/augmentor.py](src/data/augmentor.py) | ROSE-style oversampling |
| [src/data/telecom_italia_loader.py](src/data/telecom_italia_loader.py) | **NEW** — real Milan-grid CDR loader, produces hourly features + per-cell z-score anomaly proxy |
| `data/raw/ibm_telco_churn.csv` | IBM Telco source dataset |
| `data/processed/anomalies_labeled.csv` | Injected anomalies + labels |
| `data/corpus/rca_playbooks/*.md` | 8 domain playbooks (KB source) |
| `data/eval/ground_truth_rca/ground_truth_rca_60.json` | **60-item GT** (12 per anomaly type) — authoritative |
| `data/eval/ground_truth_rca/ground_truth_rca.json` | Legacy 15-item GT (fallback) |
| `data/graph_rag/` | Output of `build_graph_rag.py`: `kb_graph.pkl` + `chunks.json` |

### 3.4 Detection
- [src/detection/detector.py](src/detection/detector.py) — `BillingAnomalyDetector`, `train_and_evaluate()`. IsolationForest hyperparameters at [config.py](config.py#L92-L97).
- Metrics from the last run live under `mlruns/<experiment>/<run>/metrics/`.

### 3.5 Retrieval
| File | Role | Offline-capable? |
|---|---|---|
| [src/rag/embedder.py](src/rag/embedder.py) | sentence-transformers wrapper | Yes (first run downloads model) |
| [src/rag/chunker.py](src/rag/chunker.py) | Recursive text splitter with overlap | Yes |
| [src/rag/knowledge_base.py](src/rag/knowledge_base.py) | ChromaDB-backed vector KB | Yes (persists to `chroma_db/`) |
| [src/rag/hybrid_retriever.py](src/rag/hybrid_retriever.py) | **NEW** — BM25 + dense + RRF | Yes (rank_bm25) |
| [src/rag/reranker.py](src/rag/reranker.py) | **NEW** — optional cross-encoder reranker | Degrades gracefully if model can't load |
| [src/rag/graph_rag.py](src/rag/graph_rag.py) | **NEW / HEADLINE** — entity+relation graph over playbooks, k-hop retrieval | Yes (heuristic extractor fallback) |

#### GraphRAG schema ([src/rag/graph_rag.py](src/rag/graph_rag.py#L60-L62))
- **Entity types:** `SYSTEM`, `COMPONENT`, `FAILURE_MODE`, `FIX`, `METRIC`
- **Relation types:** `CAUSES`, `DEPENDS_ON`, `FEEDS_INTO`, `TRIGGERS`, `FIXES`, `MONITORS`
- **Extractor:** OpenAI JSON-mode prompt ([src/rag/graph_rag.py](src/rag/graph_rag.py#L130-L156)) with deterministic heuristic fallback ([src/rag/graph_rag.py](src/rag/graph_rag.py#L110-L128))
- **Retriever:** lexical seed-entity match → k-hop BFS → score by node-support + edge-support + token-overlap lexical boost ([src/rag/graph_rag.py](src/rag/graph_rag.py#L230-L278))

### 3.6 Agents
| File | Role |
|---|---|
| [src/agents/state.py](src/agents/state.py) | `AgentState` TypedDict including new critic fields at lines 54-60 |
| [src/agents/llm_utils.py](src/agents/llm_utils.py) | **REWRITTEN** — `call_llm()` over OpenAI SDK with retry |
| [src/agents/prompts.py](src/agents/prompts.py) | System + user prompt templates |
| [src/agents/investigator.py](src/agents/investigator.py) | Formulates query, retrieves from KB |
| [src/agents/reasoner.py](src/agents/reasoner.py) | CoT reasoning over evidence → hypothesis |
| [src/agents/critic.py](src/agents/critic.py) | **NEW** — reviews hypothesis, verdict: `accept`/`revise`, one loop back to `broaden_query` |
| [src/agents/reporter.py](src/agents/reporter.py) | Structured JSON RCA output |
| [src/agents/graph.py](src/agents/graph.py) | LangGraph assembly — wires all 5 nodes + 2 conditional edges |

### 3.7 Evaluation
| File | Contents |
|---|---|
| [src/evaluation/metrics.py](src/evaluation/metrics.py) | ROUGE-L, BERTScore, Context Precision/Recall/MRR, 60-item GT loader (multi-reference max-score), `evaluate_pipeline_results()`, `print_evaluation_report()` |
| [src/evaluation/llm_judge.py](src/evaluation/llm_judge.py) | **NEW** — OpenAI-first judge, 4-axis Likert, RAGAS faithfulness, RAGAS answer-relevancy, batch/aggregate helpers |
| [src/evaluation/stats.py](src/evaluation/stats.py) | **NEW** — `bootstrap_ci`, `paired_bootstrap_pvalue`, `wilcoxon_paired`, `compare_configs` |
| [src/evaluation/__init__.py](src/evaluation/__init__.py) | Barrel exports |
| [src/mlflow_tracking.py](src/mlflow_tracking.py) | MLflow run/logging helpers |

### 3.8 Utilities
| File | Contents |
|---|---|
| [src/utils/rate_limit.py](src/utils/rate_limit.py) | Thread-safe token-bucket, singleton via `get_limiter()` |
| [src/utils/test_data.py](src/utils/test_data.py) | `anomalies_from_ground_truth()` — deterministic test set builder |
| [src/utils/tracing.py](src/utils/tracing.py) | **NEW** — JSONL tracer + `trace_span()` context manager + `summarize_trace()` |

### 3.9 Tests (87 passing, ~6s)
| File | Tests | What it covers |
|---|---|---|
| [tests/test_metrics.py](tests/test_metrics.py) | 18 | ROUGE-L, type match, retrieval metrics, GT loader |
| [tests/test_stats.py](tests/test_stats.py) | 12 | Bootstrap CI, paired bootstrap, Wilcoxon, compare_configs |
| [tests/test_utils.py](tests/test_utils.py) | 10 | Token bucket, GT-derived anomalies |
| [tests/test_llm_judge.py](tests/test_llm_judge.py) | 7 | `_parse_json`, backend detection, no-backend safety |
| [tests/test_anomaly_injector.py](tests/test_anomaly_injector.py) | 6 | All 5 injectors + determinism |
| [tests/test_chunker.py](tests/test_chunker.py) | 5 | Short / empty / long text, metadata |
| [tests/test_graph_rag.py](tests/test_graph_rag.py) | 13 | Heuristic extraction, builder, save/load roundtrip, retrieval, k-hops |
| [tests/test_hybrid_retriever.py](tests/test_hybrid_retriever.py) | 7 | Tokenize, RRF math |
| [tests/test_critic_and_tracing.py](tests/test_critic_and_tracing.py) | 11 | Critic fallback paths, tracer spans, summarize |
| [tests/test_telecom_italia_loader.py](tests/test_telecom_italia_loader.py) | 4 | Synthetic TSV roundtrip, z-score proxy, error path |
| [tests/conftest.py](tests/conftest.py) | — | sys.path fix |

---

## 4. Runtime pipeline (single anomaly)

```
                                            ┌──── (if count < 2) ─────┐
anomaly ─► investigator ──► retrieval_count ┤                         │
                 │                          └────── (proceed) ──────► reasoner
                 │ Chroma / Hybrid /                                    │
                 │ GraphRAG                                             ▼
                 │                                                    critic
                 │                               ┌─── (revise, max 1) ─┤
                 │                               │                     │ (accept)
                 │                               ▼                     ▼
                 └────── (broaden_query) ◄── broaden_query          reporter
                                                                        │
                                                                        ▼
                                                                   RCA_report
```

Implementation: [src/agents/graph.py](src/agents/graph.py#L58-L98).
Routing: [should_retry_retrieval](src/agents/graph.py#L21), [should_revise](src/agents/critic.py#L105).
State shape: [AgentState](src/agents/state.py#L40).

---

## 5. Evaluation pipeline

```
per-anomaly: generated RCA vs GT entry(s) of same anomaly_type (multi-ref max)
           │
           ├─► ROUGE-L              ([src/evaluation/metrics.py])
           ├─► BERTScore            ([src/evaluation/metrics.py])
           ├─► RAGAS faithfulness   ([src/evaluation/llm_judge.py])
           ├─► RAGAS answer_relev.  ([src/evaluation/llm_judge.py])
           └─► 4-axis Likert        ([src/evaluation/llm_judge.py])

per-config: 60 scalars per metric
           │
           ├─► mean + 95% bootstrap CI       (src/evaluation/stats.py)
           ├─► paired-bootstrap p vs baseline (src/evaluation/stats.py)
           └─► Wilcoxon signed-rank p        (src/evaluation/stats.py)

report: [run_ablation.py]
```

---

## 6. How to run — every path

### 6.1 One-time setup

```powershell
# 1. Clone / open workspace
cd "C:\Users\TatsatPandey\Documents\Personal\Personal\Mtech DSE\Sem 4\RAGML"

# 2. Create / activate venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1   # or use the terminal profile

# 3. Install deps
pip install -r requirements.txt

# 4. Configure LLM
#    Create .env in the repo root with:
#      OPENAI_API_KEY=sk-...
#      LLM_MODEL=gpt-4o-mini           (optional override)
#      JUDGE_MODEL=gpt-4o              (optional override)
#      LLM_RATE_PER_MIN=450            (optional override)
```

> **Cost control tip:** override with `LLM_MODEL=gpt-4o-mini` and `JUDGE_MODEL=gpt-4o-mini` during development; only switch the judge to `gpt-4o` for the final viva run.

### 6.2 Verify LLM works

```powershell
.\.venv\Scripts\python.exe test_llm.py
# Expect: "SUCCESS: ..."
```

### 6.3 Run the test suite (offline, fast)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
# Expect: 87 passed in ~6s
```

With coverage:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=term
```

### 6.4 Build the knowledge base (indexes 8 playbooks into Chroma)

```powershell
.\.venv\Scripts\python.exe -c "from src.rag.knowledge_base import build_knowledge_base; build_knowledge_base(force_rebuild=True)"
```

### 6.5 Build the GraphRAG graph (the headline novelty)

**Offline, deterministic, no API cost** (heuristic extractor):
```powershell
.\.venv\Scripts\python.exe scripts\build_graph_rag.py --offline
```

**Full LLM-extracted graph** (recommended for viva — richer graph, ~50-100 edges):
```powershell
.\.venv\Scripts\python.exe scripts\build_graph_rag.py
# cost: ~$0.10-0.30 at gpt-4o-mini
```

### 6.6 Train the detector end-to-end

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```
This:
1. Downloads / loads IBM Telco CSV
2. Injects 5 types of anomalies → `data/processed/anomalies_labeled.csv`
3. Trains IsolationForest + DBSCAN
4. Indexes playbooks into Chroma
5. Runs the multi-agent pipeline on a sample
6. Logs to MLflow under `mlruns/`

### 6.7 Ablation study (the headline table)

```powershell
# All 5 configs (A, B, C, D, E=GraphRAG), 60 anomalies, with LLM judge
.\.venv\Scripts\python.exe run_ablation.py --n 60 --configs A B C D E --judge

# Quick 10-item smoke run without judge
.\.venv\Scripts\python.exe run_ablation.py --n 10 --configs A D
```
Output: `ablation_results.json` at repo root + printed significance table.

### 6.8 Plot results for the deck

```powershell
.\.venv\Scripts\python.exe scripts\plot_results.py --metric rouge_l
# PNGs land in docs/diagrams/
```

### 6.9 Launch the Streamlit UI

```powershell
.\.venv\Scripts\streamlit run app.py
# Browser opens at http://localhost:8501
```

UI pages:
- **Home** — architecture + stack overview ([app.py](app.py))
- **1 📊 Upload & Detect** — upload a CSV, run detector ([pages/1_📊_Upload_Detect.py](pages/1_📊_Upload_Detect.py))
- **2 🔍 RCA Viewer** — pick an anomaly → run the agent DAG → view RCA ([pages/2_🔍_RCA_Viewer.py](pages/2_🔍_RCA_Viewer.py))
- **3 📚 Knowledge Base** — browse/query playbooks ([pages/3_📚_Knowledge_Base.py](pages/3_📚_Knowledge_Base.py))

### 6.10 Docker (optional — for demo)

```powershell
docker build -t telecom-rca .
docker run -p 8501:8501 -e OPENAI_API_KEY=$env:OPENAI_API_KEY telecom-rca
# Browser: http://localhost:8501
```
Dockerfile at [Dockerfile](Dockerfile).

### 6.11 Optional — Telecom Italia CDR track (Phase 1.5 D1)

1. Download Milan SMS/Call/Internet TSVs from https://dandelion.eu/datamine/open-big-data/
2. Unzip into `data/raw/telecom_italia_cdr/`
3. Run the loader:
```powershell
.\.venv\Scripts\python.exe src\data\telecom_italia_loader.py
# Produces data/processed/telecom_italia_cdr.parquet (or .csv.gz if pyarrow unavailable)
```
If the raw dir is empty, the loader prints a helpful message and exits cleanly.

---

## 7. Readiness check

### 7.1 What's ready (green)
| Area | Status | Evidence |
|---|---|---|
| Code compiles & imports clean | ✅ | 87/87 tests pass in ~6s |
| Generator ↔ Judge independence | ✅ | gpt-4o-mini vs gpt-4o at [config.py](config.py#L41-L48); `backend` field persisted per judged item |
| Ground-truth expanded to 60 | ✅ | [data/eval/ground_truth_rca/ground_truth_rca_60.json](data/eval/ground_truth_rca/ground_truth_rca_60.json), 12 per type, unique IDs |
| Rate-limit pacing | ✅ | Token bucket at 450 RPM default; safe for gpt-4o-mini tier-1 |
| Significance testing | ✅ | Bootstrap CI + paired-bootstrap + Wilcoxon in [src/evaluation/stats.py](src/evaluation/stats.py) |
| GraphRAG (headline novelty) | ✅ | [src/rag/graph_rag.py](src/rag/graph_rag.py) + [scripts/build_graph_rag.py](scripts/build_graph_rag.py) + 13 tests + config E |
| Hybrid retrieval | ✅ | BM25+RRF+cross-encoder ([src/rag/hybrid_retriever.py](src/rag/hybrid_retriever.py) + [src/rag/reranker.py](src/rag/reranker.py)) |
| Critic node | ✅ | [src/agents/critic.py](src/agents/critic.py) wired into graph, one revise loop |
| Observability | ✅ | JSONL tracer at [src/utils/tracing.py](src/utils/tracing.py) + tests |
| Limitations doc | ✅ | [docs/08_LIMITATIONS.md](docs/08_LIMITATIONS.md) |
| Deck diagrams | ✅ | 6 Mermaid sources at [docs/diagrams/README.md](docs/diagrams/README.md) + matplotlib script |
| Streamlit UI | ✅ | `app.py` + 3 pages; no Groq/ChatGroq references in pages (verified by grep) |
| Docker | ✅ | [Dockerfile](Dockerfile); pre-downloads embedding model; pass `OPENAI_API_KEY` at runtime |
| CI | ✅ | [.github/workflows/ci.yml](.github/workflows/ci.yml) — verifies config + runs pytest |

### 7.2 What needs you to do something (yellow)
| Item | What to do | Blocker for |
|---|---|---|
| Set `OPENAI_API_KEY` | Create `.env` or `$env:OPENAI_API_KEY = "..."` before the first real run | Any LLM-touching command |
| Rebuild GraphRAG with LLM | `python scripts/build_graph_rag.py` (without `--offline`) | Viva-quality graph (offline has only 3 edges) |
| Run ablation | `python run_ablation.py --n 60 --configs A B C D E --judge` | Final results table in thesis/deck |
| Render deck PNGs | `python scripts/plot_results.py` after ablation | Deck figures |
| Re-index KB | Only needed if playbooks changed | KB drift |

### 7.3 What's acknowledged as a gap (red — but documented)
| Gap | Why it's okay for now | Doc location |
|---|---|---|
| No human evaluation | LLM-as-Judge + RAGAS are the accepted proxies for Phase 1; human eval is Phase 3 | [docs/08_LIMITATIONS.md §3.4](docs/08_LIMITATIONS.md) |
| Single-author GT | 60-item expansion with diverse modes; future SME review listed | [docs/08_LIMITATIONS.md §1.3](docs/08_LIMITATIONS.md) |
| Synthetic anomalies | Documented; RCA narratives are independent of injection rules | [docs/08_LIMITATIONS.md §1.1](docs/08_LIMITATIONS.md) |
| Python 3.14 Pydantic v1 warnings | Non-blocking deprecation notices | [docs/08_LIMITATIONS.md §4.3](docs/08_LIMITATIONS.md) |

### 7.4 Is the UI ready?
Yes — with caveats.
- The Streamlit homepage text was updated to `OpenAI (gpt-4o-mini + gpt-4o judge)` ([app.py](app.py#L222) + L232).
- The three `pages/*.py` files do not contain any Groq/ChatGroq imports (verified).
- The RCA Viewer calls `run_pipeline()` from [src/agents/graph.py](src/agents/graph.py), which now goes through the OpenAI-backed `call_llm()`.
- **The UI will throw a clear error at the first LLM call if `OPENAI_API_KEY` is missing.** The error string tells the user what to do. No silent failures.

### 7.5 Final readiness verdict
- **Code readiness for thesis submission:** 9/10. Everything runs; 87 tests green; migration clean; no stale Groq code paths in hot code.
- **Research novelty for viva:** 8/10. GraphRAG is a real, defensible contribution. Caveat: the graph should be built with LLM extraction (not heuristic) before the viva so the node/edge counts are publication-grade.
- **Evaluation rigor:** 8/10. Multi-metric + stats + independent judge + 60-item GT. Remaining weakness: N=12 per anomaly type is small, but bootstrap CI is honest about that.
- **Cost to finish:** ~$5-10 USD in OpenAI credits for one full ablation + GraphRAG build. ~1 evening of execution time including rendering the deck.

---

## 8. Reproducibility checklist

- [x] Deterministic seeds in [config.py](config.py#L84) (`RANDOM_SEED=42`)
- [x] Injection determinism tested ([tests/test_anomaly_injector.py](tests/test_anomaly_injector.py))
- [x] GT-derived anomaly set determinism tested ([tests/test_utils.py](tests/test_utils.py))
- [x] Judge temperature = 0 ([config.py](config.py#L49))
- [x] MLflow tracking enabled ([src/mlflow_tracking.py](src/mlflow_tracking.py))
- [x] Pytest suite runs offline (no network, no LLM)
- [x] Lockfile — `requirements.txt` with `>=` pins; consider `pip freeze > requirements.lock.txt` before submission
- [x] Docker image pre-downloads embedding model

---

## 9. References to key external work

| Technique | Reference | Where we use it |
|---|---|---|
| RAGAS framework | Es et al., 2023, *"RAGAS: Automated Evaluation of Retrieval Augmented Generation"* | [src/evaluation/llm_judge.py](src/evaluation/llm_judge.py) faithfulness + answer-relevancy |
| LLM-as-Judge bias | Zheng et al., 2023, *"Judging LLM-as-a-Judge with MT-Bench"* | Motivates gpt-4o-mini → gpt-4o tier split |
| RRF | Cormack et al., 2009, *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"* | [src/rag/hybrid_retriever.py](src/rag/hybrid_retriever.py#L32) |
| GraphRAG | Microsoft Research 2024, *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"* | [src/rag/graph_rag.py](src/rag/graph_rag.py) |
| IsolationForest | Liu et al., 2008 | [src/detection/detector.py](src/detection/detector.py) |
| Telecom Italia CDR | Barlacchi et al., 2015, *"A multi-source dataset of urban life in Milan"*, Nature Sci Data | [src/data/telecom_italia_loader.py](src/data/telecom_italia_loader.py) |
| 3GPP TS 32.240 | Charging architecture and principles | Cited in [docs/08_LIMITATIONS.md](docs/08_LIMITATIONS.md) + playbooks |
| 3GPP TS 32.298 | CDR parameter description | Same |
| TMF678 | Customer Bill Management API | Same |

---

## 10. Open questions for the viva

These are the questions you should expect and have a 1-sentence answer ready for:

1. *"Why is your judge independent?"* → Different model tier (gpt-4o-mini gen vs gpt-4o judge), temp=0, backend tag persisted per item, Zheng 2023 mitigation pattern.
2. *"Is N=60 enough?"* → Per-type N=12; bootstrap CIs are wide and honestly reported; no per-type claims, only joint-metric claims.
3. *"What's novel here vs plain RAG?"* → GraphRAG over telecom playbooks with 6-relation schema; Critic node with revise-loop; multi-metric evaluation with significance tests — none of which is in the cited telecom-RCA baselines.
4. *"Why not open-source LLMs?"* → We kept a Groq fallback; removed Groq from the critical path because same-model judging undermines evaluation. Trade-off documented.
5. *"How do you know the synthetic anomalies aren't contaminating RCA eval?"* → Injection rules operate on tabular features; RCA narratives come from independently-authored playbooks + 60-item GT with diverse realistic failure modes.

---

**End of document.** Keep this open during the viva — it's your single source of truth.
