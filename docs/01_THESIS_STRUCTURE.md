# Thesis Structure

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

**Program:** MTech — Data Science & Engineering  
**Domain:** Telecom / Billing / Network Operations  
**Technology:** Agentic AI | RAG | LangGraph | MLflow | Streamlit  

---

## Chapter 1: Introduction (8–10 pages)

### 1.1 Background
- Overview of telecom billing infrastructure and CDR processing pipelines
- Scale of operations: AT&T processes ~400M CDR records/day
- Role of billing accuracy in revenue assurance and regulatory compliance

### 1.2 Problem Statement
- Billing anomalies: zero-billing, duplicate charges, usage spikes, CDR processing failures, SLA breaches
- The **detection-to-resolution gap** — anomalies are detected by rule-based/ML systems, but root cause analysis remains manual
- Manual RCA takes 2–4 hours per incident; requires senior engineering expertise
- MTTR (Mean-Time-To-Resolution) delays cause revenue leakage and operational overhead

### 1.3 Motivation
- At 0.01% anomaly rate on 400M CDRs = 40,000 anomalies/day requiring triage
- Over-reliance on senior engineers for initial RCA is unsustainable at scale
- LLMs + RAG offer a path to automate knowledge retrieval and reasoning

### 1.4 Research Objectives
1. Design a domain-specific multi-agent RAG architecture for telecom billing anomaly diagnosis
2. Curate an open-source telecom billing knowledge corpus for RAG benchmarking
3. Empirically compare single-agent vs. multi-agent architectures on diagnostic reasoning quality
4. Build an end-to-end deployable prototype with experiment tracking and UI

### 1.5 Research Questions
- **RQ1:** Can a multi-agent RAG system produce diagnostically accurate RCA for billing anomalies?
- **RQ2:** Does separating retrieval, reasoning, and reporting into distinct agents improve RCA quality over a monolithic approach?
- **RQ3:** What is the achievable MTTR reduction compared to manual triage baselines?

### 1.6 Scope and Limitations
- Batch anomaly triage (not real-time CDR streaming)
- Configurable OpenAI-compatible LLM backend: Groq preferred, Kimi fallback, and custom provider support; the dissertation reports the exact provider used for each experimental run
- Evaluation on synthetic + public datasets (no proprietary CDR data)

### 1.7 Thesis Organization
- Chapter-by-chapter roadmap

---

## Chapter 2: Literature Review (12–15 pages)

### 2.1 Anomaly Detection in Telecom
- Traditional rule-based systems and their limitations
- Statistical methods (z-score, Grubbs' test)
- ML approaches: Isolation Forest (Liu et al., 2008), DBSCAN (Ester et al., 1996), Autoencoders
- Telecom-specific: fraud detection, revenue assurance, billing validation

### 2.2 Retrieval-Augmented Generation (RAG)
- Original RAG architecture (Lewis et al., 2020)
- Evolution: naive RAG → advanced RAG → modular RAG
- Embedding models: BERT (Devlin et al., 2019), sentence-transformers
- Vector databases: ChromaDB, Milvus, Pinecone, FAISS
- Chunking strategies: fixed-size, semantic, recursive

### 2.3 LLM-Based Agents
- ReAct framework (Yao et al., 2022) — reasoning + acting
- Multi-agent systems survey (Dorri et al., 2018)
- Agent orchestration frameworks: LangGraph, AutoGen, CrewAI
- Tool-augmented LLMs
- Critique of autonomous agents: hallucination, reliability, determinism

### 2.4 AIOps and Automated Root Cause Analysis
- Ericsson AI-assisted NOC (2022)
- Nokia AVA network management platform
- IBM Watson AIOps
- Microsoft AIOps research
- RAG for enterprise knowledge management (IBM Research, 2023)

### 2.5 Evaluation of Generative AI Systems
- RAGAS framework (Es et al., 2023)
- BERTScore (Zhang et al., 2020)
- LLM-as-Judge (Zheng et al., 2023)
- ROUGE metrics for text evaluation

### 2.6 Research Gap
- No prior open-source system applies multi-agent RAG to telecom billing anomaly RCA
- No public benchmark dataset for telecom billing anomaly diagnosis
- Limited empirical comparison of single vs. multi-agent RAG for domain-specific reasoning

---

## Chapter 3: Methodology (15–18 pages)

### 3.1 Research Design
- Applied research with experimental evaluation
- Design Science Research methodology

### 3.2 System Architecture
- Five-layer modular architecture overview
- Layer 1 — Ingestion: Data loading, schema validation
- Layer 2 — Detection: Anomaly detection with confidence scoring
- Layer 3 — RAG Engine: Document parsing, embedding, vector storage, retrieval
- Layer 4 — Agent Layer: LangGraph multi-agent workflow
- Layer 5 — Output: Structured RCA, dashboard, experiment logging

### 3.3 Data Collection and Preparation
- Primary datasets: IBM Telco Customer Churn, Maven Analytics Telecom Churn
- Synthetic anomaly injection methodology (5 anomaly types)
- Reproducibility: seed-controlled, DVC-tracked injection scripts
- Document corpus curation: source selection, section extraction, chunking strategy

### 3.4 Anomaly Detection Module
- Feature engineering from billing data
- IsolationForest configuration and hyperparameter tuning
- DBSCAN as alternative detector
- Threshold selection and confidence score calibration

### 3.5 RAG Knowledge Base Construction
- Document collection and preprocessing
- Chunking strategy: recursive text splitting with overlap
- Embedding model: `all-MiniLM-L6-v2` (384-dim, CPU-compatible)
- ChromaDB indexing with metadata
- Retrieval strategy: top-k semantic search with re-ranking

### 3.6 Multi-Agent Design
- **Investigator Agent:** Receives anomaly context → queries RAG/GraphRAG store → retrieves top-k relevant documents
- **Reasoner Agent:** Receives anomaly context + retrieved docs → generates structured root cause hypothesis
- **Critic Agent:** Reviews the hypothesis for grounding, evidence use, hallucination risk, and optionally requests one revision
- **Reporter Agent:** Receives final hypothesis + evidence → produces JSON-schema-validated RCA report
- LangGraph StateGraph definition: nodes, edges, state schema, conditional routing, and bounded critic revision loop

### 3.7 GraphRAG Layer
- Entity and relation extraction from telecom RCA playbooks
- NetworkX graph construction and persistence
- Multi-hop graph traversal for complex RCA queries
- Comparison against flat vector retrieval in Config E

### 3.8 Prompt Engineering
- System prompts for each agent
- Few-shot examples
- Output format constraints (JSON schema)
- Grounding instructions to prevent hallucination

### 3.9 Experiment Tracking
- MLflow integration: parameters, metrics, artifacts logging
- Run comparison and reproducibility

---

## Chapter 4: Implementation (10–12 pages)

### 4.1 Development Environment
- Hardware specifications
- Software stack and versions
- Virtual environment and dependency management

### 4.2 Data Pipeline Implementation
- Data loading and preprocessing scripts
- Anomaly injection implementation (code walkthrough)
- EDA and visualization notebooks

### 4.3 Anomaly Detector Implementation
- Training pipeline code
- Hyperparameter tuning experiments
- Model serialization and integration

### 4.4 RAG Pipeline Implementation
- Document parsing with PyMuPDF
- Chunking and embedding pipeline
- ChromaDB collection setup
- Retrieval query interface

### 4.5 Agent Pipeline Implementation
- LangGraph StateGraph code walkthrough
- Individual agent node implementations
- State schema definition
- Error handling and fallback mechanisms

### 4.6 Streamlit Dashboard
- UI components and layout
- User workflow: upload → detect → investigate → report
- Integration with backend pipeline

### 4.7 MLflow Integration
- Experiment setup
- Metric and artifact logging
- Run comparison dashboard

---

## Chapter 5: Results and Evaluation (12–15 pages)

### 5.1 Anomaly Detection Results
- Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix analysis
- Comparison: IsolationForest vs. DBSCAN
- Per-anomaly-type performance breakdown

### 5.2 RAG Retrieval Quality
- Context Recall, Context Precision (RAGAS)
- MRR@5 results
- Qualitative examples of retrieval hits and misses

### 5.3 RCA Quality Assessment
- BERTScore F1 results
- ROUGE-L scores
- Anomaly Type Match accuracy
- Resolution Match (LLM-as-judge)
- Simulated MTTR reduction

### 5.4 Ablation Study
- Configuration A: No RAG baseline
- Configuration B: RAG-only single LLM call
- Configuration C: Single Agent + RAG
- Configuration D: Multi-Agent + RAG (proposed system)
- Configuration E: Multi-Agent + GraphRAG (headline novelty)
- Statistical significance testing (Wilcoxon signed-rank test, paired bootstrap, bootstrap confidence intervals)
- Results table with confidence intervals

### 5.5 Qualitative Analysis
- Case studies: 3–5 representative anomaly investigations
- Side-by-side comparison: system output vs. manual RCA
- Failure analysis: where/why the system produces incorrect diagnoses

---

## Chapter 6: Discussion (6–8 pages)

### 6.1 Interpretation of Results
- How results answer each research question
- Comparison with related work

### 6.2 Architectural Insights
- Why multi-agent outperforms single-agent (separation of concerns)
- RAG retrieval quality as the bottleneck
- Prompt engineering impact on output quality

### 6.3 Practical Implications
- Deployment readiness assessment
- Integration with existing telecom NOC workflows
- Cost-benefit analysis for production adoption

### 6.4 Threats to Validity
- Internal: synthetic data limitations, hyperparameter sensitivity
- External: generalizability to other telecom operators
- Construct: LLM-as-judge evaluation bias

---

## Chapter 7: Conclusion and Future Work (4–5 pages)

### 7.1 Summary of Contributions
1. Novel multi-agent RAG architecture for telecom billing RCA
2. Curated open-source telecom billing knowledge corpus
3. Empirical evidence: multi-agent > single-agent for domain-specific RCA
4. Reproducible end-to-end deployable system with configurable LLM provider support

### 7.2 Answers to Research Questions
- RQ1, RQ2, RQ3 answered with evidence

### 7.3 Future Work
- Real-time CDR stream processing (Kafka integration)
- Distributed vector store (Milvus) for production scale
- Batched LLM inference (vLLM) for throughput
- Operator-specific fine-tuning with proprietary knowledge bases
- Human-in-the-loop feedback for continuous improvement
- Multi-modal RCA (incorporating network topology graphs)

---

## Appendices

### Appendix A: Anomaly Injection Scripts (Code)
### Appendix B: Prompt Templates (Full Text)
### Appendix C: Corpus Manifest (Document List)
### Appendix D: Full Evaluation Results Tables
### Appendix E: Streamlit UI Screenshots
### Appendix F: MLflow Experiment Logs

---

## Estimated Page Count

| Chapter | Pages |
|---------|-------|
| Introduction | 8–10 |
| Literature Review | 12–15 |
| Methodology | 15–18 |
| Implementation | 10–12 |
| Results & Evaluation | 12–15 |
| Discussion | 6–8 |
| Conclusion & Future Work | 4–5 |
| References | 3–5 |
| Appendices | 10–15 |
| **Total** | **80–103** |
