# BITS WILP Mid-Sem Outline Report

<!--
  FORMAT INSTRUCTIONS FOR WORD CONVERSION:
  - This follows the exact format of the BITS sample "Project Abstract & Outline Report"
  - The sample is the "Rugged Laptop" document (Appendix format)
  - Font: Times New Roman, 12pt throughout
  - Double spaced
  - 1 inch margins on all four sides
  - Page numbers at bottom right (serial)
  - Title pages (pages 1-2): centered, bold titles, BITS logo placeholder
  - Abstract page (page 3): ABSTRACT heading, paragraph text, signature lines at bottom
  - Contents page (page 4): section numbers, titles, page numbers, figure/table lists
  - Sections start from page 5 onwards
  - All figures and tables must have a number and title
  - The document size must be ≤ 10 MB
  - PDF must be text-searchable (no full-page scans except signature pages)
-->

---

<!-- PAGE 1: FIRST TITLE PAGE -->

<div align="center">

# A MULTI-AGENT GRAPHRAG SYSTEM FOR AUTONOMOUS ROOT CAUSE ANALYSIS OF TELECOM BILLING ANOMALIES

**BITS ZG628T: Dissertation**

by

**(Tatsat Pandey)**  
**(Insert BITS ID Number)**

Dissertation work carried out at

**(Insert Organization Name, Location Name)**

Submitted in partial fulfilment of **MTech – Data Science & Engineering**  
degree programme

Under the Supervision of

**(Insert Supervisor Name)**  
**(Insert Organization Name, Location Name)**

<!-- BITS LOGO PLACEHOLDER: Insert BITS Pilani logo image here -->
[BITS Pilani Logo]

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE**  
**PILANI (RAJASTHAN)**

**(April 2026)**

</div>

---

<!-- PAGE 2: SECOND TITLE PAGE (identical to page 1 per the sample) -->

<div align="center">

# A MULTI-AGENT GRAPHRAG SYSTEM FOR AUTONOMOUS ROOT CAUSE ANALYSIS OF TELECOM BILLING ANOMALIES

**BITS ZG628T: Dissertation**

by

**(Tatsat Pandey)**  
**(Insert BITS ID Number)**

Dissertation work carried out at

**(Insert Organization Name, Location Name)**

Submitted in partial fulfilment of **MTech – Data Science & Engineering**  
degree programme

Under the Supervision of

**(Insert Supervisor Name)**  
**(Insert Organization Name, Location Name)**

<!-- BITS LOGO PLACEHOLDER -->
[BITS Pilani Logo]

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE**  
**PILANI (RAJASTHAN)**

**(April 2026)**

</div>

---

<!-- PAGE 3: ABSTRACT -->

## ABSTRACT

This project designs and evaluates a multi-agent Retrieval-Augmented Generation (RAG) system for autonomous root cause analysis of billing anomalies in telecom networks. Billing anomalies such as zero billing, duplicate charges, usage spikes, Call Detail Record (CDR) processing failures, and Service Level Agreement (SLA) breaches are detected by rule-based or machine learning systems, but diagnosing their root cause remains manual and time-intensive. The proposed system comprises four LLM-powered agents orchestrated via LangGraph: Investigator for retrieval, Reasoner for hypothesis generation, Critic for factual review, and Reporter for structured output. The system uses ChromaDB for vector retrieval, a GraphRAG layer for entity-relation graph traversal over telecom playbooks, scikit-learn for anomaly detection, MLflow for experiment tracking, and Streamlit for the user interface. A five-configuration ablation study evaluates no-RAG, RAG-only, single-agent RAG, multi-agent RAG, and multi-agent GraphRAG variants using ROUGE-L, BERTScore, RAGAS-style faithfulness, LLM-as-Judge scoring, bootstrap confidence intervals, and Wilcoxon significance tests. Key contributions include the multi-agent RAG architecture, GraphRAG retrieval layer, curated telecom knowledge corpus, and reproducible evaluation framework.

<br><br>

______________________________&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;______________________________  
**Signature of the Student**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Signature of the Supervisor**

**Name:** Tatsat Pandey&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Name:** __________  
**Date:** __________&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Date:** __________  
**Place:** __________&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Place:** __________

---

<!-- PAGE 4: CONTENTS -->

## Contents

1. SYSTEM OVERVIEW AND MODULES .......................................................................... 5  
2. SYSTEM ARCHITECTURE / FUNCTIONAL BLOCK DIAGRAM .................................... 8  
3. TECHNICAL SPECIFICATIONS .................................................................................... 9  
4. DESIGN CONSIDERATIONS ....................................................................................... 10  
5. FUTURE PLAN ............................................................................................................ 11  

**Figure 1:** SYSTEM MODULES AND COMPONENTS ........................................................ 5  
**Figure 2:** MULTI-AGENT PIPELINE FLOW DIAGRAM ..................................................... 8  

**Table 1:** ANOMALY TYPES AND DESCRIPTIONS ........................................................... 6  
**Table 2:** AGENT ROLES AND RESPONSIBILITIES .......................................................... 7  
**Table 3:** TECHNICAL SPECIFICATIONS ......................................................................... 9  
**Table 4:** ABLATION CONFIGURATIONS ......................................................................... 10  
**Table 5:** PROJECT TIMELINE AND STATUS ................................................................... 11  

---

<!-- PAGE 5 ONWARDS: MAIN CONTENT -->

## 1. SYSTEM OVERVIEW AND MODULES

The system is a multi-agent RAG pipeline for autonomous root cause analysis (RCA) of telecom billing anomalies. The main modules are: Data Pipeline, Anomaly Detection, Knowledge Base and Retrieval, Agent Orchestration, Evaluation Framework, and User Interface. All modules are represented in Figure 1 below.

**SYSTEM / SUB SYSTEM DESCRIPTION:** The system consists of the following major components:

(a) Data Pipeline  
(b) Anomaly Detection Module  
(c) RAG Knowledge Base (ChromaDB)  
(d) GraphRAG Retrieval Layer  
(e) Multi-Agent Orchestration (LangGraph)  
(f) Evaluation Framework  
(g) Streamlit User Interface  
(h) MLflow Experiment Tracking  

The description of each of the major components is explained below:

### (a) Data Pipeline

The data pipeline loads public telecom-style datasets (IBM Telco Churn, Maven Telecom Churn) and applies controlled synthetic anomaly injection. Five anomaly types are generated: zero billing, duplicate charges, usage spikes, CDR processing failures, and SLA breaches. Each anomaly is labeled with ground-truth type and severity for evaluation. The processed dataset is stored as `data/processed/anomalies_labeled.csv`.

### (b) Anomaly Detection Module

IsolationForest is the primary unsupervised anomaly detector, chosen for its effectiveness on high-dimensional tabular data without requiring labeled training examples. DBSCAN provides a density-based comparison baseline. Both models are persisted using joblib (`models/isolation_forest_model.joblib`, `models/dbscan_model.joblib`). Detection metrics include Precision, Recall, F1, and ROC-AUC.

### (c) RAG Knowledge Base (ChromaDB)

A curated corpus of telecom RCA playbooks covering zero billing, duplicate charges, usage spikes, CDR failures, and SLA breaches is chunked and embedded using `all-MiniLM-L6-v2` (384-dimensional sentence-transformers). Embeddings are indexed in ChromaDB for semantic similarity retrieval. At runtime, the Investigator agent generates a search query from the anomaly context and retrieves the top-K most relevant chunks.

### (d) GraphRAG Retrieval Layer

The GraphRAG module extracts entities and relations from telecom playbooks using LLM-assisted extraction and stores them in a NetworkX knowledge graph (`data/graph_rag/kb_graph.pkl`). This enables multi-hop retrieval: for example, connecting an anomaly type to a billing system component, then to an SLA clause, then to a remediation playbook. GraphRAG retrieval is activated through the `USE_GRAPH_RAG=1` environment variable.

### (e) Multi-Agent Orchestration (LangGraph)

Four specialized agents are orchestrated via LangGraph in a directed acyclic graph:

| Agent | Role | Stage |
|---|---|---|
| Investigator | Generates search queries and retrieves evidence from ChromaDB or GraphRAG | Retrieval |
| Reasoner | Synthesizes retrieved context with anomaly data to produce a root cause hypothesis | Reasoning |
| Critic | Reviews the hypothesis for factual grounding, hallucination risk, and evidence consistency | Validation |
| Reporter | Produces a structured JSON RCA report with severity, actions, and audit metadata | Output |

*Table 2: Agent Roles and Responsibilities*

The flow is: Investigator → Reasoner → Critic → Reporter. The Critic can trigger one revision loop back to the Reasoner if grounding is insufficient.

### (f) Evaluation Framework

The evaluation framework provides multi-dimensional assessment:

- **Detection metrics:** Precision, Recall, F1, ROC-AUC
- **RAG retrieval quality:** RAGAS-style Context Recall, Context Precision, Faithfulness, Answer Relevancy
- **RCA output quality:** ROUGE-L, BERTScore, LLM-as-Judge (correctness, groundedness, actionability, completeness on a 1–5 Likert scale)
- **Statistical significance:** Bootstrap confidence intervals, paired bootstrap, Wilcoxon signed-rank tests

### (g) Streamlit User Interface

A three-page Streamlit application provides:

1. **Upload & Detect** — Upload CSV data and run anomaly detection
2. **RCA Viewer** — Select detected anomalies and run the multi-agent RCA pipeline
3. **Knowledge Base** — Browse and search the telecom RCA knowledge base

### (h) MLflow Experiment Tracking

All pipeline runs and ablation experiments are tracked via MLflow with file-backed SQLite storage. Parameters, metrics, and artifacts are logged for reproducibility.

---

## 2. SYSTEM ARCHITECTURE / FUNCTIONAL BLOCK DIAGRAM

The system architecture connects the modules as follows:

```
                    ┌──────────────────────────┐
                    │    Raw Telecom Datasets   │
                    │  (IBM Telco, Maven Telco) │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Anomaly Injection &    │
                    │    Detection Module       │
                    │  (IsolationForest/DBSCAN) │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Detected Anomaly Record │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                                 │
                ▼                                 ▼
   ┌────────────────────┐            ┌────────────────────┐
   │  ChromaDB Vector   │            │  GraphRAG Entity-  │
   │  Retrieval         │            │  Relation Graph    │
   │  (all-MiniLM-L6-v2)│           │  (NetworkX)        │
   └─────────┬──────────┘            └─────────┬──────────┘
             │                                 │
             └───────────────┬─────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │    Investigator Agent     │
              │    (Retrieve Evidence)    │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    Reasoner Agent         │
              │    (Generate Hypothesis)  │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    Critic Agent           │
              │    (Review & Revise)      │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    Reporter Agent         │
              │    (Structured RCA JSON)  │
              └────────────┬─────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         Streamlit     MLflow       Evaluation
         Dashboard     Tracking     Framework
```

*Figure 2: Multi-Agent Pipeline Flow Diagram*

The configurable LLM backend supports Groq (preferred), Kimi (fallback), and any custom OpenAI-compatible endpoint. The judge model can be configured independently from the generation model.

---

## 3. TECHNICAL SPECIFICATIONS

| SL No. | Technical Parameter | Specification |
|---|---|---|
| 1. | Programming Language | Python 3.11 |
| 2. | LLM Backend | Configurable OpenAI-compatible (Groq / Kimi / Custom) |
| 3. | Default LLM Model | Llama 3.3 70B Versatile (via Groq) |
| 4. | Embedding Model | all-MiniLM-L6-v2 (384 dimensions) |
| 5. | Vector Store | ChromaDB (file-backed) |
| 6. | Graph Store | NetworkX (pickle serialized) |
| 7. | Agent Framework | LangGraph |
| 8. | Anomaly Detectors | IsolationForest, DBSCAN (scikit-learn) |
| 9. | Experiment Tracking | MLflow (file-backed SQLite) |
| 10. | User Interface | Streamlit (multi-page) |
| 11. | Containerization | Docker |
| 12. | Testing Framework | pytest (87 tests) |
| 13. | Datasets | IBM Telco Churn, Maven Telecom Churn |
| 14. | Anomaly Types | 5 (Zero Billing, Duplicate Charge, Usage Spike, CDR Failure, SLA Breach) |
| 15. | Ablation Configurations | 5 (A: No RAG, B: RAG-only, C: Single-agent+RAG, D: Multi-agent+RAG, E: Multi-agent+GraphRAG) |
| 16. | Evaluation Metrics | ROUGE-L, BERTScore, RAGAS, LLM-as-Judge, Bootstrap CI, Wilcoxon |
| 17. | Deployment | Streamlit Cloud / Docker |
| 18. | Repository | github.com/tatsat3mutee/multiagent-rag-telecom-billing-rca |

*Table 3: Technical Specifications*

---

## 4. DESIGN CONSIDERATIONS

- The system is designed to be reproducible: all data processing, model training, retrieval indexing, and evaluation can be re-run from source using documented scripts.
- The LLM backend is provider-neutral: switching from Groq to Kimi or any custom OpenAI-compatible endpoint requires only changing environment variables, not code.
- The judge model can be configured independently from the generation model to mitigate same-model evaluation bias (following Zheng et al., 2023).
- The multi-agent decomposition separates retrieval, reasoning, critique, and reporting into independent, testable agents rather than a monolithic prompt chain.
- Anomaly detection uses unsupervised methods (IsolationForest, DBSCAN) to avoid dependence on labeled training data that may not be available in production telecom environments.
- The GraphRAG layer provides multi-hop reasoning capability beyond flat vector retrieval, enabling connections between anomaly types, billing systems, SLA clauses, and remediation procedures.

| Config | Description | Purpose |
|---|---|---|
| A | No RAG — direct LLM generation | Lower bound baseline |
| B | RAG-only — retrieve + generate in single prompt | Measures retrieval value |
| C | Single-agent + RAG — one agent does everything | Tests agent decomposition need |
| D | Multi-agent + RAG — proposed 4-agent pipeline | Core proposed system |
| E | Multi-agent + GraphRAG — graph-traversal retrieval | Headline novelty |

*Table 4: Ablation Configurations*

---

## 5. FUTURE PLAN

| Sl No. | Phases | Start Date – End Date | Work to be done | Status |
|---|---|---|---|---|
| 1 | Literature Review & Project Outline | Jan 2026 – Feb 2026 | Literature survey, problem definition, architecture design | COMPLETED |
| 2 | Data Pipeline & Detection | Feb 2026 – Mar 2026 | Dataset preparation, anomaly injection, IsolationForest/DBSCAN implementation | COMPLETED |
| 3 | RAG & Agent Implementation | Mar 2026 – Apr 2026 | ChromaDB knowledge base, GraphRAG, 4-agent LangGraph pipeline, Streamlit UI | COMPLETED |
| 4 | Evaluation & Ablation | Apr 2026 – May 2026 | Run 5-config ablation, ground truth evaluation, LLM-as-Judge, statistical tests | PENDING |
| 5 | Final Report Writing | May 2026 – Jun 2026 | Write full dissertation chapters, format per BITS guidelines, create figures/tables | PENDING |
| 6 | Dissertation Review | Jun 2026 – Jul 2026 | Submit to supervisor & additional examiner for review and feedback | PENDING |
| 7 | Final Submission & Viva | Jul 2026 | Final review, PPT preparation, submission, viva | PENDING |

*Table 5: Project Timeline and Status*

---

## 6. ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| AIOps | Artificial Intelligence for IT Operations |
| BERTScore | Bidirectional Encoder Representations from Transformers Score |
| CDR | Call Detail Record |
| CI | Confidence Interval |
| CSV | Comma-Separated Values |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| GraphRAG | Graph-based Retrieval-Augmented Generation |
| JSON | JavaScript Object Notation |
| LLM | Large Language Model |
| LVDS | Low Voltage Differential Signalling |
| MLflow | Machine Learning Flow |
| MTTR | Mean Time To Resolution |
| NLP | Natural Language Processing |
| RAG | Retrieval-Augmented Generation |
| RAGAS | Retrieval-Augmented Generation Assessment |
| RCA | Root Cause Analysis |
| ROC-AUC | Receiver Operating Characteristic – Area Under Curve |
| ROUGE-L | Recall-Oriented Understudy for Gisting Evaluation – Longest Common Subsequence |
| SLA | Service Level Agreement |
| UI | User Interface |
