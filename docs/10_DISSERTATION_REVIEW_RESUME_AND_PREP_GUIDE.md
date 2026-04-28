# Dissertation Review, Resume Positioning, and Viva Preparation Guide

## Project Title

**A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks**

## 1. Executive Verdict

This project is a strong MTech dissertation topic if it is positioned correctly. It should be presented as an **applied AI systems research project**, not as a new machine learning algorithm. The core contribution is the design, implementation, and evaluation of a domain-specific GenAI system that combines anomaly detection, retrieval-augmented generation, multi-agent orchestration, GraphRAG, evaluation, and a working user interface for telecom billing root cause analysis.

Current dissertation strength: **8/10**  
Potential strength after final documentation and evaluation cleanup: **9/10**

The project is also strong enough to include on a resume for roles in AI engineering, GenAI, machine learning engineering, data science, AIOps, and applied research engineering.

## 2. Why This Is a Good MTech Dissertation Topic

### 2.1 It Solves a Real Operational Problem

Telecom billing systems process very large volumes of Call Detail Records (CDRs), customer usage events, invoices, and service-level commitments. Billing anomalies such as zero billing, duplicate charges, sudden usage spikes, CDR processing failures, and Service Level Agreement (SLA) breaches can cause revenue leakage and customer dissatisfaction.

The important research gap is not only anomaly detection. Many systems can detect suspicious records. The harder operational problem is explaining **why** the anomaly occurred and what an engineer should do next. Your project targets this detection-to-resolution gap.

### 2.2 It Has a Clear Applied Research Question

The central question is defensible:

> Can a multi-agent Retrieval-Augmented Generation system improve telecom billing root cause analysis compared with direct LLM generation, simple RAG, and single-agent RAG?

This is a good MTech-level question because it is specific, measurable, and implementable within one semester.

### 2.3 It Is More Than a Chatbot

The project includes:

- Data loading and preprocessing
- Synthetic billing anomaly injection
- IsolationForest and DBSCAN based anomaly detection
- ChromaDB vector retrieval
- GraphRAG over telecom playbooks
- LangGraph multi-agent workflow
- Critic-based review and revision
- Structured Root Cause Analysis (RCA) report generation
- MLflow experiment tracking
- Streamlit dashboard
- Evaluation with ROUGE-L, BERTScore, RAGAS-style metrics, LLM-as-Judge, bootstrap confidence intervals, and Wilcoxon testing

This makes the work an end-to-end applied AI system, which is much stronger than a simple prompt demo.

### 2.4 It Has a Defensible Novelty Angle

The strongest novelty angle is:

> A domain-specific multi-agent RAG and GraphRAG architecture for telecom billing RCA, evaluated through ablation across no-RAG, RAG-only, single-agent RAG, multi-agent RAG, and multi-agent GraphRAG configurations.

You should not claim novelty in IsolationForest, DBSCAN, LangGraph, ROUGE, BERTScore, or RAG itself. The novelty is in the **system composition, domain adaptation, GraphRAG retrieval layer, evaluation design, and telecom RCA use case**.

## 3. What Has Been Built So Far

### 3.1 Data and Anomaly Detection

The project uses telecom-style customer billing datasets and injects five anomaly categories:

| Anomaly Type | Meaning | Example Failure |
|---|---|---|
| Zero billing | Active customer receives a zero bill | Tariff/rating failure, missing CDRs |
| Duplicate charge | Same usage or account is charged more than once | Replay, retry, duplicate billing job |
| Usage spike | Sudden abnormal usage or charge increase | Fraud, roaming issue, metering error |
| CDR failure | Call Detail Record processing fails | Null records, parser failure, ingestion issue |
| SLA breach | Contractual or service-level threshold exceeded | Wrong cap, plan mismatch, billing limit violation |

Detection is implemented through IsolationForest and DBSCAN. This is useful for building the pipeline, but it should be presented as a supporting module, not the main research contribution.

### 3.2 Retrieval Layer

The system uses ChromaDB and sentence-transformers for dense vector retrieval over telecom RCA playbooks and operational documents. This enables the LLM agents to ground their answers in retrieved context instead of relying only on model memory.

### 3.3 GraphRAG Layer

GraphRAG is the headline extension. It converts telecom playbook content into an entity-relation graph and retrieves context through graph traversal. This is useful when the answer depends on relationships between systems, failure modes, metrics, and fixes.

Example:

CDR parser failure -> missing usage records -> zero-rated invoice -> revenue leakage -> trigger CDR reprocessing

This is stronger than flat vector search when the root cause requires multi-hop reasoning.

### 3.4 Multi-Agent RCA Pipeline

The current agent pipeline should be explained as:

1. **Investigator Agent**: Converts the anomaly into a retrieval query and fetches relevant context.
2. **Reasoner Agent**: Uses anomaly data and retrieved evidence to generate a root cause hypothesis.
3. **Critic Agent**: Checks whether the hypothesis is grounded, complete, and safe enough to proceed.
4. **Reporter Agent**: Produces a structured RCA report with root cause, evidence, severity, and recommended actions.

The strongest defense point is that each agent has a clear role, making the workflow auditable and easier to evaluate than a single long prompt.

### 3.5 Evaluation

The evaluation design is one of the strongest parts of the project. It includes:

- Ablation study across Config A, B, C, D, and E
- ROUGE-L for lexical overlap
- BERTScore for semantic similarity
- RAGAS-style faithfulness and answer relevancy
- LLM-as-Judge for correctness, groundedness, completeness, and actionability
- Bootstrap confidence intervals
- Wilcoxon signed-rank testing

This gives the dissertation a credible experimental structure.

## 4. Current Risks and How to Handle Them

| Risk | Severity | How to Defend It |
|---|---:|---|
| Synthetic anomalies | High | Present as controlled benchmark construction; production validation is future work |
| Churn dataset is not real CDR data | High | State clearly that it is a proxy dataset for billing behavior, not a production CDR stream |
| Self-authored playbooks and ground truth | High | Call this a limitation; propose SME validation as future work |
| LLM-as-Judge bias | Medium | Use separate judge model where possible; disclose backend and temperature |
| Small sample size | Medium | Use confidence intervals and non-parametric tests; avoid overclaiming per-type superiority |
| API provider inconsistency in docs | Medium | Align all docs to one statement: configurable OpenAI-compatible backend |
| MTTR reduction not measured in production | Medium | Say projected triage reduction, not achieved production MTTR reduction |

The safest viva position is:

> This system is a reproducible research prototype demonstrating the feasibility and measurable benefit of multi-agent RAG and GraphRAG for telecom billing RCA. It is not yet a production-validated telecom operations platform.

## 5. Should You Continue With This Topic?

Yes. Continue with this topic.

It is beneficial because it gives you:

- A real domain problem
- A modern GenAI architecture
- Strong resume value
- A working application demo
- A clear thesis structure
- A defensible evaluation plan
- Multiple future work directions

The only reason to change the topic would be if your department strictly requires mathematical algorithm novelty. If your department accepts applied AI systems, data science pipelines, and experimental evaluation, this topic is suitable.

## 6. How To Position This In the Dissertation

Recommended framing:

> This dissertation proposes and evaluates a domain-specific multi-agent RAG architecture for telecom billing root cause analysis. The system combines anomaly detection, vector retrieval, GraphRAG retrieval, agentic reasoning, critic-based revision, and structured reporting. It is evaluated through ablation studies and multi-dimensional text-quality and grounding metrics.

Avoid these claims:

- Do not say it is fully production-ready.
- Do not say it proves real MTTR reduction unless you measure a manual baseline.
- Do not say it uses only open-source LLMs if API providers are used.
- Do not say GraphRAG is always better unless Config E results support it.
- Do not say the ground truth is industry-validated unless an SME reviews it.

Use these claims instead:

- Reproducible research prototype
- Domain-specific telecom RCA benchmark setup
- Multi-agent RAG architecture
- GraphRAG retrieval extension
- Evidence-backed ablation study
- Production-oriented design, not production-validated deployment

## 7. Resume Positioning

### 7.1 Full Resume Entry

**Multi-Agent RAG System for Telecom Billing Root Cause Analysis**  
Built an end-to-end GenAI system for telecom billing anomaly diagnosis using LangGraph, ChromaDB, sentence-transformers, Streamlit, and MLflow. Designed a four-agent RCA workflow with Investigator, Reasoner, Critic, and Reporter agents for grounded root-cause generation and corrective action recommendation. Implemented GraphRAG over telecom playbooks using NetworkX-based entity-relation retrieval and compared it against flat vector RAG through ablation studies. Evaluated RCA quality using ROUGE-L, BERTScore, RAGAS-style faithfulness/relevancy, LLM-as-Judge scoring, bootstrap confidence intervals, and Wilcoxon testing.

### 7.2 Short Resume Bullets

- Built a multi-agent RAG and GraphRAG system for telecom billing root cause analysis using LangGraph, ChromaDB, NetworkX, MLflow, and Streamlit.
- Designed an Investigator-Reasoner-Critic-Reporter workflow to generate grounded RCA reports from detected billing anomalies.
- Implemented Config A-E ablation comparing no RAG, RAG-only, single-agent RAG, multi-agent RAG, and multi-agent GraphRAG.
- Evaluated outputs using ROUGE-L, BERTScore, RAGAS-style faithfulness/relevancy, LLM-as-Judge, bootstrap confidence intervals, and Wilcoxon tests.
- Built a reproducible anomaly detection and RCA pipeline using IsolationForest, DBSCAN, synthetic anomaly injection, ChromaDB retrieval, and MLflow tracking.

### 7.3 LinkedIn Version

Built a multi-agent RAG + GraphRAG system for telecom billing anomaly root cause analysis. The project combines anomaly detection, ChromaDB retrieval, NetworkX graph retrieval, LangGraph agent orchestration, critic-based revision, MLflow tracking, Streamlit UI, and multi-metric evaluation through ablation studies.

## 8. Topics To Prepare For Viva and Interviews

### 8.1 Telecom Domain

- Call Detail Record (CDR)
- Service Level Agreement (SLA)
- Mean Time To Resolution (MTTR)
- Root Cause Analysis (RCA)
- Revenue assurance
- Billing pipeline
- Mediation system
- Rating engine
- Charging system
- Invoice generation
- Zero billing
- Duplicate billing
- Usage spike
- CDR ingestion failure
- SLA breach

### 8.2 Machine Learning and Detection

- IsolationForest intuition
- DBSCAN intuition
- Anomaly score and confidence score
- Precision, recall, F1-score, ROC-AUC
- Synthetic anomaly injection
- Data leakage
- Class imbalance
- Feature engineering
- Why customer churn data is only a proxy for billing data
- Limitations of point-in-time anomaly detection

### 8.3 RAG Fundamentals

- What is Retrieval-Augmented Generation
- Chunking
- Embeddings
- Vector databases
- Cosine similarity
- Top-k retrieval
- ChromaDB
- sentence-transformers
- Hallucination
- Grounding
- Faithfulness
- Answer relevancy
- Context precision
- Context recall

### 8.4 GraphRAG

- Difference between vector RAG and GraphRAG
- Entity extraction
- Relation extraction
- NetworkX graph
- k-hop traversal
- Seed entity matching
- Node and edge support
- Multi-hop RCA reasoning
- When GraphRAG may fail
- Why Config E is important

### 8.5 Multi-Agent Systems

- What is an LLM agent
- LangGraph StateGraph
- Agent state
- Conditional routing
- Investigator Agent
- Reasoner Agent
- Critic Agent
- Reporter Agent
- Revision loop
- Why multi-agent can outperform single prompt
- Tradeoff: better structure vs higher latency and complexity

### 8.6 Evaluation

- Ablation study
- Config A: no RAG
- Config B: RAG-only
- Config C: single-agent RAG
- Config D: multi-agent RAG
- Config E: multi-agent GraphRAG
- ROUGE-L
- BERTScore
- RAGAS-style metrics
- LLM-as-Judge
- Judge bias
- Bootstrap confidence interval
- Wilcoxon signed-rank test
- Statistical significance
- Why lexical metrics alone are insufficient

### 8.7 System Design and Engineering

- End-to-end pipeline architecture
- Streamlit UI
- MLflow tracking
- ChromaDB persistence
- SQLite inference logging
- Environment variables and API configuration
- Fallback behavior when LLM is unavailable
- Reproducibility through seeds and saved artifacts
- Local deployment vs production deployment
- Security and privacy concerns for telecom data

### 8.8 Limitations and Future Work

- Synthetic data limitation
- No proprietary CDR stream
- No SME validation
- Single-author ground truth
- Small evaluation sample
- LLM-as-Judge limitations
- Domain shift
- Need for real incident tickets
- Need for human evaluation
- Future production integration with ticketing systems

## 9. Most Likely Viva Questions

1. What problem are you solving?
2. Why is this problem important in telecom?
3. What is a CDR?
4. What is RCA?
5. Why is anomaly detection alone insufficient?
6. Why did you choose RAG instead of fine-tuning?
7. Why did you use multiple agents?
8. What does the Critic agent add?
9. What is GraphRAG?
10. How is GraphRAG different from ChromaDB vector search?
11. What are Config A, B, C, D, and E?
12. What metrics did you use and why?
13. How do you know the generated RCA is correct?
14. What are the limitations of LLM-as-Judge?
15. What is your ground truth dataset?
16. Who validated the ground truth?
17. What are the limitations of synthetic anomalies?
18. Is this system production-ready?
19. What would change if you had real telecom data?
20. What is your main contribution?

## 10. Better Ideas or Alternative Directions

The current project is worth continuing. However, if you want to make it sharper, you can choose one of these improved directions instead of changing everything.

### Option A: Keep Current Topic, Strengthen It With Human Evaluation

This is the best option.

Add a small human validation layer:

- Ask 2-3 people with telecom, networking, cloud, or operations knowledge to rate 10 RCA outputs.
- Use a simple 1-5 rubric: correctness, usefulness, groundedness, actionability.
- Compare human scores with LLM-as-Judge scores.

Why this is better: it directly addresses the biggest limitation.

### Option B: Shift Title Toward AIOps Instead of Telecom Billing Only

Possible title:

**A Multi-Agent GraphRAG Framework for AIOps Root Cause Analysis in Telecom Billing Systems**

This makes the topic broader and more aligned with industry terms like AIOps, incident management, and observability.

Why this is better: more resume-friendly and less dependent on telecom-specific data.

Risk: too broad if not controlled.

### Option C: Make GraphRAG the Main Thesis Contribution

Possible title:

**GraphRAG-Enhanced Multi-Agent Root Cause Analysis for Telecom Billing Anomalies**

This puts the most novel part in the title.

Why this is better: faculty and recruiters immediately see the advanced contribution.

Risk: you must have solid Config D vs Config E results.

### Option D: Focus on Evaluation Framework for GenAI RCA

Possible title:

**Evaluating Multi-Agent RAG Systems for Telecom Root Cause Analysis**

This emphasizes ablation, metrics, judge bias, faithfulness, and statistical testing.

Why this is better: academically safer if system novelty is questioned.

Risk: less flashy as a demo.

### Option E: Use Real Telecom Italia CDR Data as Secondary Validation

If the Telecom Italia loader is stable, use it as a secondary experiment track.

Why this is better: it reduces the criticism that the project only uses churn-style data.

Risk: more engineering time and more explanation required.

## 11. Best Recommended Final Direction

The best direction is not to abandon the project. The best direction is:

> Keep the current project, but reframe the final dissertation around Multi-Agent GraphRAG for AIOps-style Telecom Billing RCA, and add one small human-evaluation or D-vs-E ablation result before final submission.

Recommended final title:

**A Multi-Agent GraphRAG System for Root Cause Analysis of Telecom Billing Anomalies**

This title is shorter, clearer, and stronger than the current version. It highlights the two best parts: multi-agent architecture and GraphRAG.

## 12. Immediate Action Checklist

Before final submission, complete these items:

- Align all docs to four agents: Investigator, Reasoner, Critic, Reporter.
- Align all docs to five ablation configs: A, B, C, D, E.
- Choose one LLM-provider story and use it everywhere.
- Replace "open-source" with "reproducible" unless every major runtime component is truly open-source.
- Replace "40x achieved" with "projected triage reduction" unless a manual baseline is measured.
- Run Config D vs Config E comparison.
- Add one slide explaining GraphRAG with a simple entity-relation example.
- Add one limitations slide.
- Add one viva slide answering: "What is your actual contribution?"
- Prepare 15-20 likely viva questions from this guide.

## 13. Final Positioning Statement

Use this in viva, resume discussion, and project explanation:

> My dissertation builds a reproducible multi-agent GraphRAG system for telecom billing anomaly root cause analysis. The system detects billing anomalies, retrieves relevant operational knowledge from vector and graph-based stores, reasons through a four-agent LangGraph workflow, critiques the hypothesis for grounding, and generates structured RCA reports. I evaluate the architecture through ablation studies and multiple quality metrics, while explicitly documenting limitations around synthetic data, LLM-as-Judge bias, and the need for future SME validation.

This is the strongest, safest, and most professional way to present the work.