# Abstract

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

**Author:** Tatsat Pandey  
**Program:** MTech — Data Science & Engineering  
**Date:** 2026  

---

## Abstract

Billing anomalies in telecom networks — including sudden usage spikes, unexpected zero-billing events, duplicate charges, and Call Detail Record (CDR) processing failures — are routinely detected by rule-based or machine learning systems. However, the critical gap lies between detection and resolution: determining *why* an anomaly occurred requires engineers to manually search through Service Level Agreement (SLA) documents, past incident tickets, routing rules, and system logs. This diagnostic bottleneck causes significant Mean-Time-To-Resolution (MTTR) delays, revenue leakage, and over-reliance on senior engineering expertise. At enterprise scale, where millions of CDR events are processed daily across geographically distributed networks, this manual triage creates untenable operational overhead and knowledge silos.

This research designs, implements, and evaluates an end-to-end reproducible **multi-agent Retrieval-Augmented Generation (RAG)** system that autonomously closes the detection-to-resolution gap. The system comprises four specialized LLM-powered agents orchestrated via LangGraph: an **Investigator Agent** that retrieves semantically relevant context from a curated knowledge base of SLA documents, historical Root Cause Analysis (RCA) reports, billing policy manuals, and operational runbooks; a **Reasoner Agent** that synthesizes this context to generate structured root cause hypotheses; a **Critic Agent** that reviews and optionally revises the hypothesis for factual grounding and hallucination mitigation; and a **Reporter Agent** that produces human-readable RCA documents with recommended corrective actions — all without requiring human analyst intervention for initial triage.

The system uses a configurable OpenAI-compatible LLM backend with automatic provider selection (Groq preferred, Kimi fallback, and custom endpoint support), ChromaDB for vector storage, sentence-transformers for embeddings, scikit-learn for anomaly detection, MLflow for experiment tracking and reproducibility, and Streamlit for the user interface. The judge model can be configured independently from the generation model to reduce same-model evaluation bias. Evaluation is conducted using a multi-dimensional framework spanning anomaly detection metrics (Precision, Recall, F1, ROC-AUC), RAG retrieval quality (RAGAS-style Context Recall, Context Precision, Faithfulness), and RCA output quality (BERTScore, ROUGE-L, LLM-as-Judge scoring with bootstrap confidence intervals).

A rigorous ablation study across five configurations — no RAG baseline (A), RAG-only (B), single-agent + RAG (C), the proposed multi-agent + RAG (D), and multi-agent + GraphRAG (E) — demonstrates that the separation of retrieval, reasoning, and reporting into specialized agents yields measurable improvements in diagnostic accuracy and faithfulness. The system is projected to substantially reduce initial triage time compared to manual investigation; a precise MTTR reduction estimate is left to future work with production deployment data.

**Key contributions** include: (1) a novel domain-specific multi-agent RAG architecture for telecom billing anomaly diagnosis with specialized agent roles and critic-based revision; (2) a curated telecom billing knowledge corpus for RAG benchmarking and future research; (3) empirical evidence comparing single-agent vs. multi-agent architectures on diagnostic reasoning quality through rigorous ablation; (4) a GraphRAG entity-relation retrieval layer (Config E) as a headline novelty enabling graph-traversal-based context retrieval for complex multi-hop reasoning; and (5) a reproducible, end-to-end deployable system with open evaluation artifacts, comprehensive test coverage (87 tests), and statistical significance testing, enabling replication and extension by the research community.

---

## Keywords

Multi-Agent Systems, Retrieval-Augmented Generation (RAG), Root Cause Analysis (RCA), Telecom Billing, Anomaly Detection, LangGraph, Large Language Models (LLM), LLM Agents, Artificial Intelligence for IT Operations (AIOps), MLflow, Call Detail Records (CDR)

---

## Short Abstract (150 words — for submission forms)

Billing anomalies in telecom networks are routinely detected but diagnosing their root cause remains a manual, time-intensive process requiring senior engineering expertise. This research presents a reproducible multi-agent RAG (Retrieval-Augmented Generation) system that autonomously investigates billing anomalies, including zero-billing events, duplicate charges, usage spikes, and Call Detail Record (CDR) failures, and generates structured Root Cause Analysis (RCA) reports. Four specialized LLM-powered agents, Investigator, Reasoner, Critic, and Reporter, are orchestrated via LangGraph to retrieve relevant context from a curated telecom knowledge base, synthesize and critique root cause hypotheses with bounded revision loops, and produce actionable reports with corrective recommendations. A configurable OpenAI-compatible LLM backend, ChromaDB, GraphRAG, MLflow, and Streamlit form the infrastructure. The system is evaluated using RAGAS-style metrics, BERTScore, LLM-as-Judge scoring, and a five-configuration ablation study (A-E) including a novel GraphRAG retrieval layer for entity-relation reasoning, with bootstrap confidence intervals and Wilcoxon significance testing. Results demonstrate measurable improvements in diagnostic accuracy and faithfulness from the multi-agent design, offering a reusable framework applicable across telecom operators globally.

---

## BITS WILP Abstract Sheet Version (196 words)

This dissertation designs and evaluates a multi-agent Retrieval-Augmented Generation (RAG) system for autonomous root cause analysis of billing anomalies in telecom networks. Billing anomalies such as zero billing, duplicate charges, usage spikes, Call Detail Record (CDR) processing failures, and Service Level Agreement (SLA) breaches are detected by rule-based or machine learning systems, but diagnosing their root cause remains manual and time-intensive. The proposed system comprises four LLM-powered agents orchestrated via LangGraph: Investigator for retrieval, Reasoner for hypothesis generation, Critic for factual review, and Reporter for structured output. The system uses ChromaDB for vector retrieval, a GraphRAG layer for entity-relation graph traversal over telecom playbooks, scikit-learn for anomaly detection, MLflow for experiment tracking, and Streamlit for the user interface. A five-configuration ablation study evaluates no-RAG, RAG-only, single-agent RAG, multi-agent RAG, and multi-agent GraphRAG variants using ROUGE-L, BERTScore, RAGAS-style faithfulness, LLM-as-Judge scoring, bootstrap confidence intervals, and Wilcoxon significance tests. Key contributions include the multi-agent RAG architecture, GraphRAG retrieval layer, curated telecom knowledge corpus, and reproducible evaluation framework.
