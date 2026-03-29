# Abstract

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

**Author:** Tatsat Pandey  
**Program:** MTech — Data Science & Engineering  
**Date:** 2026  

---

## Abstract

Billing anomalies in telecom networks — including sudden usage spikes, unexpected zero-billing events, duplicate charges, and CDR processing failures — are routinely detected by rule-based or machine learning systems. However, the critical gap lies between detection and resolution: determining *why* an anomaly occurred requires engineers to manually search through SLA documents, past incident tickets, routing rules, and system logs. This diagnostic bottleneck causes significant Mean-Time-To-Resolution (MTTR) delays, revenue leakage, and over-reliance on senior engineering expertise. At enterprise scale, where millions of CDR events are processed daily, this manual triage creates untenable operational overhead.

This research designs, implements, and evaluates an end-to-end open-source **multi-agent Retrieval-Augmented Generation (RAG)** system that autonomously closes the detection-to-resolution gap. The system comprises three specialized LLM-powered agents orchestrated via LangGraph: an **Investigator Agent** that retrieves semantically relevant context from a curated knowledge base of SLA documents, historical RCA reports, billing policy manuals, and operational runbooks; a **Reasoning Agent** that synthesizes this context to generate structured root cause hypotheses; and a **Reporter Agent** that produces human-readable RCA documents with recommended corrective actions — all without requiring human analyst intervention for initial triage.

The system is built entirely on open-source technologies: Llama 3.3 70B for language modeling (served via Groq), ChromaDB for vector storage, sentence-transformers for embeddings, scikit-learn for anomaly detection, MLflow for experiment tracking, and Streamlit for the user interface. Evaluation is conducted using a multi-dimensional framework spanning anomaly detection metrics (Precision, Recall, F1, ROC-AUC), RAG retrieval quality (RAGAS — Context Recall, Context Precision, Faithfulness), and RCA output quality (BERTScore, ROUGE-L, LLM-as-Judge).

A rigorous ablation study across four configurations — no RAG baseline, RAG-only, single-agent + RAG, and the proposed multi-agent + RAG — demonstrates that the separation of retrieval, reasoning, and reporting into specialized agents yields measurable improvements in diagnostic accuracy and faithfulness. The system achieves an estimated 40x reduction in initial triage time compared to manual investigation baselines.

**Key contributions** include: (1) a novel domain-specific multi-agent RAG architecture for telecom billing anomaly diagnosis; (2) a curated open-source telecom billing knowledge corpus for RAG benchmarking; (3) empirical evidence comparing single-agent vs. multi-agent architectures on diagnostic reasoning quality; and (4) an open-source, fully deployable end-to-end system.

---

## Keywords

Multi-Agent Systems, Retrieval-Augmented Generation (RAG), Root Cause Analysis, Telecom Billing, Anomaly Detection, LangGraph, Large Language Models, LLM Agents, AIOps, MLflow, Open-Source AI

---

## Short Abstract (150 words — for submission forms)

Billing anomalies in telecom networks are routinely detected but diagnosing their root cause remains a manual, time-intensive process requiring senior engineering expertise. This research presents an open-source multi-agent RAG (Retrieval-Augmented Generation) system that autonomously investigates billing anomalies and generates structured Root Cause Analysis reports. Three specialized LLM-powered agents — Investigator, Reasoner, and Reporter — are orchestrated via LangGraph to retrieve relevant context from a curated telecom knowledge base, synthesize root cause hypotheses, and produce actionable reports. Built entirely on open-source tools (Llama 3.3 70B via Groq, ChromaDB, MLflow, Streamlit), the system is evaluated using RAGAS, BERTScore, and ablation studies comparing single vs. multi-agent architectures. Results demonstrate measurable improvements in diagnostic accuracy from the multi-agent design and an estimated 40x reduction in initial triage time, offering a reusable framework applicable across telecom operators globally.
