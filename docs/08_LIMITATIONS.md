# Limitations & Threats to Validity

This chapter documents honest limitations of the current system and evaluation. It is intended to support viva defense and to flag areas for follow-up work.

## 1. Data Limitations

### 1.1 Synthetic anomaly injection
Anomalies in `data/processed/anomalies_labeled.csv` are **synthetically injected** into the IBM Telco Customer Churn dataset via rule-based transforms (`src/data/anomaly_injector.py`):

| Anomaly type    | Injection rule                              | Risk                                   |
|-----------------|---------------------------------------------|----------------------------------------|
| zero_billing    | `MonthlyCharges := 0`                       | Too trivially separable by IForest     |
| duplicate_charge| Row cloned                                  | Exact duplicates are unusually clean   |
| usage_spike     | `MonthlyCharges *= U(5, 15)`                | Spike magnitude may exceed production  |
| cdr_failure     | `TotalCharges := NaN`                       | Production NaNs are partial, not full  |
| sla_breach      | Charges pushed above empirical p95          | Label leakage: test uses same quantile |

**Implication**: the reported detection F1 = 0.81 and ROC-AUC = 0.877 are likely **upper bounds** on what the same pipeline would achieve on production billing logs. The injection rules also create label leakage for `sla_breach` because the detector indirectly sees the quantile used to inject. Partial mitigation: the 60-item GT deliberately includes realistic narrative failure modes (Kafka rebalance replay, mTLS cert expiry, DST empty window, etc.) that the retriever/reasoner must produce, so the RAG-side evaluation is **not** contaminated by the injection rules.

### 1.2 Schema mismatch: churn features ≠ billing KPIs
IBM Telco is a **churn** dataset; its features (tenure, InternetService, PaperlessBilling, …) are only a weak proxy for production telecom-billing signals (CDR counts per mediation pipeline, rating latency p99, duplicate-key rate, tax-engine deltas). Production deployment would require re-featurization on the real CDR feature space. **Phase 1.5 D1** (Telecom Italia CDR ingestion) addresses this partially by introducing a real multi-source CDR feature space as a secondary track.

### 1.3 Ground truth circularity
The initial 15-entry ground truth `ground_truth_rca.json` was authored by the same person who wrote the 8 playbooks the retriever is indexed over. This creates **single-author circularity**: high ROUGE-L reflects shared vocabulary, not objectively correct RCA. The expanded `ground_truth_rca_60.json` (12 per type, 60 total) introduces a second layer of independent narrative failure modes, but every entry still carries `"author": "seed"`. Full mitigation requires SME review (≥2 reviewers) and replacement of ≥20 entries with SME-authored RCAs (reserved for future work).

### 1.4 No proprietary telco data
No ticket data, no production RCAs, no real incident postmortems. All SME content is public-domain (3GPP TS 32.240 / 32.298, TMF678 Customer Bill Management, public vendor docs). This is a hard constraint from the industrial-placement NDA and is documented as a scope limitation, not a methodological flaw.

## 2. Modeling Limitations

### 2.1 Static, point-in-time detector
`IsolationForest` treats each row independently. It does not model:
- **Temporal drift** (billing behaviour changes seasonally and with campaign launches)
- **Customer-level baselines** (per-account z-score would catch usage_spike earlier)
- **Causal chains** (mediation failure → downstream zero-billing)

A temporal baseline (e.g. Prophet per account, or a simple EWMA deviation detector) is listed as future work.

### 2.2 Single embedding model
The KB is indexed with `all-MiniLM-L6-v2` (384-dim) — chosen for free/local operation. Domain-adapted embeddings (e.g. `bge-large-en`, SBERT fine-tuned on telecom docs) would likely improve Recall@5, but require GPU compute not available in the project envelope.

### 2.3 No structured RCA schema
Outputs are free-text. A structured schema (`{cause_system, contributing_factors[], recommended_fix, priority}`) would make downstream tooling and offline evaluation more reliable but was descoped after Phase 1.

## 3. Evaluation Limitations

### 3.1 Small N
Even at 60 ground-truth items, per-type sample size is 12. Paired-bootstrap CIs are wide (typically ±0.08–0.15 on ROUGE-L). Statements about configuration ranking (A vs B vs C vs D) are supported by Wilcoxon and paired-bootstrap p-values on the joint metric, but **per-type claims are not statistically supported**.

### 3.2 ROUGE-L and BERTScore are lexical/semantic surface metrics
They reward overlap, not correctness. This is why we added **LLM-as-Judge** (correctness / groundedness / actionability / completeness on a 1–5 Likert) plus **RAGAS-style faithfulness + answer_relevancy**. The judge is `gpt-4o-mini` by default with Groq Llama-3.3-70B as fallback; the `backend` tag is persisted in results for transparent bias reporting.

### 3.3 LLM-as-Judge bias
The judge is itself an LLM; it may prefer responses stylistically similar to its own outputs. Mitigations applied:
- Temperature = 0.0 (deterministic)
- Explicit rubric per axis with anchor examples in prompt
- Judge model ≠ generator model (mini vs 70B)
- Fallback disclosure: every judged sample carries its `backend` tag so readers can stratify

### 3.4 No human evaluation
Phase 1 does not include human-rated RCA quality. This is called out as the single biggest open risk and is listed as Phase 3 (post-thesis) work.

## 4. System Limitations

### 4.1 Free-tier LLM constraints
Groq free tier is capped at ~30 RPM for Llama-3.3-70B-Versatile. The token-bucket limiter (`src/utils/rate_limit.py`) is set conservatively at 25 RPM. Ablation runs (4 configs × 60 items × up to 2 LLM calls = ~480 requests) therefore take ~20 minutes minimum wall-clock. This is a deployment constraint, not a methodological one, but it bounds how many experimental configurations can be run per day.

### 4.2 Single-region deployment
Everything runs locally (ChromaDB on disk, Groq API over internet). No distributed retrieval, no sharded vector store, no failover. Production scale-out is out of scope.

### 4.3 Python 3.14 + Pydantic v1
The project runs on Python 3.14; several LangChain sub-packages still emit Pydantic v1 deprecation warnings. These are non-blocking but will require a migration to Pydantic v2 before long-term support.

## 5. Research-Novelty Framing

This project is positioned as **applied systems research**, not ML-algorithm research. The contributions claimed are:

1. **GraphRAG over telecom playbooks** (Phase 2 headline): entity+relation graph extracted from public telecom-billing playbooks, used for multi-hop retrieval, evaluated against flat-vector and hybrid baselines on the 60-item GT.
2. **Honest LLM-as-Judge evaluation** with bias disclosure and statistical-significance testing (bootstrap CI + paired-bootstrap + Wilcoxon), applied to a domain (telecom billing RCA) where published benchmarks do not exist.
3. **Reproducible pipeline** with MLflow tracking, deterministic seeding, and a ≥50-test pytest suite.

We explicitly **do not** claim novelty in: the IsolationForest detector, the LangGraph agent pattern, or the ROUGE/BERTScore metrics themselves.

## 6. Threats to Validity Summary

| Threat                  | Severity | Mitigation                                                   |
|-------------------------|----------|--------------------------------------------------------------|
| Synthetic anomalies     | High     | Documented; RCA evaluation uses realistic narrative GT       |
| Single-author GT        | High     | 60-item expansion with diverse failure modes; future SME review |
| Small N                 | Medium   | Bootstrap CI + Wilcoxon + paired-bootstrap reported          |
| LLM-judge bias          | Medium   | Temp=0, cross-model judge, backend tag persisted             |
| Schema mismatch         | Medium   | Phase 1.5 Telecom Italia CDR track                           |
| Free-tier rate limits   | Low      | Token-bucket pacing; runs reproducible if slow               |
| No human eval           | High     | Acknowledged as Phase 3 future work                          |
