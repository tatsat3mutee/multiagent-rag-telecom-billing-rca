# Data Sources

## A Multi-Agent RAG System for Autonomous Root Cause Analysis of Billing Anomalies in Telecom Networks

---

## 1. Primary Structured Datasets

### 1.1 IBM Telco Customer Churn Dataset

| Property | Value |
|----------|-------|
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Format** | CSV |
| **Records** | 7,043 |
| **License** | Open / Public Domain |
| **Key Features** | `CustomerID`, `MonthlyCharges`, `TotalCharges`, `Churn`, `Contract`, `tenure`, `PaymentMethod`, `InternetService`, `PhoneService`, `TotalCharges` |
| **Use in Project** | Primary dataset for anomaly injection baseline. Real customer billing profiles used as foundation for synthetic anomaly generation. |

**Why This Dataset:**
- Real telecom CRM data with billing-relevant features
- Well-documented, widely cited in academic literature
- Contains continuous billing features ideal for anomaly injection
- 7,000+ records provide sufficient statistical power for evaluation

### 1.2 Maven Analytics Telecom Churn Dataset

| Property | Value |
|----------|-------|
| **Source** | Maven Analytics |
| **URL** | https://www.mavenanalytics.io/data-playground |
| **Format** | CSV |
| **Records** | ~7,000 |
| **License** | Open / Educational Use |
| **Key Features** | Usage metrics, data plan, customer lifetime, support calls, monthly charge |
| **Use in Project** | Cross-domain validation dataset; feature diversity; confirms methodology generalizes beyond single dataset |

**Why This Dataset:**
- Complementary features to IBM dataset
- Different provider context validates methodology generalizability
- Usage metrics enable more diverse anomaly type injection

---

## 2. Document Corpus for RAG Knowledge Base

### 2.1 Telecom Standards & SLA Documents

| Document | Source | URL | Format | Role in RAG |
|----------|--------|-----|--------|-------------|
| ETSI SLA Templates | ETSI | https://www.etsi.org | PDF | Contractual obligation extraction; penalty clause retrieval for SLA breach anomalies |
| ITU-T Billing Recommendations | ITU-T | https://www.itu.int | PDF | International billing standards; dispute resolution procedures |
| 3GPP TS 32.240 (Charging Architecture) | 3GPP | https://www.3gpp.org/specifications | PDF/HTML | CDR format specifications; billing event definitions; charging mechanism descriptions |
| ATIS Billing Standards | ATIS | https://www.atis.org | PDF | North American billing event taxonomy; billing validation rules |

### 2.2 Regulatory & Compliance Documents

| Document | Source | URL | Format | Role in RAG |
|----------|--------|-----|--------|-------------|
| FCC Billing Complaint Reports | FCC | https://www.fcc.gov/consumers/guides/filing-informal-complaint | CSV + PDF | Historical billing dispute patterns; consumer complaint classifications |
| FCC Consumer Complaint Database | FCC | https://opendata.fcc.gov | CSV | Structured complaint data with billing issue categories |
| TRAI Tariff/Billing Regulations | TRAI India | https://www.trai.gov.in | PDF | Regulatory billing constraints for Indian telecom context |
| AT&T SEC Filings (10-K) | SEC EDGAR | https://www.sec.gov/cgi-bin/browse-edgar | PDF | Revenue recognition policies; billing methodology documentation |

### 2.3 Incident Management & RCA Resources

| Document | Source | URL | Format | Role in RAG |
|----------|--------|-----|--------|-------------|
| Public Incident Postmortems | GitHub Collections | https://github.com/danluu/post-mortems | Markdown | Real-world RCA patterns; incident resolution playbooks |
| Awesome Incident Management | GitHub | https://github.com/topics/incident-management | Markdown | Incident response frameworks; escalation procedures |
| SRE Book (Google) — Postmortem Chapter | Google | https://sre.google/sre-book/ | HTML | Structured postmortem methodology; RCA best practices |

### 2.4 Synthetic RCA Playbooks (Self-Created)

| Document Type | Count | Format | Purpose |
|---------------|-------|--------|---------|
| Zero-Billing RCA Playbook | 3–5 | Markdown | Known root causes for zero-billing events, remediation steps |
| Duplicate Charge RCA Playbook | 3–5 | Markdown | CDR deduplication failure patterns, resolution procedures |
| Usage Spike RCA Playbook | 3–5 | Markdown | Legitimate vs. anomalous spike patterns, investigation steps |
| CDR Processing Failure Playbook | 3–5 | Markdown | Common CDR parsing errors, data pipeline failure modes |
| SLA Breach RCA Playbook | 3–5 | Markdown | SLA violation root causes, penalty calculation procedures |

**Total synthetic playbooks: 15–25 documents**

These playbooks serve dual purpose:
1. **Gold retrieval targets** — the RAG system should retrieve these for matching anomaly types
2. **Ground truth for evaluation** — BERTScore/ROUGE-L reference documents

---

## 3. Synthetic Anomaly Injection Methodology

### 3.1 Anomaly Types

| # | Anomaly Type | Injection Method | Prevalence | Severity |
|---|-------------|-----------------|------------|----------|
| 1 | **Zero-Billing** | Set `MonthlyCharges = 0` for random active customers with active services | 2–5% of records | HIGH |
| 2 | **Duplicate Charges** | Duplicate entire billing row with same timestamp; double `MonthlyCharges` | 1–3% of records | HIGH |
| 3 | **Usage Spike** | Multiply usage features (data/voice) by 10x for random accounts | 2–4% of records | MEDIUM |
| 4 | **CDR Processing Failure** | Introduce `NaN`/null values in critical fields (`TotalCharges`, `tenure`) | 1–2% of records | MEDIUM |
| 5 | **SLA Breach** | Generate usage patterns exceeding contract thresholds (e.g., `MonthlyCharges > contract_limit`) | 1–3% of records | HIGH |

### 3.2 Injection Parameters

```python
# All injection is seed-controlled for reproducibility
RANDOM_SEED = 42
ANOMALY_RATIOS = {
    "zero_billing": 0.03,      # 3% of records
    "duplicate_charge": 0.02,  # 2% of records
    "usage_spike": 0.03,       # 3% of records
    "cdr_failure": 0.015,      # 1.5% of records
    "sla_breach": 0.02,        # 2% of records
}
# Total anomaly rate: ~11.5% (realistic for evaluation)
```

### 3.3 Academic Justification

- Base customer profiles remain **real** (IBM/Maven) — only anomaly labels are synthetic
- Methodology follows Chandola et al. (2009) — controlled injection for benchmarking
- All injection logic is **reproducible** via seed-controlled numpy/random
- Scripts are **open-sourced** as part of project release
- Industry standard: AT&T, Verizon use synthetic data for model training in production

---

## 4. Ground Truth RCA Dataset (For Evaluation)

For each anomaly type, 3–5 **manually written reference RCA documents** are created following this schema:

```json
{
  "anomaly_id": "ZB-001",
  "anomaly_type": "zero_billing",
  "affected_pattern": "Active customer with InternetService=Fiber, MonthlyCharges suddenly 0",
  "root_cause": "Billing system failed to process CDR for current cycle. Likely cause: CDR ingestion pipeline timeout during batch processing window, resulting in zero-rated billing record.",
  "supporting_evidence": [
    "3GPP TS 32.240 Section 5.2 — CDR processing timeout handling",
    "SLA Document — Section 3.1: Billing accuracy guarantee ≥ 99.5%"
  ],
  "recommended_action": [
    "Trigger CDR reprocessing for affected account",
    "Verify CDR pipeline health for batch window",
    "Issue corrective billing statement"
  ],
  "severity": "HIGH",
  "estimated_revenue_impact": "MonthlyCharges × affected_accounts"
}
```

**Total ground truth RCAs: 15–25 documents**

---

## 5. Data Directory Structure

```
data/
├── raw/
│   ├── ibm_telco_churn.csv          # Original IBM dataset
│   └── maven_telecom_churn.csv      # Original Maven dataset
├── processed/
│   ├── anomalies_labeled.csv        # Combined dataset with injected anomalies
│   ├── train.csv                    # Training split (80%)
│   └── test.csv                     # Test split (20%)
├── corpus/
│   ├── standards/                   # ETSI, ITU-T, 3GPP PDFs
│   ├── regulatory/                  # FCC, TRAI documents
│   ├── incidents/                   # Public postmortem documents
│   └── rca_playbooks/               # Self-created RCA playbooks (15-25)
├── eval/
│   ├── ground_truth_rca/            # Reference RCA documents (15-25)
│   └── test_anomalies.csv           # 100+ test anomaly cases with ground truth
└── corpus_manifest.csv              # Master list of all corpus documents
```
