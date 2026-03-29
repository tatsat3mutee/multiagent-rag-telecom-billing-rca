# Telecom Billing Architecture Overview

## CDR (Call Detail Record) Processing Pipeline

### What is a CDR?
A Call Detail Record (CDR) is a data record produced by telephone exchanges, VoIP systems, or other telecommunications equipment. Each CDR contains metadata about a telecommunications transaction: who called whom, when, for how long, and through which network elements.

### CDR Processing Pipeline Stages

#### Stage 1: Collection
Network elements (switches, routers, GGSN/PGW nodes) generate raw CDR files. These are collected by mediation devices at regular intervals (typically every 5-15 minutes).

#### Stage 2: Mediation
The mediation layer:
- Normalizes CDR formats from different network elements
- Performs deduplication to remove duplicate records
- Validates mandatory fields (timestamp, MSISDN, duration, data volume)
- Enriches records with additional metadata (cell tower location, service type)
- Filters test/internal calls

#### Stage 3: Rating
The rating engine:
- Maps each CDR to the customer's active tariff plan
- Calculates charges based on usage type, duration, volume, time-of-day
- Applies discounts, bundles, promotions
- Handles overage calculations
- Produces rated CDRs with monetary amounts

#### Stage 4: Billing
The billing system:
- Aggregates rated CDRs per customer per billing cycle
- Applies recurring charges (monthly plan fees, equipment charges)
- Calculates taxes and regulatory fees
- Generates invoice
- Posts to accounts receivable

#### Stage 5: Revenue Assurance
Revenue assurance systems:
- Reconcile CDR counts between network and billing (should match ≥99.9%)
- Identify billing leakage (unbilled usage)
- Detect overbilling patterns
- Monitor KPIs: billing completeness, accuracy, timeliness

## Common Billing Anomaly Categories

### 1. Revenue Leakage Anomalies
- Zero-billing: Active services generating $0 charges
- Missing CDRs: Usage occurred but CDR not captured
- Underrating: CDR rated below correct tariff

### 2. Overbilling Anomalies
- Duplicate charges: Same usage billed multiple times
- Overrating: CDR rated above correct tariff
- Ghost charges: Charges for unused/inactive services

### 3. Service Quality Anomalies
- SLA breaches: Service metrics below guaranteed thresholds
- Usage spikes: Abnormal usage patterns indicating fraud or system error
- CDR processing failures: Pipeline failures causing data loss

## Key Performance Indicators (KPIs)

| KPI | Target | Description |
|-----|--------|-------------|
| Billing Accuracy | ≥99.5% | Percentage of correctly rated transactions |
| CDR Processing Rate | ≥99.9% | Percentage of CDRs successfully processed |
| Revenue Leakage Rate | <0.5% | Revenue lost due to billing errors |
| Bill Cycle Timeliness | 100% | Bills generated on schedule |
| Dispute Rate | <0.1% | Customer billing disputes as percentage of total bills |

## Regulatory Framework

### FCC (United States)
- Truth-in-Billing rules (47 CFR § 64.2401): Clear, non-misleading billing
- Anti-cramming rules: Prohibit unauthorized third-party charges
- Bill shock rules: Notification requirements for exceeding plan limits

### TRAI (India)
- Metering and Billing accuracy: ≥99.5%
- Maximum permissible billing error: 0.1%
- Customer right to itemized billing

### EU
- Roaming Regulation (EU 531/2012): Rate caps for roaming
- BEREC billing guidelines: Transparent charging practices
