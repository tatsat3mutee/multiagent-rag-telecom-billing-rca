# Duplicate Charge Anomaly — RCA Playbook

## Overview
Duplicate charges occur when a customer is billed more than once for the same service usage within a billing cycle. This leads to overbilling complaints and regulatory risk.

## Common Root Causes

### 1. CDR Deduplication Engine Failure
**Description:** The CDR deduplication process, which normally filters out duplicate records from network elements, fails or is bypassed during processing.
**Indicators:** Exact duplicate CDR records with identical timestamps, same calling/called party, same duration. Multiple duplicate events in the same batch.
**Investigation Steps:**
1. Check deduplication engine logs for error/failure events
2. Verify deduplication rules are active and correctly configured
3. Identify the CDR source (which network element produced duplicates)
4. Check if duplicates originated from the mediation device or billing system
**Resolution:** Remove duplicate CDRs; restart deduplication engine; reprocess affected billing records.

### 2. Mediation Device Re-transmission
**Description:** The mediation device between the network and billing system re-transmits CDRs due to acknowledgment timeout, but the original CDRs were already processed.
**Indicators:** CDR records with same content but slightly different ingestion timestamps; pattern correlates with network congestion periods.
**Investigation Steps:**
1. Check mediation device re-transmission logs
2. Verify acknowledgment (ACK) protocol between mediation and billing
3. Check network congestion reports for the affected time window
**Resolution:** Implement idempotent CDR processing; fix ACK timeout configuration.

### 3. Billing System Retry Logic Error
**Description:** The billing system's retry mechanism for failed transactions duplicates the charge when the original transaction actually succeeded but the confirmation was lost.
**Indicators:** Two identical charge entries with slightly different processing timestamps; second entry appears after system timeout period.
**Investigation Steps:**
1. Review billing system transaction logs for retry events
2. Check for transaction ID uniqueness enforcement
3. Verify retry logic configuration and timeout settings
**Resolution:** Implement transaction-level idempotency; add unique transaction ID checks before charge posting.

### 4. Cross-System Synchronization Failure
**Description:** When multiple billing systems handle different service types (voice, data, VAS), synchronization failures can cause the same usage to be billed by multiple systems.
**Indicators:** Duplicate charges from different billing system IDs for the same usage event; typically affects bundled service plans.
**Investigation Steps:**
1. Check cross-system billing reconciliation reports
2. Verify service-to-system routing rules
3. Audit recent changes to billing system orchestration
**Resolution:** Fix routing rules; reconcile and reverse duplicate charges; strengthen inter-system deduplication.

## Severity Assessment
- **Revenue Impact:** HIGH — Overbilling leads to customer complaints, chargebacks, and regulatory fines
- **Regulatory Risk:** HIGH — FCC/TRAI regulations mandate billing accuracy; systematic duplicates can trigger investigation
- **Customer Impact:** HIGH — Direct financial harm to customers; trust erosion

## Reference Standards
- 3GPP TS 32.240 Section 4.3 — CDR uniqueness requirements
- 3GPP TS 32.297 — CDR file transfer and deduplication
- FCC Truth-in-Billing rules (47 CFR § 64.2401)
- TRAI billing accuracy guidelines

## Recommended Actions
1. Immediately reverse duplicate charges for affected customers
2. Audit and repair CDR deduplication engine
3. Implement transaction-level idempotency in billing system
4. Add real-time monitoring for duplicate charge rate
5. File regulatory compliance report if threshold exceeded
