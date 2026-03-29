# Zero-Billing Anomaly — RCA Playbook

## Overview
Zero-billing events occur when an active customer's billing record shows $0 charges despite having active services. This is a critical revenue assurance issue.

## Common Root Causes

### 1. CDR Ingestion Pipeline Timeout
**Description:** The CDR (Call Detail Record) processing pipeline times out during the batch processing window, resulting in zero-rated billing records.
**Indicators:** Multiple zero-billing records appearing in the same billing cycle; correlation with batch processing timestamps.
**Investigation Steps:**
1. Check CDR pipeline health logs for the affected billing cycle
2. Verify batch processing window completion status
3. Cross-reference with network element CDR exports
**Resolution:** Trigger CDR reprocessing for affected accounts; verify pipeline health.

### 2. Provisioning System Mismatch
**Description:** Customer was provisioned for services but the billing system was not updated. The service activation event did not propagate to the billing platform.
**Indicators:** Customer has active services in CRM but billing record shows no usage.
**Investigation Steps:**
1. Compare CRM service status with billing system records
2. Check provisioning event queue for failed/stuck messages
3. Verify service activation date vs. billing cycle start date
**Resolution:** Resync provisioning system with billing; issue corrective bill.

### 3. Rating Engine Configuration Error
**Description:** The rating engine has an incorrect tariff plan configuration, rating certain service usage at $0.
**Indicators:** Specific service type always rated at $0; pattern across multiple customers on same plan.
**Investigation Steps:**
1. Review tariff plan configuration for affected service type
2. Compare with approved rate card
3. Check recent configuration changes to rating engine
**Resolution:** Correct tariff configuration; reprocess affected CDRs.

### 4. Billing Cycle Boundary Error
**Description:** CDRs fall between two billing cycles due to timezone or boundary calculation errors, resulting in records not being picked up by either cycle.
**Indicators:** Zero billing at cycle boundaries; usage appears in next cycle's records.
**Investigation Steps:**
1. Check CDR timestamps vs. billing cycle boundaries
2. Verify timezone handling in CDR processor
3. Look for duplicate billing in adjacent cycles
**Resolution:** Adjust billing cycle boundary logic; reprocess missed CDRs.

## Severity Assessment
- **Revenue Impact:** HIGH — Direct revenue leakage per affected account
- **Regulatory Risk:** MEDIUM — May violate billing accuracy SLA requirements (typically ≥99.5%)
- **Customer Impact:** LOW (customer benefits from zero billing) but HIGH for provider revenue

## Reference Standards
- 3GPP TS 32.240 Section 5.2 — CDR processing and timeout handling
- ETSI billing accuracy requirements — minimum 99.5% accuracy threshold
- ITU-T E.800 — Quality of service metrics for billing

## Recommended Actions
1. Trigger CDR reprocessing for all affected accounts
2. Verify CDR pipeline end-to-end health
3. Issue corrective billing statements
4. Add monitoring alert for zero-billing anomaly rate exceeding 0.1%
