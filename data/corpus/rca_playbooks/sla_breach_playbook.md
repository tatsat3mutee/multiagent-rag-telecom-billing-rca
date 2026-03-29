# SLA Breach Anomaly — RCA Playbook

## Overview
SLA (Service Level Agreement) breach occurs when a customer's actual billing exceeds the contractual thresholds defined in their service agreement, or when service quality metrics fall below guaranteed levels. Billing-related SLA breaches typically involve charges exceeding contract caps or guaranteed rate violations.

## Common Root Causes

### 1. Contract Threshold Not Applied in Rating Engine
**Description:** The customer's contractual billing cap or rate ceiling was not properly configured in the rating engine, allowing charges to exceed the agreed-upon maximum.
**Indicators:** Charges consistently above contract cap; other customers on same plan are correctly capped; configuration gap between CRM contract and billing system.
**Investigation Steps:**
1. Pull customer's contract details from CRM — identify billing cap and rate terms
2. Check rating engine configuration for this customer/plan
3. Compare contract terms with rating rules
4. Verify when the configuration divergence started
**Resolution:** Correct rating engine configuration; reprocess bills from divergence date; issue refund for overcharges.

### 2. Plan Migration Error
**Description:** During a plan migration or upgrade, the old plan's billing rules were deactivated before the new plan's rules were fully configured, resulting in uncapped billing.
**Indicators:** SLA breach starts immediately after plan change event; gap between old plan end and new plan start; default or fallback rates applied.
**Investigation Steps:**
1. Check customer's plan change history
2. Verify plan migration workflow execution
3. Check if rollback/fallback rates are higher than contract rates
4. Review plan migration testing checklist
**Resolution:** Apply correct plan retroactively; refund excess charges; fix migration workflow.

### 3. Usage Type Reclassification
**Description:** A change in usage classification rules causes previously included usage (within plan) to be reclassified as out-of-plan or premium usage, incurring additional charges.
**Indicators:** Charges from specific usage types that were previously included; rule change event correlates with breach; affects multiple customers on similar plans.
**Investigation Steps:**
1. Review recent changes to usage classification rules
2. Compare pre/post classification for affected usage types
3. Verify if change was approved via change management
4. Assess the scope of affected customers
**Resolution:** Revert incorrect classification; reprocess affected bills; communicate with affected customers.

### 4. Overage Charge Calculation Error
**Description:** The overage (excess usage) charge calculation has a bug, applying per-unit charges that are higher than contractually agreed overage rates.
**Indicators:** Overage rates don't match contract; specific calculation formula produces incorrect results at certain usage levels; affects customers who exceed base plan limits.
**Investigation Steps:**
1. Compare billed overage rate with contractual rate
2. Test overage calculation formula with known inputs
3. Check for recent changes to overage calculation logic
4. Verify if rounding/precision errors contribute
**Resolution:** Fix calculation logic; reprocess overage charges; issue credits.

### 5. Third-Party / Premium Service Charges Not Excluded
**Description:** Third-party or premium service charges (e.g., content subscriptions, premium SMS) are being counted toward the plan usage when they should be billed separately, causing the combined total to breach the SLA cap.
**Indicators:** SLA cap comparison includes third-party charges; contract specifies cap applies only to core services; premium services billed through same account.
**Investigation Steps:**
1. Break down charges by service type (core vs. premium vs. third-party)
2. Review contract language on what's included in the cap
3. Check billing system's treatment of third-party charges
**Resolution:** Separate third-party charges from SLA cap calculation; adjust bill accordingly.

## Severity Assessment
- **Revenue Impact:** MEDIUM — Overcharges require refund; undercharges mean revenue leakage
- **Regulatory Risk:** HIGH — SLA violations can trigger regulatory penalties, especially for government/enterprise contracts
- **Customer Impact:** HIGH — Breach of contractual terms damages trust; enterprise customers may invoke penalty clauses

## SLA Penalty Calculation
Per standard telecom SLA frameworks:
- **Category A (Core Service):** Penalty = (breach amount) × penalty multiplier (typically 2x–5x)
- **Category B (Support/Quality):** Service credit = monthly charge × downtime percentage
- Penalty caps typically at 30% of monthly contract value

## Reference Standards
- ETSI GS NFV-REL 001 — Network service reliability requirements
- ITU-T E.860 — SLA framework for telecom services
- 3GPP TS 32.240 Section 8 — Charging architecture SLA compliance
- TRAI Quality of Service regulations
- FCC — Service quality and billing obligations

## Recommended Actions
1. Immediately verify customer's contract terms and billing cap
2. Compare actual charges against contractual threshold
3. Identify root cause: configuration error vs. usage reclassification vs. calculation error
4. Issue corrective billing and penalties/credits as per contract terms
5. Conduct root cause fix to prevent recurrence
6. Notify account management team for relationship preservation
7. File compliance report if regulatory reporting threshold is triggered
