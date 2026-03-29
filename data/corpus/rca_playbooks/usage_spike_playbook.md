# Usage Spike Anomaly — RCA Playbook

## Overview
Usage spikes occur when a customer's data, voice, or messaging usage suddenly increases by an order of magnitude (10x+) compared to their historical baseline. Not all spikes are anomalous — some represent legitimate changes in behavior.

## Common Root Causes

### 1. Fraudulent SIM Cloning / Account Compromise
**Description:** The customer's SIM has been cloned or account credentials compromised, allowing unauthorized usage under their account.
**Indicators:** Sudden 10x+ usage spike; usage from unusual geographic locations; concurrent usage sessions from different cell towers; customer reports not making those calls.
**Investigation Steps:**
1. Check usage location data — is it consistent with customer's profile?
2. Look for concurrent sessions from different locations
3. Verify SIM IMEI history for changes
4. Cross-reference with fraud detection alerts
**Resolution:** Suspend account; issue replacement SIM; reverse fraudulent charges; file fraud report.

### 2. Device Malware / Background Data Consumption
**Description:** Customer's device has malware or a malfunctioning app that generates excessive background data usage.
**Indicators:** High data usage but normal voice/SMS; usage pattern is continuous (24/7) rather than burst; customer unaware of excessive usage.
**Investigation Steps:**
1. Analyze usage pattern — is it continuous or burst?
2. Check if data usage is upload-heavy (malware indicator)
3. Verify if customer recently installed new apps
**Resolution:** Advise customer on device security; apply retroactive data cap if plan allows; consider goodwill credit.

### 3. Roaming Without Data Cap Awareness
**Description:** Customer traveling internationally incurs massive roaming data charges without realizing their plan does not include international data.
**Indicators:** Usage spike coincides with roaming events; high per-MB charge rates; geographic change in cell tower data.
**Investigation Steps:**
1. Check roaming events in CDR data
2. Verify customer's plan roaming provisions
3. Check if roaming rate notification SMS was sent
**Resolution:** Apply roaming rate cap if applicable; issue bill shock credit per regulatory requirements; advise on roaming plans.

### 4. Metering / Rating Error
**Description:** A rating engine bug multiplies usage quantities by an incorrect factor, making normal usage appear as a spike.
**Indicators:** Multiple customers on same plan show similar spike pattern; spike is exactly Nx normal (round multiplier); resolved after rating engine patch.
**Investigation Steps:**
1. Check if spike affects multiple customers on the same plan
2. Verify recent rating engine configuration changes
3. Compare raw CDR usage quantities with rated amounts
**Resolution:** Correct rating engine; reprocess affected CDRs; issue corrective billing.

### 5. Legitimate Behavior Change
**Description:** Customer genuinely changed their usage pattern (e.g., work from home, streaming habits, new household members).
**Indicators:** Spike is sustained over multiple days; usage types are diverse; no suspicious location changes.
**Investigation Steps:**
1. Check if usage pattern is sustained or one-time
2. Verify usage diversity (is it one type or multiple?)
3. Compare with similar customer segments
**Resolution:** No corrective action needed; may recommend plan upgrade; update baseline for future detection.

## Severity Assessment
- **Revenue Impact:** VARIABLE — legitimate spikes generate revenue; fraudulent spikes require reversal
- **Regulatory Risk:** MEDIUM — bill shock regulations require proactive notification
- **Customer Impact:** HIGH — unexpected large bills cause significant customer dissatisfaction

## Reference Standards
- 3GPP TS 32.240 — Usage metering and rating accuracy
- EU Roaming Regulation (EU 531/2012) — Bill shock protection
- FCC Bill Shock rules — notification requirements
- TRAI — Maximum billing threshold guidelines

## Recommended Actions
1. Cross-reference with fraud detection system
2. Check usage location and pattern for legitimacy indicators
3. Verify rating engine accuracy for affected plan
4. If fraudulent: suspend and reverse; if legitimate: recommend plan upgrade
5. Ensure bill shock notification was sent per regulatory requirements
