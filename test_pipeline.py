"""Quick test of the full pipeline."""
import sys
import json
sys.path.insert(0, ".")

from src.agents.graph import run_pipeline

# Test with a zero-billing anomaly
test = {
    "account_id": "CUST-00123",
    "anomaly_type": "zero_billing",
    "confidence": 0.95,
    "monthly_charges": 0.0,
    "total_charges": 2500.0,
    "tenure": 36,
    "features": {"InternetService": "Fiber optic", "Contract": "Two year"},
}
print("Running RCA pipeline...")
result = run_pipeline(test)
print(f"Status: {result.get('pipeline_status')}")
print(f"Latency: {result.get('latency_ms', 0):.0f}ms")
print(f"Docs retrieved: {result.get('retrieval_count', 0)}")
rca = result.get("rca_report", {})
if rca:
    print(f"\nSeverity: {rca.get('severity')}")
    print(f"Root Cause: {rca.get('root_cause', 'N/A')[:300]}")
    print(f"\nRecommended Actions:")
    for a in rca.get("recommended_actions", []):
        print(f"  - {a}")
    print(f"\nSummary: {rca.get('summary', 'N/A')[:300]}")
else:
    print("No RCA report generated")
    if result.get("error_message"):
        print(f"Error: {result['error_message']}")

# Test with duplicate charge
print("\n" + "="*50)
test2 = {
    "account_id": "CUST-00456",
    "anomaly_type": "duplicate_charge",
    "confidence": 0.88,
    "monthly_charges": 190.50,
    "total_charges": 3200.0,
    "tenure": 24,
    "features": {},
}
print("Running RCA for duplicate charge...")
result2 = run_pipeline(test2)
rca2 = result2.get("rca_report", {})
if rca2:
    print(f"Status: {result2.get('pipeline_status')}")
    print(f"Severity: {rca2.get('severity')}")
    print(f"Root Cause: {rca2.get('root_cause', 'N/A')[:300]}")
