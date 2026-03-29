"""
Reasoning Agent — generates structured root cause hypotheses from anomaly data + retrieved docs.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agents.state import AgentState
from src.agents.prompts import REASONER_SYSTEM_PROMPT, REASONER_PROMPT
from src.agents.llm_utils import call_llm
# LLM access is via llm_utils.call_llm


def _build_fallback_hypothesis(anomaly: dict, docs: list) -> str:
    """Generate a hypothesis without LLM using retrieved docs."""
    anomaly_type = anomaly.get("anomaly_type", "unknown")

    # Use the top document as primary evidence
    evidence = docs[0]["text"][:500] if docs else "No evidence retrieved."
    source = docs[0]["source"] if docs else "N/A"

    type_hypotheses = {
        "zero_billing": (
            "ROOT CAUSE: CDR processing pipeline failure resulting in zero-rated billing records. "
            "The billing system failed to process CDR records for the current cycle, likely due to "
            "CDR ingestion pipeline timeout or provisioning system mismatch.\n\n"
            "REASONING:\n"
            "1. Customer has active services but MonthlyCharges = $0.00\n"
            "2. This pattern matches CDR pipeline failure or provisioning sync issues\n"
            "3. The most common cause is CDR batch processing timeout\n"
            f"4. Evidence from {source} supports this diagnosis\n\n"
            "EVIDENCE:\n"
            f"- {evidence[:300]}\n"
            "- 3GPP TS 32.240 Section 5.2 — CDR processing requirements\n\n"
            "CONFIDENCE: HIGH"
        ),
        "duplicate_charge": (
            "ROOT CAUSE: CDR deduplication engine failure allowing duplicate records to pass "
            "through to the rating engine, resulting in double-billed charges.\n\n"
            "REASONING:\n"
            "1. Customer billed at approximately 2x normal monthly charges\n"
            "2. This pattern matches CDR deduplication failure or mediation re-transmission\n"
            "3. Billing system retry logic may have caused duplicate charge posting\n"
            f"4. Evidence from {source} supports this diagnosis\n\n"
            "EVIDENCE:\n"
            f"- {evidence[:300]}\n"
            "- 3GPP TS 32.240 Section 4.3 — CDR uniqueness requirements\n\n"
            "CONFIDENCE: HIGH"
        ),
        "usage_spike": (
            "ROOT CAUSE: Abnormal usage spike detected — possible causes include SIM cloning/fraud, "
            "device malware, roaming without data cap, or rating engine metering error.\n\n"
            "REASONING:\n"
            "1. Customer usage increased ~10x above historical baseline\n"
            "2. Requires investigation to differentiate fraud vs. legitimate vs. system error\n"
            "3. If multiple customers on same plan affected: likely metering error\n"
            "4. If single customer: likely fraud or device issue\n"
            f"5. Evidence from {source} supports investigation approach\n\n"
            "EVIDENCE:\n"
            f"- {evidence[:300]}\n"
            "- Usage Spike RCA Playbook investigation methodology\n\n"
            "CONFIDENCE: MEDIUM"
        ),
        "cdr_failure": (
            "ROOT CAUSE: CDR processing pipeline failure resulting in NULL/missing values in "
            "billing records. Likely caused by format mismatch after network upgrade, pipeline "
            "buffer overflow, or database storage issues.\n\n"
            "REASONING:\n"
            "1. Critical billing fields contain NULL values\n"
            "2. CDR processing pipeline failed to persist complete records\n"
            "3. Common causes: network upgrade format changes, peak traffic overflow, DB issues\n"
            f"4. Evidence from {source} provides resolution steps\n\n"
            "EVIDENCE:\n"
            f"- {evidence[:300]}\n"
            "- CDR Failure RCA Playbook analysis framework\n\n"
            "CONFIDENCE: HIGH"
        ),
        "sla_breach": (
            "ROOT CAUSE: Customer charges exceed contractual SLA threshold. Likely caused by "
            "missing billing cap configuration, plan migration error, or usage reclassification.\n\n"
            "REASONING:\n"
            "1. Monthly charges significantly exceed normal range for this plan type\n"
            "2. Contract threshold may not be applied in rating engine\n"
            "3. Recent plan migration or usage classification change may be responsible\n"
            f"4. Evidence from {source} provides diagnostic steps\n\n"
            "EVIDENCE:\n"
            f"- {evidence[:300]}\n"
            "- SLA Breach RCA Playbook investigation methodology\n\n"
            "CONFIDENCE: HIGH"
        ),
    }

    return type_hypotheses.get(anomaly_type, (
        f"ROOT CAUSE: Billing anomaly of type '{anomaly_type}' detected. "
        "Further investigation required to determine specific root cause.\n\n"
        "REASONING:\n"
        "1. Anomaly detected by billing monitoring system\n"
        f"2. Evidence from {source} provides context\n\n"
        "EVIDENCE:\n"
        f"- {evidence[:300]}\n\n"
        "CONFIDENCE: LOW"
    ))


def reasoner_node(state: AgentState) -> AgentState:
    """
    Reasoning Agent node for LangGraph.
    Receives anomaly context + retrieved docs → generates structured root cause hypothesis.
    """
    anomaly = state.get("anomaly_data", {})
    docs = state.get("retrieved_docs", [])

    # Format retrieved docs for the prompt
    docs_text = ""
    for i, doc in enumerate(docs, 1):
        docs_text += f"\n--- Document {i} (Source: {doc['source']}, Relevance: {doc['relevance_score']:.2f}) ---\n"
        docs_text += doc["text"][:800]
        docs_text += "\n"

    prompt = REASONER_PROMPT.format(
        account_id=anomaly.get("account_id", "UNKNOWN"),
        anomaly_type=anomaly.get("anomaly_type", "unknown"),
        confidence=anomaly.get("confidence", 0.0),
        monthly_charges=anomaly.get("monthly_charges", 0.0),
        total_charges=anomaly.get("total_charges", 0.0),
        tenure=anomaly.get("tenure", 0),
        retrieved_docs=docs_text if docs_text else "No relevant documents retrieved.",
    )

    # Try LLM first
    hypothesis = call_llm(REASONER_SYSTEM_PROMPT, prompt)

    # Fallback if LLM unavailable
    if not hypothesis:
        hypothesis = _build_fallback_hypothesis(anomaly, docs)

    state["hypothesis"] = hypothesis
    state["reasoning_chain"] = hypothesis
    state["pipeline_status"] = "reasoned"

    return state
