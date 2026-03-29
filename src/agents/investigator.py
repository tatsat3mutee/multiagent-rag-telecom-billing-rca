"""
Investigator Agent — retrieves relevant documents from the RAG knowledge base.
"""
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agents.state import AgentState
from src.agents.prompts import INVESTIGATOR_SYSTEM_PROMPT, INVESTIGATOR_PROMPT
from src.agents.llm_utils import call_llm
from src.rag.knowledge_base import KnowledgeBase
from config import TOP_K


def investigator_node(state: AgentState) -> AgentState:
    """
    Investigator Agent node for LangGraph.
    Receives anomaly context → formulates query → retrieves top-k docs from ChromaDB.
    """
    anomaly = state.get("anomaly_data", {})

    # Format the prompt
    prompt = INVESTIGATOR_PROMPT.format(
        account_id=anomaly.get("account_id", "UNKNOWN"),
        anomaly_type=anomaly.get("anomaly_type", "unknown"),
        confidence=anomaly.get("confidence", 0.0),
        monthly_charges=anomaly.get("monthly_charges", 0.0),
        total_charges=anomaly.get("total_charges", 0.0),
        tenure=anomaly.get("tenure", 0),
        features=anomaly.get("features", {}),
    )

    # Try to get a refined query from the LLM
    llm_query = call_llm(INVESTIGATOR_SYSTEM_PROMPT, prompt)

    # Build search query — use LLM query if available, otherwise fallback
    anomaly_type = anomaly.get("anomaly_type", "billing anomaly")
    if llm_query:
        search_query = llm_query.strip()
    else:
        # Fallback query based on anomaly type
        query_map = {
            "zero_billing": "zero billing anomaly root cause CDR processing failure provisioning mismatch",
            "duplicate_charge": "duplicate charge billing deduplication failure retry logic error",
            "usage_spike": "usage spike anomaly fraud SIM cloning metering error roaming",
            "cdr_failure": "CDR processing failure data pipeline NULL records format mismatch",
            "sla_breach": "SLA breach contract threshold billing cap exceeded rate violation",
        }
        search_query = query_map.get(anomaly_type,
                                      f"{anomaly_type} billing anomaly root cause analysis telecom")

    # Query knowledge base
    kb = KnowledgeBase()
    results = kb.search(search_query, n_results=TOP_K)

    retrieved_docs = []
    for r in results:
        retrieved_docs.append({
            "text": r["text"],
            "source": r["source"],
            "relevance_score": r["relevance_score"],
            "metadata": r["metadata"],
        })

    state["retrieval_query"] = search_query
    state["retrieved_docs"] = retrieved_docs
    state["retrieval_count"] = len(retrieved_docs)
    state["pipeline_status"] = "investigated"

    return state
