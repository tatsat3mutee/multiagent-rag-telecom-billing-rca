"""
LangGraph StateGraph orchestration for the multi-agent RCA pipeline.
Wires Investigator → Reasoner → Reporter as graph nodes.
"""
import time
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.investigator import investigator_node
from src.agents.reasoner import reasoner_node
from src.agents.reporter import reporter_node
from config import TOP_K


def should_retry_retrieval(state: AgentState) -> str:
    """Conditional routing: retry with broader query if insufficient docs retrieved."""
    retrieval_count = state.get("retrieval_count", 0)
    if retrieval_count < 2:
        return "broaden_query"
    return "proceed"


def broaden_query_node(state: AgentState) -> AgentState:
    """Broadens the retrieval query and re-searches."""
    anomaly = state.get("anomaly_data", {})
    anomaly_type = anomaly.get("anomaly_type", "billing anomaly")

    # Use a broader fallback query
    state["retrieval_query"] = (
        f"telecom billing anomaly root cause analysis investigation "
        f"{anomaly_type} CDR processing failure resolution incident management"
    )

    from src.rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    results = kb.search(state["retrieval_query"], n_results=TOP_K)

    retrieved_docs = []
    for r in results:
        retrieved_docs.append({
            "text": r["text"],
            "source": r["source"],
            "relevance_score": r["relevance_score"],
            "metadata": r["metadata"],
        })

    state["retrieved_docs"] = retrieved_docs
    state["retrieval_count"] = len(retrieved_docs)

    return state


def build_graph() -> StateGraph:
    """Build the LangGraph StateGraph for the multi-agent pipeline."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("broaden_query", broaden_query_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("reporter", reporter_node)

    # Set entry point
    workflow.set_entry_point("investigator")

    # Add conditional edge after investigator
    workflow.add_conditional_edges(
        "investigator",
        should_retry_retrieval,
        {
            "broaden_query": "broaden_query",
            "proceed": "reasoner",
        },
    )

    # Broaden query always goes to reasoner
    workflow.add_edge("broaden_query", "reasoner")

    # Reasoner → Reporter → END
    workflow.add_edge("reasoner", "reporter")
    workflow.add_edge("reporter", END)

    return workflow.compile()


def run_pipeline(anomaly_record: dict) -> dict:
    """
    Run the complete multi-agent RCA pipeline for a single anomaly.

    Args:
        anomaly_record: dict with keys: account_id, anomaly_type, confidence,
                       monthly_charges, total_charges, tenure, features

    Returns:
        Complete agent state including rca_report.
    """
    graph = build_graph()

    initial_state: AgentState = {
        "anomaly_data": anomaly_record,
        "retrieved_docs": [],
        "retrieval_count": 0,
        "pipeline_status": "started",
    }

    start_time = time.time()

    try:
        result = graph.invoke(initial_state)
        result["latency_ms"] = (time.time() - start_time) * 1000
    except Exception as e:
        result = initial_state.copy()
        result["pipeline_status"] = "error"
        result["error_message"] = str(e)
        result["latency_ms"] = (time.time() - start_time) * 1000

    return result


def run_batch_pipeline(anomaly_records: List[dict]) -> List[dict]:
    """Run the pipeline for a batch of anomalies."""
    results = []
    for i, record in enumerate(anomaly_records):
        print(f"Processing anomaly {i+1}/{len(anomaly_records)}: {record.get('account_id', 'N/A')} ({record.get('anomaly_type', 'unknown')})")
        result = run_pipeline(record)
        results.append(result)
    return results


if __name__ == "__main__":
    # Test with a sample anomaly
    test_anomaly = {
        "account_id": "CUST-00123",
        "anomaly_type": "zero_billing",
        "confidence": 0.95,
        "monthly_charges": 0.0,
        "total_charges": 2500.0,
        "tenure": 36,
        "features": {"InternetService": "Fiber optic", "Contract": "Two year"},
    }

    print("Running multi-agent RCA pipeline...")
    result = run_pipeline(test_anomaly)

    print(f"\nPipeline Status: {result.get('pipeline_status')}")
    print(f"Latency: {result.get('latency_ms', 0):.0f}ms")
    print(f"Retrieved Docs: {result.get('retrieval_count', 0)}")

    rca = result.get("rca_report", {})
    if rca:
        import json
        print(f"\nRCA Report:")
        print(json.dumps(rca, indent=2))
