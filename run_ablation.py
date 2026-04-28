"""
Ablation Study — Compares 5 configurations:
  Config A: No RAG (direct LLM generation)
  Config B: RAG only (retrieve + generate in single prompt)
  Config C: Single Agent + RAG (one agent does everything)
    Config D: Multi-Agent + RAG (proposed 4-agent pipeline)
  Config E: Multi-Agent + GraphRAG (headline novelty — graph-traversal retrieval)
            Requires: data/graph_rag/kb_graph.pkl
            Build with: python scripts/build_graph_rag.py --offline

Usage: python run_ablation.py [--n N] [--configs A,B,C,D,E] [--gt]
   --n N        limit to N anomalies per type (default 12 when --gt, else 3)
   --configs    subset of configs to run (default A,B,C,D)
   --gt         build test anomalies from the 60-item ground truth
                (enables per-item ground_truth_id matching for fair eval)
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Fix Python 3.14 + transformers import hang on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    LLM_MODEL, PROCESSED_DATA_DIR,
    TOP_K, ABLATION_CONFIGS, GROUND_TRUTH_DIR,
)
from config import LLM_API_KEY, LLM_BASE_URL
from src.utils.rate_limit import get_limiter

# Pre-import to avoid repeated lazy-import overhead inside loops
from openai import OpenAI


# ── Shared LLM helper ──

def _build_client():
    if not LLM_API_KEY:
        return None
    kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


_client = _build_client()
_limiter = get_limiter()


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the configured OpenAI-compatible LLM (OpenAI / Kimi / MiniMax / Groq / etc.)
    with token-bucket pacing + retry on rate limits."""
    if _client is None:
        raise RuntimeError(
            "No LLM API key set — configure GROQ_API_KEY or KIMI_API_KEY in .env "
            "before running the ablation."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    max_retries = 5
    for attempt in range(max_retries):
        _limiter.acquire()  # proactive pacing
        try:
            resp = _client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.1,
                messages=messages,
                timeout=60,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "timeout" in msg:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f" [rate-limit, retry in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit exceeded after max retries")


def parse_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


# ── Config A: No RAG — Direct LLM ──

def run_config_a(anomaly: dict) -> dict:
    """Direct LLM generation — no RAG, no agents."""
    system_prompt = (
        "You are a telecom billing anomaly expert. Analyze the anomaly and produce a JSON RCA report. "
        "You must respond ONLY with a valid JSON object with these keys: "
        "anomaly_id, anomaly_type, root_cause, supporting_evidence (list), "
        "recommended_actions (list), severity (HIGH/MEDIUM/LOW), confidence_score (0-1), summary."
    )
    user_prompt = (
        f"Analyze this billing anomaly:\n"
        f"- Account: {anomaly['account_id']}\n"
        f"- Type: {anomaly['anomaly_type']}\n"
        f"- Detection Confidence: {anomaly['confidence']}\n"
        f"- Monthly Charges: ${anomaly['monthly_charges']:.2f}\n"
        f"- Total Charges: ${anomaly['total_charges']:.2f}\n"
        f"- Tenure: {anomaly['tenure']} months\n\n"
        f"Generate a comprehensive root cause analysis report as JSON."
    )

    start = time.time()
    try:
        response = call_llm(system_prompt, user_prompt)
        rca = parse_json_from_llm(response)
    except Exception as e:
        rca = {
            "anomaly_id": anomaly["account_id"],
            "anomaly_type": anomaly["anomaly_type"],
            "root_cause": f"LLM generation failed: {e}",
            "supporting_evidence": [],
            "recommended_actions": [],
            "severity": "UNKNOWN",
            "confidence_score": 0.0,
            "summary": "Failed to generate RCA.",
        }
    latency = (time.time() - start) * 1000

    return {
        "anomaly_data": anomaly,
        "rca_report": rca,
        "pipeline_status": "completed",
        "latency_ms": latency,
        "retrieval_count": 0,
        "retrieved_docs": [],
        "config": "no_rag",
    }


# ── Config B: RAG Only — Retrieve + single-prompt generation ──

def run_config_b(anomaly: dict) -> dict:
    """RAG + LLM in a single prompt — no agent decomposition."""
    from src.rag.knowledge_base import KnowledgeBase

    start = time.time()
    # Retrieve
    kb = KnowledgeBase()
    query = f"{anomaly['anomaly_type']} billing anomaly root cause telecom CDR"
    results = kb.search(query, n_results=TOP_K)

    docs_text = ""
    for i, r in enumerate(results, 1):
        docs_text += f"\n--- Document {i} (Source: {r['source']}) ---\n{r['text'][:600]}\n"

    system_prompt = (
        "You are a telecom billing anomaly expert. You are given retrieved knowledge base documents "
        "and an anomaly to analyze. "
        "Use the provided documents as evidence to produce a JSON RCA report. "
        "You must respond ONLY with a valid JSON object with these keys: "
        "anomaly_id, anomaly_type, root_cause, supporting_evidence (list), "
        "recommended_actions (list), severity (HIGH/MEDIUM/LOW), confidence_score (0-1), summary."
    )
    user_prompt = (
        f"Analyze this billing anomaly using the retrieved documents:\n\n"
        f"ANOMALY:\n"
        f"- Account: {anomaly['account_id']}\n"
        f"- Type: {anomaly['anomaly_type']}\n"
        f"- Detection Confidence: {anomaly['confidence']}\n"
        f"- Monthly Charges: ${anomaly['monthly_charges']:.2f}\n"
        f"- Total Charges: ${anomaly['total_charges']:.2f}\n"
        f"- Tenure: {anomaly['tenure']} months\n\n"
        f"RETRIEVED DOCUMENTS:\n{docs_text}\n\n"
        f"Generate a comprehensive root cause analysis report as JSON, grounded in the retrieved documents."
    )

    try:
        response = call_llm(system_prompt, user_prompt)
        rca = parse_json_from_llm(response)
    except Exception as e:
        rca = {
            "anomaly_id": anomaly["account_id"],
            "anomaly_type": anomaly["anomaly_type"],
            "root_cause": f"LLM generation failed: {e}",
            "supporting_evidence": [],
            "recommended_actions": [],
            "severity": "UNKNOWN",
            "confidence_score": 0.0,
            "summary": "Failed to generate RCA.",
        }
    latency = (time.time() - start) * 1000

    retrieved_docs = [{"text": r["text"], "source": r["source"],
                       "relevance_score": r["relevance_score"], "metadata": r["metadata"]}
                      for r in results]

    return {
        "anomaly_data": anomaly,
        "rca_report": rca,
        "pipeline_status": "completed",
        "latency_ms": latency,
        "retrieval_count": len(results),
        "retrieved_docs": retrieved_docs,
        "config": "rag_only",
    }


# ── Config C: Single Agent + RAG ──

def run_config_c(anomaly: dict) -> dict:
    """Single agent that does retrieval + reasoning + reporting in one pass."""
    from src.rag.knowledge_base import KnowledgeBase

    start = time.time()
    # Step 1: Retrieval with LLM-refined query
    kb = KnowledgeBase()

    refine_prompt = (
        f"You are a telecom billing investigation agent. Given this anomaly, output ONLY a concise "
        f"search query (one line, no explanation) to find relevant root cause documents:\n"
        f"Type: {anomaly['anomaly_type']}, Charges: ${anomaly['monthly_charges']:.2f}, "
        f"Tenure: {anomaly['tenure']} months"
    )
    try:
        refined_query = call_llm("You output only search queries, nothing else.", refine_prompt).strip()
    except Exception:
        refined_query = f"{anomaly['anomaly_type']} billing anomaly root cause telecom"

    results = kb.search(refined_query, n_results=TOP_K)

    docs_text = ""
    for i, r in enumerate(results, 1):
        docs_text += f"\n--- Document {i} (Source: {r['source']}) ---\n{r['text'][:600]}\n"

    # Step 2: Single-agent analysis + report generation
    system_prompt = (
        "You are an expert telecom billing RCA analyst. Perform the full investigation:\n"
        "1. Analyze the anomaly and retrieved evidence\n"
        "2. Determine the root cause with structured reasoning\n"
        "3. Produce a JSON RCA report\n\n"
        "You must respond ONLY with a valid JSON object with these keys: "
        "anomaly_id, anomaly_type, root_cause, supporting_evidence (list), "
        "recommended_actions (list), severity (HIGH/MEDIUM/LOW), confidence_score (0-1), summary."
    )
    user_prompt = (
        f"ANOMALY:\n"
        f"- Account: {anomaly['account_id']}\n"
        f"- Type: {anomaly['anomaly_type']}\n"
        f"- Detection Confidence: {anomaly['confidence']}\n"
        f"- Monthly Charges: ${anomaly['monthly_charges']:.2f}\n"
        f"- Total Charges: ${anomaly['total_charges']:.2f}\n"
        f"- Tenure: {anomaly['tenure']} months\n\n"
        f"RETRIEVED DOCUMENTS:\n{docs_text}\n\n"
        f"Perform full root cause analysis. Reason step-by-step, then produce the JSON report."
    )

    try:
        response = call_llm(system_prompt, user_prompt)
        rca = parse_json_from_llm(response)
    except Exception as e:
        rca = {
            "anomaly_id": anomaly["account_id"],
            "anomaly_type": anomaly["anomaly_type"],
            "root_cause": f"LLM generation failed: {e}",
            "supporting_evidence": [],
            "recommended_actions": [],
            "severity": "UNKNOWN",
            "confidence_score": 0.0,
            "summary": "Failed to generate RCA.",
        }
    latency = (time.time() - start) * 1000

    retrieved_docs = [{"text": r["text"], "source": r["source"],
                       "relevance_score": r["relevance_score"], "metadata": r["metadata"]}
                      for r in results]

    return {
        "anomaly_data": anomaly,
        "rca_report": rca,
        "pipeline_status": "completed",
        "latency_ms": latency,
        "retrieval_count": len(results),
        "retrieved_docs": retrieved_docs,
        "config": "single_agent_rag",
    }


# ── Config D: Multi-Agent + RAG (proposed system) ──

def run_config_d(anomaly: dict) -> dict:
    """Full multi-agent pipeline: Investigator → Reasoner → Critic → Reporter."""
    import time as _time
    from src.agents.graph import run_pipeline

    start = _time.time()
    try:
        result = run_pipeline(anomaly)
    except Exception as e:
        latency = (_time.time() - start) * 1000
        result = {
            "anomaly_data": anomaly,
            "rca_report": {
                "anomaly_id": anomaly["account_id"],
                "anomaly_type": anomaly["anomaly_type"],
                "root_cause": f"Multi-agent pipeline error: {e}",
                "supporting_evidence": [],
                "recommended_actions": [],
                "severity": "UNKNOWN",
                "confidence_score": 0.0,
                "summary": "Pipeline failed due to LLM timeout or error.",
            },
            "pipeline_status": "completed",
            "latency_ms": latency,
            "retrieval_count": 0,
            "retrieved_docs": [],
        }
    result["config"] = "multi_agent_rag"
    return result


def run_config_e(anomaly: dict) -> dict:
    """Multi-Agent + GraphRAG retrieval (Config E — headline novelty).

    Identical to Config D (full multi-agent pipeline) except the Investigator
    uses GraphRAG entity-relation graph traversal instead of ChromaDB dense
    retrieval.  This isolates the contribution of GraphRAG over vector search
    in an otherwise identical pipeline.

    Requires that the graph has been built first:
      python scripts/build_graph_rag.py --offline   # fast, heuristic extraction
      python scripts/build_graph_rag.py             # LLM-based extraction (better)
    """
    from src.rag.graph_rag import GRAPH_PATH
    if not GRAPH_PATH.exists():
        raise RuntimeError(
            "GraphRAG graph not built. Run one of:\n"
            "  python scripts/build_graph_rag.py --offline\n"
            "  python scripts/build_graph_rag.py\n"
            f"Expected file: {GRAPH_PATH}"
        )
    os.environ["USE_GRAPH_RAG"] = "1"
    try:
        result = run_config_d(anomaly)
    finally:
        os.environ.pop("USE_GRAPH_RAG", None)
    result["config"] = "graph_rag"
    return result


# ── Test Data ──

def get_test_anomalies() -> List[dict]:
    """Get 15 test anomalies (3 per type) with varied parameters for robust evaluation."""
    anomalies = []

    # Zero Billing — 3 variants
    for i, (tenure, total, contract, isp) in enumerate([
        (36, 2500.0, "Two year", "Fiber optic"),
        (6, 450.0, "Month-to-month", "DSL"),
        (60, 5800.0, "Two year", "Fiber optic"),
    ], 1):
        anomalies.append({
            "account_id": f"ABL-ZERO-{i:03d}",
            "anomaly_type": "zero_billing",
            "confidence": round(0.85 + i * 0.04, 2),
            "monthly_charges": 0.0,
            "total_charges": total,
            "tenure": tenure,
            "features": {"InternetService": isp, "Contract": contract},
        })

    # Duplicate Charge — 3 variants
    for i, (charges, tenure, total, contract) in enumerate([
        (159.90, 24, 3200.0, "One year"),
        (89.50, 12, 1100.0, "Month-to-month"),
        (210.40, 48, 9500.0, "Two year"),
    ], 1):
        anomalies.append({
            "account_id": f"ABL-DUP-{i:03d}",
            "anomaly_type": "duplicate_charge",
            "confidence": round(0.82 + i * 0.05, 2),
            "monthly_charges": charges,
            "total_charges": total,
            "tenure": tenure,
            "features": {"InternetService": "Fiber optic", "Contract": contract},
        })

    # Usage Spike — 3 variants
    for i, (charges, tenure, total, isp) in enumerate([
        (850.0, 18, 4500.0, "Fiber optic"),
        (620.0, 8, 1200.0, "DSL"),
        (1100.0, 30, 7200.0, "Fiber optic"),
    ], 1):
        anomalies.append({
            "account_id": f"ABL-SPIKE-{i:03d}",
            "anomaly_type": "usage_spike",
            "confidence": round(0.88 + i * 0.03, 2),
            "monthly_charges": charges,
            "total_charges": total,
            "tenure": tenure,
            "features": {"InternetService": isp, "Contract": "Month-to-month"},
        })

    # CDR Failure — 3 variants
    for i, (tenure, contract, isp) in enumerate([
        (12, "Month-to-month", "DSL"),
        (24, "One year", "Fiber optic"),
        (3, "Month-to-month", "DSL"),
    ], 1):
        anomalies.append({
            "account_id": f"ABL-CDR-{i:03d}",
            "anomaly_type": "cdr_failure",
            "confidence": round(0.86 + i * 0.04, 2),
            "monthly_charges": 0.0,
            "total_charges": 0.0,
            "tenure": tenure,
            "features": {"InternetService": isp, "Contract": contract},
        })

    # SLA Breach — 3 variants
    for i, (charges, tenure, total, contract) in enumerate([
        (250.0, 48, 6000.0, "Two year"),
        (180.0, 24, 3500.0, "One year"),
        (320.0, 60, 10500.0, "Two year"),
    ], 1):
        anomalies.append({
            "account_id": f"ABL-SLA-{i:03d}",
            "anomaly_type": "sla_breach",
            "confidence": round(0.80 + i * 0.05, 2),
            "monthly_charges": charges,
            "total_charges": total,
            "tenure": tenure,
            "features": {"InternetService": "Fiber optic", "Contract": contract},
        })

    return anomalies


# ── Evaluation ──

def evaluate_config_results(results: List[dict]) -> dict:
    """Evaluate a single config's results with comprehensive metrics."""
    from src.evaluation.metrics import (
        compute_rouge_l, compute_bert_score, anomaly_type_match, load_ground_truth,
    )

    gt_list = load_ground_truth()
    gt_lookup = {gt["anomaly_type"]: gt for gt in gt_list}

    metrics = {
        "total": len(results),
        "successful": sum(1 for r in results if r.get("pipeline_status") == "completed"),
        "avg_latency_ms": np.mean([r.get("latency_ms", 0) for r in results]),
        "avg_retrieval_count": np.mean([r.get("retrieval_count", 0) for r in results]),
    }

    # Type accuracy
    type_matches = 0
    total_typed = 0
    for r in results:
        rca = r.get("rca_report", {})
        pred_type = rca.get("anomaly_type", "")
        true_type = r.get("anomaly_data", {}).get("anomaly_type", "")
        if true_type:
            total_typed += 1
            if anomaly_type_match(pred_type, true_type):
                type_matches += 1
    metrics["type_accuracy"] = type_matches / max(total_typed, 1)

    # Retrieval quality: avg relevance score of retrieved docs
    all_relevance_scores = []
    for r in results:
        docs = r.get("retrieved_docs", [])
        for doc in docs:
            score = doc.get("relevance_score", 0)
            all_relevance_scores.append(score)
    metrics["avg_retrieval_relevance"] = float(np.mean(all_relevance_scores)) if all_relevance_scores else 0.0

    # ROUGE-L & BERTScore (per-sample for statistical analysis)
    hypotheses = []
    references = []
    rouge_per_sample = []
    for r in results:
        rca = r.get("rca_report", {})
        atype = r.get("anomaly_data", {}).get("anomaly_type", "")
        if atype in gt_lookup:
            hyp = rca.get("root_cause", "")
            ref = gt_lookup[atype].get("root_cause", "")
            if hyp and ref:
                hypotheses.append(hyp)
                references.append(ref)
                rouge = compute_rouge_l(hyp, ref)
                rouge_per_sample.append(rouge["fmeasure"])

    if hypotheses:
        metrics["rouge_l_f1"] = float(np.mean(rouge_per_sample))
        metrics["rouge_l_std"] = float(np.std(rouge_per_sample))
        metrics["rouge_l_per_sample"] = rouge_per_sample  # Keep for significance testing
        bert = compute_bert_score(hypotheses, references)
        metrics["bert_score_f1"] = float(bert["f1"])
        metrics["bert_per_sample"] = bert.get("individual_f1", [])
    else:
        metrics["rouge_l_f1"] = 0.0
        metrics["rouge_l_std"] = 0.0
        metrics["rouge_l_per_sample"] = []
        metrics["bert_score_f1"] = 0.0
        metrics["bert_per_sample"] = []

    # Per-anomaly-type breakdown
    type_metrics = {}
    for atype in ["zero_billing", "duplicate_charge", "usage_spike", "cdr_failure", "sla_breach"]:
        type_results = [r for r in results if r.get("anomaly_data", {}).get("anomaly_type") == atype]
        if type_results:
            type_rouge = []
            for r in type_results:
                rca = r.get("rca_report", {})
                hyp = rca.get("root_cause", "")
                ref = gt_lookup.get(atype, {}).get("root_cause", "")
                if hyp and ref:
                    type_rouge.append(compute_rouge_l(hyp, ref)["fmeasure"])
            type_metrics[atype] = {
                "count": len(type_results),
                "rouge_l_f1": float(np.mean(type_rouge)) if type_rouge else 0.0,
                "avg_latency_ms": float(np.mean([r.get("latency_ms", 0) for r in type_results])),
            }
    metrics["per_type"] = type_metrics

    # Avg evidence & action counts
    evidence_counts = []
    action_counts = []
    for r in results:
        rca = r.get("rca_report", {})
        ev = rca.get("supporting_evidence", [])
        ac = rca.get("recommended_actions", [])
        evidence_counts.append(len(ev) if isinstance(ev, list) else 0)
        action_counts.append(len(ac) if isinstance(ac, list) else 0)
    metrics["avg_evidence_count"] = float(np.mean(evidence_counts)) if evidence_counts else 0.0
    metrics["avg_action_count"] = float(np.mean(action_counts)) if action_counts else 0.0

    return metrics


# ── Main Runner ──

def run_ablation(test_anomalies: List[dict] = None,
                 config_keys: List[str] = None,
                 run_judge: bool = False):
    """Run the selected configs and produce comparison results.

    Args:
        test_anomalies: If None, falls back to the legacy 15-item hard-coded set.
        config_keys: Subset of ('A_no_rag','B_rag_only','C_single_agent_rag',
            'D_multi_agent_rag'). Defaults to all four.
        run_judge: If True, runs LLM-as-Judge + faithfulness + answer-relevancy
            on each config's results after generation. Requires a configured LLM key.
    """
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — Multi-Agent RAG System")
    print(f"  LLM: {LLM_MODEL}")
    print("=" * 70)

    if test_anomalies is None:
        test_anomalies = get_test_anomalies()
    print(f"\nTest set: {len(test_anomalies)} anomalies")

    all_configs = {
        "A_no_rag": ("Config A: No RAG (Direct LLM)", run_config_a),
        "B_rag_only": ("Config B: RAG + LLM (single prompt)", run_config_b),
        "C_single_agent_rag": ("Config C: Single Agent + RAG", run_config_c),
        "D_multi_agent_rag": ("Config D: Multi-Agent + RAG [proposed]", run_config_d),
        "E_graph_rag": ("Config E: Multi-Agent + GraphRAG [headline novelty]", run_config_e),
    }
    if config_keys:
        configs = {k: v for k, v in all_configs.items() if k in config_keys}
    else:
        configs = all_configs

    all_results = {}
    all_metrics = {}

    for config_key, (config_name, run_fn) in configs.items():
        print(f"\n{'─' * 70}")
        print(f"  Running: {config_name}")
        print(f"{'─' * 70}")

        results = []
        for anomaly in test_anomalies:
            atype = anomaly["anomaly_type"]
            print(f"  Processing: {anomaly['account_id']} ({atype})...", end=" ", flush=True)
            try:
                result = run_fn(anomaly)
                status = "✓" if result.get("pipeline_status") == "completed" else "✗"
                latency = result.get("latency_ms", 0)
                print(f"{status} ({latency:.0f}ms)")
                results.append(result)
            except Exception as e:
                print(f"✗ ERROR: {e}")
                results.append({
                    "anomaly_data": anomaly,
                    "rca_report": {"anomaly_type": atype, "root_cause": f"Error: {e}"},
                    "pipeline_status": "error",
                    "latency_ms": 0,
                    "retrieval_count": 0,
                    "retrieved_docs": [],
                    "config": config_key,
                })
            # Pacing is handled by the token bucket inside call_llm(); no
            # fixed sleep here. Rate-limit 429 retries still protect bursts.

        metrics = evaluate_config_results(results)
        all_results[config_key] = results
        all_metrics[config_key] = metrics

        if run_judge:
            try:
                from src.evaluation.llm_judge import judge_batch, aggregate_judge_scores
                from src.evaluation.metrics import load_ground_truth
                gt_by_type = {}
                for gt in load_ground_truth():
                    gt_by_type.setdefault(gt["anomaly_type"], gt)  # type->any row
                print(f"  [judge] scoring {len(results)} results for {config_key}...")
                judge_batch(results, gt_by_type, verbose=False)
                metrics.update(aggregate_judge_scores(results))
            except Exception as e:
                print(f"  [judge] pass failed for {config_key}: {e}")

        print(f"\n  Results: ROUGE-L={metrics['rouge_l_f1']:.3f}±{metrics.get('rouge_l_std', 0):.3f} | "
              f"BERTScore={metrics['bert_score_f1']:.3f} | "
              f"TypeAcc={metrics['type_accuracy']:.1%} | "
              f"AvgLatency={metrics['avg_latency_ms']:.0f}ms")

    # ── Print comparison table ──
    print("\n\n" + "=" * 90)
    print("  ABLATION RESULTS COMPARISON (n=15 per config)")
    print("=" * 90)

    header = f"{'Config':<35} {'ROUGE-L':>10} {'BERT-F1':>8} {'TypeAcc':>8} {'Latency':>10} {'Retrieval':>10} {'Evidence':>9}"
    print(header)
    print("─" * len(header))

    for config_key, (config_name, _) in configs.items():
        m = all_metrics[config_key]
        rouge_str = f"{m['rouge_l_f1']:.3f}±{m.get('rouge_l_std', 0):.3f}"
        print(f"{config_name:<35} {rouge_str:>10} {m['bert_score_f1']:>8.3f} "
              f"{m['type_accuracy']:>7.1%} {m['avg_latency_ms']:>9.0f}ms "
              f"{m.get('avg_retrieval_relevance', 0):>10.3f} "
              f"{m['avg_evidence_count']:>9.1f}")

    # ── Statistical Significance Tests ──
    print(f"\n{'─' * 90}")
    print("  STATISTICAL SIGNIFICANCE (paired-bootstrap + Wilcoxon) on ROUGE-L")
    print(f"{'─' * 90}")

    try:
        from src.evaluation.stats import bootstrap_ci, paired_bootstrap_pvalue, wilcoxon_paired

        # 95% CI per config
        for ck in configs:
            vals = all_metrics[ck].get("rouge_l_per_sample", [])
            if vals:
                m, lo, hi = bootstrap_ci(vals)
                print(f"  {ck}: ROUGE-L mean={m:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  n={len(vals)}")
                all_metrics[ck]["rouge_ci95"] = [round(lo, 4), round(hi, 4)]

        # Pairwise D vs {A,B,C}
        d_rouge = all_metrics.get("D_multi_agent_rag", {}).get("rouge_l_per_sample", [])
        for compare_key in ["A_no_rag", "B_rag_only", "C_single_agent_rag"]:
            if compare_key not in all_metrics:
                continue
            c_rouge = all_metrics[compare_key].get("rouge_l_per_sample", [])
            if len(d_rouge) == len(c_rouge) and len(d_rouge) >= 5:
                p_boot = paired_bootstrap_pvalue(d_rouge, c_rouge)
                w = wilcoxon_paired(d_rouge, c_rouge)
                sig = "***" if p_boot < 0.001 else "**" if p_boot < 0.01 else "*" if p_boot < 0.05 else "n.s."
                cname = configs[compare_key][0]
                print(f"  D vs {cname}: Δmean={np.mean(d_rouge)-np.mean(c_rouge):+.3f}  "
                      f"p_bootstrap={p_boot:.4f}  p_wilcoxon={w['pvalue']:.4f}  {sig}")
    except Exception as e:
        print(f"  Significance test error: {e}")

    # ── Per-type breakdown ──
    print(f"\n{'─' * 90}")
    print("  PER-TYPE ROUGE-L BREAKDOWN (Config D)")
    print(f"{'─' * 90}")
    d_per_type = all_metrics["D_multi_agent_rag"].get("per_type", {})
    for atype, tm in d_per_type.items():
        print(f"  {atype:<25} ROUGE-L={tm['rouge_l_f1']:.3f}  n={tm['count']}  latency={tm['avg_latency_ms']:.0f}ms")

    # ── Improvement ──
    print(f"\n{'─' * 90}")
    baseline = all_metrics["A_no_rag"]
    proposed = all_metrics["D_multi_agent_rag"]

    if baseline["rouge_l_f1"] > 0:
        rouge_imp = ((proposed["rouge_l_f1"] - baseline["rouge_l_f1"]) / baseline["rouge_l_f1"]) * 100
        print(f"  ROUGE-L improvement (D vs A): {rouge_imp:+.1f}%")
    if baseline["bert_score_f1"] > 0:
        bert_imp = ((proposed["bert_score_f1"] - baseline["bert_score_f1"]) / baseline["bert_score_f1"]) * 100
        print(f"  BERTScore improvement (D vs A): {bert_imp:+.1f}%")

    # ── Save results ──
    output = {
        "model": LLM_MODEL,
        "test_anomalies_count": len(test_anomalies),
        "anomalies_per_type": 3,
        "configs": {},
    }
    for config_key in configs:
        m = all_metrics[config_key]
        # Filter out non-serializable fields for JSON
        metrics_clean = {}
        for k, v in m.items():
            if k in ("rouge_l_per_sample", "bert_per_sample"):
                metrics_clean[k] = [round(x, 4) if isinstance(x, float) else x for x in v]
            elif k == "per_type":
                metrics_clean[k] = v
            elif isinstance(v, float):
                metrics_clean[k] = round(v, 4)
            else:
                metrics_clean[k] = v
        output["configs"][config_key] = {
            "description": configs[config_key][0],
            "metrics": metrics_clean,
        }
        output["configs"][config_key]["rca_reports"] = []
        for r in all_results[config_key]:
            output["configs"][config_key]["rca_reports"].append({
                "account_id": r.get("anomaly_data", {}).get("account_id"),
                "anomaly_type": r.get("anomaly_data", {}).get("anomaly_type"),
                "latency_ms": round(r.get("latency_ms", 0), 1),
                "rca_report": r.get("rca_report", {}),
            })

    output_path = Path("ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    # ── Log to MLflow ──
    try:
        from src.mlflow_tracking import setup_mlflow
        import mlflow
        setup_mlflow()

        with mlflow.start_run(run_name="ablation_study_n15"):
            mlflow.log_param("model", LLM_MODEL)
            mlflow.log_param("test_anomalies", len(test_anomalies))
            mlflow.log_param("anomalies_per_type", 3)

            for config_key in configs:
                m = all_metrics[config_key]
                for metric_name, metric_val in m.items():
                    if isinstance(metric_val, (int, float)):
                        mlflow.log_metric(f"{config_key}_{metric_name}", metric_val)

            mlflow.log_artifact(str(output_path))
        print("  Results logged to MLflow.")
    except Exception as e:
        print(f"  MLflow logging skipped: {e}")

    return all_results, all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RCA ablation study.")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to N anomalies per type (default: 12 with --gt, 3 without).")
    parser.add_argument("--configs", type=str, default="A,B,C,D",
                        help="Comma-separated subset of configs: A,B,C,D,E "
                             "(E requires data/graph_rag/kb_graph.pkl — "
                             "build with: python scripts/build_graph_rag.py --offline)")
    parser.add_argument("--gt", action="store_true",
                        help="Use 60-item ground truth as the test set (per-item GT matching).")
    parser.add_argument("--judge", action="store_true",
                        help="Also run LLM-as-Judge + faithfulness + relevancy.")
    args = parser.parse_args()

    key_map = {
        "A": "A_no_rag", "B": "B_rag_only",
        "C": "C_single_agent_rag", "D": "D_multi_agent_rag",
        "E": "E_graph_rag",
    }
    config_keys = [key_map[c.strip().upper()] for c in args.configs.split(",") if c.strip().upper() in key_map]

    if args.gt:
        from src.utils.test_data import anomalies_from_ground_truth
        n_per_type = args.n if args.n is not None else 12
        test_set = anomalies_from_ground_truth(limit_per_type=n_per_type)
        print(f"\n[mode] Using GT-derived test set: {len(test_set)} anomalies "
              f"({n_per_type} per type, ground_truth_id preserved).")
    else:
        base = get_test_anomalies()
        if args.n is not None:
            # Keep `args.n` per type from the 15-item hardcoded set
            by_type: Dict[str, List[dict]] = {}
            for a in base:
                by_type.setdefault(a["anomaly_type"], []).append(a)
            test_set = []
            for rows in by_type.values():
                test_set.extend(rows[: args.n])
        else:
            test_set = base
        print(f"\n[mode] Using legacy hardcoded test set: {len(test_set)} anomalies.")

    run_ablation(test_anomalies=test_set, config_keys=config_keys, run_judge=args.judge)
