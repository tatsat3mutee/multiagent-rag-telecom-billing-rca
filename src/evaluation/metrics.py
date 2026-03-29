"""
Evaluation metrics for the Multi-Agent RAG RCA System.
Covers anomaly detection, RAG retrieval quality, and RCA output quality.
"""
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import GROUND_TRUTH_DIR


def load_ground_truth() -> List[dict]:
    """Load ground truth RCA documents."""
    gt_path = GROUND_TRUTH_DIR / "ground_truth_rca.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found at {gt_path}")
    with open(gt_path, "r") as f:
        return json.load(f)


# ── Anomaly Detection Metrics ──

def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
    """Compute anomaly detection metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, classification_report,
    )

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_scores is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
            metrics["average_precision"] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["average_precision"] = 0.0

    return metrics


# ── RAG Retrieval Metrics ──

def context_recall(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """Compute context recall: proportion of relevant docs that were retrieved."""
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def context_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """Compute context precision: proportion of retrieved docs that are relevant."""
    if not retrieved_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def mrr_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
    """Mean Reciprocal Rank at k."""
    relevant_set = set(relevant_docs)
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ── RCA Quality Metrics ──

def compute_rouge_l(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-L score between hypothesis and reference."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "precision": scores["rougeL"].precision,
            "recall": scores["rougeL"].recall,
            "fmeasure": scores["rougeL"].fmeasure,
        }
    except ImportError:
        # Fallback: simple token overlap
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        overlap = len(hyp_tokens & ref_tokens)
        precision = overlap / len(hyp_tokens) if hyp_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "fmeasure": f1}


def compute_bert_score(hypotheses: List[str], references: List[str]) -> dict:
    """Compute BERTScore for a batch of hypothesis-reference pairs."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
            "individual_f1": F1.tolist(),
        }
    except ImportError:
        # Fallback: use ROUGE-L as proxy
        f1_scores = []
        for hyp, ref in zip(hypotheses, references):
            rouge = compute_rouge_l(hyp, ref)
            f1_scores.append(rouge["fmeasure"])
        return {
            "precision": np.mean(f1_scores),
            "recall": np.mean(f1_scores),
            "f1": np.mean(f1_scores),
            "individual_f1": f1_scores,
            "note": "Using ROUGE-L as fallback (BERTScore not installed)"
        }


def anomaly_type_match(predicted_type: str, ground_truth_type: str) -> bool:
    """Check if predicted anomaly type matches ground truth."""
    return predicted_type.lower().strip() == ground_truth_type.lower().strip()


# ── Comprehensive Evaluation ──

def evaluate_pipeline_results(results: List[dict], ground_truths: List[dict] = None) -> dict:
    """
    Evaluate a batch of pipeline results against ground truth.

    Args:
        results: List of pipeline outputs from run_pipeline()
        ground_truths: Optional list of ground truth RCA documents
    """
    if ground_truths is None:
        try:
            ground_truths = load_ground_truth()
        except FileNotFoundError:
            ground_truths = []

    # Build ground truth lookup
    gt_lookup = {}
    for gt in ground_truths:
        gt_lookup[gt["anomaly_type"]] = gt

    metrics = {
        "total_processed": len(results),
        "successful": sum(1 for r in results if r.get("pipeline_status") == "completed"),
        "avg_latency_ms": np.mean([r.get("latency_ms", 0) for r in results]),
        "avg_retrieval_count": np.mean([r.get("retrieval_count", 0) for r in results]),
    }

    # Type matching
    type_matches = 0
    total_with_type = 0
    for r in results:
        rca = r.get("rca_report", {})
        pred_type = rca.get("anomaly_type", "")
        true_type = r.get("anomaly_data", {}).get("anomaly_type", "")
        if true_type:
            total_with_type += 1
            if anomaly_type_match(pred_type, true_type):
                type_matches += 1

    metrics["anomaly_type_accuracy"] = type_matches / max(total_with_type, 1)

    # RCA quality against ground truth
    if ground_truths:
        hypotheses = []
        references = []
        for r in results:
            rca = r.get("rca_report", {})
            anomaly_type = r.get("anomaly_data", {}).get("anomaly_type", "")
            if anomaly_type in gt_lookup:
                hypotheses.append(rca.get("root_cause", ""))
                references.append(gt_lookup[anomaly_type].get("root_cause", ""))

        if hypotheses:
            # ROUGE-L
            rouge_scores = [compute_rouge_l(h, r) for h, r in zip(hypotheses, references)]
            metrics["rouge_l_f1"] = np.mean([s["fmeasure"] for s in rouge_scores])

            # BERTScore
            bert_scores = compute_bert_score(hypotheses, references)
            metrics["bert_score_f1"] = bert_scores["f1"]

    return metrics


def print_evaluation_report(metrics: dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)

    print(f"\n  Pipeline Performance:")
    print(f"    Total Processed:      {metrics.get('total_processed', 0)}")
    print(f"    Successful:           {metrics.get('successful', 0)}")
    print(f"    Avg Latency:          {metrics.get('avg_latency_ms', 0):.0f}ms")
    print(f"    Avg Retrieval Count:  {metrics.get('avg_retrieval_count', 0):.1f}")

    print(f"\n  RCA Quality:")
    print(f"    Anomaly Type Accuracy: {metrics.get('anomaly_type_accuracy', 0):.2%}")

    if "rouge_l_f1" in metrics:
        print(f"    ROUGE-L F1:           {metrics.get('rouge_l_f1', 0):.4f}")
    if "bert_score_f1" in metrics:
        print(f"    BERTScore F1:         {metrics.get('bert_score_f1', 0):.4f}")

    print("\n" + "=" * 60)
