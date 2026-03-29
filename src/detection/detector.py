"""
Anomaly Detection module using IsolationForest and DBSCAN.
Detects billing anomalies and outputs structured results.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ISOLATION_FOREST_PARAMS, MODELS_DIR, RANDOM_SEED


# Features used for anomaly detection
DETECTION_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


class BillingAnomalyDetector:
    """Anomaly detection for telecom billing data."""

    def __init__(self, method: str = "isolation_forest"):
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = DETECTION_FEATURES

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features for detection."""
        features = df[self.feature_names].copy()
        # Handle NaN — fill with median for detection
        features = features.fillna(features.median())
        return features

    def fit(self, df: pd.DataFrame):
        """Train the anomaly detector."""
        features = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(features)

        if self.method == "isolation_forest":
            self.model = IsolationForest(**ISOLATION_FOREST_PARAMS)
            self.model.fit(X_scaled)
        elif self.method == "dbscan":
            self.model = DBSCAN(eps=1.5, min_samples=10)
            self.model.fit(X_scaled)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies. Returns DataFrame with predictions and confidence scores.
        """
        features = self._prepare_features(df)
        X_scaled = self.scaler.transform(features)

        result = df.copy()

        if self.method == "isolation_forest":
            # IsolationForest: -1 = anomaly, 1 = normal
            predictions = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)

            result["predicted_anomaly"] = (predictions == -1).astype(int)
            # Convert decision function to confidence (0-1 range)
            result["anomaly_confidence"] = 1 - (scores - scores.min()) / (scores.max() - scores.min())

        elif self.method == "dbscan":
            labels = self.model.fit_predict(X_scaled)
            result["predicted_anomaly"] = (labels == -1).astype(int)
            result["anomaly_confidence"] = (labels == -1).astype(float)

        return result

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate detector performance if ground truth labels exist."""
        if "is_anomaly" not in df.columns:
            raise ValueError("DataFrame must have 'is_anomaly' column for evaluation")

        result = self.predict(df)
        y_true = result["is_anomaly"].values
        y_pred = result["predicted_anomaly"].values

        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
        }

        # ROC-AUC if confidence scores available
        if "anomaly_confidence" in result.columns:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, result["anomaly_confidence"].values)
            except ValueError:
                metrics["roc_auc"] = 0.0

        return metrics

    def get_anomalous_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the detected anomalous records with metadata."""
        result = self.predict(df)
        anomalies = result[result["predicted_anomaly"] == 1].copy()

        # Add feature-based anomaly type estimation
        anomalies["estimated_type"] = anomalies.apply(self._estimate_anomaly_type, axis=1)

        return anomalies

    def _estimate_anomaly_type(self, row: pd.Series) -> str:
        """Heuristic to estimate anomaly type from features."""
        if pd.isna(row.get("TotalCharges")):
            return "cdr_failure"
        if row.get("MonthlyCharges", 0) == 0:
            return "zero_billing"

        # Check for duplicate first (higher threshold) before usage_spike
        if row.get("MonthlyCharges", 0) > 200:
            return "duplicate_charge"

        # Check for SLA breach (charges > 95th percentile range)
        if row.get("MonthlyCharges", 0) > 150:
            return "sla_breach"

        # Usage spike: monthly charges disproportionate to total/tenure history
        if row.get("MonthlyCharges", 0) > row.get("TotalCharges", 0) * 0.5 and row.get("tenure", 1) > 1:
            return "usage_spike"

        return "unknown"

    def save(self, filepath: Path = None):
        """Save model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.method}_model.joblib"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "method": self.method}, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: Path = None):
        """Load model from disk."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.method}_model.joblib"
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.method = data["method"]
        print(f"Model loaded from {filepath}")
        return self


def train_and_evaluate(df: pd.DataFrame, method: str = "isolation_forest",
                       test_size: float = 0.3) -> dict:
    """Complete training and evaluation pipeline with proper train/test split."""
    from sklearn.model_selection import train_test_split

    # Stratified train/test split to avoid data leakage
    if "is_anomaly" in df.columns:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=RANDOM_SEED,
            stratify=df["is_anomaly"],
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=RANDOM_SEED,
        )

    detector = BillingAnomalyDetector(method=method)
    detector.fit(train_df)

    # Evaluate on HELD-OUT test set only
    metrics = detector.evaluate(test_df)
    detector.save()

    print(f"\n{method.upper()} Results (test set, n={len(test_df)}):")
    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return {"detector": detector, "metrics": metrics}
