"""
Synthetic anomaly injection into telecom billing datasets.
Implements 5 anomaly types with seed-controlled reproducibility.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED, ANOMALY_RATIOS, PROCESSED_DATA_DIR


def inject_zero_billing(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """Set MonthlyCharges=0 for random active customers."""
    n = int(len(df) * ratio)
    # Only inject into customers with active services
    active_mask = df["MonthlyCharges"] > 0
    candidates = df[active_mask].index.tolist()
    if len(candidates) < n:
        n = len(candidates)
    selected = rng.choice(candidates, size=n, replace=False)
    df.loc[selected, "MonthlyCharges"] = 0.0
    df.loc[selected, "anomaly_type"] = "zero_billing"
    df.loc[selected, "is_anomaly"] = 1
    return df


def inject_duplicate_charges(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """Duplicate billing rows with doubled charges."""
    n = int(len(df) * ratio)
    selected_idx = rng.choice(df.index.tolist(), size=n, replace=False)
    duplicates = df.loc[selected_idx].copy()
    duplicates["MonthlyCharges"] = duplicates["MonthlyCharges"] * 2
    duplicates["anomaly_type"] = "duplicate_charge"
    duplicates["is_anomaly"] = 1
    duplicates.index = range(len(df), len(df) + len(duplicates))
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


def inject_usage_spike(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """Multiply charges by 10x for random accounts."""
    n = int(len(df) * ratio)
    candidates = df[df["is_anomaly"] == 0].index.tolist()
    if len(candidates) < n:
        n = len(candidates)
    selected = rng.choice(candidates, size=n, replace=False)
    df.loc[selected, "MonthlyCharges"] = df.loc[selected, "MonthlyCharges"] * 10
    df.loc[selected, "TotalCharges"] = df.loc[selected, "TotalCharges"] * 5
    df.loc[selected, "anomaly_type"] = "usage_spike"
    df.loc[selected, "is_anomaly"] = 1
    return df


def inject_cdr_failure(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """Introduce NaN values in critical fields."""
    n = int(len(df) * ratio)
    candidates = df[df["is_anomaly"] == 0].index.tolist()
    if len(candidates) < n:
        n = len(candidates)
    selected = rng.choice(candidates, size=n, replace=False)
    # Null out TotalCharges and tenure to simulate CDR processing failure
    df.loc[selected, "TotalCharges"] = np.nan
    df.loc[selected, "anomaly_type"] = "cdr_failure"
    df.loc[selected, "is_anomaly"] = 1
    return df


def inject_sla_breach(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """Generate charges exceeding contract thresholds."""
    n = int(len(df) * ratio)
    candidates = df[df["is_anomaly"] == 0].index.tolist()
    if len(candidates) < n:
        n = len(candidates)
    selected = rng.choice(candidates, size=n, replace=False)
    # Set charges far above the 95th percentile to simulate SLA breach
    p95 = df["MonthlyCharges"].quantile(0.95)
    df.loc[selected, "MonthlyCharges"] = p95 * rng.uniform(1.5, 3.0, size=n)
    df.loc[selected, "anomaly_type"] = "sla_breach"
    df.loc[selected, "is_anomaly"] = 1
    return df


def inject_all_anomalies(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    ratios: dict = None,
) -> pd.DataFrame:
    """
    Inject all 5 anomaly types into the dataset.
    Returns DataFrame with 'is_anomaly' and 'anomaly_type' columns.
    """
    if ratios is None:
        ratios = ANOMALY_RATIOS

    rng = np.random.default_rng(seed)

    # Initialize anomaly columns
    df = df.copy()
    df["is_anomaly"] = 0
    df["anomaly_type"] = "normal"

    # Inject in order
    df = inject_zero_billing(df, rng, ratios["zero_billing"])
    df = inject_duplicate_charges(df, rng, ratios["duplicate_charge"])
    df = inject_usage_spike(df, rng, ratios["usage_spike"])
    df = inject_cdr_failure(df, rng, ratios["cdr_failure"])
    df = inject_sla_breach(df, rng, ratios["sla_breach"])

    return df


def create_labeled_dataset(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Full pipeline: inject anomalies and save labeled dataset."""
    labeled_df = inject_all_anomalies(df, seed=seed)

    # Save
    output_path = PROCESSED_DATA_DIR / "anomalies_labeled.csv"
    labeled_df.to_csv(output_path, index=False)
    print(f"Labeled dataset saved to {output_path}")
    print(f"Total records: {len(labeled_df)}")
    print(f"Anomalies: {labeled_df['is_anomaly'].sum()} ({labeled_df['is_anomaly'].mean()*100:.1f}%)")
    print(f"\nAnomaly type distribution:")
    print(labeled_df["anomaly_type"].value_counts())

    return labeled_df


if __name__ == "__main__":
    from loader import load_ibm_telco
    df = load_ibm_telco()
    create_labeled_dataset(df)
