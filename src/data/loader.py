"""
Dataset loader for IBM Telco and Maven Telecom churn datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, RANDOM_SEED


def load_ibm_telco(filepath: Path = None) -> pd.DataFrame:
    """Load and clean the IBM Telco Customer Churn dataset."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "ibm_telco_churn.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"IBM Telco dataset not found at {filepath}. "
            "Run `python scripts/download_datasets.py` first."
        )

    df = pd.read_csv(filepath)

    # Clean TotalCharges — contains spaces for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Ensure numeric types
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["tenure"] = df["tenure"].astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Encode Churn as binary
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    return df


def load_maven_telecom(filepath: Path = None) -> pd.DataFrame:
    """Load and clean the Maven Analytics Telecom Churn dataset."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "maven_telecom_churn.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Maven Telecom dataset not found at {filepath}. "
            "Run `python scripts/download_datasets.py` first."
        )

    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def get_billing_features(df: pd.DataFrame, dataset_type: str = "ibm") -> pd.DataFrame:
    """Extract billing-relevant features for anomaly detection."""
    if dataset_type == "ibm":
        feature_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        # Encode categorical features relevant to billing
        billing_df = df[feature_cols].copy()

        # Derive features
        billing_df["charges_per_month"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"],
        )
        billing_df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=[0, 12, 24, 48, 72, 100],
            labels=["0-12", "12-24", "24-48", "48-72", "72+"],
        )

        # Has active services indicator
        service_cols = [c for c in df.columns if "Service" in c or "service" in c]
        if service_cols:
            billing_df["active_services"] = (
                df[service_cols].apply(lambda x: x == "Yes").sum(axis=1)
            )
        else:
            billing_df["active_services"] = 1

        # Contract type encoding
        if "Contract" in df.columns:
            billing_df["contract_month"] = (df["Contract"] == "Month-to-month").astype(int)

        return billing_df

    elif dataset_type == "maven":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols].copy()

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
