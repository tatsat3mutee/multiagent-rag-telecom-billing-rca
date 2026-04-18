"""
Telecom Italia Big Data Challenge CDR loader (Milan grid, Nov-Dec 2013).

Reference: https://dandelion.eu/datamine/open-big-data/ (and the Nature SD
paper "A multi-source dataset of urban life in Milan"). The raw files are
per-day TSVs with columns:
    CellID, datetime, countrycode, smsin, smsout, callin, callout, internet

This loader:
  - Reads a directory of daily TSVs (or a single file) and concatenates.
  - Aggregates per (CellID, hour) to produce a billing-like feature frame:
      sms_total, call_total, internet, anomaly_proxy_z
    where anomaly_proxy_z is a z-score of hourly activity vs same-weekday
    rolling baseline — a proxy for "unexpected usage".
  - Writes the processed frame to `data/processed/telecom_italia_cdr.parquet`.

Design notes:
  - No hard dep on pyarrow: falls back to `.csv.gz` if parquet fails.
  - Will skip gracefully if no raw files present and print a helpful
    message pointing to the Dandelion catalog download.

This is Phase 1.5 D1 — it augments the synthetic IBM Telco dataset with a
real multi-source CDR feature space, per the Limitations-doc schema-mismatch
mitigation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

CDR_RAW_DIR = RAW_DATA_DIR / "telecom_italia_cdr"
CDR_PROCESSED = PROCESSED_DATA_DIR / "telecom_italia_cdr.parquet"


COLUMNS = [
    "cell_id", "datetime", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, na_values=[""])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
    for c in ("sms_in", "sms_out", "call_in", "call_out", "internet"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df.dropna(subset=["datetime"])


def _load_all(raw_dir: Path) -> pd.DataFrame:
    files: List[Path] = sorted(list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.tsv")))
    if not files:
        raise FileNotFoundError(
            f"No TSV files under {raw_dir}. Download from "
            "https://dandelion.eu/datamine/open-big-data/ "
            "(Telecommunications - SMS, Call, Internet - MI) and unzip to this folder."
        )
    parts = [_read_one(f) for f in files]
    return pd.concat(parts, ignore_index=True)


def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(hour=df["datetime"].dt.floor("h"))
    g = df.groupby(["cell_id", "hour"], as_index=False).agg(
        sms_total=("sms_in", "sum"),
        sms_out_total=("sms_out", "sum"),
        call_total=("call_in", "sum"),
        call_out_total=("call_out", "sum"),
        internet=("internet", "sum"),
    )
    g["sms_total"] = g["sms_total"] + g["sms_out_total"]
    g["call_total"] = g["call_total"] + g["call_out_total"]
    return g.drop(columns=["sms_out_total", "call_out_total"])


def _add_anomaly_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day z-score per cell: high |z| suggests unusual usage."""
    df = df.sort_values(["cell_id", "hour"]).copy()
    df["hod"] = df["hour"].dt.hour
    stats = df.groupby(["cell_id", "hod"])["internet"].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "_mu", "std": "_sigma"})
    merged = df.merge(stats, on=["cell_id", "hod"], how="left")
    merged["_sigma"] = merged["_sigma"].replace(0, np.nan).fillna(merged["_mu"].std())
    merged["anomaly_proxy_z"] = (merged["internet"] - merged["_mu"]) / merged["_sigma"]
    return merged.drop(columns=["_mu", "_sigma", "hod"])


def build(raw_dir: Path = CDR_RAW_DIR, out_path: Path = CDR_PROCESSED) -> Optional[Path]:
    try:
        raw = _load_all(raw_dir)
    except FileNotFoundError as e:
        print(f"[telecom-italia] {e}")
        return None

    print(f"[telecom-italia] loaded {len(raw):,} rows from {raw_dir}")
    hourly = _aggregate_hourly(raw)
    hourly = _add_anomaly_proxy(hourly)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        hourly.to_parquet(out_path, index=False)
        print(f"[telecom-italia] wrote {out_path} ({len(hourly):,} rows)")
        return out_path
    except Exception as e:
        alt = out_path.with_suffix(".csv.gz")
        hourly.to_csv(alt, index=False, compression="gzip")
        print(f"[telecom-italia] parquet failed ({e}) — wrote {alt}")
        return alt


if __name__ == "__main__":
    build()
