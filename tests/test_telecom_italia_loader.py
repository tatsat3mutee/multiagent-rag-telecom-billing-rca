"""Tests for src.data.telecom_italia_loader — synthetic input, no network."""
import numpy as np
import pandas as pd
import pytest

from src.data import telecom_italia_loader as ti


def _synthetic_raw(tmp_path, n_cells=3, n_hours=48):
    rng = np.random.default_rng(0)
    rows = []
    base_ts = pd.Timestamp("2013-11-01 00:00:00").value // 10**6  # ms epoch
    for c in range(n_cells):
        for h in range(n_hours):
            ts = base_ts + h * 3600 * 1000
            rows.append((
                c, ts, 39,  # country
                float(rng.integers(0, 50)), float(rng.integers(0, 50)),
                float(rng.integers(0, 30)), float(rng.integers(0, 30)),
                float(rng.integers(0, 1000)),
            ))
    df = pd.DataFrame(rows, columns=ti.COLUMNS)
    p = tmp_path / "day1.tsv"
    df.to_csv(p, sep="\t", header=False, index=False)
    return p


class TestReadOne:
    def test_reads_and_parses_dates(self, tmp_path):
        p = _synthetic_raw(tmp_path, n_cells=2, n_hours=5)
        df = ti._read_one(p)
        assert len(df) == 10
        assert df["datetime"].dtype.kind == "M"
        assert (df["internet"] >= 0).all()


class TestAggregateAndProxy:
    def test_hourly_aggregation(self, tmp_path):
        _synthetic_raw(tmp_path, n_cells=2, n_hours=6)
        raw = ti._load_all(tmp_path)
        hourly = ti._aggregate_hourly(raw)
        # 2 cells * 6 hours
        assert len(hourly) == 12
        assert {"sms_total", "call_total", "internet"} <= set(hourly.columns)

    def test_anomaly_proxy_has_z(self, tmp_path):
        _synthetic_raw(tmp_path, n_cells=2, n_hours=48)
        raw = ti._load_all(tmp_path)
        hourly = ti._aggregate_hourly(raw)
        withz = ti._add_anomaly_proxy(hourly)
        assert "anomaly_proxy_z" in withz.columns
        # z-scores should have approximately zero mean
        assert abs(withz["anomaly_proxy_z"].mean()) < 1.0


class TestBuildErrorPath:
    def test_missing_dir_prints_and_returns_none(self, tmp_path):
        out = ti.build(raw_dir=tmp_path / "does_not_exist", out_path=tmp_path / "x.parquet")
        assert out is None
