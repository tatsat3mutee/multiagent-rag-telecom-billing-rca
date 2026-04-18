"""Tests for src.utils.*"""
import time

import pytest

from src.utils.rate_limit import TokenBucket, get_limiter
from src.utils.test_data import anomalies_from_ground_truth


class TestTokenBucket:
    def test_full_bucket_allows_immediate_first_acquire(self):
        b = TokenBucket(rate_per_minute=60, capacity=3)
        waited = b.acquire()
        assert waited == 0.0

    def test_depleted_bucket_waits_at_least_expected_interval(self):
        # 120 rpm = 2 rps → interval ~0.5s
        b = TokenBucket(rate_per_minute=120, capacity=1)
        b.acquire()  # deplete
        start = time.monotonic()
        b.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.3  # generous lower bound for CI jitter

    def test_singleton_shared(self):
        a = get_limiter()
        b = get_limiter()
        assert a is b


class TestGTDerivedAnomalies:
    def test_default_shape(self):
        xs = anomalies_from_ground_truth(limit_per_type=3)
        assert len(xs) == 15

    def test_has_ground_truth_id(self):
        xs = anomalies_from_ground_truth(limit_per_type=2)
        assert all("ground_truth_id" in a for a in xs)

    def test_deterministic_under_seed(self):
        xs = anomalies_from_ground_truth(limit_per_type=3, seed=123)
        ys = anomalies_from_ground_truth(limit_per_type=3, seed=123)
        assert xs == ys

    def test_types_present(self):
        xs = anomalies_from_ground_truth(limit_per_type=3)
        types = {a["anomaly_type"] for a in xs}
        assert len(types) == 5

    def test_ground_truth_id_is_unique_per_anomaly(self):
        xs = anomalies_from_ground_truth(limit_per_type=12)
        gt_ids = [a["ground_truth_id"] for a in xs]
        assert len(gt_ids) == len(set(gt_ids))
