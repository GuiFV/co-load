"""Learned VRAM estimates: config-seeded, refined by observation, cached to disk."""

import pytest

from coload.estimates import EstimateStore

GIB = 2**30


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "learned.json"


class TestSeeds:
    def test_seed_used_when_nothing_learned(self, store_path):
        store = EstimateStore(store_path, seeds={"m": 8 * GIB})
        assert store.estimate("m") == 8 * GIB

    def test_unknown_model_raises(self, store_path):
        store = EstimateStore(store_path, seeds={})
        with pytest.raises(KeyError):
            store.estimate("nope")


class TestLearning:
    def test_observed_value_takes_precedence_over_seed(self, store_path):
        store = EstimateStore(store_path, seeds={"m": 8 * GIB})
        store.observe("m", 9 * GIB)
        assert store.estimate("m") == 9 * GIB

    def test_learned_persists_across_instances(self, store_path):
        EstimateStore(store_path, seeds={"m": 8 * GIB}).observe("m", 9 * GIB)
        fresh = EstimateStore(store_path, seeds={"m": 8 * GIB})
        assert fresh.estimate("m") == 9 * GIB

    def test_keeps_peak_observation(self, store_path):
        """KV cache can grow; never shrink the learned figure."""
        store = EstimateStore(store_path, seeds={})
        store.observe("m", 10 * GIB)
        store.observe("m", 7 * GIB)
        assert store.estimate("m") == 10 * GIB

    def test_ignores_nonpositive_observations(self, store_path):
        store = EstimateStore(store_path, seeds={"m": 8 * GIB})
        store.observe("m", 0)
        store.observe("m", -5)
        assert store.estimate("m") == 8 * GIB


class TestPersistenceRobustness:
    def test_corrupt_file_starts_fresh(self, store_path):
        store_path.write_text("{not json!", encoding="utf-8")
        store = EstimateStore(store_path, seeds={"m": 8 * GIB})
        assert store.estimate("m") == 8 * GIB

    def test_creates_parent_directories(self, tmp_path):
        deep = tmp_path / "a" / "b" / "learned.json"
        store = EstimateStore(deep, seeds={})
        store.observe("m", GIB)
        assert deep.exists()
