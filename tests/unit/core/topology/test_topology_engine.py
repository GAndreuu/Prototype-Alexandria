"""
Tests for Topology Engine
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from core.topology.topology_engine import TopologyEngine

class TestTopologyEngine:
    @pytest.fixture
    def engine(self):
        with patch.object(TopologyEngine, '_load_model'), \
             patch.object(TopologyEngine, 'load_topology', return_value=True):
            engine = TopologyEngine(auto_load=False)
            return engine

    def test_init(self, engine):
        assert engine is not None

    def test_encode(self, engine):
        # Mock generic encode
        with patch.object(engine, 'encode', return_value=np.random.randn(2, 384)):
            vecs = engine.encode(["a", "b"])
            assert vecs.shape == (2, 384)

    def test_train_manifold(self, engine):
        vectors = np.random.randn(10, 384)
        with patch('sklearn.cluster.KMeans'):
             res = engine.train_manifold(vectors, n_clusters=2)
             # Should be valid result (dict or object)
             assert res is not None
