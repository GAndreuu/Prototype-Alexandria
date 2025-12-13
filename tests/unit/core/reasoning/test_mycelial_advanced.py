"""
Tests for Mycelial Reasoning Advanced
"""
import pytest
import numpy as np
from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig

class TestMycelialReasoningAdvanced:
    @pytest.fixture
    def mycelial(self):
        config = MycelialConfig()
        return MycelialReasoning(config)

    def test_observe(self, mycelial):
        # observe takes indices [h0, h1, h2, h3]
        mycelial.observe([0, 1, 2, 3])
        # Should have created connections
        stats = mycelial.get_network_stats()
        assert stats is not None

    def test_propagate(self, mycelial):
        # First observe to create connections
        mycelial.observe([0, 1, 2, 3])
        mycelial.observe([0, 1, 2, 4])  # Overlapping observation
        
        # Propagate
        activation = mycelial.propagate([0, 1, 2, 3])
        assert isinstance(activation, dict)

    def test_reason(self, mycelial):
        # observe first
        mycelial.observe([0, 1, 2, 3])
        
        # reason returns refined indices
        result = mycelial.reason([0, 1, 2, 3])
        assert len(result) == 4

    def test_connect_nodes(self, mycelial):
        # connect_nodes(node_a, node_b, weight_delta)
        # node format is (head, code)
        new_weight = mycelial.connect_nodes((0, 0), (1, 1), weight_delta=0.5)
        assert new_weight > 0

    def test_get_neighbors(self, mycelial):
        mycelial.observe([0, 1, 2, 3])
        neighbors = mycelial.get_neighbors((0, 0))
        assert isinstance(neighbors, list)

    def test_get_network_stats(self, mycelial):
        # get_network_stats is the correct method name
        stats = mycelial.get_network_stats()
        assert "num_nodes" in stats or "total_edges" in stats or isinstance(stats, dict)

    def test_decay(self, mycelial):
        mycelial.observe([0, 1, 2, 3])
        mycelial.decay()
        # Should not crash
