"""
Tests for core/reasoning/symbol_grounding.py
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from core.reasoning.symbol_grounding import SymbolGrounder

class TestSymbolGrounder:
    """Tests for SymbolGrounder class."""
    
    @pytest.fixture
    def grounder(self):
        """Create SymbolGrounder with mocked dependencies."""
        mock_topology = Mock()
        # Mock encode to return a list containing one vector
        mock_topology.encode.return_value = [np.random.randn(384)]
        
        mock_vqvae = Mock()
        # Mock encode to return tensor indices (batch x seq)
        mock_vqvae.encode.return_value = torch.tensor([[0, 10], [1, 20]])
        # Mock internal device attribute access
        mock_vqvae.vqvae.device = 'cpu'
        
        return SymbolGrounder(topology_engine=mock_topology, vqvae_wrapper=mock_vqvae)

    def test_init(self, grounder):
        """Test initialization."""
        assert grounder is not None
        assert grounder.topology is not None
        assert grounder.vqvae is not None

    def test_ground(self, grounder):
        """Test grounding text to nodes."""
        nodes = grounder.ground("test text")
        
        assert len(nodes) > 0
        assert isinstance(nodes[0], tuple)
        # Check node structure (head, code)
        assert len(nodes[0]) == 2
        
    def test_ground_empty(self, grounder):
        """Test grounding empty text."""
        nodes = grounder.ground("")
        assert nodes == []

    def test_ground_gap(self, grounder):
        """Test gap grounding wrapper."""
        nodes = grounder.ground_gap("gap desc")
        assert len(nodes) > 0

    def test_ground_params(self, grounder):
        """Verify calls to dependencies."""
        grounder.ground("test")
        
        # Verify topology called
        grounder.topology.encode.assert_called_with(["test"])
        # Verify vqvae called
        grounder.vqvae.encode.assert_called()
