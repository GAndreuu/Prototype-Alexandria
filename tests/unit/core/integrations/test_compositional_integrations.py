"""
Tests for core/integrations/test_compositional_integrations.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from core.integrations.abduction_compositional_integration import AbductionCompositionalIntegration
from core.integrations.agents_compositional_integration import AgentsCompositionalIntegration

class TestAbductionCompositionalIntegration:
    """Tests for AbductionCompositionalIntegration."""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance."""
        bridge = Mock()
        abduction = Mock()
        compositional = Mock()
        return AbductionCompositionalIntegration(bridge, abduction, compositional)
    
    def test_init(self, integration):
        """Test initialization."""
        assert integration is not None
        assert integration.bridge is not None

    def test_detect_gaps_geometric(self, integration):
        """Test gap detection."""
        embeddings = [np.random.randn(384) for _ in range(5)]
        
        # Mock internal detection methods to avoid complex logic
        with patch.object(integration, '_detect_curvature_anomalies', return_value=[]), \
             patch.object(integration, '_detect_energy_barriers', return_value=[]), \
             patch.object(integration, '_detect_disconnections', return_value=[]):
            
            gaps = integration.detect_gaps_geometric(embeddings)
            assert isinstance(gaps, list)

class TestAgentsCompositionalIntegration:
    """Tests for AgentsCompositionalIntegration."""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance."""
        agents_layer = Mock()
        compositional_layer = Mock()
        return AgentsCompositionalIntegration(agents_layer, compositional_layer)
        
    def test_init(self, integration):
        """Test initialization."""
        assert integration is not None
