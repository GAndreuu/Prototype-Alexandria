"""
Tests for core/loop/nemesis_integration.py
"""
import pytest
from unittest.mock import Mock, patch
from core.loop.nemesis_integration import NemesisIntegration

class TestNemesisIntegration:
    """Tests for NemesisIntegration class."""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance with mocked dependencies."""
        with patch('core.learning.active_inference.ActiveInferenceAgent') as MockAI, \
             patch('core.learning.predictive_coding.PredictiveCodingNetwork') as MockPC, \
             patch('core.learning.meta_hebbian.MetaHebbianPlasticity') as MockMH, \
             patch('core.learning.free_energy.VariationalFreeEnergy') as MockFE:
             
             # Configure mocks if needed
             MockPC.return_value.process.return_value = {}
             
             integration = NemesisIntegration()
             return integration
    
    def test_init(self, integration):
        """Test initialization of all modules."""
        assert integration is not None
        assert integration.active_inference is not None
        assert integration.predictive_coding is not None
        assert integration.meta_hebbian is not None
        assert integration.free_energy is not None

    def test_select_action_with_active_inference(self, integration):
        """Test selection delegates to ActiveInferenceAgent."""
        # Setup mock return
        mock_ai = integration.active_inference
        
        mock_action = Mock()
        mock_action.target = "cluster_1"
        mock_action.parameters = {}
        mock_action.action_type.name = "EXPLORE_CLUSTER"
        mock_action.expected_information_gain = 0.5
        mock_action.expected_risk = 0.1
        
        # select_action returns (action, info_dict)
        mock_ai.select_action.return_value = (mock_action, {"selected_EFE": -0.5, "selection_prob": 0.9})

        gap = {"id": "gap1", "description": "test gap"}
        hypotheses = [{"id": "hyp1", "confidence_score": 0.8, "target_cluster": "cluster_1"}]
        
        # Act
        result_action = integration.select_action(gap, hypotheses)
        
        # Assert
        assert result_action.target == "cluster_1"
        assert result_action.confidence == 0.8
        
    def test_update_after_action(self, integration):
        """Test update propagation to all sub-modules."""
        # Setup mocks return values
        integration.predictive_coding.process.return_value = {'total_error': 0.1}
        integration.free_energy.compute.return_value = (0.5, {'complexity': 0.2})
        
        action = Mock()
        observation = Mock()
        reward = 1.0
        
        metrics = integration.update_after_action(action, observation, reward)
        
        assert metrics['free_energy'] == 0.5
        assert metrics['accuracy'] == 1.0
        
        # Verify calls
        integration.predictive_coding.process.assert_called()
        integration.free_energy.compute.assert_called()
        integration.meta_hebbian.evolve_rules.assert_called()

    def test_get_metrics(self, integration):
        """Test metrics retrieval."""
        metrics = integration.get_metrics()
        assert 'free_energy' in metrics
        assert metrics['modules_active']['active_inference'] is True
