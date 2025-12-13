"""
Tests for core/loop/active_inference_adapter.py
"""
import pytest
from unittest.mock import Mock, patch
from core.loop.active_inference_adapter import ActiveInferenceActionAdapter
from core.loop.action_selection import LoopState, ActionType

class TestActiveInferenceAdapter:
    """Tests for ActiveInferenceActionAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance with mocked internal agent."""
        # We need to mock the import inside __init__
        with patch('core.learning.active_inference.ActiveInferenceAgent') as MockAgent:
            mock_instance = MockAgent.return_value
            
            # Mock select_action return value tuple (action, info)
            mock_ai_action = Mock()
            mock_ai_action.action_type.name = "QUERY_SEARCH"
            mock_ai_action.target = "test_target"
            mock_ai_action.parameters = {}
            mock_ai_action.expected_information_gain = 0.5
            mock_ai_action.expected_risk = 0.1
            
            mock_instance.select_action.return_value = (mock_ai_action, {"selected_EFE": -1.0, "selection_prob": 0.8})
            
            adapter = ActiveInferenceActionAdapter()
            return adapter

    def test_init(self, adapter):
        """Test initialization."""
        assert adapter is not None
        assert adapter.agent is not None

    def test_select_action(self, adapter):
        """Test action selection conversion."""
        loop_state = LoopState(cycle=1, gaps=[{"gap_id": "gap1", "description": "test"}])
        
        action = adapter.select_action(loop_state)
        
        assert action is not None
        assert action.action_type == ActionType.QUERY_SEARCH
        assert action.target == "test_target"
        assert action.source == "active_inference"
        assert action.expected_free_energy == -1.0
        
    def test_fallback_action_when_agent_none(self):
        """Test fallback when agent is not initialized."""
        # Create adapter but force agent to None
        with patch('core.learning.active_inference.ActiveInferenceAgent', side_effect=ImportError("Fail")):
            adapter = ActiveInferenceActionAdapter()
            
        assert adapter.agent is None
        
        loop_state = LoopState(cycle=1)
        action = adapter.select_action(loop_state)
        
        assert action.source == "fallback"
        assert action.action_type == ActionType.QUERY_SEARCH
