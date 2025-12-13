"""
Tests for core/loop/action_selection.py
"""
import pytest
from core.loop.action_selection import ActionType, AgentAction, LoopState

class TestActionSelectionTypes:
    """Tests for Action Selection Types and Structures."""
    
    def test_action_types_exist(self):
        """Test defined action types exist."""
        expected = [
            'QUERY_SEARCH', 'EXPLORE_CLUSTER', 'DEEPEN_TOPIC',
            'BRIDGE_CONCEPTS', 'FILL_GAP', 'CONSOLIDATE',
            'FOLLOW_CONNECTION', 'REST'
        ]
        for name in expected:
            assert hasattr(ActionType, name)
        
    def test_agent_action_creation(self):
        """Test creation of AgentAction dataclass."""
        action = AgentAction(
            action_type=ActionType.EXPLORE_CLUSTER,
            target="cluster_123",
            parameters={"depth": 2}
        )
        assert action.target == "cluster_123"
        assert action.confidence == 0.5 # default
        assert action.source == "heuristic" # default
        
    def test_agent_action_to_dict(self):
        """Test serialization of AgentAction."""
        action = AgentAction(
            action_type=ActionType.QUERY_SEARCH,
            target="query",
            parameters={"q": "test"},
            expected_free_energy=-1.5,
            source="test_source"
        )
        data = action.to_dict()
        assert data['action_type'] == "QUERY_SEARCH"
        assert data['target'] == "query"
        assert data['expected_free_energy'] == -1.5
        assert data['source'] == "test_source"
        
    def test_loop_state_init(self):
        """Test LoopState container initialization."""
        state = LoopState(cycle=1)
        assert state.cycle == 1
        assert isinstance(state.gaps, list)
        assert isinstance(state.hypotheses, list)
        assert isinstance(state.recent_actions, list)
