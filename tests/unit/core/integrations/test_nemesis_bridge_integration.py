
import pytest
import numpy as np
from unittest.mock import MagicMock
from core.integrations.nemesis_bridge_integration import (
    NemesisBridgeIntegration, NemesisBridgeConfig, GeometricAction, GeometricEFE
)

# Mock classes
class MockAnchor:
    def __init__(self, idx):
        self.global_idx = idx

@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    bridge.config.embedding_dim = 128
    
    # Mock projection
    bridge._project_to_latent.side_effect = lambda x: x
    
    # Mock nearest anchors
    def get_nearest(point, k=1):
        # Return k mock anchors with random distances
        return [(MockAnchor(i), 0.1 * i) for i in range(k)]
    
    bridge.get_nearest_anchors.side_effect = get_nearest
    
    # Mock compute_metric_deformation
    bridge.compute_metric_deformation.return_value = np.eye(128)
    
    return bridge

@pytest.fixture
def nbi(mock_bridge):
    return NemesisBridgeIntegration(mock_bridge)

def test_initialization(nbi, mock_bridge):
    assert nbi.bridge == mock_bridge
    assert isinstance(nbi.config, NemesisBridgeConfig)
    assert nbi.nemesis is None

def test_select_action_geometric(nbi):
    gaps = [{'gap_id': 'g1', 'description': 'desc', 'priority_score': 0.8}]
    hypotheses = [{'id': 'h1', 'hypothesis_text': 'hyp', 'confidence_score': 0.7}]
    current_state = np.zeros(128)
    
    action = nbi.select_action_geometric(gaps, hypotheses, current_state)
    
    assert isinstance(action, GeometricAction)
    assert isinstance(action.geometric_efe, GeometricEFE)
    assert action.geometric_efe.total != float('inf')
    
    # Verify history
    assert len(nbi.get_action_history()) == 1

def test_select_action_geometric_fallback(nbi, mock_bridge):
    # Test with no bridge response or failure
    mock_bridge.get_nearest_anchors.side_effect = Exception("Bridge fail")
    
    gaps = []
    hypotheses = []
    current_state = np.zeros(128)
    
    # Should default to explore action
    action = nbi.select_action_geometric(gaps, hypotheses, current_state)
    
    assert action.action_type == "explore"

def test_update_beliefs_geometric(nbi):
    obs = np.random.randn(128)
    action = GeometricAction(
        action_type="test", target="t", parameters={},
        geometric_efe=MagicMock(), geodesic_path=np.zeros((5, 128))
    )
    
    metrics = nbi.update_beliefs_geometric(obs, action, 1.0)
    
    assert isinstance(metrics, dict)
    assert "prediction_error" in metrics
    assert nbi._current_belief is obs

def test_action_sorting(nbi):
    # Verify that higher priority action is selected
    action1 = GeometricAction(
        action_type="a1", target="t1", parameters={}, 
        geometric_efe=GeometricEFE(total=10.0, risk=0, ambiguity=0, curvature_term=0, field_energy=0, geodesic_length=0, attractors_visited=0), 
        geometric_value=0.0
    ) # Priority = -10
    
    action2 = GeometricAction(
        action_type="a2", target="t2", parameters={}, 
        geometric_efe=GeometricEFE(total=1.0, risk=0, ambiguity=0, curvature_term=0, field_energy=0, geodesic_length=0, attractors_visited=0), 
        geometric_value=5.0
    ) # Priority = -1 + 5 = 4
    
    # Manually testing priority property since we can't easily mock internal selection logic without mocking generate_candidates
    assert action2.priority > action1.priority

def test_stats(nbi):
    stats = nbi.stats()
    assert stats["has_bridge"] is True
    assert stats["action_history_size"] == 0
