
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.integrations.geodesic_bridge_integration import (
    GeodesicBridgeIntegration, GeodesicBridgeConfig, SemanticPath, ActivationMap
)

# Mock classes for VQVAEManifoldBridge and Anchor
class MockAnchor:
    def __init__(self, idx, coords):
        self.global_idx = idx
        self.coordinates = coords

@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    bridge.latent_dim = 128
    
    # Mock anchors
    anchors = [MockAnchor(i, np.random.randn(128)) for i in range(10)]
    bridge.anchors = anchors
    
    # Mock methods
    bridge._project_to_latent.side_effect = lambda x: x
    
    def get_nearest(point, k=1):
        dists = [(a, np.linalg.norm(point - a.coordinates)) for a in anchors]
        dists.sort(key=lambda x: x[1])
        return dists[:k]
    
    bridge.get_nearest_anchors.side_effect = get_nearest
    
    return bridge

@pytest.fixture
def gbi(mock_bridge):
    # Mocking all field imports to avoid dependency issues or real implementation side-effects
    with patch.dict('sys.modules', {
        'core.field.geodesic_flow': MagicMock(),
        'core.field.manifold': MagicMock(),
        'core.field.metric': MagicMock()
    }):
         gbi = GeodesicBridgeIntegration(mock_bridge)
         # Force fallback by ensuring geodesic is None for basic tests first
         gbi.geodesic = None 
         return gbi

def test_initialization(gbi, mock_bridge):
    assert gbi.bridge == mock_bridge
    assert isinstance(gbi.config, GeodesicBridgeConfig)
    assert gbi.geodesic is None # As forced in fixture

def test_semantic_path_fallback(gbi):
    start = np.random.randn(128)
    end = np.random.randn(128)
    
    path = gbi.semantic_path(start, end, use_cache=False)
    
    assert isinstance(path, SemanticPath)
    assert path.converged is True
    assert len(path.points) > 0
    # Fallback uses linear path, so geodesic ratio should be 1.0
    assert np.isclose(path.geodesic_ratio, 1.0)

def test_propagate_concept_fallback(gbi):
    source = np.random.randn(128)
    
    activations = gbi.propagate_concept(source, steps=3)
    
    assert isinstance(activations, ActivationMap)
    # With random data and fallback, we might get some activations
    # We check structure mostly
    assert isinstance(activations.activations, dict)
    assert activations.total_energy >= 0.0

def test_geodesic_field_fallback(gbi):
    center = np.random.randn(128)
    field = gbi.geodesic_field(center, n_directions=4)
    
    assert len(field.directions) == 4
    assert len(field.paths) == 4
    assert len(field.lengths) == 4

def test_distance(gbi):
    a = np.random.randn(128)
    b = np.random.randn(128)
    
    dist_geo = gbi.distance(a, b, geodesic=True)
    dist_euc = gbi.distance(a, b, geodesic=False)
    
    # In fallback (linear), they should be equal
    assert np.isclose(dist_geo, dist_euc)

def test_caching(gbi):
    start = np.random.randn(128)
    end = np.random.randn(128)
    
    path1 = gbi.semantic_path(start, end, use_cache=True)
    path2 = gbi.semantic_path(start, end, use_cache=True)
    
    assert path1 is path2 # Should be same object reference
    
    gbi.clear_cache()
    path3 = gbi.semantic_path(start, end, use_cache=True)
    assert path1 is not path3 # Should be new object
