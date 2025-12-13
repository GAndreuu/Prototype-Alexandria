"""
Tests for core/field/manifold.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestDynamicManifold:
    """Tests for DynamicManifold class."""
    
    @pytest.fixture
    def manifold(self):
        """Create manifold instance."""
        from core.field.manifold import DynamicManifold, ManifoldConfig
        config = ManifoldConfig(base_dim=384, num_heads=4, codebook_size=256)
        return DynamicManifold(config)
    
    def test_init_dimensions(self, manifold):
        """Test initial dimension setup."""
        assert manifold.config.base_dim == 384
        assert manifold.current_dim == 384
    
    def test_embed_point(self, manifold):
        """Test embedding a high-dim vector."""
        high_dim = np.random.randn(384)
        point = manifold.embed(high_dim)
        
        assert point.coordinates.shape[0] == manifold.current_dim
    
    def test_add_point(self, manifold):
        """Test adding named point."""
        from core.field.manifold import ManifoldPoint
        coords = np.random.randn(384)
        point = ManifoldPoint(coordinates=coords, discrete_codes=np.array([1,2,3,4]))
        manifold.add_point("test_point", point)
        
        assert "test_point" in manifold.points
    
    def test_activate_point(self, manifold):
        """Test point activation."""
        from core.field.manifold import ManifoldPoint
        coords = np.random.randn(384)
        point = ManifoldPoint(coordinates=coords, discrete_codes=np.array([1,2,3,4]))
        manifold.add_point("test", point)
        manifold.activate_point("test", intensity=0.8)
        
        active = manifold.get_active_points()
        assert len(active) > 0
    
    def test_expand_dimension(self, manifold):
        """Test dimension expansion."""
        initial_dim = manifold.current_dim
        manifold.expand_dimension(n_dims=4)
        
        assert manifold.current_dim == initial_dim + 4
    
    def test_contract_dimension(self, manifold):
        """Test dimension contraction."""
        manifold.expand_dimension(8)  # First expand
        expanded_dim = manifold.current_dim
        manifold.contract_dimension(n_dims=4)
        
        assert manifold.current_dim == expanded_dim - 4
    
    def test_decay_activations(self, manifold):
        """Test activation decay."""
        from core.field.manifold import ManifoldPoint
        coords = np.random.randn(384)
        point = ManifoldPoint(coordinates=coords, discrete_codes=np.array([1,2,3,4]), activation=1.0)
        manifold.add_point("decay_test", point)
        manifold.activate_point("decay_test", intensity=1.0)
        
        manifold.decay_activations(rate=0.5)
        
        point_after = manifold.get_point("decay_test")
        assert point_after.activation < 1.0
    
    def test_get_neighbors(self, manifold):
        """Test neighbor retrieval."""
        from core.field.manifold import ManifoldPoint
        # Add multiple points
        for i in range(10):
            coords = np.random.randn(384)
            point = ManifoldPoint(coordinates=coords, discrete_codes=np.array([i%256,0,0,0]))
            manifold.add_point(f"p{i}", point)
        
        query_point = manifold.get_point("p0")
        neighbors = manifold.get_neighbors(query_point, k=3)
        
        assert len(neighbors) <= 3
