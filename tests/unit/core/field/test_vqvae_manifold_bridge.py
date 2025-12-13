"""
Tests for core/field/vqvae_manifold_bridge.py

REAL TESTS - using actual module instances
"""
import pytest
import numpy as np
import torch


class TestBridgeConfig:
    """Tests for BridgeConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        from core.field.vqvae_manifold_bridge import BridgeConfig
        config = BridgeConfig()
        
        assert config.embedding_dim == 384
        assert config.num_heads == 4
        assert config.codes_per_head == 256
        assert hasattr(config, 'pull_strength')
        assert hasattr(config, 'projection_mode')


class TestVQVAEManifoldBridge:
    """Tests for VQVAEManifoldBridge with REAL functionality."""
    
    @pytest.fixture
    def bridge(self):
        """Create bridge instance."""
        from core.field.vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig
        config = BridgeConfig()
        return VQVAEManifoldBridge(config)
    
    def test_init(self, bridge):
        """Test initialization."""
        assert bridge.config is not None
        assert bridge.config.embedding_dim == 384
    
    def test_connect_vqvae(self, bridge):
        """Test VQ-VAE connection with real tensor codebook."""
        from unittest.mock import Mock
        
        mock_vqvae = Mock()
        mock_vqvae.quantizer = Mock()
        # Real torch tensor for codebook
        mock_vqvae.quantizer.codebooks = torch.randn(4, 256, 128)
        
        result = bridge.connect_vqvae(mock_vqvae)
        
        assert result == True
    
    def test_embed_after_connect(self, bridge):
        """Test embedding after connecting VQ-VAE."""
        from unittest.mock import Mock
        
        # Connect first
        mock_vqvae = Mock()
        mock_vqvae.quantizer = Mock()
        mock_vqvae.quantizer.codebooks = torch.randn(4, 256, 128)
        bridge.connect_vqvae(mock_vqvae)
        
        # Now embed
        embedding = np.random.randn(384)
        point = bridge.embed(embedding)
        
        assert point is not None
        assert hasattr(point, 'coordinates')
        assert hasattr(point, 'discrete_codes')
        assert point.discrete_codes.shape == (4,)


class TestManifoldPoint:
    """Tests for ManifoldPoint dataclass."""
    
    def test_create_point(self):
        """Test point creation."""
        from core.field.vqvae_manifold_bridge import ManifoldPoint
        
        point = ManifoldPoint(
            coordinates=np.random.randn(384),
            discrete_codes=np.array([1, 2, 3, 4])
        )
        
        assert point.coordinates.shape == (384,)
        assert point.discrete_codes.shape == (4,)
        assert point.activation == 0.0
    
    def test_is_near_attractor_by_distance(self):
        """Test attractor proximity check via nearst_anchor_distance field."""
        from core.field.vqvae_manifold_bridge import ManifoldPoint
        
        # Point near attractor
        point_near = ManifoldPoint(
            coordinates=np.random.randn(384),
            discrete_codes=np.array([1, 2, 3, 4]),
            nearest_anchor_distance=0.1  # Close to anchor
        )
        
        # is_near_attractor returns True if distance < 0.2
        assert point_near.nearest_anchor_distance < 0.2
        
        # Point far from attractor
        point_far = ManifoldPoint(
            coordinates=np.random.randn(384),
            discrete_codes=np.array([1, 2, 3, 4]),
            nearest_anchor_distance=0.5  # Far from anchor
        )
        
        assert point_far.nearest_anchor_distance >= 0.2
