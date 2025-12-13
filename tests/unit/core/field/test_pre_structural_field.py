"""
Tests for core/field/pre_structural_field.py

REAL INTEGRATION TESTS - using actual module instances
"""
import pytest
import numpy as np


class TestPreStructuralConfig:
    """Tests for PreStructuralConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        from core.field.pre_structural_field import PreStructuralConfig
        config = PreStructuralConfig()
        
        assert config.base_dim == 32
        assert config.input_dim == 384
        assert hasattr(config, 'max_expansion')
        assert hasattr(config, 'deformation_strength')
    
    def test_custom_values(self):
        """Test custom configuration."""
        from core.field.pre_structural_field import PreStructuralConfig
        config = PreStructuralConfig(base_dim=64, input_dim=512)
        
        assert config.base_dim == 64
        assert config.input_dim == 512


class TestPreStructuralField:
    """Tests for PreStructuralField with REAL components."""
    
    @pytest.fixture
    def field(self):
        """Create real field instance."""
        from core.field.pre_structural_field import PreStructuralField, PreStructuralConfig
        config = PreStructuralConfig(base_dim=32, input_dim=384)
        return PreStructuralField(config)
    
    def test_init(self, field):
        """Test initialization creates all components."""
        assert field.config is not None
        assert field.manifold is not None
        assert field.metric is not None
        assert field.field is not None
        assert field.flow is not None
        assert field.cycle is not None
    
    def test_trigger(self, field):
        """Test triggering a concept activates the field."""
        embedding = np.random.randn(384)
        
        state = field.trigger(embedding, intensity=1.0)
        
        assert state is not None
        # Should have field state properties
        assert hasattr(state, 'free_energy_field') or isinstance(state, dict)
    
    def test_run_cycle(self, field):
        """Test running a full expansion-configuration-compression cycle."""
        embedding = np.random.randn(384)
        
        result = field.run_cycle(trigger_embedding=embedding)
        
        assert result is not None
    
    def test_get_free_energy_at(self, field):
        """Test free energy computation at a point."""
        point = np.random.randn(32)  # base_dim
        
        F = field.get_free_energy_at(point)
        
        assert isinstance(F, (float, np.floating))
    
    def test_get_gradient_at(self, field):
        """Test gradient computation."""
        point = np.random.randn(32)
        
        grad = field.get_gradient_at(point)
        
        assert grad.shape == (32,)
    
    def test_get_attractors(self, field):
        """Test getting current attractors."""
        attractors = field.get_attractors()
        
        assert isinstance(attractors, list)
    
    def test_stats(self, field):
        """Test statistics retrieval."""
        stats = field.stats()
        
        assert isinstance(stats, dict)
        assert 'manifold' in stats or 'temperature' in stats or len(stats) > 0
    
    def test_set_temperature(self, field):
        """Test temperature adjustment."""
        field.set_temperature(2.5)
        
        # Should update field temperature
        assert field.field.config.temperature == 2.5
    
    def test_connect_vqvae(self, field):
        """Test VQ-VAE connection with mock model."""
        import torch
        from unittest.mock import Mock
        
        # Create mock VQ-VAE with real tensor structure
        mock_vqvae = Mock()
        mock_vqvae.quantizer = Mock()
        codebooks = torch.randn(4, 256, 128)
        mock_vqvae.quantizer.codebooks = codebooks
        
        # Should connect without error
        field.connect_vqvae(mock_vqvae)
        
        # Bridge should be connected
        assert field.vqvae_bridge is not None
