"""
Tests for core/field/free_energy_field.py

REAL TESTS - using actual module instances
"""
import pytest
import numpy as np


class TestFieldConfig:
    """Tests for FieldConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        from core.field.free_energy_field import FieldConfig
        config = FieldConfig()
        
        assert hasattr(config, 'temperature')
        assert hasattr(config, 'energy_scale')
        assert config.temperature == 1.0


class TestFreeEnergyField:
    """Tests for FreeEnergyField with REAL components."""
    
    @pytest.fixture
    def manifold(self):
        """Create real manifold instance."""
        from core.field.manifold import DynamicManifold, ManifoldConfig
        config = ManifoldConfig(base_dim=32)
        return DynamicManifold(config)
    
    @pytest.fixture
    def metric(self, manifold):
        """Create real metric instance."""
        from core.field.metric import RiemannianMetric, MetricConfig
        config = MetricConfig()
        return RiemannianMetric(manifold, config)
    
    @pytest.fixture
    def field(self, manifold, metric):
        """Create real field instance."""
        from core.field.free_energy_field import FreeEnergyField, FieldConfig
        config = FieldConfig(temperature=1.0)
        return FreeEnergyField(manifold, metric, config)
    
    def test_init(self, field):
        """Test initialization."""
        assert field.config is not None
        assert field.config.temperature == 1.0
        assert field.manifold is not None
        assert field.metric is not None
    
    def test_energy_at(self, field):
        """Test energy computation at a point."""
        point = np.random.randn(32)
        
        energy = field.energy_at(point)
        
        assert isinstance(energy, (float, np.floating))
    
    def test_entropy_at(self, field):
        """Test entropy computation at a point."""
        point = np.random.randn(32)
        
        entropy = field.entropy_at(point)
        
        assert isinstance(entropy, (float, np.floating))
        assert entropy >= 0  # Entropy should be non-negative
    
    def test_free_energy_formula(self, field):
        """Test F = E - T*S relationship."""
        point = np.random.randn(32)
        
        E = field.energy_at(point)
        S = field.entropy_at(point)
        F = field.free_energy_at(point)
        T = field.config.temperature
        
        # F should approximately equal E - T*S
        expected = E - T * S
        assert np.isclose(F, expected, rtol=0.01)
    
    def test_gradient_shape(self, field):
        """Test gradient returns correct shape."""
        point = np.random.randn(32)
        
        grad = field.gradient_at(point)
        
        assert grad.shape == (32,)
    
    def test_set_temperature(self, field):
        """Test temperature adjustment."""
        field.set_temperature(2.5)
        
        assert field.config.temperature == 2.5
    
    def test_descend(self, field):
        """Test gradient descent."""
        start = np.random.randn(32)
        
        final = field.descend(start, step_size=0.01, steps=5)
        
        assert final.shape == (32,)
        # Final point should be different from start (unless at minimum)
    
    def test_compute_field(self, field):
        """Test full field computation."""
        # Use random grid points
        grid = np.random.randn(10, 32)
        
        state = field.compute_field(grid)
        
        assert state is not None
        assert hasattr(state, 'free_energy_field')
        assert state.free_energy_field.shape == (10,)
