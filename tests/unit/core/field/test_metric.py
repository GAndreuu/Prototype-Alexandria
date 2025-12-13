"""
Tests for core/field/metric.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestRiemannianMetric:
    """Tests for RiemannianMetric class."""
    
    @pytest.fixture
    def manifold(self):
        """Create mock manifold."""
        from core.field.manifold import DynamicManifold, ManifoldConfig
        config = ManifoldConfig(base_dim=32)
        return DynamicManifold(config)
    
    @pytest.fixture
    def metric(self, manifold):
        """Create metric instance."""
        from core.field.metric import RiemannianMetric, MetricConfig
        config = MetricConfig(deformation_radius=0.8, deformation_strength=0.5)
        return RiemannianMetric(manifold, config)
    
    def test_init_with_manifold(self, metric):
        """Test metric initializes with manifold."""
        assert metric.manifold is not None
        assert metric.config.deformation_strength == 0.5
    
    def test_metric_at(self, metric):
        """Test metric tensor computation."""
        point = np.random.randn(32)
        g = metric.metric_at(point)
        
        assert g.shape == (32, 32)
        # Should be symmetric
        assert np.allclose(g, g.T)
    
    def test_deform_at(self, metric):
        """Test local deformation of metric."""
        point = np.random.randn(32)
        
        # Add a deformation
        metric.deform_at(point, intensity=1.0)
        
        # Should have deformations
        assert len(metric.deformations) > 0
    
    def test_distance(self, metric):
        """Test distance computation."""
        p1 = np.random.randn(32)
        p2 = np.random.randn(32)
        
        dist = metric.distance(p1, p2)
        
        assert dist >= 0
    
    def test_relax_deformations(self, metric):
        """Test relaxation of metric deformations."""
        point = np.random.randn(32)
        metric.deform_at(point, intensity=1.0)
        
        metric.relax(rate=0.5)
        # Should still have some deformations but reduced
    
    def test_clear_deformations(self, metric):
        """Test clearing deformations."""
        point = np.random.randn(32)
        metric.deform_at(point, intensity=1.0)
        
        metric.clear_deformations()
        
        assert len(metric.deformations) == 0


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        from core.field.metric import MetricConfig
        config = MetricConfig()
        
        assert hasattr(config, 'deformation_radius')
        assert hasattr(config, 'deformation_strength')
        assert hasattr(config, 'decay_rate')
