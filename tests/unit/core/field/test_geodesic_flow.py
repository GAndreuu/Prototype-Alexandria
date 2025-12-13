"""
Tests for core/field/geodesic_flow.py

REAL TESTS - using actual module instances
"""
import pytest
import numpy as np


class TestGeodesicConfig:
    """Tests for GeodesicConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        from core.field.geodesic_flow import GeodesicConfig
        config = GeodesicConfig()
        
        assert hasattr(config, 'max_steps')
        assert hasattr(config, 'dt')
        assert hasattr(config, 'active_dims')
        assert hasattr(config, 'use_scipy_integrator')
        assert config.max_steps == 200
    
    def test_custom_values(self):
        """Test custom configuration."""
        from core.field.geodesic_flow import GeodesicConfig
        config = GeodesicConfig(max_steps=100, dt=0.01, active_dims=16)
        
        assert config.max_steps == 100
        assert config.dt == 0.01
        assert config.active_dims == 16


class TestGeodesicFlow:
    """Tests for GeodesicFlow with REAL components."""
    
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
    def flow(self, manifold, metric):
        """Create real flow instance."""
        from core.field.geodesic_flow import GeodesicFlow, GeodesicConfig
        config = GeodesicConfig(max_steps=50, active_dims=32)
        return GeodesicFlow(manifold, metric, config)
    
    def test_init(self, flow):
        """Test initialization."""
        assert flow.config is not None
        assert flow.manifold is not None
        assert flow.metric is not None
    
    def test_shortest_path(self, flow):
        """Test shortest path computation."""
        start = np.random.randn(32) * 0.1
        end = np.random.randn(32) * 0.1
        
        path = flow.shortest_path(start, end, max_iterations=5)
        
        assert path is not None
        assert hasattr(path, 'points')
        assert hasattr(path, 'length')
        assert path.points.shape[1] == 32


class TestGeodesicPath:
    """Tests for GeodesicPath dataclass."""
    
    def test_create_path(self):
        """Test path creation."""
        from core.field.geodesic_flow import GeodesicPath
        
        path = GeodesicPath(
            points=np.random.randn(10, 32),
            length=1.5,
            converged=True,
            end_error=0.01,
            best_step=9
        )
        
        assert path.length == 1.5
        assert path.converged == True
        assert path.n_steps == 10
    
    def test_n_steps_property(self):
        """Test n_steps property."""
        from core.field.geodesic_flow import GeodesicPath
        
        path = GeodesicPath(
            points=np.random.randn(25, 64),
            length=2.0,
            converged=False,
            end_error=0.1,
            best_step=20
        )
        
        assert path.n_steps == 25
