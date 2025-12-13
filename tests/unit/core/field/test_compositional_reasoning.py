"""
Tests for core/field/compositional_reasoning.py

REAL TESTS - using actual VQVAEManifoldBridge
"""
import pytest
import numpy as np


class TestCompositionConfig:
    """Tests for CompositionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        from core.field.compositional_reasoning import CompositionConfig
        config = CompositionConfig()
        
        assert hasattr(config, 'residual_mode')
        assert hasattr(config, 'composition_strategy')
        assert hasattr(config, 'residual_scale')
    
    def test_custom_values(self):
        """Test custom configuration."""
        from core.field.compositional_reasoning import CompositionConfig, ResidualMode
        config = CompositionConfig(
            residual_mode=ResidualMode.ATTENTION,
            residual_scale=0.2
        )
        
        assert config.residual_mode == ResidualMode.ATTENTION
        assert config.residual_scale == 0.2


class TestCompositionalReasoner:
    """Tests for CompositionalReasoner with REAL bridge."""
    
    @pytest.fixture
    def bridge(self):
        """Create real bridge instance."""
        from core.field.vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig
        config = BridgeConfig()
        return VQVAEManifoldBridge(config)
    
    @pytest.fixture
    def reasoner(self, bridge):
        """Create reasoner with real bridge."""
        from core.field.compositional_reasoning import CompositionalReasoner, CompositionConfig
        config = CompositionConfig()
        return CompositionalReasoner(bridge=bridge, config=config)
    
    def test_init(self, reasoner):
        """Test initialization."""
        assert reasoner is not None
        assert reasoner.config is not None
        assert reasoner.bridge is not None
    
    def test_reason_basic(self, reasoner):
        """Test basic reasoning from start to target."""
        start = np.random.randn(384)
        target = np.random.randn(384)
        
        result = reasoner.reason(start, target)
        
        assert result is not None
        assert hasattr(result, 'cumulative_vector')
        assert result.cumulative_vector.shape == (384,)
    
    def test_analogy(self, reasoner):
        """Test analogy a:b :: c:?"""
        a = np.random.randn(384)
        b = np.random.randn(384)
        c = np.random.randn(384)
        
        d, path = reasoner.analogy(a, b, c)
        
        assert d is not None
        assert d.shape == (384,)
        assert path is not None


class TestCompositionalPath:
    """Tests for CompositionalPath dataclass."""
    
    def test_required_fields(self):
        """Test path has required fields."""
        from core.field.compositional_reasoning import CompositionalPath
        
        path = CompositionalPath(
            start_vector=np.random.randn(384),
            target_vector=np.random.randn(384),
            cumulative_vector=np.random.randn(384),
            points=np.random.randn(10, 384),
            residuals=np.random.randn(10, 384),
            path_length=1.5,
            total_residual_magnitude=0.3,
            n_steps=10,
            converged=True,
            composition_trace=["node_0", "node_1"],
            steps=[],
            dominant_contributions=[(0, 0.8), (1, 0.2)]
        )
        
        assert path.path_length == 1.5
        assert len(path.composition_trace) == 2
        assert path.n_steps == 10
        assert path.converged == True
    
    def test_summary(self):
        """Test path summary generation."""
        from core.field.compositional_reasoning import CompositionalPath
        
        path = CompositionalPath(
            start_vector=np.zeros(384),
            target_vector=np.ones(384),
            cumulative_vector=np.ones(384) * 0.5,
            points=np.random.randn(5, 384),
            residuals=np.random.randn(5, 384),
            path_length=2.0,
            total_residual_magnitude=0.5,
            n_steps=5,
            converged=True,
            composition_trace=["step_0"],
            steps=[],
            dominant_contributions=[]
        )
        
        summary = path.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
