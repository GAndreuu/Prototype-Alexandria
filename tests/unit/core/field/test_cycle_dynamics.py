"""
Tests for core/field/cycle_dynamics.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestCycleDynamics:
    """Tests for CycleDynamics class."""
    
    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies."""
        manifold = Mock()
        manifold.current_dim = 32
        manifold.config = Mock(base_dim=32)
        manifold.points = {}
        manifold.get_active_points = Mock(return_value=[])
        manifold.expand_dimension = Mock()
        manifold.contract_dimension = Mock()
        manifold.decay_activations = Mock()
        
        metric = Mock()
        metric.relax = Mock()
        metric.deform_at_point = Mock()
        metric.stats = Mock(return_value={})
        
        field = Mock()
        field.stats = Mock(return_value={'mean_F': 0.5})
        field.compute_field = Mock(return_value=Mock(mean_free_energy=0.5, num_attractors=3))
        field.set_temperature = Mock()
        field.descend = Mock(return_value=np.random.randn(32))
        field.get_state = Mock(return_value=Mock(attractors=[]))
        
        flow = Mock()
        
        return manifold, metric, field, flow
    
    @pytest.fixture
    def dynamics(self, mock_deps):
        """Create cycle dynamics instance."""
        from core.field.cycle_dynamics import CycleDynamics, CycleConfig
        manifold, metric, field, flow = mock_deps
        config = CycleConfig()
        return CycleDynamics(manifold, metric, field, flow, config)
    
    def test_init(self, dynamics):
        """Test initialization."""
        assert dynamics.cycle_count == 0
        assert len(dynamics.history) == 0
    
    def test_run_cycle_increments_count(self, dynamics):
        """Test cycle count increments."""
        initial = dynamics.cycle_count
        dynamics.run_cycle()
        
        assert dynamics.cycle_count == initial + 1
    
    def test_run_cycle_returns_state(self, dynamics):
        """Test cycle returns state."""
        state = dynamics.run_cycle()
        
        assert hasattr(state, 'phase')
        assert hasattr(state, 'cycle_number')
        assert hasattr(state, 'free_energy_delta')
    
    def test_trigger_cycle(self, dynamics):
        """Test triggered cycle."""
        trigger = np.random.randn(384)
        dynamics.manifold.embed = Mock(return_value=Mock(coordinates=np.random.randn(32)))
        dynamics.manifold.add_point = Mock()
        dynamics.manifold.activate_point = Mock()
        
        state = dynamics.trigger_cycle(trigger)
        
        assert state.cycle_number > 0
    
    def test_continuous_cycles(self, dynamics):
        """Test multiple cycles."""
        states = dynamics.continuous_cycles(n_cycles=3)
        
        assert len(states) == 3
        assert dynamics.cycle_count == 3
    
    def test_get_current_state(self, dynamics):
        """Test current state retrieval."""
        state = dynamics.get_current_state()
        
        assert isinstance(state, dict)
        assert 'cycle_count' in state
    
    def test_get_history_summary(self, dynamics):
        """Test history summary."""
        dynamics.run_cycle()
        dynamics.run_cycle()
        
        summary = dynamics.get_history_summary()
        
        assert isinstance(summary, dict)
        assert summary['cycles'] == 2
    
    def test_reset(self, dynamics):
        """Test reset method."""
        dynamics.run_cycle()
        dynamics.reset()
        
        assert dynamics.cycle_count == 0
        assert len(dynamics.history) == 0
