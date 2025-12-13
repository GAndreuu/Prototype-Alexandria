"""
Tests for core/learning/meta_hebbian.py
"""
import pytest
import numpy as np
from core.learning.meta_hebbian import MetaHebbianPlasticity, MetaHebbianConfig

class TestMetaHebbian:
    """Tests for MetaHebbianPlasticity class."""
    
    @pytest.fixture
    def meta(self):
        """Create MetaHebbian instance."""
        config = MetaHebbianConfig(num_heads=2)
        return MetaHebbianPlasticity(config)

    def test_init(self, meta):
        """Test initialization."""
        assert meta is not None
        assert len(meta.rules) >= 1

    def test_compute_weight_update(self, meta):
        """Test weight update calculation."""
        weights = np.zeros((10, 10))
        pre = np.random.randn(10)
        post = np.random.randn(10)
        
        updated = meta.compute_weight_update(weights, pre, post, head_idx=0)
        
        assert updated.shape == (10, 10)
        # Weights should change (unless lr is 0, which it isn't)
        assert not np.allclose(weights, updated)

    def test_evolve_rules(self, meta):
        """Test rule evolution."""
        fitness_scores = [0.1, 0.2, 0.3, 0.4]
        stats = meta.evolve_rules(fitness_scores)
        
        assert stats['rules_updated'] is True
        assert 'generation' in stats

    def test_save_load_state(self, meta, tmp_path):
        """Test state persistence."""
        save_path = tmp_path / "meta_state.pkl"
        
        # Save
        path = meta.save_state(str(save_path))
        assert str(path) == str(save_path)
        
        # Load
        new_meta = MetaHebbianPlasticity(meta.config)
        success = new_meta.load_state(str(save_path))
        
        assert success is True
        assert new_meta._generation == meta._generation
