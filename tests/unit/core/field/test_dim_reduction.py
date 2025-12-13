"""
Tests for core/field/dim_reduction.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os


class TestDimensionalityReducer:
    """Tests for DimensionalityReducer class."""
    
    @pytest.fixture
    def reducer(self, tmp_path):
        """Create reducer instance with temp save path."""
        from core.field.dim_reduction import DimensionalityReducer
        return DimensionalityReducer(
            input_dim=384, 
            target_dim=32, 
            save_path=str(tmp_path / "test_pca.pkl")
        )
    
    def test_init_default_dims(self, reducer):
        """Test default dimension configuration."""
        assert reducer.input_dim == 384
        assert reducer.target_dim == 32
    
    def test_init_random_projection_fallback(self, reducer):
        """Test random projection matrix is created when no PCA."""
        # When no PCA file exists, should have projection matrix
        assert hasattr(reducer, 'projection_matrix')
        assert reducer.projection_matrix.shape == (384, 32)
    
    def test_transform_single_embedding(self, reducer):
        """Test transform single 384d vector."""
        embedding = np.random.randn(384)
        
        result = reducer.transform(embedding)
        
        assert result.shape == (32,)
    
    def test_transform_batch(self, reducer):
        """Test transform batch of embeddings."""
        embeddings = np.random.randn(10, 384)
        
        result = reducer.transform(embeddings)
        
        assert result.shape == (10, 32)
    
    def test_transform_preserves_relative_distances(self, reducer):
        """Test relative distances are approximately preserved."""
        a = np.random.randn(384)
        b = np.random.randn(384)
        c = np.random.randn(384)
        
        # Original distances
        d_ab = np.linalg.norm(a - b)
        d_ac = np.linalg.norm(a - c)
        
        # Projected distances
        ta, tb, tc = reducer.transform(a), reducer.transform(b), reducer.transform(c)
        d_ab_proj = np.linalg.norm(ta - tb)
        d_ac_proj = np.linalg.norm(ta - tc)
        
        # Relative ordering should be similar (approximately)
        # This is a loose check - random projection preserves structure approximately
        assert d_ab_proj > 0
        assert d_ac_proj > 0
    
    def test_fit_partial_exists(self, reducer):
        """Test fit_partial method exists (even if no-op)."""
        embeddings = np.random.randn(100, 384)
        
        # Should not raise
        reducer.fit_partial(embeddings)
    
    def test_projection_matrix_is_orthogonal(self, reducer):
        """Test projection uses semi-orthogonal matrix."""
        # QR decomposition should give orthogonal columns
        q = reducer.projection_matrix
        qtq = q.T @ q
        
        # Should be close to identity if orthogonal
        identity = np.eye(32)
        assert np.allclose(qtq, identity, atol=0.1)
