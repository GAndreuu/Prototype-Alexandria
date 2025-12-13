"""
Tests for core/reasoning/vqvae/layers.py
"""
import pytest
import torch

class TestOrthogonalProductQuantizer:
    """Tests for OrthogonalProductQuantizer class."""
    
    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance."""
        from core.reasoning.vqvae.layers import OrthogonalProductQuantizer
        return OrthogonalProductQuantizer(
            embedding_dim=256,
            num_embeddings=256,
            num_heads=4
        )
    
    def test_init(self, quantizer):
        """Test initialization."""
        assert quantizer.num_heads == 4
        assert quantizer.num_embeddings == 256
        assert quantizer.head_dim == 64  # 256 / 4
    
    def test_codebooks_shape(self, quantizer):
        """Test codebook tensor shape."""
        # Each head has its own codebook
        assert hasattr(quantizer, 'codebooks')
        assert quantizer.codebooks.shape == (4, 256, 64)  # heads, codes, head_dim
    
    def test_forward_returns_tuple(self, quantizer):
        """Test forward returns (z_q, indices, distances)."""
        x = torch.randn(16, 256)
        
        result = quantizer(x)
        
        assert len(result) == 3
        z_q, indices, distances = result
        assert z_q.shape == (16, 256)
        assert indices.shape == (16, 4)
        assert distances.shape == (16, 4, 256)
    
    def test_quantization_deterministic(self, quantizer):
        """Test same input gives same output (no randomness)."""
        x = torch.randn(8, 256)
        
        z_q1, indices1, _ = quantizer(x)
        z_q2, indices2, _ = quantizer(x)
        
        assert torch.allclose(z_q1, z_q2)
        assert torch.equal(indices1, indices2)
    
    def test_indices_in_range(self, quantizer):
        """Test all indices are valid codebook indices."""
        x = torch.randn(32, 256)
        
        _, indices, _ = quantizer(x)
        
        assert indices.min() >= 0
        assert indices.max() < 256
    
    def test_straight_through_gradient(self, quantizer):
        """Test gradients flow via straight-through estimator."""
        x = torch.randn(8, 256, requires_grad=True)
        
        z_q, _, _ = quantizer(x)
        total_loss = z_q.sum()
        total_loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

class TestQuantizerOrthogonality:
    """Tests for orthogonality properties."""
    
    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance."""
        from core.reasoning.vqvae.layers import OrthogonalProductQuantizer
        return OrthogonalProductQuantizer(embedding_dim=256, num_embeddings=256, num_heads=4)
    
    def test_heads_are_independent(self, quantizer):
        """Test different heads can have different codes."""
        x = torch.randn(100, 256)
        
        _, indices, _ = quantizer(x)
        
        # Correlation between heads should be low because input is random
        for i in range(4):
            for j in range(i+1, 4):
                corr = torch.corrcoef(torch.stack([
                    indices[:, i].float(),
                    indices[:, j].float()
                ]))[0, 1]
                # Not perfectly correlated
                assert abs(corr.item()) < 0.95
