"""
Tests for Model Wiki (VQ-VAE)
"""
import pytest
import torch
from core.reasoning.vqvae.model_wiki import ProductQuantizerSimple, MonolithWiki

class TestMonolithWiki:
    def test_init(self):
        model = MonolithWiki()
        assert model is not None

    def test_forward(self):
        model = MonolithWiki()
        x = torch.randn(2, 384)
        out = model(x)
        assert "reconstructed" in out
        assert out["reconstructed"].shape == x.shape

    def test_dimensions_match_wiki(self):
        model = MonolithWiki(input_dim=384, hidden_dim=512)
        assert model.encoder[0].in_features == 384

class TestProductQuantizerSimple:
    @pytest.fixture
    def quantizer(self):
        # Correct signature: num_heads, codebook_size, head_dim
        return ProductQuantizerSimple(num_heads=4, codebook_size=256, head_dim=128)

    def test_init(self, quantizer):
        assert quantizer.num_heads == 4
        assert quantizer.codebook_size == 256

    def test_forward_shape(self, quantizer):
        # Input dim = num_heads * head_dim = 4 * 128 = 512
        x = torch.randn(2, 512)
        z_q, indices, _ = quantizer(x)
        assert z_q.shape == x.shape
        assert indices.shape == (2, 4)

    def test_codes_in_range(self, quantizer):
        x = torch.randn(2, 512)
        _, indices, _ = quantizer(x)
        assert indices.max() < 256
        assert indices.min() >= 0
