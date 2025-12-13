"""
Tests for VQ-VAE Reasoning
"""
import pytest
import torch
from core.reasoning.vqvae.model import MonolithV13
from core.reasoning.vqvae.layers import OrthogonalProductQuantizer

class TestOrthogonalProductQuantizer:
    def test_init(self):
        # num_embeddings, embedding_dim, num_heads
        quantizer = OrthogonalProductQuantizer(256, 256, 4)
        assert quantizer is not None

    def test_forward_shape(self):
        quantizer = OrthogonalProductQuantizer(256, 256, 4)
        x = torch.randn(2, 256)
        out, diff, ind = quantizer(x)
        assert out.shape == x.shape

class TestMonolithV13:
    def test_init(self):
        model = MonolithV13()
        assert model is not None
        
    def test_forward(self):
        model = MonolithV13()
        x = torch.randn(2, 384)
        out = model(x)
        # Check dict keys or type
        assert isinstance(out, dict)
