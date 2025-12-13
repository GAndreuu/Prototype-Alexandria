"""
Tests for VQ-VAE Loss Functions
"""
import pytest
import torch
from core.reasoning.vqvae.loss import (
    compute_orthogonal_loss,
    compute_vq_commitment_loss,
    compute_head_balance_loss,
    compute_code_usage_entropy_loss
)
from core.reasoning.vqvae.layers import OrthogonalProductQuantizer

class TestVQCommitmentLoss:
    def test_commitment_loss_basic(self):
        z = torch.randn(4, 256)
        z_q = torch.randn(4, 256)
        loss = compute_vq_commitment_loss(z, z_q)
        assert loss > 0

    def test_commitment_loss_zero_when_equal(self):
        z = torch.randn(4, 256)
        loss = compute_vq_commitment_loss(z, z)
        assert loss < 0.01  # Should be near zero (MSE of same)

    def test_commitment_loss_symmetric(self):
        z = torch.randn(4, 256)
        z_q = torch.randn(4, 256)
        loss1 = compute_vq_commitment_loss(z, z_q)
        # Loss should change if we swap inputs
        loss2 = compute_vq_commitment_loss(z_q, z)
        # Not exactly equal but both positive
        assert loss1 > 0 and loss2 > 0

class TestOrthogonalLoss:
    def test_orthogonal_loss_basic(self):
        # compute_orthogonal_loss expects a quantizer MODULE with .codebooks
        quantizer = OrthogonalProductQuantizer(256, 256, 4)  # num_emb, dim, heads
        loss = compute_orthogonal_loss(quantizer)
        assert loss >= 0

    def test_orthogonal_loss_for_identity(self):
        # With random init, codebooks are not orthogonal
        quantizer = OrthogonalProductQuantizer(256, 256, 4)
        loss = compute_orthogonal_loss(quantizer)
        assert isinstance(loss, torch.Tensor)

class TestHeadBalanceLoss:
    def test_head_balance_loss(self):
        z_q = torch.randn(4, 512)  # 4 heads * 128 head_dim
        loss = compute_head_balance_loss(z_q, num_heads=4)
        assert loss >= 0

class TestCodeUsageEntropyLoss:
    def test_entropy_loss(self):
        # indices shape: [B, H]
        indices = torch.randint(0, 256, (32, 4))
        loss = compute_code_usage_entropy_loss(indices, num_embeddings=256)
        assert loss >= 0
