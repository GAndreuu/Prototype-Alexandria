
"""
Integration Test: VQ-VAE Model Loading
Verifies model architecture and weight loading with real PyTorch.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure core is importable
from core.reasoning.vqvae.model import MonolithV13

@pytest.mark.integration
class TestModelLoadingIntegration:

    def test_old_architecture(self):
        """Test Old architecture (256D hidden, 384D latent)."""
        model_old = MonolithV13(input_dim=384, hidden_dim=256)
        test_input = torch.randn(1, 384)
        output = model_old(test_input)
        
        assert output['reconstructed'].shape == (1, 384)
        assert output['indices'].shape == (1, 4)

    def test_new_architecture(self):
        """Test New architecture (1024D hidden, 512D latent)."""
        model_new = MonolithV13(input_dim=384, hidden_dim=1024, latent_dim=512)
        test_input = torch.randn(1, 384)
        output = model_new(test_input)
        
        assert output['reconstructed'].shape == (1, 384)
        assert output['indices'].shape == (1, 4)

    def test_wiki_weights_loading(self):
        """Test loading of wiki-trained weights if available."""
        # Locate weights
        project_root = Path(__file__).parent.parent.parent.parent
        wiki_path = project_root / "data" / "monolith_v13_wiki_trained.pth"
        
        if not wiki_path.exists():
            print(f"\nSKIPPING: Wiki model not found at {wiki_path}")
            return # Just return to pass/skip softly
        
        try:
            model_wiki = MonolithV13(input_dim=384, hidden_dim=1024, latent_dim=512)
            state_dict = torch.load(str(wiki_path), map_location='cpu', weights_only=False)
            model_wiki.load_state_dict(state_dict)
            
            # Inference Check
            test_input = torch.randn(10, 384)
            with torch.no_grad():
                output = model_wiki(test_input)
                indices = output['indices'].cpu().numpy()
            
            unique_codes = np.unique(indices)
            assert len(unique_codes) > 0
            
        except Exception as e:
            pytest.fail(f"Failed to load wiki model: {e}")
