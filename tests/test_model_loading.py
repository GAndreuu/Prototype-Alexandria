"""
Test script to verify VQ-VAE model loading and codebook usage.

Usage:
    python scripts/test_model_loading.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.reasoning.vqvae.model import MonolithV13

def test_model_architectures():
    """Test both old and new model architectures."""
    print("=" * 60)
    print("VQ-VAE Model Architecture Test")
    print("=" * 60)
    
    # Test 1: Old architecture (256D hidden, 384D latent)
    print("\n[1/3] Testing OLD architecture (256D hidden, 384D latent)...")
    try:
        model_old = MonolithV13(input_dim=384, hidden_dim=256)
        test_input = torch.randn(1, 384)
        output = model_old(test_input)
        
        assert output['reconstructed'].shape == (1, 384), "Output shape mismatch"
        assert output['indices'].shape == (1, 4), "Indices shape mismatch"
        print("   ✅ Old architecture works correctly")
        print(f"   - Parameters: {sum(p.numel() for p in model_old.parameters()):,}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: New architecture (1024D hidden, 512D latent)
    print("\n[2/3] Testing NEW architecture (1024D hidden, 512D latent)...")
    try:
        model_new = MonolithV13(input_dim=384, hidden_dim=1024, latent_dim=512)
        test_input = torch.randn(1, 384)
        output = model_new(test_input)
        
        assert output['reconstructed'].shape == (1, 384), "Output shape mismatch"
        assert output['indices'].shape == (1, 4), "Indices shape mismatch"
        print("   ✅ New architecture works correctly")
        print(f"   - Parameters: {sum(p.numel() for p in model_new.parameters()):,}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Load wiki-trained weights
    print("\n[3/3] Testing wiki-trained model loading...")
    wiki_path = Path(__file__).parent.parent / "data" / "monolith_v13_wiki_trained.pth"
    
    if not wiki_path.exists():
        print(f"   ⚠️  Wiki model not found at {wiki_path}")
        print("   Skipping weight loading test")
    else:
        try:
            model_wiki = MonolithV13(input_dim=384, hidden_dim=1024, latent_dim=512)
            state_dict = torch.load(str(wiki_path), map_location='cpu', weights_only=False)
            model_wiki.load_state_dict(state_dict)
            
            # Test inference
            test_input = torch.randn(10, 384)
            with torch.no_grad():
                output = model_wiki(test_input)
                indices = output['indices'].cpu().numpy()
            
            # Check codebook usage
            unique_codes = np.unique(indices)
            usage_pct = len(unique_codes) / (4 * 256) * 100
            
            print("   ✅ Wiki-trained model loaded successfully")
            print(f"   - Codebook usage on 10 samples: {len(unique_codes)}/1024 codes ({usage_pct:.1f}%)")
            print(f"   - Reconstruction loss: {torch.nn.functional.mse_loss(output['reconstructed'], test_input):.6f}")
        except Exception as e:
            print(f"   ❌ FAILED to load wiki model: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_architectures()
    sys.exit(0 if success else 1)
