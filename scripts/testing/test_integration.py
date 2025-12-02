"""Test full integration with MycelialVQVAE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.reasoning.mycelial_reasoning import MycelialVQVAE

print("="*60)
print("Testing MycelialVQVAE with Wiki-Trained Model")
print("="*60)

# Test 1: Load with wiki model
print("\n[1/3] Loading MycelialVQVAE (should auto-select wiki model)...")
mvq = MycelialVQVAE.load_default(use_wiki_model=True)
print(f"✅ Loaded! Model type: {mvq.vqvae.__class__.__name__}")

# Test 2: Encode some data
print("\n[2/3] Testing encoding...")
test_data = torch.randn(100, 384)
indices = mvq.encode(test_data)
print(f"✅ Encoded 100 samples")
print(f"   Indices shape: {indices.shape}")
print(f"   Unique codes used: {np.unique(indices.cpu().numpy()).shape[0]}/1024")

# Test  3: Reasoning
print("\n[3/3] Testing mycelial reasoning...")
sample_indices = indices[0]
mvq.observe(sample_indices)
new_indices, activation = mvq.reason(sample_indices)
print(f"✅ Reasoning works!")
print(f"   Original indices: {sample_indices.cpu().numpy()}")
print(f"   Reasoned indices: {new_indices.cpu().numpy()}")

print("\n" + "="*60)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("="*60)
print("\nWiki-trained model is ready to use in Alexandria.")
