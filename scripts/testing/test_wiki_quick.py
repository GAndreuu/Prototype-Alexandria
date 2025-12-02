"""Quick test for wiki model loading."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from core.reasoning.vqvae.model_wiki import MonolithWiki

# Load model
print("Loading MonolithWiki...")
model = MonolithWiki()

# Load weights
print("Loading weights...")
state = torch.load('data/monolith_v13_wiki_trained.pth', map_location='cpu', weights_only=False)
model.load_state_dict(state, strict=False)

print("✅ Wiki model loaded successfully!")

# Test forward pass
print("\nTesting forward pass...")
x = torch.randn(10, 384)
out = model(x)

print(f"✅ Forward pass works!")
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {out['reconstructed'].shape}")
print(f"   Indices shape: {out['indices'].shape}")
print(f"   Unique codes: {torch.unique(out['indices']).numel()}/1024")
