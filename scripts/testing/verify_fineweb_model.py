import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.reasoning.vqvae.model_wiki import MonolithWiki

def verify_model():
    model_path = Path("monolith_v3_fineweb (1)/monolith_v3.pt")
    
    if not model_path.exists():
        print(f"❌ Model file not found at: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    
    try:
        # Load state dict
        # weights_only=False needed for some older checkpoints or full models
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            print("Found 'model_state_dict' in checkpoint. Extracting...")
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Instantiate model
        model = MonolithWiki()
        
        print(f"Checkpoint keys (first 20):")
        keys = list(state_dict.keys())
        for k in keys[:20]:
            print(f"  - {k}")
            
        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("✅ Model loaded successfully (Strict match)!")
        else:
            print("⚠️ Model loaded with mismatches:")
            if missing:
                print(f"   Missing keys: {len(missing)}")
                for k in missing[:5]: print(f"    - {k}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
                for k in unexpected[:5]: print(f"    - {k}")
                
            # Check if critical layers are present
            critical_layers = ['quantizer.codebooks', 'encoder.0.weight', 'decoder.0.weight']
            all_good = True
            for layer in critical_layers:
                if layer in missing:
                    print(f"❌ Critical layer missing: {layer}")
                    all_good = False
            
            if all_good:
                print("✅ Critical layers present. Mismatches might be negligible (e.g. unused buffers).")
            else:
                print("❌ Model structure incompatible.")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    verify_model()
