"""
Experiment B: Re-scaling Test
Tests decoder dependency on head asymmetry by normalizing norms without retraining.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from core.reasoning.vqvae.model import MonolithV13

def experiment_B_rescaling(model_path="data/monolith_v13_trained.pth",
                           data_path="data/training_embeddings.npy",
                           sample_size=10000):
    """
    Force all heads to have equal mean norm and measure MSE impact.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MonolithV13()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    # Load data
    full_data = np.load(data_path)
    idx = np.random.choice(len(full_data), min(sample_size, len(full_data)), replace=False)
    data = torch.from_numpy(full_data[idx]).float().to(device)
    
    print(f"🧪 Experiment B: Re-scaling Test")
    print(f"   Samples: {len(data)}")
    print()
    
    with torch.no_grad():
        # Forward pass
        out = model(data)
        z_q = out['z_q']
        recon_orig = out['reconstructed']
        mse_orig = torch.nn.functional.mse_loss(recon_orig, data).item()
        
        # Reshape to heads
        B, D = z_q.shape
        H = 4
        Hd = D // H
        z_q_heads = z_q.view(B, H, Hd)
        
        # Compute current norms
        norms = (z_q_heads ** 2).sum(dim=-1).mean(dim=0)  # [H]
        
        print("📊 Original Head Norms:")
        for h in range(H):
            print(f"   Head {h}: {norms[h].item():.2f}")
        print()
        
        # Compute scaling factors for equal norms
        target_norm = norms.mean()
        scale = torch.sqrt(target_norm / (norms + 1e-8))
        
        print("🔧 Scaling Factors (to equalize norms):")
        for h in range(H):
            print(f"   Head {h}: {scale[h].item():.4f}")
        print()
        
        # Apply scaling
        z_q_bal = z_q_heads * scale.view(1, H, 1)
        z_q_bal = z_q_bal.view(B, D)
        
        # Decode with balanced z_q
        recon_bal = model.decoder(z_q_bal)
        mse_bal = torch.nn.functional.mse_loss(recon_bal, data).item()
        
        # Verify balanced norms
        norms_bal = (z_q_bal.view(B, H, Hd) ** 2).sum(dim=-1).mean(dim=0)
        
        print("✅ Balanced Head Norms:")
        for h in range(H):
            print(f"   Head {h}: {norms_bal[h].item():.2f}")
        print()
        
    delta_mse = mse_bal - mse_orig
    pct_increase = (delta_mse / mse_orig) * 100
    
    print("📈 Results:")
    print(f"   Original MSE: {mse_orig:.6f}")
    print(f"   Balanced MSE: {mse_bal:.6f}")
    print(f"   ΔMSE: {delta_mse:+.6f} ({pct_increase:+.2f}%)")
    print()
    
    if abs(pct_increase) < 5:
        print("💡 Decoder is NOT heavily dependent on asymmetry (ΔMSE < 5%)")
    elif pct_increase > 0:
        print("⚠️  Decoder is optimized for asymmetry (MSE increased)")
    else:
        print("✨ Balancing actually IMPROVED performance!")
    
    return {
        'mse_orig': mse_orig,
        'mse_bal': mse_bal,
        'delta_mse': delta_mse,
        'pct_increase': pct_increase,
        'norms_orig': norms.cpu().numpy(),
        'norms_bal': norms_bal.cpu().numpy()
    }

if __name__ == "__main__":
    experiment_B_rescaling()
