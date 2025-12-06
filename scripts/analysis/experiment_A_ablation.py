"""
Experiment A: Real Head Ablation via Decoder Masking
Measures true impact of each head on reconstruction MSE.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from core.reasoning.vqvae.model import MonolithV13

def experiment_A_ablation(model_path="data/monolith_v13_trained.pth",
                          data_path="data/training_embeddings.npy",
                          sample_size=10000):
    """
    Ablate each head individually and measure ΔMSE.
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
    
    print(f"🧪 Experiment A: Real Head Ablation")
    print(f"   Samples: {len(data)}")
    print()
    
    # Baseline (no masking)
    with torch.no_grad():
        out_full = model(data)
        recon_full = out_full['reconstructed']
        mse_full = torch.nn.functional.mse_loss(recon_full, data).item()
    
    print(f"📊 Baseline MSE (all heads): {mse_full:.6f}")
    print()
    
    # Ablate each head
    results = {}
    for h in range(4):
        # Create mask: all 1s except head h
        mask = torch.ones(4, device=device)
        mask[h] = 0.0
        
        with torch.no_grad():
            out_masked = model.forward_with_head_mask(data, mask)
            recon_masked = out_masked['reconstructed']
            mse_masked = torch.nn.functional.mse_loss(recon_masked, data).item()
        
        delta_mse = mse_masked - mse_full
        pct_increase = (delta_mse / mse_full) * 100
        
        results[h] = {
            'mse': mse_masked,
            'delta_mse': delta_mse,
            'pct_increase': pct_increase
        }
        
        print(f"   Head {h} ablated:")
        print(f"      MSE: {mse_masked:.6f}")
        print(f"      ΔMSE: {delta_mse:+.6f} ({pct_increase:+.2f}%)")
        print()
    
    # Rank by importance
    ranked = sorted(results.items(), key=lambda x: x[1]['delta_mse'], reverse=True)
    
    print("🏆 Head Importance Ranking (by ΔMSE):")
    for rank, (h, res) in enumerate(ranked, 1):
        print(f"   {rank}. Head {h}: ΔMSE = {res['delta_mse']:+.6f}")
    
    return results

if __name__ == "__main__":
    experiment_A_ablation()
