
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.reasoning.vqvae.model import MonolithV13
from core.reasoning.vqvae.loss import compute_vq_commitment_loss

def analyze_metrics():
    # Load Data
    print("Loading data...")
    data = np.load("data/training_embeddings.npy")
    # Take a sample
    sample = torch.tensor(data[:2000], dtype=torch.float32)
    
    # Load Model
    print("Loading model...")
    device = "cpu"
    model = MonolithV13().to(device)
    model.load_state_dict(torch.load("data/monolith_v13_trained.pth", map_location=device))
    model.eval()
    
    # Forward
    print("Running forward pass...")
    with torch.no_grad():
        output = model(sample)
        
    z_e = output['z_e']
    z_q = output['z_q']
    indices = output['indices'].cpu().numpy().flatten()
    
    # Analyze Loss Components
    mse_recon = torch.nn.functional.mse_loss(output['reconstructed'], sample).item()
    mse_codebook = torch.nn.functional.mse_loss(z_q, z_e).item() # z_q should match z
    # mse_commitment = torch.nn.functional.mse_loss(z_q, z_e) # Same logic
    
    print(f"\n--- Loss Component Analysis ---")
    print(f"Reconstruction MSE: {mse_recon:.6f}")
    print(f"Codebook/Commitment MSE (||z_e - z_q||^2): {mse_codebook:.6f}")
    
    z_e_norm = torch.norm(z_e, dim=-1).mean().item()
    z_q_norm = torch.norm(z_q, dim=-1).mean().item()
    print(f"Average ||z_e||: {z_e_norm:.4f}")
    print(f"Average ||z_q||: {z_q_norm:.4f}")
    
    # Distribution Analysis (Power Law check)
    print(f"\n--- Codebook Distribution ---")
    counts = np.bincount(indices, minlength=256)
    
    # Sort for Power Law curve
    sorted_counts = np.sort(counts)[::-1]
    
    print(f"Top 5 Codes Usage: {sorted_counts[:5]}")
    print(f"Bottom 5 Codes Usage: {sorted_counts[-5:]}")
    print(f"Std Dev of Counts: {np.std(counts):.2f}")
    print(f"Entropy: {compute_entropy(counts):.4f}")

    # Theoretical Max Entropy (Uniform)
    max_ent = np.log(256)
    print(f"Max Entropy (Uniform): {max_ent:.4f}")

def compute_entropy(counts):
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

if __name__ == "__main__":
    analyze_metrics()
