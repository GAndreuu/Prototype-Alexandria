"""
Experiments C & D Combined:
C: Coarse-to-Fine Semantics (terminal codes)
D: Fuzzy Retrieval (Hamming distributions)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from core.reasoning.vqvae.model import MonolithV13
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def experiment_C_semantics(model, data, device, num_codes=256):
    """Find terminal Head 1 codes (low modifier entropy)."""
    print("🧬 Experiment C: Coarse-to-Fine Semantics")
    print()
    
    with torch.no_grad():
        out = model(data)
        indices = out['indices'].cpu().numpy()  # [B, 4]
    
    # Compute conditional entropy for each Head 1 code
    results = []
    
    for code_id in range(num_codes):
        mask = indices[:, 1] == code_id
        n_samples = mask.sum()
        
        if n_samples < 10:
            continue
        
        # Modifier indices (heads 0, 2, 3)
        modifiers = indices[mask][:, [0, 2, 3]]  # [N, 3]
        
        # Compute joint entropy of modifiers
        # Treat as single value: h0*256^2 + h2*256 + h3
        joint_codes = modifiers[:, 0] * 256**2 + modifiers[:, 1] * 256 + modifiers[:, 2]
        _, counts = np.unique(joint_codes, return_counts=True)
        p = counts / counts.sum()
        entropy = -np.sum(p * np.log2(p + 1e-9))
        
        results.append({
            'code': code_id,
            'samples': n_samples,
            'entropy': entropy
        })
    
    # Sort by entropy (ascending = most collapsed)
    results = sorted(results, key=lambda x: x['entropy'])
    
    print("📉 Top 10 Most Collapsed Codes (Terminal):")
    for i, res in enumerate(results[:10], 1):
        print(f"   {i}. Code {res['code']:3d}: "
              f"N={res['samples']:5d}, Entropy={res['entropy']:.2f} bits")
    
    print()
    print("📈 Top 10 Most Diverse Codes:")
    for i, res in enumerate(results[-10:][::-1], 1):
        print(f"   {i}. Code {res['code']:3d}: "
              f"N={res['samples']:5d}, Entropy={res['entropy']:.2f} bits")
    
    return results


def experiment_D_fuzzy(model, data, device, sample_size=5000):
    """Compute Hamming distributions and AUC."""
    print("\n🧷 Experiment D: Fuzzy Retrieval Deep Dive")
    print()
    
    # Sample subset
    idx = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
    subset = data[idx].to(device)
    
    with torch.no_grad():
        out = model(subset)
        indices = out['indices'].cpu().numpy()  # [N, 4]
        z_e = out['z_e'].cpu().numpy()  # [N, D]
    
    # Find neighbors in continuous space
    nn = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nn.fit(z_e)
    dists, neighbors = nn.kneighbors(z_e)
    
    # Compute Hamming distances
    hamming_neighbors = []
    hamming_random = []
    
    for i in range(min(1000, len(subset))):
        # Neighbors (skip self at index 0)
        for j in range(1, 6):
            nb_idx = neighbors[i, j]
            hamming = (indices[i] != indices[nb_idx]).sum()
            hamming_neighbors.append(hamming)
        
        # Random pairs
        for _ in range(5):
            rand_idx = np.random.randint(0, len(subset))
            hamming = (indices[i] != indices[rand_idx]).sum()
            hamming_random.append(hamming)
    
    # Statistics
    mean_nb = np.mean(hamming_neighbors)
    mean_rand = np.mean(hamming_random)
    
    print(f"   Hamming Distance (Neighbors): {mean_nb:.2f}")
    print(f"   Hamming Distance (Random): {mean_rand:.2f}")
    print(f"   Signal Ratio: {mean_rand / (mean_nb + 1e-6):.2f}x")
    print()
    
    # Plot histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 5, 0.5)
    ax.hist(hamming_neighbors, bins=bins, alpha=0.5, label='Neighbors', density=True)
    ax.hist(hamming_random, bins=bins, alpha=0.5, label='Random', density=True)
    ax.set_xlabel('Hamming Distance')
    ax.set_ylabel('Density')
    ax.set_title('Hamming Distance Distribution: Neighbors vs Random')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = "docs/reports/experiment_D_hamming.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Plot saved: {plot_path}")
    
    return {
        'hamming_neighbors': hamming_neighbors,
        'hamming_random': hamming_random,
        'mean_nb': mean_nb,
        'mean_rand': mean_rand
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MonolithV13()
    state = torch.load("data/monolith_v13_trained.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    # Load data
    full_data = np.load("data/training_embeddings.npy")
    idx = np.random.choice(len(full_data), min(50000, len(full_data)), replace=False)
    data = torch.from_numpy(full_data[idx]).float()
    
    print(f"🔬 Experiments C & D")
    print(f"   Samples: {len(data)}")
    print()
    
    # Run experiments
    results_C = experiment_C_semantics(model, data, device)
    results_D = experiment_D_fuzzy(model, data, device)
    
    print("\n✅ Experiments C & D Complete")

if __name__ == "__main__":
    main()
