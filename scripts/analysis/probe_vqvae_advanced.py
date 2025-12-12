import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.reasoning.vqvae.model import MonolithV13

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AdvancedAblation:
    def __init__(self, model_path="data/monolith_v13_trained.pth", 
                 data_path="data/training_embeddings.npy"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MonolithV13()
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Load data
        full_data = np.load(data_path)
        # Sample for performance
        idx = np.random.choice(len(full_data), min(50000, len(full_data)), replace=False)
        self.data = torch.from_numpy(full_data[idx]).float()
        
        logger.info(f"✅ Loaded Model & Data (N={len(self.data)})")
        
    def run_forward(self, indices_override=None):
        """Run forward with optional index override for ablation."""
        all_recon = []
        all_indices = []
        
        with torch.no_grad():
            for i in range(0, len(self.data), 2048):
                batch = self.data[i:i+2048].to(self.device)
                
                if indices_override is None:
                    out = self.model(batch)
                    all_indices.append(out['indices'].cpu())
                    all_recon.append(out['reconstructed'].cpu())
                else:
                    # Override indices for ablation
                    # Must manually encode -> override -> decode
                    z_e = self.model.encoder(batch)
                    z_q, idx_natural, _ = self.model.quantizer(z_e)
                    
                    # Override specific head
                    idx_modified = idx_natural.clone()
                    for h, val in indices_override.items():
                        idx_modified[:, h] = val
                    
                    # Re-quantize with overridden indices
                    # Need to manually get z_q from indices
                    # This requires accessing quantizer internals
                    # For simplicity, we'll just use the natural path but note this limitation
                    
                    # Workaround: Just use natural indices (this needs model modification for true ablation)
                    # For now, report this limitation
                    all_indices.append(idx_natural.cpu())
                    recon = self.model.decoder(z_q)
                    all_recon.append(recon.cpu())
                    
        return torch.cat(all_recon), torch.cat(all_indices)
    
    def ablation_A_indispensability(self):
        """A. Head Ablation Analysis"""
        logger.info("\n🔹 A. Head Indispensability (Ablation)")
        
        # Baseline
        recon_base, indices_base = self.run_forward()
        mse_base = torch.nn.functional.mse_loss(self.data, recon_base)
        
        logger.info(f"   Baseline MSE: {mse_base:.6f}")
        
        results = {}
        
        # For each head, try to "freeze" it
        # Note: True ablation requires modifying quantizer forward pass
        # Placeholder: We'll measure impact by zeroing out that head's contribution
        
        for h in range(4):
            logger.info(f"   Ablating Head {h}...")
            
            # Simplified: Measure Hamming distance when one head is constant
            # Get most frequent code for this head
            most_frequent = torch.mode(indices_base[:, h]).values.item()
            
            # Create override dict
            # NOTE: This is a placeholder - true ablation needs model surgery
            delta_mse = 0.0 # Placeholder
            hamming_dist = 0.0 # Placeholder
            
            results[f'head_{h}'] = {
                'delta_mse': delta_mse,
                'most_frequent_code': most_frequent
            }
            
        logger.info("   ⚠️  Full ablation requires model modification (decoder input surgery)")
        return results
    
    def ablation_B_norm_balance(self):
        """B. Norm Balancing Analysis"""
        logger.info("\n🔹 B. Norm Balancing")
        
        # Run forward and inspect z_q contribution per head
        with torch.no_grad():
            batch = self.data[:2048].to(self.device)
            out = self.model(batch)
            z_q = out['z_q'] # (B, 256) - concatenated from 4 heads
            
            # Split z_q back into heads (each head contributes 64 dims)
            head_dim = 256 // 4
            norms_per_head = []
            
            for h in range(4):
                z_q_h = z_q[:, h*head_dim:(h+1)*head_dim]
                norm_h = torch.norm(z_q_h, dim=1).mean()
                norm_sq = torch.sum(z_q_h**2, dim=1).mean()
                norms_per_head.append((norm_h.item(), norm_sq.item()))
                
            total_norm_sq = torch.sum(z_q**2, dim=1).mean().item()
            
            logger.info("   Norm Contribution per Head:")
            for h, (n, nsq) in enumerate(norms_per_head):
                frac = nsq / total_norm_sq
                logger.info(f"      Head {h}: ||z_q(h)|| = {n:.2f}, Fraction = {frac:.2%}")
                
        return {'norms': norms_per_head, 'total_norm_sq': total_norm_sq}
    
    def ablation_C_coarse_to_fine(self):
        """C. Coarse-to-Fine Structure Analysis"""
        logger.info("\n🔹 C. Coarse-to-Fine (Head 1 Clustering)")
        
        _, indices = self.run_forward()
        
        # Group by Head 1 codes
        head1_codes = indices[:, 1].numpy()
        unique_codes = np.unique(head1_codes)
        
        # Sample top 5 most frequent codes from Head 1
        counts = np.bincount(head1_codes, minlength=256)
        top_codes = np.argsort(counts)[::-1][:5]
        
        logger.info(f"   Analyzing Top 5 Head 1 Codes:")
        
        for code in top_codes:
            mask = head1_codes == code
            n_samples = mask.sum()
            
            if n_samples < 10:
                continue
                
            # Conditional distribution on other heads
            other_heads = indices[mask][:, [0, 2, 3]].numpy()
            
            # Entropy of modifiers given this Head 1 code
            entropies = []
            for i, h_idx in enumerate([0, 2, 3]):
                codes_h = other_heads[:, i]
                _, c = np.unique(codes_h, return_counts=True)
                p = c / c.sum()
                ent = -np.sum(p * np.log2(p + 1e-9))
                entropies.append(ent)
                
            avg_ent = np.mean(entropies)
            logger.info(f"      Code {code}: N={n_samples}, Modifier Entropy={avg_ent:.2f} bits")
            
        return {}
    
    def ablation_D_fuzzy_matching(self):
        """D. Fuzzy Matching / Hamming Distance Analysis"""
        logger.info("\n🔹 D. Fuzzy Matching for Retrieval")
        
        _, indices = self.run_forward()
        
        # Find nearest neighbors in continuous space
        sample = self.data[:5000].numpy()
        sample_idx = indices[:5000].numpy()
        
        nn = NearestNeighbors(n_neighbors=11, metric='euclidean')
        nn.fit(sample)
        dists, neighbors = nn.kneighbors(sample)
        
        # Compute Hamming distance for neighbors vs random
        hamming_neighbors = []
        hamming_random = []
        
        for i in range(1000):
            # Neighbors
            for j in range(1, 6): # Top 5 neighbors
                nb_idx = neighbors[i, j]
                hamming = (sample_idx[i] != sample_idx[nb_idx]).sum()
                hamming_neighbors.append(hamming)
                
            # Random
            for _ in range(5):
                rand_idx = np.random.randint(0, len(sample))
                hamming = (sample_idx[i] != sample_idx[rand_idx]).sum()
                hamming_random.append(hamming)
                
        avg_nb = np.mean(hamming_neighbors)
        avg_rand = np.mean(hamming_random)
        
        logger.info(f"   Hamming Distance (Neighbors): {avg_nb:.2f}")
        logger.info(f"   Hamming Distance (Random): {avg_rand:.2f}")
        logger.info(f"   Ratio (Signal): {avg_rand / (avg_nb + 1e-6):.2f}x")
        
        return {'hamming_neighbors': avg_nb, 'hamming_random': avg_rand}
    
    def ablation_E_hyperparams(self):
        """E. Hyperparameter Criticality (Placeholder)"""
        logger.info("\n🔹 E. Hyperparameter Criticality")
        logger.info("   ⚠️  Requires multiple training runs with varied hyperparameters")
        logger.info("   Current model: Single configuration")
        return {}
    
    def run_all(self):
        results = {
            'A': self.ablation_A_indispensability(),
            'B': self.ablation_B_norm_balance(),
            'C': self.ablation_C_coarse_to_fine(),
            'D': self.ablation_D_fuzzy_matching(),
            'E': self.ablation_E_hyperparams()
        }
        
        # Save report
        report_path = "docs/reports/vqvae_advanced_ablation.md"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Advanced VQ-VAE Ablation Report\n\n")
            f.write("## B. Norm Balancing\n")
            f.write(f"Norms: {results['B']}\n\n")
            f.write("## D. Fuzzy Matching\n")
            f.write(f"Hamming (Neighbors): {results['D']['hamming_neighbors']:.2f}\n")
            f.write(f"Hamming (Random): {results['D']['hamming_random']:.2f}\n")
            
        logger.info(f"\n✅ Report saved: {report_path}")
        return results

if __name__ == "__main__":
    ablation = AdvancedAblation()
    ablation.run_all()
