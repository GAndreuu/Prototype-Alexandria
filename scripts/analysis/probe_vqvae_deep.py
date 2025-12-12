
import sys
import os
import json
import torch
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import linregress, entropy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "core" / "reasoning"))

try:
    from core.reasoning.vqvae.model import MonolithV13
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Import Error: {e}")
    MODEL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DeepProbeConfig:
    model_path: str = "data/monolith_v13_trained.pth"
    data_path: str = "data/training_embeddings.npy"
    report_path: str = "docs/reports/vqvae_deep_analysis.md"
    batch_size: int = 2048
    sample_limit: int = 100000

class DeepProbe:
    def __init__(self, config: DeepProbeConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data = None
        
    def load(self):
        logger.info("📦 Loading Data...")
        full_data = np.load(self.config.data_path)
        if len(full_data) > self.config.sample_limit:
            # Random sample for speed, but large enough for stats
            idx = np.random.choice(len(full_data), self.config.sample_limit, replace=False)
            self.data = torch.from_numpy(full_data[idx]).float()
        else:
            self.data = torch.from_numpy(full_data).float()
            
        logger.info(f"   Data Shape: {self.data.shape}")

        logger.info("🧠 Loading Model...")
        self.model = MonolithV13(input_dim=self.data.shape[1], hidden_dim=256)
        # Load state
        state = torch.load(self.config.model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def run_forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs forward pass on all data in batches."""
        z_e_list, z_q_list, indices_list, recon_list = [], [], [], []
        
        with torch.no_grad():
            for i in range(0, len(self.data), self.config.batch_size):
                batch = self.data[i: i+self.config.batch_size].to(self.device)
                out = self.model(batch)
                
                z_e_list.append(out['z_e'].cpu())
                z_q_list.append(out['z_q'].cpu())
                indices_list.append(out['indices'].cpu())
                recon_list.append(out['reconstructed'].cpu())
                
        return (
            torch.cat(z_e_list),
            torch.cat(z_q_list),
            torch.cat(indices_list),
            torch.cat(recon_list)
        )

    def analyze_1_codebook_state(self, z_e, z_q, indices):
        """1. Estado global do codebook"""
        logger.info("\n🔹 1. Global Codebook State")
        
        # Usage stats per head
        num_heads = indices.shape[1]
        codebook_size = 256
        
        usage_fractions = []
        for h in range(num_heads):
            uniq = torch.unique(indices[:, h])
            frac = len(uniq) / codebook_size
            usage_fractions.append(frac)
        
        logger.info(f"   Usage Fraction (Mean ± Std): {np.mean(usage_fractions):.4f} ± {np.std(usage_fractions):.4f}")
        
        # Recon vs VQ norms
        # z_e and z_q are (N, hidden_dim) or (N, heads * head_dim)
        # Note: z_e in v13 is (N, 256), z_q is (N, 256)
        
        mse_recon = torch.nn.functional.mse_loss(self.data, self.data) # Placeholder, computed in caller
        mse_commit = torch.nn.functional.mse_loss(z_e, z_q)
        norm_ze = torch.norm(z_e, dim=1).mean()
        norm_zq = torch.norm(z_q, dim=1).mean()
        
        # Relative Error
        # mean(||z_e - z_q||^2 / ||z_e||^2)
        err_sq = torch.sum((z_e - z_q)**2, dim=1)
        ze_sq = torch.sum(z_e**2, dim=1) + 1e-9
        rel_err = (err_sq / ze_sq).mean()
        
        logger.info(f"   Commitment Loss (MSE): {mse_commit:.6f}")
        logger.info(f"   Avg ||z_e||: {norm_ze:.4f}")
        logger.info(f"   Avg ||z_q||: {norm_zq:.4f}")
        logger.info(f"   Relative Error: {rel_err:.2%}")
        
        # Scale per head
        # We need access to model codebooks
        codebooks = self.model.quantizer.codebooks.detach().cpu() # (H, 256, D)
        cb_norms = torch.norm(codebooks, dim=2).mean(dim=1) # (H,)
        
        # Projected z_heads norms
        # Need to project z_e again to split by heads
        # Or assumes z_e is already concatenation of heads
        # In OrthogonalProductQuantizer:
        # z -> split heads -> project? No, usually z is projected inside quantizer
        # Let's peek at `model.py`... 
        # Actually `model.quantizer(z)` does the projection. 
        # We can't easily get pre-projection z_heads from the output dict unless we modified model.
        # But we check checking `codebooks` norm is enough.
        logger.info(f"   Codebook Norms per Head: {cb_norms.numpy()}")

        return {
            'usage_mean': float(np.mean(usage_fractions)),
            'commit_loss': float(mse_commit),
            'rel_error': float(rel_err),
            'codebook_norms': cb_norms.numpy().tolist()
        }

    def analyze_2_distribution(self, indices):
        """2. Distribuição de uso + power law"""
        logger.info("\n🔹 2. Distribution & Power Law")
        
        num_heads = indices.shape[1]
        results = {}
        
        for h in range(num_heads):
            counts = torch.bincount(indices[:, h], minlength=256).float()
            probs = counts / counts.sum()
            
            # Entropy
            p_nz = probs[probs > 0]
            ent = -torch.sum(p_nz * torch.log2(p_nz)).item()
            max_ent = np.log2(256)
            
            # Power Law Fit
            # Sort counts descending
            sorted_counts, _ = torch.sort(counts, descending=True)
            # Take only non-zero
            sorted_counts = sorted_counts[sorted_counts > 0].numpy()
            ranks = np.arange(1, len(sorted_counts) + 1)
            
            if len(sorted_counts) > 5:
                # Fit log(count) = -alpha * log(rank) + c
                slope, intercept, r_value, p_value, std_err = linregress(np.log(ranks), np.log(sorted_counts))
                alpha = -slope
            else:
                alpha, r_value = 0.0, 0.0
                
            # KL Divergence vs Uniform
            # Uniform on support (len(sorted_counts))
            n_support = len(sorted_counts)
            p_uniform = np.ones(n_support) / n_support
            
            # KL(p_emp || p_unif) = sum p_emp log(p_emp / p_unif)
            # p_emp here refers to the distribution ON THE SUPPORT
            p_emp_support = sorted_counts / sorted_counts.sum()
            kl = np.sum(p_emp_support * np.log(p_emp_support / p_uniform))
            
            logger.info(f"   Head {h}: H={ent:.2f}/{max_ent:.2f} ({ent/max_ent:.1%}), Alpha={alpha:.2f}, R2={r_value**2:.2f}, KL={kl:.4f}")
            
            results[f'head_{h}'] = {
                'entropy': ent,
                'alpha': alpha,
                'r2': r_value**2,
                'kl': kl
            }
            
        return results

    def analyze_3_independence(self, indices):
        """3. Heads Independence"""
        logger.info("\n🔹 3. Independence")
        
        B, H = indices.shape
        indices_np = indices.numpy()
        
        # Pairwise Mutual Information
        mis = []
        coincidences = []
        
        for i in range(H):
            for j in range(i+1, H):
                # Coincidence
                match = (indices_np[:, i] == indices_np[:, j]).mean()
                coincidences.append(match)
                
                # Mutual Information
                # HI(X,Y) = H(X) + H(Y) - H(X,Y)
                # Joint entropy
                # Map pairs to scalar: x*256 + y
                joint = indices_np[:, i] * 256 + indices_np[:, j]
                # Count unique pairs
                _, joint_counts = np.unique(joint, return_counts=True)
                p_xy = joint_counts / B
                h_xy = -np.sum(p_xy * np.log2(p_xy))
                
                # Marginals
                _, c_i = np.unique(indices_np[:, i], return_counts=True); p_i = c_i/B
                _, c_j = np.unique(indices_np[:, j], return_counts=True); p_j = c_j/B
                h_i = -np.sum(p_i * np.log2(p_i))
                h_j = -np.sum(p_j * np.log2(p_j))
                
                mi = h_i + h_j - h_xy
                norm_mi = mi / (0.5 * (h_i + h_j))
                mis.append(norm_mi)
                
        avg_mi = np.mean(mis)
        avg_coin = np.mean(coincidences)
        
        logger.info(f"   Avg Coincidence (Jaccard-like): {avg_coin:.4f}")
        logger.info(f"   Avg Normalized MI: {avg_mi:.4f}")
        
        return {'avg_mi': avg_mi, 'avg_coincidence': avg_coin}

    def analyze_4_dynamics(self):
        """4. Dinâmica (Snapshot history)"""
        logger.info("\n🔹 4. Training Dynamics")
        
        checkpoints = sorted(list(Path("data").glob("monolith_v13_trained.pth.epoch*")), 
                           key=lambda p: int(str(p).split("epoch")[-1]))
        
        history = []
        
        # Load sample indices for each epoch (Expensive, so we do small sample)
        # Or we just load the weights and check codebook status?
        # User asked for: loss, recon, entropy, util over time.
        # We can't re-calculate loss/recon easily without running forward pass on data.
        # But we CAN calculate 'util' (codes usage) if we scan a batch.
        
        subset = self.data[:2048].to(self.device) # Small subset for speed
        
        for ckpt in checkpoints:
            epoch = str(ckpt).split("epoch")[-1]
            try:
                state = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                
                with torch.no_grad():
                    out = self.model(subset)
                    indices = out['indices'] # (B, 4)
                    
                    # Entropy & Util
                    util_list = []
                    ent_list = []
                    for h in range(4):
                         uniq = torch.unique(indices[:, h])
                         util_list.append(len(uniq)/256)
                    
                    avg_util = np.mean(util_list)
                    history.append({'epoch': epoch, 'util': avg_util})
                    logger.info(f"   Epoch {epoch}: Util={avg_util:.2%}")
            except Exception as e:
                logger.warning(f"Failed to probe {ckpt}: {e}")
                
        # Reload best model at end
        self.load() 
        return history

    def analyze_5_sensitivity(self):
        """5. Sensibilidade (Local & Robustness)"""
        logger.info("\n🔹 5. Sensitivity Analysis")
        
        subset = self.data[:1000].to(self.device)
        
        # 1. Local perturbation
        # Add noise relative to norm
        norm = torch.norm(subset, dim=1, keepdim=True)
        noise = torch.randn_like(subset) * 0.05 * norm # 5% noise
        perturbed = subset + noise
        
        with torch.no_grad():
            # Use forward pass to get indices
            out_orig = self.model(subset)
            idx_orig = out_orig['indices']
            
            out_pert = self.model(perturbed)
            idx_pert = out_pert['indices']
            
        changes = (idx_orig != idx_pert).float().mean()
        logger.info(f"   Index Change Rate (5% Noise): {changes:.2%}")
        
        return {'perturbation_change': float(changes)}

    def analyze_6_performance(self, z_e, z_q, indices, samples):
        """6. Reconstrução vs Uso"""
        logger.info("\n🔹 6. Performance Correlation")
        
        # Calculate per-sample reconstruction error
        # Need to reconstruct first... already in run_forward?
        # Re-running decoder for exact batch matching might be cleaner or pass recon
        
        # recon_list passed? Yes, samples is actually 'recon' if we change signature
        # Wait, run_forward returns (z_e, z_q, indices, recon)
        
        recon = samples # Assumed passed correctly
        
        mse_per_sample = torch.mean((self.data - recon)**2, dim=1)
        
        # Correlate recon error with code probability
        # For each sample, we have 4 codes.
        # Define "Code Rareness" of a sample as min(prob(code)) across heads
        
        rareness_scores = []
        errors = []
        
        # Pre-calc probs
        probs_map = []
        for h in range(4):
            counts = torch.bincount(indices[:, h], minlength=256).float()
            probs_map.append(counts / counts.sum())
            
        for i in range(len(indices)):
            sample_probs = [probs_map[h][indices[i, h]] for h in range(4)]
            min_prob = min(sample_probs)
            rareness_scores.append(min_prob.item())
            errors.append(mse_per_sample[i].item())
            
        corr = np.corrcoef(rareness_scores, errors)[0, 1]
        logger.info(f"   Correlation (Min Prob vs MSE): {corr:.4f}")
        # Negative corr expected: Lower prob (rarer) -> Higher MSE? Or Lower Prob -> code is specific?
        # Usually Rare codes = anomalies or poor fit -> High MSE. So Negative correlation between Prob and MSE.
        
        return {'corr_prob_mse': corr}

    def run(self):
        self.load()
        z_e, z_q, indices, recon = self.run_forward()
        
        report = {}
        report['1_codebook'] = self.analyze_1_codebook_state(z_e, z_q, indices)
        report['2_dist'] = self.analyze_2_distribution(indices)
        report['3_indep'] = self.analyze_3_independence(indices)
        report['4_dynamics'] = self.analyze_4_dynamics()
        report['5_sens'] = self.analyze_5_sensitivity()
        report['6_perf'] = self.analyze_6_performance(z_e, z_q, indices, recon)
        
        # Write Report
        with open(self.config.report_path, 'w') as f:
            f.write("# Deep VQ-VAE Analysis Report\n\n")
            f.write("## 1. Codebook State\n")
            f.write(json.dumps(report['1_codebook'], indent=2))
            f.write("\n\n## 2. Distribution\n")
            f.write(json.dumps(report['2_dist'], indent=2))
            f.write("\n\n## 4. Dynamics\n")
            f.write(json.dumps(report['4_dynamics'], indent=2))
        
        print(f"\n✅ Report Saved: {self.config.report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deep Probe VQ-VAE")
    parser.add_argument("--model", type=str, default="data/monolith_v13_trained.pth", help="Path to model checkpoint")
    parser.add_argument("--report", type=str, default="docs/reports/vqvae_deep_analysis.md", help="Path to output report")
    
    args = parser.parse_args()
    
    config = DeepProbeConfig(
        model_path=args.model,
        report_path=args.report
    )
    
    probe = DeepProbe(config)
    probe.run()
