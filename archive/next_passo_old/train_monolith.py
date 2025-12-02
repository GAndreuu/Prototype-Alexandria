"""
Monolith Encoder Training Script
Trains the new VQ-VAE architecture (512D hidden) on Alexandria's data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURE (same as monolith_encoder.py)
# ============================================================================

class ProductQuantizer(nn.Module):
    """Product Quantizer with EMA and entropy regularization for training."""
    
    def __init__(self, num_heads=4, codebook_size=256, head_dim=128, ema_decay=0.99):
        super().__init__()
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.head_dim = head_dim
        self.ema_decay = ema_decay
        
        # Codebooks (will be learned)
        self.codebooks = nn.Parameter(
            torch.randn(num_heads, codebook_size, head_dim) * 0.01
        )
        
        # EMA tracking for codebook updates
        self.register_buffer('ema_count', torch.ones(num_heads, codebook_size))
        self.register_buffer('ema_weight', self.codebooks.data.clone())
    
    def forward(self, z):
        """
        Quantize with straight-through estimator.
        
        Args:
            z: [B, hidden_dim] (will be split into heads)
            
        Returns:
            z_q: [B, hidden_dim] — quantized
            indices: [B, num_heads] — which codes were used
            commit_loss: scalar — commitment loss
        """
        B = z.shape[0]
        z_heads = z.view(B, self.num_heads, self.head_dim)
        
        z_q_parts = []
        indices_list = []
        commit_loss = 0.0
        
        for h in range(self.num_heads):
            z_h = z_heads[:, h, :]  # [B, head_dim]
            cb_h = self.codebooks[h]  # [codebook_size, head_dim]
            
            # Find nearest codes
            dist = torch.cdist(z_h, cb_h, p=2)  # [B, codebook_size]
            idx = dist.argmin(dim=1)  # [B]
            
            # Quantize
            z_q_h = cb_h[idx]  # [B, head_dim]
            
            # Commitment loss (encourage encoder to commit to codes)
            commit_loss += F.mse_loss(z_h, z_q_h.detach())
            
            # Straight-through estimator
            z_q_h = z_h + (z_q_h - z_h).detach()
            
            z_q_parts.append(z_q_h)
            indices_list.append(idx)
            
            # EMA update (only in training)
            if self.training:
                with torch.no_grad():
                    # Count usage
                    onehot = F.one_hot(idx, self.codebook_size).float()  # [B, codebook_size]
                    counts = onehot.sum(0)  # [codebook_size]
                    
                    # EMA count
                    self.ema_count[h] = self.ema_decay * self.ema_count[h] + (1 - self.ema_decay) * counts
                    
                    # EMA weight
                    sum_z = (onehot.T @ z_h)  # [codebook_size, head_dim]
                    self.ema_weight[h] = self.ema_decay * self.ema_weight[h] + (1 - self.ema_decay) * sum_z
                    
                    # Update codebook
                    n = self.ema_count[h].unsqueeze(1)
                    self.codebooks.data[h] = self.ema_weight[h] / (n + 1e-5)
        
        z_q = torch.cat(z_q_parts, dim=1)  # [B, hidden_dim]
        indices = torch.stack(indices_list, dim=1)  # [B, num_heads]
        
        return z_q, indices, commit_loss / self.num_heads


class MonolithModel(nn.Module):
    """Monolith VQ-VAE for training."""
    
    def __init__(self, input_dim=384, hidden_dim=512, num_heads=4, codebook_size=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: 384 → 1024 → 512
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Quantizer
        head_dim = hidden_dim // num_heads
        self.quantizer = ProductQuantizer(num_heads, codebook_size, head_dim)
        
        # Decoder: 512 → 1024 → 384
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, indices, commit_loss = self.quantizer(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return {
            'reconstructed': x_recon,
            'z_e': z,
            'z_q': z_q,
            'indices': indices,
            'commit_loss': commit_loss
        }


# ============================================================================
# TRAINING
# ============================================================================

def compute_orthogonal_loss(quantizer):
    """
    Encourage codebooks in different heads to be diverse.
    """
    codebooks = quantizer.codebooks  # [num_heads, codebook_size, head_dim]
    
    loss = 0.0
    for h1 in range(quantizer.num_heads):
        for h2 in range(h1 + 1, quantizer.num_heads):
            cb1 = codebooks[h1]  # [codebook_size, head_dim]
            cb2 = codebooks[h2]
            
            # Normalize
            cb1_norm = F.normalize(cb1, dim=1)
            cb2_norm = F.normalize(cb2, dim=1)
            
            # Similarity matrix
            sim = cb1_norm @ cb2_norm.T  # [codebook_size, codebook_size]
            
            # Penalize high similarity
            loss += (sim ** 2).mean()
    
    return loss / (quantizer.num_heads * (quantizer.num_heads - 1) / 2)


def train_monolith(
    data_path="data/training_embeddings.npy",
    output_dir="monolith_export",
    epochs=20,
    batch_size=256,
    lr=1e-3,
    device=None
):
    """
    Train Monolith encoder.
    
    Args:
        data_path: Path to embeddings (.npy file)
        output_dir: Where to save weights
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
    """
    
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info("="*80)
    logger.info(f"MONOLITH TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_path}")
    
    # Load data
    logger.info(f"\nLoading embeddings from {data_path}...")
    embeddings = np.load(data_path)
    logger.info(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # Model
    model = MonolithModel(
        input_dim=384,
        hidden_dim=512,
        num_heads=4,
        codebook_size=256
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Training loop
    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    
    history = {'losses': [], 'codebook_usage': []}
    
    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0
        epoch_recon = 0
        epoch_commit = 0
        epoch_ortho = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch, in pbar:
            batch = batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            output = model(batch)
            
            # Losses
            recon_loss = F.mse_loss(output['reconstructed'], batch)
            commit_loss = output['commit_loss']
            ortho_loss = compute_orthogonal_loss(model.quantizer)
            
            # Total loss
            loss = recon_loss + 0.25 * commit_loss + 0.01 * ortho_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Track
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_commit += commit_loss.item()
            epoch_ortho += ortho_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })
        
        # Epoch stats
        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_commit = epoch_commit / n_batches
        avg_ortho = epoch_ortho / n_batches
        
        # Codebook usage
        model.eval()
        with torch.no_grad():
            sample = torch.tensor(embeddings[:10000], dtype=torch.float32).to(device)
            output = model(sample)
            indices = output['indices'].cpu().numpy()
            
            usage_per_head = []
            for h in range(4):
                unique = len(np.unique(indices[:, h]))
                usage_per_head.append(unique)
            
            total_usage = sum(usage_per_head)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"Commit: {avg_commit:.4f} | "
            f"Ortho: {avg_ortho:.4f} | "
            f"Codebook: {total_usage}/1024 ({total_usage/1024*100:.1f}%)"
        )
        
        history['losses'].append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'recon': avg_recon,
            'commit': avg_commit,
            'ortho': avg_ortho
        })
        history['codebook_usage'].append({
            'epoch': epoch + 1,
            'total': total_usage,
            'per_head': usage_per_head
        })
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_path / f"weights_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_weights = output_path / "weights.pth"
    torch.save(model.state_dict(), final_weights)
    logger.info(f"\n✅ Final weights saved: {final_weights}")
    
    # Save config
    config = {
        'input_dim': 384,
        'hidden_dim': 512,
        'num_heads': 4,
        'codebook_size': 256,
        'head_dim': 128,
        'trained_on': len(embeddings),
        'epochs': epochs,
        'final_loss': avg_loss,
    }
    
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Config saved: {config_path}")
    
    # Save codebooks separately for easy access
    codebooks = model.quantizer.codebooks.detach().cpu().numpy()
    codebooks_path = output_path / "codebooks.npz"
    np.savez(codebooks_path, codebooks=codebooks)
    logger.info(f"✅ Codebooks saved: {codebooks_path}")
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"✅ History saved: {history_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Final reconstruction loss: {avg_recon:.6f}")
    logger.info(f"Final codebook usage: {total_usage}/1024 ({total_usage/1024*100:.1f}%)")
    logger.info(f"\nFiles saved to: {output_path.absolute()}")
    logger.info("\nNext steps:")
    logger.info("1. Test integration: python core/encoders/monolith_encoder.py test-integration")
    logger.info("2. Update Alexandria to use new encoder")
    logger.info("3. Reset Mycelial state: rm data/mycelial_state.npz")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Monolith VQ-VAE")
    parser.add_argument("--data", default="data/training_embeddings.npy", help="Path to embeddings")
    parser.add_argument("--output", default="monolith_export", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    train_monolith(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
