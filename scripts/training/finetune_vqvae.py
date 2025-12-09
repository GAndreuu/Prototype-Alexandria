"""
VQ-VAE Fine-Tuning Script
=========================
Fine-tunes the Monolith V13 model to repair codebook collapse and reduce quantization error.
Uses existing weights as a starting point.

Improvements over standard training:
1. Loads pre-trained weights.
2. Uses lower Learning Rate (1e-4).
3. Implements Dead Code Revival (Metric-based reset).
4. Uses 16 workers for data loading (User Request).
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

try:
    from core.reasoning.vqvae.model import MonolithV13
    from core.reasoning.vqvae.loss import (
        compute_orthogonal_loss, 
        compute_vq_commitment_loss,
        compute_head_balance_loss,
        compute_code_usage_entropy_loss
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(PROJECT_ROOT))
    from core.reasoning.vqvae.model import MonolithV13
    from core.reasoning.vqvae.loss import (
        compute_orthogonal_loss, 
        compute_vq_commitment_loss,
        compute_head_balance_loss,
        compute_code_usage_entropy_loss
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_dead_codes(model, dataloader, device, threshold=3):
    """Identify dead or lazy codes (used less than threshold times)"""
    model.eval()
    usage_counts = torch.zeros(model.quantizer.num_heads, model.quantizer.num_embeddings).to(device)
    
    with torch.no_grad():
        for batch, in dataloader:
            batch = batch.to(device)
            out = model(batch)
            indices = out['indices'] # [B, H]
            
            for h in range(model.quantizer.num_heads):
                counts = torch.bincount(indices[:, h], minlength=model.quantizer.num_embeddings)
                usage_counts[h] += counts
    
    # Dead codes are those with usage < threshold
    dead_mask = usage_counts < threshold
    total_dead = dead_mask.sum().item()
    return dead_mask, total_dead

def revive_dead_codes(model, dead_mask, data_batch):
    """Reset dead codes to random vectors from the current data batch"""
    # data_batch: [B, D]
    revived_count = 0
    
    with torch.no_grad():
        device = model.quantizer.codebooks.device
        z_e = model.encoder(data_batch.to(device)) # [B, D]
        
        # Reshape to [B, H, D//H] to match codebook heads
        B, D = z_e.shape
        H = model.quantizer.num_heads
        Hd = D // H
        
        # Ensure z_e is reshaped correctly for the heads
        # z_e output from encoder is flat.
        # OrthogonalQuantizer splits it straightforwardly: .view(bsz, num_heads, head_dim)
        z_e_reshaped = z_e.view(B, H, Hd)
        
        for h in range(H):
            # dead_mask is [H, Num_Emb]
            dead_indices = torch.where(dead_mask[h])[0]  # Indices of dead codes
            n_dead = len(dead_indices)
            
            if n_dead == 0:
                continue
                
            # Select random samples from batch to replace dead codes
            # We need n_dead samples
            # If batch is smaller than n_dead, replace=True needed, but usually B=256, n_dead<=256
            if B >= n_dead:
                rand_idx = torch.randperm(B)[:n_dead]
            else:
                rand_idx = torch.randint(0, B, (n_dead,))
                
            replacements = z_e_reshaped[rand_idx, h, :] # [n_dead, Hd]
            
            # Verify shapes match exactly
            if replacements.shape != model.quantizer.codebooks.data[h, dead_indices].shape:
                 logger.error(f"Shape mismatch! Src: {replacements.shape}, Tgt: {model.quantizer.codebooks.data[h, dead_indices].shape}")
                 continue

            model.quantizer.codebooks.data[h, dead_indices] = replacements
            revived_count += n_dead
            
    return revived_count

def finetune_vqvae(
    data_path="data/training_embeddings.npy",
    model_path="data/monolith_v13_trained.pth",
    output_path="data/monolith_v13_finetuned.pth",
    epochs=10,
    batch_size=256,
    lr=1e-4,
    num_workers=16
):
    # Check data
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    # Load data
    logger.info(f"Loading embeddings from {data_path}...")
    try:
        embeddings = np.load(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
        
    logger.info(f"Loaded {len(embeddings)} embeddings.")
    
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load Model
    model = MonolithV13()
    if os.path.exists(model_path):
        logger.info(f"Loading existing weights from {model_path}")
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            logger.warning(f"Could not load weights cleanly: {e}. Starting fresh? No. Aborting.")
            return
    else:
        logger.warning(f"Model path {model_path} not found. Cannot fine-tune. Aborting.")
        return
        
    model.to(device)
    model.train()
    
    # Optimizer (Lower LR for fine-tuning)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # DataLoader
    data_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    
    # Windows-safe num_workers check
    if os.name == 'nt' and num_workers > 0:
        logger.info(f"Using {num_workers} workers on Windows. Ensure script is guarded with 'if __name__ == \"__main__\":'")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"Starting Fine-Tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Accumulate batch for dead code revival
        last_batch = None
        
        for batch, in pbar:
            batch = batch.to(device)
            last_batch = batch
            
            optimizer.zero_grad()
            output = model(batch)
            
            # --- Losses ---
            # 1. Reconstruction (MSE)
            recon_loss = torch.nn.functional.mse_loss(output['reconstructed'], batch)
            
            # 2. Commitment (Strict Mode: beta=5.0)
            # Force encoder to commit to the codebook
            vq_loss = compute_vq_commitment_loss(output['z_e'], output['z_q'], beta=5.0)
            
            # 3. Aux losses
            ortho_loss = compute_orthogonal_loss(model.quantizer)
            balance_loss = compute_head_balance_loss(output['z_q'], num_heads=4)
            entropy_loss = compute_code_usage_entropy_loss(
                output['indices'], 
                num_embeddings=256,
                target_entropy_frac=0.95 
            )
            
            loss = (
                recon_loss + 
                vq_loss + 
                0.1 * ortho_loss + 
                0.1 * balance_loss + 
                0.5 * entropy_loss # Boosted entropy weight
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'recon': f'{recon_loss.item():.4f}'})
        
        # --- End of Epoch Logic ---
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        logger.info(f"Epoch {epoch+1} Stats: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}")
        
        # Dead Code Revival (Strict Mode)
        if (epoch + 1) % 1 == 0: # Check EVERY epoch
            logger.info("Scanning for dead/lazy codes...")
            dead_mask, total_dead = evaluate_dead_codes(model, dataloader, device, threshold=5)
            
            if total_dead > 0:
                logger.info(f"Found {total_dead} codes with usage < 5. Reviving...")
                revived = revive_dead_codes(model, dead_mask, last_batch)
                logger.info(f"Revived {revived} codes.")
            else:
                logger.info("No dead codes found.")

        # Save Checkpoint
        torch.save(model.state_dict(), f"{output_path}.checkpoint")

    # Save Final
    torch.save(model.state_dict(), output_path)
    logger.info(f"✅ Fine-Tuning Complete. Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    
    finetune_vqvae(epochs=args.epochs, lr=args.lr, num_workers=args.workers)
