"""
VQ-VAE Training Script
Re-train the Monolith V13 model to fix codebook collapse.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm

from core.reasoning.vqvae.model import MonolithV13
from core.reasoning.vqvae.loss import compute_orthogonal_loss, compute_vq_commitment_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_vqvae(data_path="data/training_embeddings.npy", 
                output_path="data/monolith_v13_trained.pth",
                epochs=20, 
                batch_size=256, 
                lr=1e-3):
    """
    Train VQ-VAE model on exported embeddings.
    
    Args:
        data_path: Path to .npy file with embeddings
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    
    # Load data
    logger.info(f"Loading embeddings from {data_path}...")
    embeddings = np.load(data_path)
    logger.info(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Model
    model = MonolithV13().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Convert to tensor
    data = torch.tensor(embeddings, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_vq = 0
        total_ortho = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch, in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            
            # Losses
            recon_loss = torch.nn.functional.mse_loss(output['reconstructed'], batch)
            vq_loss = compute_vq_commitment_loss(output['z_e'], output['z_q'])
            ortho_loss = compute_orthogonal_loss(model.quantizer)
            
            loss = recon_loss + vq_loss + 0.1 * ortho_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_ortho += ortho_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_vq = total_vq / len(dataloader)
        avg_ortho = total_ortho / len(dataloader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | "
                   f"Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | Ortho: {avg_ortho:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{output_path}.epoch{epoch+1}"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), output_path)
    logger.info(f"✅ Training complete! Model saved to {output_path}")
    
    # Test codebook usage
    model.eval()
    with torch.no_grad():
        test_batch = data[:1000].to(device)
        output = model(test_batch)
        indices = output['indices'].cpu().numpy()
        unique_codes = np.unique(indices)
        logger.info(f"Codebook usage: {len(unique_codes)}/256 codes used")
        logger.info(f"Code distribution: min={indices.min()}, max={indices.max()}, "
                   f"mean={indices.mean():.1f}, std={indices.std():.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VQ-VAE Monolith V13")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data", type=str, default="data/training_embeddings.npy", help="Path to embeddings")
    parser.add_argument("--output", type=str, default="data/monolith_v13_trained.pth", help="Output model path")
    
    args = parser.parse_args()
    
    train_vqvae(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
