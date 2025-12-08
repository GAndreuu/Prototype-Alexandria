"""
Quick test of balanced training with 3 epochs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.train_vqvae import train_vqvae

if __name__ == "__main__":
    train_vqvae(
        data_path="data/training_embeddings.npy",
        output_path="data/monolith_v13_balanced_test.pth",
        epochs=3,
        batch_size=256,
        lr=1e-3
    )
