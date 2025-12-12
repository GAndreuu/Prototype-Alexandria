"""
Train Topology Engine & Mycelial Network
=========================================
Uses the exported embeddings and fine-tuned VQ-VAE to:
1. Re-cluster the Topology Engine (KMeans)
2. Build Hebbian connections in the Mycelial Network
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import logging
from tqdm import tqdm

from core.topology.topology_engine import TopologyEngine
from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
from core.reasoning.vqvae.model import MonolithV13

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_topology(embeddings: np.ndarray, n_clusters: int = 256):
    """Train Topology Engine with KMeans clustering"""
    logger.info("=== Training Topology Engine ===")
    
    engine = TopologyEngine()
    results = engine.train_manifold(embeddings, n_clusters=n_clusters)
    
    if "error" in results:
        logger.error(f"Topology training failed: {results['error']}")
        return None
    
    # Save topology state
    engine.save_topology("data/topology.json")
    logger.info(f"Topology trained: {results['n_clusters']} clusters, silhouette={results.get('silhouette_score', 'N/A')}")
    
    return engine

def train_mycelial(embeddings: np.ndarray, model_path: str = "data/monolith_v13_finetuned.pth", batch_size: int = 256):
    """Train Mycelial Network by observing VQ-VAE indices"""
    logger.info("=== Training Mycelial Network ===")
    
    # Load VQ-VAE model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MonolithV13()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize Mycelial Network (fresh or load existing)
    config = MycelialConfig(
        num_heads=4,
        codebook_size=256,
        learning_rate=0.1,
        decay_rate=0.995,
        save_path="data/mycelial_state.npz"
    )
    mycelial = MycelialReasoning(config)
    
    # Process all embeddings through VQ-VAE and observe
    n_samples = len(embeddings)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    logger.info(f"Processing {n_samples} embeddings through VQ-VAE...")
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Building Mycelial Connections"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            batch = torch.tensor(embeddings[start:end], dtype=torch.float32).to(device)
            output = model(batch)
            indices = output['indices'].cpu().numpy()  # [B, 4]
            
            # Observe each sample (batch observe)
            for idx_row in indices:
                mycelial.observe(idx_row)
            
            # Periodic decay to maintain sparsity
            if (i + 1) % 100 == 0:
                mycelial.decay()
    
    # Final decay
    mycelial.decay()
    
    # Save state
    mycelial.save_state()
    
    # Stats
    stats = mycelial.get_stats()
    logger.info(f"Mycelial Network trained: {stats['n_nodes']} nodes, {stats['n_edges']} edges, {stats['total_observations']} observations")
    
    return mycelial

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Topology & Mycelial")
    parser.add_argument("--data", type=str, default="data/training_embeddings.npy")
    parser.add_argument("--model", type=str, default="data/monolith_v13_finetuned.pth")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--skip-topology", action="store_true")
    parser.add_argument("--skip-mycelial", action="store_true")
    args = parser.parse_args()
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.data}...")
    embeddings = np.load(args.data)
    logger.info(f"Loaded {len(embeddings)} embeddings.")
    
    # Train Topology
    if not args.skip_topology:
        train_topology(embeddings, n_clusters=args.clusters)
    
    # Train Mycelial
    if not args.skip_mycelial:
        train_mycelial(embeddings, model_path=args.model)
    
    logger.info("✅ Training Complete!")

if __name__ == "__main__":
    main()
