"""
Populate Mycelial Network and Generate Visualizations
Processes sample documents to build Hebbian connections, then visualizes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from core.reasoning.mycelial_reasoning import MycelialVQVAE
from core.memory.storage import LanceDBStorage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_and_visualize(num_samples=1000):
    """Process samples to build network, then visualize."""
    
    logger.info("="*80)
    logger.info("POPULATING MYCELIAL NETWORK")
    logger.info("="*80)
    
    # 1. Load models
    logger.info("\n1. Loading models...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    wrapper = MycelialVQVAE.load_default()
    storage = LanceDBStorage()
    
    # 2. Sample embeddings from database
    logger.info(f"\n2. Sampling {num_samples} embeddings from LanceDB...")
    table = storage.table
    sample_data = table.to_pandas().sample(n=min(num_samples, len(table.to_pandas())))
    
    # 3. Process embeddings
    logger.info("\n3. Processing embeddings (building Hebbian connections)...")
    for idx, row in sample_data.iterrows():
        vector = row['vector']
        indices = wrapper.encode(vector)
        wrapper.observe(indices)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"   Processed {idx + 1}/{num_samples}")
    
    # 4. Save state
    wrapper.save_state()
    logger.info("\n4. Mycelial state saved")
    
    # 5. Show stats
    stats = wrapper.get_network_stats()
    logger.info("\n5. Network Statistics:")
    logger.info(f"   Observations: {stats['total_observations']}")
    logger.info(f"   Active connections: {stats['active_connections']}")
    logger.info(f"   Density: {stats['density']:.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("NETWORK POPULATED!")
    logger.info("="*80)
    logger.info("\nNow run: python scripts/visualize_mycelial.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to process")
    args = parser.parse_args()
    
    populate_and_visualize(args.samples)
